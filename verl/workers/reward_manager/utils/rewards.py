import random
import re
import concurrent.futures

from tqdm import tqdm

from .apis import request_verifier


verification_prompt_template = """You are required to act as a logic checker. Your task is to analyze the provided Reasoning and determine the correct option for the Question based **STRICTLY AND ONLY** on the information presented within that Reasoning.

Question and Options:
{question_and_options}

Reasoning to Analyze:
{cot}

---
Based on the provided Reasoning alone, what is the final answer? Your output MUST strictly follow the required format.

**STRICT OUTPUT FORMAT:**
<answer>
[The single letter corresponding to your final choice (A, B, C, or D). NOTHING ELSE.]
</answer>""".strip()


def shuffle_options_and_answer(original_options, correct_answer_letter):
    """
    Randomly shuffle the options list, ensuring the shuffled order differs
    as much as possible from the original order.
    Returns the shuffled options list and the corresponding new correct answer letter.
    """
    # Create indexed options list [(0, A), (1, B), (2, C), (3, D)]
    indexed_options = list(enumerate(original_options))

    # Track the best shuffle result
    best_shuffle = None
    best_difference_score = -1

    N = len(original_options)

    # Try multiple shuffles and select the one with the greatest difference
    for _ in range(100):
        # Randomly shuffle
        shuffled = indexed_options.copy()
        random.shuffle(shuffled)

        # Calculate the degree of difference from original positions
        # Score 1 for each element not in its original position, 0 if in original position
        difference_score = sum(1 for i, (orig_idx, _) in enumerate(shuffled) if orig_idx != i)

        # Update best result if this shuffle has greater difference
        if difference_score > best_difference_score:
            best_difference_score = difference_score
            best_shuffle = shuffled.copy()

        # Stop if maximum difference is reached (all elements displaced)
        if difference_score == N:
            break

    # Unpack best shuffle result, keeping only option contents
    shuffled_options = [option for _, option in best_shuffle]
    # Create mapping from position to letter

    # position_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    position_to_letter = {i: chr(ord("A") + i) for i in range(N)}

    # Find the original position of the correct answer
    original_position = ord(correct_answer_letter) - ord("A")  # A->0, B->1, C->2, D->3

    # Find the position of the original correct answer content in the shuffled list
    original_correct_content = original_options[original_position]  # Content of the original correct answer
    new_position = shuffled_options.index(
        original_correct_content
    )  # Position of original correct answer in shuffled list

    # Get the new answer letter
    new_answer_letter = position_to_letter[new_position]

    return shuffled_options, new_answer_letter


def create_vqa_prompt(question, choices):
    # Dynamically generate list in "A. option content" format using list comprehension
    options_str = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])
    # Concatenate to form the final prompt
    return f"Question: {question}\nOptions:\n{options_str}"


def make_prompt(question, cot, choices, ground_truth):
    # Shuffle options
    shuffled_options, new_answer = shuffle_options_and_answer(choices, ground_truth)
    # Regenerate prompt
    text = create_vqa_prompt(question, shuffled_options)
    # Replace placeholder here
    N = len(choices)
    if N == 2:
        sub_text = "A or B"
    elif N == 3:
        sub_text = "A, B or C"
    elif N == 4:
        sub_text = "A, B, C, or D"
    elif N == 5:
        sub_text = "A, B, C, D, or E"
    else:
        raise ValueError("N error")

    _verification_prompt_template = verification_prompt_template.replace("A, B, C, or D", sub_text)
    prompt = _verification_prompt_template.format(question_and_options=text, cot=cot)
    return prompt, shuffled_options, new_answer


def get_format_reward(response):
    response = response.strip()
    """
    ^	     : Start of the string
    <think>  : Literal <think> tag
    .*?	     : Any characters (non-greedy)
    </think> : Closing think tag
    \s*	     : allows any amount of whitespace between </think> and <answer>
    <answer> : Literal <answer> tag
    .*?	     : Any characters (non-greedy)
    </answer>: Closing answer tag
    $        : End of the string
    """
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    if not re.match(pattern, response, re.DOTALL):
        return -1.0
    if len(re.findall(r"<think>", response)) != 1 or len(re.findall(r"<answer>", response)) != 1:
        return -1.0

    return 0


def extract_answer(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    # 1
    """
    <answer>  : Opening tag
    (.*?)     : Capture anything (non-greedy)
    </answer> : Closing tag
    DOTALL    : . matches newlines
    IGNORECASE: <ANSWER> also works
    """
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    # 2 Missing complete answer tag(<answer> and </answer>) try to extract left string after processing
    # Removes reasoning content entirely.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove leftover tags
    text = text.replace("<think>", "").replace("</think>", "")
    text = text.replace("<answer>", "").replace("</answer>", "")

    return text.strip()


def cot_length_reward(length, l_min, l_opt, l_max):
    if length < l_min:
        return -1
    elif length < l_opt:
        return -1 + (length - l_min) / (l_opt - l_min)
    elif length <= l_max:
        return 0
    else:
        return -1


def get_token_num(text, tokenizer):
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    return len(tokens)


def get_length_penalty(items, tokenizer):
    length_rewards = []
    cot_lengths = []
    for item in items:
        cot = item["cot"]
        cot_len = get_token_num(cot, tokenizer)
        # at least greater than 50, best is 100
        reward = cot_length_reward(cot_len, 50, 100, 200)
        length_rewards.append(reward)
        cot_lengths.append(cot_len)

    answer_lengths = [get_token_num(item["response"], tokenizer) for item in items]

    return answer_lengths, cot_lengths, length_rewards


def process_reward(item):
    question = item["question"]
    cot = item["cot"]
    choices = item["choices"]
    ground_truth = item["ground_truth"]
    response = item["response"]

    N = len(choices)
    g_truth = [chr(ord("A") + i) for i in range(N)]

    ground_truth_text = choices[ord(ground_truth) - ord("A")]  # The string of the correct answer
    if response in g_truth:
        response_text = choices[ord(response) - ord("A")]  # The string of the first prediction
    else:
        response_text = "T"

    prompt, shuffled_options, new_ground_truth = make_prompt(question, cot, choices, ground_truth)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    result = request_verifier(messages)
    shuffled_answer = extract_answer(result)
    if shuffled_answer in g_truth:
        shuffled_answer_text = shuffled_options[
            ord(shuffled_answer) - ord("A")
        ]  # The string of the shuffled prediction
    else:
        shuffled_answer_text = "F"
    # Calculate reward
    # 1. Accuracy reward
    if response == ground_truth:
        acc_reward = 1.0
    else:
        acc_reward = 0.0

    # Consistency reward
    # 1. Both attempts are correct
    if acc_reward == 1.0 and shuffled_answer == new_ground_truth:
        consistency_reward = 1.0
    # First attempt correct, but incorrect after shuffling
    elif acc_reward == 1.0 and shuffled_answer != new_ground_truth:
        consistency_reward = 0.5
    # First attempt incorrect, but text-only version got it right
    elif acc_reward == 0.0 and shuffled_answer == new_ground_truth:
        consistency_reward = 0.5  # Text-only corrected the visual error
    # Both attempts incorrect, but answers are consistent
    elif response_text == shuffled_answer_text:
        consistency_reward = 0.1
    else:
        consistency_reward = 0.0

    return acc_reward, consistency_reward


def get_rewards(items):
    # concurrent execution
    max_workers = 128
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"Start calculating rewards, number of tasks {len(items)}, number of concurrent workers {max_workers}")
        for item in items:
            futures.append(executor.submit(process_reward, item))

        futures = [future.result() for future in tqdm(futures)]

    acc_rewards, consistency_rewards = [], []
    for acc_reward, consistency_reward in futures:
        acc_rewards.append(acc_reward)
        consistency_rewards.append(consistency_reward)

    return acc_rewards, consistency_rewards
