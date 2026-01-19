import argparse
import base64
import json

import numpy as np
import pandas as pd


def get_prompt_template(N):
    letters = [chr(ord("A") + i) for i in range(N)]
    prompt_template = """Analyze the provided image and answer the following multiple-choice question.

Your task is to first generate a step-by-step reasoning process, and then provide only the final chosen letter (A, B, C, or D).

**STRICT OUTPUT FORMAT:**
You must strictly adhere to the following structure:
<think>
[Your comprehensive, step-by-step reasoning process here.]
</think>
<answer>
[The single letter corresponding to your final choice (A, B, C, or D). NOTHING ELSE.]
</answer>

{text}""".strip()

    # 2<=N<=5
    if N == 2:
        text = "A or B"
    elif N == 3:
        text = "A, B or C"
    elif N == 4:
        text = "A, B, C, or D"
    elif N == 5:
        text = "A, B, C, D, or E"
    else:
        raise ValueError("N error")

    return prompt_template.replace("A, B, C, or D", text)


def get_non_thinking_prompt_template(N):
    letters = [chr(ord("A") + i) for i in range(N)]
    prompt_template = """Analyze the provided image and answer the following multiple-choice question.

Your task is to provide only the final chosen letter (A, B, C, or D).

**STRICT OUTPUT FORMAT:**
You must strictly adhere to the following structure:
<answer>
[The single letter corresponding to your final choice (A, B, C, or D). NOTHING ELSE.]
</answer>

{text}""".strip()
    # 2<=N<=5
    if N == 2:
        text = "A or B"
    elif N == 3:
        text = "A, B or C"
    elif N == 4:
        text = "A, B, C, or D"
    elif N == 5:
        text = "A, B, C, D, or E"
    else:
        raise ValueError("N error")

    return prompt_template.replace("A, B, C, or D", text)


def create_vqa_prompt(question, choices):
    options_str = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])

    return f"Question: {question}\nOptions:\n{options_str}"


def preproc_data(input_path, output_path):
    df = pd.read_parquet(input_path)
    datas = df.to_dict(orient="records")
    filter_datas = []
    all_num = []
    for data in datas:
        choices = data["choices"].tolist()
        if data["image"] is None or len(choices) < 2:
            continue
        filter_datas.append(data)
        all_num.append(len(choices))

    print(min(all_num), max(all_num))

    all_data = []
    for data in filter_datas:
        choices = data["choices"].tolist()
        question = data["question"]
        ground_truth = chr(ord("A") + data["answer"])

        text = create_vqa_prompt(question, choices)
        prompt_template = get_prompt_template(len(choices))
        text_prompt = prompt_template.format(text=text)

        content = "<image>" + text_prompt

        images = np.array([data["image"]], dtype=object)
        item = {
            "images": images,
            "prompt": [{"role": "user", "content": content}],
            "question": question,
            "ground_truth": ground_truth,
            "choices": choices,
            "data_source": "science_qa",
        }

        all_data.append(item)

    print(f"filter data: {len(all_data)}")
    df = pd.DataFrame(all_data)
    df.to_parquet(output_path, index=False)


def generate_test_data(input_path, output_path):
    df = pd.read_parquet(input_path)
    datas = df.to_dict(orient="records")
    filter_datas = []
    all_num = []
    for data in datas:
        if data["image"] is None or len(data["choices"]) < 2:
            continue
        filter_datas.append(data)
        all_num.append(len(data["choices"]))

    print(min(all_num), max(all_num))
    all_data = []
    mime_type = "image/jpeg"
    for data in filter_datas:
        choices = data["choices"].tolist()
        question = data["question"]
        ground_truth = chr(ord("A") + data["answer"])

        text = create_vqa_prompt(question, choices)
        prompt_template = get_prompt_template(len(choices))
        text_prompt = prompt_template.format(text=text)
        non_think_prompt_template = get_non_thinking_prompt_template(len(choices))
        non_thinking_text_prompt = non_think_prompt_template.format(text=text)
        base64_encoded_bytes = base64.b64encode(data["image"]["bytes"])
        base64_image = base64_encoded_bytes.decode("utf-8")
        content = [
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
            {"type": "text", "text": text_prompt},
        ]
        non_content = [
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
            {"type": "text", "text": non_thinking_text_prompt},
        ]
        item = {
            "prompt": [{"role": "user", "content": content}],
            "non_thinking_prompt": [{"role": "user", "content": non_content}],
            "question": question,
            "ground_truth": ground_truth,
            "choices": choices,
            "data_source": "science_qa",
        }

        all_data.append(item)

    print(f"filter data: {len(all_data)}")
    with open(output_path, "w") as fout:
        for item in all_data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ScienceQA data for VQA training and evaluation.")
    parser.add_argument(
        "--source_dir",
        type=str,
        default="ScienceQA",
        help="Directory containing source parquet files (default: ScienceQA)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data",
        help="Directory to save output files (default: ../data)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        choices=["train", "test"],
        default=["train", "test"],
        help="Data splits to process (default: train test)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["preproc", "generate_test", "all"],
        default="all",
        help="Processing mode: preproc (parquet), generate_test (jsonl), or all (default: all)",
    )

    args = parser.parse_args()

    for split in args.splits:
        input_path = f"{args.source_dir}/{split}.parquet"
        parquet_output = f"{args.output_dir}/scienceqa_{split}_data.parquet"
        jsonl_output = f"{args.output_dir}/scienceqa_{split}_data.jsonl"

        if args.mode in ["preproc", "all"]:
            print(f"[preproc] {input_path} -> {parquet_output}")
            preproc_data(input_path, parquet_output)

        if args.mode in ["generate_test", "all"]:
            print(f"[generate_test] {input_path} -> {jsonl_output}")
            generate_test_data(input_path, jsonl_output)
