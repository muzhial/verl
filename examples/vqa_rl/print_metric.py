import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测评")
    parser.add_argument("--model_id", type=str, default="Qwen2.5-VL-7B-Instruct", help="model name")
    parser.add_argument("--dataset", type=str, default="", help="dataset name")

    args = parser.parse_args()

    datas = []
    with open(f"./output/{args.dataset}.{args.model_id}.output.jsonl") as fin:
        for line in fin:
            line = line.strip()
            data = json.loads(line)
            datas.append(data)

    print(f"model name: {args.model_id}")

    cnt = 0
    total_cnt = 0
    for data in datas:
        pred = data["answer"]
        ground_truth = data["ground_truth"]
        if pred == ground_truth:
            cnt += 1

        total_cnt += 1

    print(f"ACC: {cnt / total_cnt}")
