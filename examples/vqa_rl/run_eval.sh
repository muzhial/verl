set -x

model_id=$1

export MODEL_SERVER=127.0.0.1:8000
export MODEL_PATH=RobustVQA-RL-ckpt-step194-ckpt # to val model
# export MODEL_PATH=Qwen2.5-VL-7B-Instruct

dataset="scienceqa_test_data.jsonl"

python3 get_model_response.py \
    --model_id $model_id \
    --dataset $dataset

python3 print_metric.py \
    --model_id $model_id \
    --dataset $dataset
