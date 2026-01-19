CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
    --model /mnt/projs/model/Qwen2.5-VL-7B-Instruct \
    --served-model-name Qwen2.5-VL-7B-Instruct \
    --max-model-len=10000 \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.9