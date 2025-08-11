set -x

# if [ "$#" -lt 3 ]; then
#     echo "Usage: run_sft.sh <cuda_visible_devices> <save_path> <model_path> [other_configs...]"
#     exit 1
# fi

export VERL_SFT_LOGGING_LEVEL=INFO

cuda_visible_devices=$1
save_path=${2:-"outputs/results/train/qwen2.5vl_3b-verl-sft"}
model_path=${3:-"/mnt/nas_data2/chenjn_workspace/datasets/HF/Qwen/Qwen/Qwen2.5-VL-3B-Instruct"}

# Parse the comma-separated string into an array
IFS=',' read -ra number_array <<< "$cuda_visible_devices"
# Get the length of the array
array_length=${#number_array[@]}

nproc_per_node=$array_length

# Shift the arguments so $@ refers to the rest
# shift 3

PROJECT_NAME=verl-sft-qwen2.5vl
EXPERIMENT_NAME=qwen2.5vl-3b


CUDA_VISIBLE_DEVICES=$cuda_visible_devices torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_vlsft_trainer \
    data.train_files=/mnt/nas_data2/chenjn_workspace/datasets/project/shengong-multimodality-agent/power_facility-climbing_falling/datas/annos/20250716-qwen25vl/train_bboxes_caption_qw_fmt.json \
    data.val_files=/mnt/nas_data2/chenjn_workspace/datasets/project/shengong-multimodality-agent/power_facility-climbing_falling/datas/annos/20250716-qwen25vl/train_bboxes_caption_qw_fmt.json \
    data.media_dir=/mnt/nas_data2/chenjn_workspace/datasets/project \
    optim.lr=1e-5 \
    data.micro_batch_size_per_gpu=2 \
    data.train_batch_size=4 \
    model.partial_pretrain=$model_path \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=['console'] \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null \
    trainer.profile=false \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true
