set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_sp2.sh <cuda_visible_devices> <save_path> <model_path> [other_configs...]"
    exit 1
fi

cuda_visible_devices=$1
save_path=$2
model_path=$3

# Parse the comma-separated string into an array
IFS=',' read -ra number_array <<< "$cuda_visible_devices"
# Get the length of the array
array_length=${#number_array[@]}

nproc_per_node=$array_length

# Shift the arguments so $@ refers to the rest
shift 3

CUDA_VISIBLE_DEVICES=$cuda_visible_devices torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=dataset/gsm8k/train.parquet \
    data.val_files=dataset/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size=4 \
    model.partial_pretrain=$model_path \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen-2.5 \
    trainer.logger=['console'] \
    trainer.total_training_steps=1 \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true
