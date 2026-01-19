set -x

ulimit -n 65535

# export RAY_DEBUG_POST_MORTEM=1

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HYDRA_FULL_ERROR=1

# wandb config
# export WANDB_API_KEY=""  # Replace with your actual API key
# export WANDB_ENTITY="your_username"      # Optional: your wandb username or team name
# export WANDB_MODE="offline"              # Optional: use "offline" for offline mode

# verifier config
export SELF_VERIFIER_SERVER=127.0.0.1:8000
export SELF_VERIFIER_SERVER_NAME=Qwen2.5-VL-7B-Instruct

TRAIN_FILES="/home/ubuntu/practice/project/VQA/data/scienceqa_train_data.parquet"
VAL_FILES="/home/ubuntu/practice/project/VQA/data/scienceqa_test_data.parquet"

MODEL_PATH="/mnt/projs/model/Qwen2.5-VL-7B-Instruct"
OUTPUT_PATH="/mnt/projs/outputs/VQA-RL/exp2"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.prompt_key=prompt \
    data.image_key=images \
    data.train_batch_size=64 \
    data.max_prompt_length=6144 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    data.truncation='right' \
    data.return_raw_chat=True \
    data.shuffle=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    actor_rollout_ref.rollout.update_weights_bucket_megabytes=512 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.use_reward_loop=False \
    reward_model.reward_manager=custom \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name="VQA-RL" \
    trainer.experiment_name="grpo" \
    trainer.default_local_dir="$OUTPUT_PATH" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.resume_mode="disable" \
    trainer.total_epochs=2 $@
