set -x

# Colocated GRPO training+generation for ARG/ARC-AGI-3.
# Override defaults with environment variables, e.g.:
#   NUM_GPUS=1 LOGGER=console bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

: "${DATA_DIR:="$HOME/data/arc_agi3"}"
: "${CKPT_PATH:="$HOME/ckpts/arc_agi3_1.5B"}"
: "${NUM_GPUS:=1}"
: "${LOGGER:=wandb}"
: "${MODEL_PATH:="Qwen/Qwen2.5-1.5B-Instruct"}"
: "${INFERENCE_BACKEND:=vllm}"
: "${MAX_TURNS:=64}"
: "${N_SAMPLES_PER_PROMPT:=8}"
: "${TRAIN_BATCH_SIZE:=8}"
: "${POLICY_MINI_BATCH_SIZE:=8}"

uv run --isolated --extra fsdp --with arc-agi --with python-dotenv -m examples.train_integrations.arc_agi3.entrypoints.main_arc_agi3 \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.algorithm.grpo_norm_by_std=true \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.epochs=20 \
  trainer.eval_batch_size=4 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$POLICY_MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=1024 \
  generator.sampling_params.max_generate_length=128 \
  generator.sampling_params.temperature=0.7 \
  generator.sampling_params.top_p=0.95 \
  generator.sampling_params.stop='["</action>"]' \
  generator.eval_sampling_params.stop='["</action>"]' \
  generator.eval_sampling_params.max_generate_length=128 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  generator.max_turns=$MAX_TURNS \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  environment.env_class=arc_agi3 \
  trainer.logger="$LOGGER" \
  trainer.project_name="arc_agi3" \
  trainer.run_name="arc_agi3_latest" \
  trainer.resume_mode=latest \
  trainer.ckpt_path=$CKPT_PATH \
  trainer.dump_data_batch=true \
  "$@"

