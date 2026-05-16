set -x

# Colocated GRPO training+generation for ALE-Bench.
# Override defaults with environment variables, e.g.:
#   NUM_GPUS=1 LOGGER=console bash examples/train_integrations/ale_bench/run_ale_bench_grpo.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

: "${DATA_DIR:="$HOME/data/ale_bench"}"
: "${CKPT_PATH:="/usr/project/xtmp/yz1051/ckpts/ale_bench_3B"}"
: "${NUM_GPUS:=1}"
: "${LOGGER:=wandb}"
: "${MODEL_PATH:="Qwen/Qwen2.5-3B-Instruct"}"
: "${INFERENCE_BACKEND:=vllm}"
: "${MAX_TURNS:=1}"
: "${N_SAMPLES_PER_PROMPT:=4}"
: "${TRAIN_BATCH_SIZE:=4}"
: "${POLICY_MINI_BATCH_SIZE:=4}"
: "${MAX_INPUT_LENGTH:=24576}"
: "${MAX_GENERATE_LENGTH:=8192}"
: "${GPU_MEMORY_UTILIZATION:=0.5}"
: "${CKPT_INTERVAL:=20}"
: "${EVAL_INTERVAL:=50}"
: "${EVAL_BEFORE_TRAIN:=false}"
: "${RUN_NAME:=ale_bench_latest}"
: "${RUN_ID:="${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"}"
: "${EXPORT_ROOT:="$HOME/exports/ale_bench"}"
: "${EXPORT_PATH:="$EXPORT_ROOT/$RUN_ID"}"
: "${MAX_MODEL_LEN:=32768}"
: "${MAX_ENV_WORKERS:=4}"
: "${PYTHON_BIN:="$REPO_ROOT/.venv/bin/python"}"
: "${COLOCATE_ALL:=true}"
: "${INFERENCE_NUM_ENGINES:="$NUM_GPUS"}"
: "${ALE_BENCH_REPO:="/home/users/yz1051/ALE-Bench"}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found: $PYTHON_BIN"
  echo "Create the local environment first: uv venv --python 3.12 .venv && source .venv/bin/activate && uv sync --extra dev --extra fsdp && uv pip install -e $ALE_BENCH_REPO python-dotenv datasets"
  exit 1
fi

export ALE_BENCH_REPO
if [ -d "$ALE_BENCH_REPO/src" ]; then
  export PYTHONPATH="$ALE_BENCH_REPO/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi
export SKYRL_LOG_RUN_ID="${SKYRL_LOG_RUN_ID:-$RUN_ID}"

echo "ALE-Bench export path: $EXPORT_PATH"
echo "ALE-Bench log run id: $SKYRL_LOG_RUN_ID"
echo "ALE-Bench python: $PYTHON_BIN"
echo "ALE-Bench repo: $ALE_BENCH_REPO"

"$PYTHON_BIN" -m examples.train_integrations.ale_bench.entrypoints.main_ale_bench \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.algorithm.grpo_norm_by_std=true \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=$COLOCATE_ALL \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$INFERENCE_NUM_ENGINES \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  trainer.epochs=20 \
  trainer.eval_batch_size=2 \
  trainer.eval_before_train=$EVAL_BEFORE_TRAIN \
  trainer.eval_interval=$EVAL_INTERVAL \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$POLICY_MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=$CKPT_INTERVAL \
  trainer.max_prompt_length=$MAX_INPUT_LENGTH \
  generator.max_input_length=$MAX_INPUT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.eval_sampling_params.temperature=0 \
  generator.eval_sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  generator.max_turns=$MAX_TURNS \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  environment.env_class=ale_bench \
  environment.skyrl_gym.max_env_workers=$MAX_ENV_WORKERS \
  trainer.logger="$LOGGER" \
  trainer.project_name="ale_bench" \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=null \
  trainer.ckpt_path=$CKPT_PATH \
  trainer.export_path="$EXPORT_PATH" \
  trainer.dump_data_batch=true \
  trainer.dump_rollout_logs=true \
  "$@"
