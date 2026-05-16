#!/bin/bash
#SBATCH --job-name=arc_agi3_qwen3_8b_reward_v2
#SBATCH --partition=compsci-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/home/users/yz1051/skyrl_logs/arc_agi3/slurm-%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yingjie.zhang@duke.edu

set -euo pipefail

cd /home/users/yz1051/SkyRL

mkdir -p /home/users/yz1051/skyrl_logs/arc_agi3
mkdir -p /usr/project/xtmp/yz1051/ckpts/arc_agi3_qwen3_8B_structured_diff

if [ ! -x .venv/bin/python ]; then
  echo "Missing .venv/bin/python. Build the SkyRL virtualenv before submitting this job." >&2
  exit 1
fi

source .venv/bin/activate

RUN_ID="${RUN_ID:-${SLURM_JOB_NAME:-arc_agi3_qwen3_8b_reward_v2}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
DATA_DIR="${DATA_DIR:-$HOME/data/arc_agi3_qwen3_8b_structured_diff}"

echo "Started at $(date)"
echo "Host: $(hostname)"
echo "Run ID: $RUN_ID"
echo "Slurm job ID: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATA_DIR=$DATA_DIR"
nvidia-smi

echo "Preparing ARC-AGI3 parquet with structured diff observation settings..."
ARC_AGI3_FRAME_OBSERVATION_MODE=initial_full_then_diff \
ARC_AGI3_FULL_FRAME_INTERVAL=0 \
ARC_AGI3_PATCH_RADIUS=4 \
ARC_AGI3_MAX_DIFF_EXAMPLES=32 \
ARC_AGI3_INVALID_ACTION_REWARD=-0.1 \
ARC_AGI3_LEVEL_REWARD=3.0 \
ARC_AGI3_DONE_REWARD=0.0 \
ARC_AGI3_MEANINGFUL_DIFF_REWARD=0.005 \
ARC_AGI3_MIN_MEANINGFUL_DIFF_CHANGES=1 \
ARC_AGI3_MAX_MEANINGFUL_DIFF_CHANGES=512 \
ARC_AGI3_REPEAT_CLICK_PENALTY=-0.02 \
ARC_AGI3_REPEAT_CLICK_RADIUS=2 \
python examples/train_integrations/arc_agi3/prepare_dataset.py \
  --output_dir "$DATA_DIR" \
  --task_ids ft09 \
  --train_size 32 \
  --val_size 8 \
  --max_steps 10

PYTORCH_ALLOC_CONF=expandable_segments:True \
DATA_DIR="$DATA_DIR" \
CKPT_PATH=/usr/project/xtmp/yz1051/ckpts/arc_agi3_qwen3_8B_structured_diff \
COLOCATE_ALL=false \
NUM_GPUS=3 \
INFERENCE_NUM_ENGINES=1 \
LOGGER=console \
MODEL_PATH=Qwen/Qwen3-8B \
MAX_TURNS=10 \
MAX_INPUT_LENGTH=8192 \
MAX_MODEL_LEN=12288 \
MAX_GENERATE_LENGTH=128 \
GPU_MEMORY_UTILIZATION=0.55 \
MAX_ENV_WORKERS=8 \
N_SAMPLES_PER_PROMPT=2 \
TRAIN_BATCH_SIZE=2 \
POLICY_MINI_BATCH_SIZE=1 \
CKPT_INTERVAL=10 \
EVAL_INTERVAL=10 \
RUN_NAME=arc_agi3_qwen3_8b_structured_diff \
RUN_ID="$RUN_ID" \
bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh \
  trainer.epochs=25 \
  trainer.eval_before_train=false \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.max_ckpts_to_keep=2 \
  trainer.log_path=$HOME/skyrl_logs/arc_agi3

echo "Finished at $(date)"
