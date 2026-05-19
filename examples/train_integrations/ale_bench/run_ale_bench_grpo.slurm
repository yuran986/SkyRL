#!/bin/bash
#SBATCH --job-name=ale_bench_grpo
#SBATCH --partition=compsci-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/yz1051/skyrl_logs/ale_bench/slurm-%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yingjie.zhang@duke.edu

set -euo pipefail

cd /home/users/yz1051/SkyRL

mkdir -p /home/users/yz1051/skyrl_logs/ale_bench
mkdir -p /usr/project/xtmp/yz1051/ckpts/ale_bench_3B
mkdir -p "$HOME/exports/ale_bench"

if [ ! -x .venv/bin/python ]; then
  echo "Missing .venv/bin/python. Build the SkyRL virtualenv before submitting this job." >&2
  echo "Example: uv venv --python 3.12 .venv && source .venv/bin/activate && uv sync --extra dev --extra fsdp" >&2
  exit 1
fi

source .venv/bin/activate

RUN_ID="${RUN_ID:-${SLURM_JOB_NAME:-ale_bench_grpo}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
ALE_BENCH_REPO="${ALE_BENCH_REPO:-/home/users/yz1051/ALE-Bench}"
DATA_DIR="${DATA_DIR:-/usr/project/xtmp/yz1051/data/ale_bench}"
ALE_BENCH_CONTAINER_BACKEND="${ALE_BENCH_CONTAINER_BACKEND:-apptainer}"
ALE_BENCH_APPTAINER_IMAGE_DIR="${ALE_BENCH_APPTAINER_IMAGE_DIR:-/usr/project/xtmp/yz1051/ale-bench-sif}"
ALE_BENCH_CACHE="${ALE_BENCH_CACHE:-/usr/project/xtmp/yz1051/ale-bench-cache}"
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$ALE_BENCH_APPTAINER_IMAGE_DIR/.apptainer-cache}"
SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-$APPTAINER_CACHEDIR}"
TMPDIR="$ALE_BENCH_APPTAINER_IMAGE_DIR/.tmp"

mkdir -p "$ALE_BENCH_CACHE" "$APPTAINER_CACHEDIR" "$TMPDIR"

if [ ! -d "$DATA_DIR" ]; then
  echo "Missing DATA_DIR: $DATA_DIR" >&2
  echo "Prepare it first with examples/train_integrations/ale_bench/prepare_dataset.py." >&2
  exit 1
fi

if [ ! -d "$ALE_BENCH_REPO/src" ]; then
  echo "Missing ALE_BENCH_REPO/src: $ALE_BENCH_REPO/src" >&2
  exit 1
fi

if [ "$ALE_BENCH_CONTAINER_BACKEND" = "apptainer" ] && [ ! -d "$ALE_BENCH_APPTAINER_IMAGE_DIR" ]; then
  echo "Missing ALE_BENCH_APPTAINER_IMAGE_DIR: $ALE_BENCH_APPTAINER_IMAGE_DIR" >&2
  echo "Create it first with:" >&2
  echo "  bash $ALE_BENCH_REPO/scripts/apptainer_pull_202301.sh yimjk/ale-bench $ALE_BENCH_APPTAINER_IMAGE_DIR" >&2
  exit 1
fi

echo "Started at $(date)"
echo "Host: $(hostname)"
echo "Run ID: $RUN_ID"
echo "Slurm job ID: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATA_DIR=$DATA_DIR"
echo "ALE_BENCH_REPO=$ALE_BENCH_REPO"
echo "ALE_BENCH_CONTAINER_BACKEND=$ALE_BENCH_CONTAINER_BACKEND"
echo "ALE_BENCH_APPTAINER_IMAGE_DIR=$ALE_BENCH_APPTAINER_IMAGE_DIR"
echo "ALE_BENCH_CACHE=$ALE_BENCH_CACHE"
echo "APPTAINER_CACHEDIR=$APPTAINER_CACHEDIR"
echo "TMPDIR=$TMPDIR"
nvidia-smi

PYTORCH_ALLOC_CONF=expandable_segments:True \
DATA_DIR="$DATA_DIR" \
ALE_BENCH_REPO="$ALE_BENCH_REPO" \
ALE_BENCH_CONTAINER_BACKEND="$ALE_BENCH_CONTAINER_BACKEND" \
ALE_BENCH_APPTAINER_IMAGE_DIR="$ALE_BENCH_APPTAINER_IMAGE_DIR" \
ALE_BENCH_CACHE="$ALE_BENCH_CACHE" \
APPTAINER_CACHEDIR="$APPTAINER_CACHEDIR" \
SINGULARITY_CACHEDIR="$SINGULARITY_CACHEDIR" \
TMPDIR="$TMPDIR" \
CKPT_PATH=/usr/project/xtmp/yz1051/ckpts/ale_bench_3B \
COLOCATE_ALL=false \
NUM_GPUS=2 \
INFERENCE_NUM_ENGINES=2 \
LOGGER=console \
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
MAX_TURNS=1 \
MAX_INPUT_LENGTH=24576 \
MAX_MODEL_LEN=32768 \
MAX_GENERATE_LENGTH=8192 \
MAX_ENV_WORKERS=4 \
N_SAMPLES_PER_PROMPT=4 \
TRAIN_BATCH_SIZE=4 \
POLICY_MINI_BATCH_SIZE=2 \
GPU_MEMORY_UTILIZATION=0.5 \
CKPT_INTERVAL=10 \
EVAL_INTERVAL=10 \
RUN_NAME=ale_bench_grpo \
RUN_ID="$RUN_ID" \
bash examples/train_integrations/ale_bench/run_ale_bench_grpo.sh \
  trainer.epochs=20 \
  trainer.eval_before_train=false \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.max_ckpts_to_keep=3 \
  trainer.log_path=$HOME/skyrl_logs/ale_bench

echo "Finished at $(date)"
