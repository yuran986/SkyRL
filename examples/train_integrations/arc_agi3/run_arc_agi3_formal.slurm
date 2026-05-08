#!/bin/bash
#SBATCH --job-name=arc_agi3_formal
#SBATCH --partition=compsci-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=/home/users/yz1051/skyrl_logs/arc_agi3/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yingjie.zhang@duke.edu

set -euo pipefail

cd /home/users/yz1051/SkyRL

mkdir -p /home/users/yz1051/skyrl_logs/arc_agi3
mkdir -p /usr/project/xtmp/yz1051/ckpts/arc_agi3_3B_formal

if [ ! -x .venv/bin/python ]; then
  echo "Missing .venv/bin/python. Build the SkyRL virtualenv before submitting this job." >&2
  exit 1
fi

source .venv/bin/activate

echo "Started at $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

PYTORCH_ALLOC_CONF=expandable_segments:True \
DATA_DIR=$HOME/data/arc_agi3 \
CKPT_PATH=/usr/project/xtmp/yz1051/ckpts/arc_agi3_3B_reward_v2 \
COLOCATE_ALL=false \
NUM_GPUS=2 \
INFERENCE_NUM_ENGINES=2 \
LOGGER=console \
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
MAX_TURNS=10 \
MAX_INPUT_LENGTH=16384 \
MAX_MODEL_LEN=18432 \
MAX_GENERATE_LENGTH=192 \
N_SAMPLES_PER_PROMPT=4 \
TRAIN_BATCH_SIZE=4 \
POLICY_MINI_BATCH_SIZE=2 \
CKPT_INTERVAL=10 \
EVAL_INTERVAL=10 \
RUN_NAME=arc_agi3_reward_v2 \
bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh \
  trainer.epochs=25 \
  trainer.eval_before_train=false \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.max_ckpts_to_keep=3 \
  trainer.log_path=$HOME/skyrl_logs/arc_agi3

echo "Finished at $(date)"
