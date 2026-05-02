# ARG/ARC-AGI-3 + SkyRL

This integration wraps the ARC-AGI-3 Toolkit as a `skyrl-gym` text environment so SkyRL can train with GRPO.

## Setup

From the repository root:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv sync --extra dev
uv pip install arc-agi python-dotenv
```

Create or edit `.env`:

```bash
OPERATION_MODE=OFFLINE
ARC_AGI3_ENVIRONMENTS_DIR=/home/users/yz1051/rlm/environment_files
```

## Prepare Data

```bash
set -a
source .env
set +a
python examples/train_integrations/arc_agi3/prepare_dataset.py \
  --output_dir $HOME/data/arc_agi3 \
  --task_ids ft09
```

The generated parquet files contain `prompt`, `env_class`, and per-episode extras such as `task_id`, `seed`, `operation_mode`, and `environments_dir`.

## Train

For a small smoke run:

```bash
DATA_DIR=$HOME/data/arc_agi3 \
NUM_GPUS=1 \
LOGGER=console \
MAX_TURNS=16 \
N_SAMPLES_PER_PROMPT=4 \
TRAIN_BATCH_SIZE=2 \
POLICY_MINI_BATCH_SIZE=2 \
bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh \
  trainer.epochs=1 \
  trainer.eval_before_train=false
```

The model should emit exactly one action per turn:

```text
<action>ACTION1</action>
<action>{"action":"ACTION6","x":32,"y":32}</action>
```

`ACTION6` requires `x` and `y` coordinates in `[0, 63]`.
