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
ARC_AGI3_FRAME_OBSERVATION_MODE=initial_full_then_diff
ARC_AGI3_FULL_FRAME_INTERVAL=8
ARC_AGI3_PATCH_RADIUS=4
ARC_AGI3_MAX_DIFF_EXAMPLES=32
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
By default, observations include a full frame at initialization, compact diff summaries every turn, local changed patches around the diff bbox, and a full-frame refresh every 8 turns.

## Train

For a small smoke run:

```bash
DATA_DIR=$HOME/data/arc_agi3 \
NUM_GPUS=1 \
LOGGER=console \
MAX_TURNS=8 \
MAX_INPUT_LENGTH=8192 \
N_SAMPLES_PER_PROMPT=5 \
TRAIN_BATCH_SIZE=8 \
POLICY_MINI_BATCH_SIZE=8 \
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

## Debug One Rollout

Run a local rollout without starting SkyRL training:

```bash
PYTHONPATH=$PWD/skyrl-gym:$PWD \
python examples/train_integrations/arc_agi3/debug_rollout.py \
  --task_id ft09 \
  --action '<action>ACTION1</action>' \
  --action '<action>{"action":"ACTION6","x":32,"y":32}</action>'
```

The script prints JSON records for `init` and each `step`, including parsed action metadata, reward components, diff stats, observations, and metrics.
