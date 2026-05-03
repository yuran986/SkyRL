# ARG/ARC-AGI-3 + SkyRL

This integration wraps the ARC-AGI-3 Toolkit as a `skyrl-gym` text environment so SkyRL can train with GRPO.

## Start Here

Use this file for the day-to-day workflow. The full environment walkthrough is in
`/home/users/yz1051/SkyRL/ARC_AGI3_ENV_SETUP.md`; the actual launcher is
`examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh`.

The normal order is:

1. Prepare `$HOME/data/arc_agi3/{train,validation}.parquet`.
2. Run `run_arc_agi3_grpo.sh` with `CKPT_PATH` on `/usr/project/xtmp`.
3. Inspect metrics, console examples, and structured rollouts under `$EXPORT_PATH`.

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
Regenerate these parquet files after changing the prompt or action protocol; existing files keep the old prompt text.
By default, observations include a full frame at initialization, compact diff summaries every turn, local changed patches around the diff bbox, and a full-frame refresh every 8 turns.

## Train

For a small smoke run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
DATA_DIR=$HOME/data/arc_agi3 \
CKPT_PATH=/usr/project/xtmp/yz1051/ckpts/arc_agi3_3B_2gpu \
NUM_GPUS=2 \
LOGGER=console \
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
MAX_TURNS=10 \
MAX_INPUT_LENGTH=6144 \
MAX_GENERATE_LENGTH=128 \
N_SAMPLES_PER_PROMPT=4 \
TRAIN_BATCH_SIZE=4 \
POLICY_MINI_BATCH_SIZE=2 \
CKPT_INTERVAL=1 \
EVAL_INTERVAL=-1 \
bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh \
  trainer.epochs=1 \
  trainer.eval_before_train=false \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.log_path=$HOME/skyrl_logs/arc_agi3
```

Infrastructure logs are written under `trainer.log_path`; with the command above, check
`$HOME/skyrl_logs/arc_agi3/infra-*.log` and `router-*.log`. If the smoke run runs out of memory,
first reduce `TRAIN_BATCH_SIZE`, `POLICY_MINI_BATCH_SIZE`, and `N_SAMPLES_PER_PROMPT`.
By default, `run_arc_agi3_grpo.sh` creates a unique export directory for each run:
`$HOME/exports/arc_agi3/${RUN_NAME}_YYYYmmdd_HHMMSS`. The script prints the resolved
`ARC-AGI-3 export path` at startup. Structured training rollouts are written to
`$EXPORT_PATH/dumped_rollouts/global_step_*_rollouts.jsonl`.
Each JSONL row contains one trajectory with per-turn action, observation, reward, reward components,
diff stats, and environment state.

Useful rollout checks:

```bash
EXPORT_PATH=$HOME/exports/arc_agi3/arc_agi3_latest_YYYYmmdd_HHMMSS

python examples/train_integrations/arc_agi3/visualize_rollouts.py \
  "$EXPORT_PATH" \
  -o "$EXPORT_PATH/rollout_viewer.html"

jq -r '.steps[].model_output' "$EXPORT_PATH/dumped_rollouts/global_step_1_rollouts.jsonl" \
  | sort | uniq -c | sort -nr

jq '.steps[] | {turn, reward, reward_components: .metadata.reward_components, diff_stats: .metadata.diff_stats}' \
  "$EXPORT_PATH/dumped_rollouts/global_step_1_rollouts.jsonl"
```

Open `rollout_viewer.html` in a browser to inspect each trajectory. The viewer shows rollout-derived
training curves by `global_step`, summary metrics, trajectory filters, per-turn `<think>`,
`<action>`, reward components, diff stats, observations, and raw step JSON.

## Metric Loggers

`LOGGER` is passed to `trainer.logger` and controls the metric tracker. It is separate from
`trainer.log_path`, which stores infrastructure logs such as `infra-*.log` and `router-*.log`.
For smoke tests, keep `LOGGER=console`; for longer runs, prefer `tensorboard`, `wandb`, or
`swanlab`.

```bash
LOGGER=console bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh

TENSORBOARD_DIR=$HOME/skyrl_logs/arc_agi3/tensorboard \
LOGGER=tensorboard \
bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh

WANDB_API_KEY=... LOGGER=wandb bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh
SWANLAB_API_KEY=... LOGGER=swanlab bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh
MLFLOW_TRACKING_URI=http://host:5000 LOGGER=mlflow bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh
```

`console` needs no setup. `tensorboard` writes event files to `TENSORBOARD_DIR` or
`tensorboard_log` if unset. `wandb` requires `WANDB_API_KEY`; `swanlab` can use
`SWANLAB_API_KEY`, `SWANLAB_LOG_DIR`, and `SWANLAB_MODE`; `mlflow` uses
`MLFLOW_TRACKING_URI` when set. These trackers are for normal training curves such as reward,
KL, entropy, loss, response length, timing, and eval pass rate. Use the static rollout viewer to
debug behavior and reward assignment inside individual trajectories.

The model should emit brief reasoning followed by exactly one executable action per turn:

```text
<think>Briefly state what changed or what to try next.</think>
<action>ACTION1</action>

<think>Click the center to test whether the selected region changes.</think>
<action>{"action":"ACTION6","x":32,"y":32}</action>
```

`ACTION6` requires `x` and `y` coordinates in `[0, 63]`. The environment executes only the
last `<action>...</action>` block; keep `<think>` short so rollout context stays within
`MAX_INPUT_LENGTH`.

## Debug One Rollout

Run a local rollout without starting SkyRL training:

```bash
PYTHONPATH=$PWD/skyrl-gym:$PWD \
python examples/train_integrations/arc_agi3/debug_rollout.py \
  --task_id ft09 \
  --action '<think>Try a simple action.</think><action>ACTION1</action>' \
  --action '<think>Try a center click.</think><action>{"action":"ACTION6","x":32,"y":32}</action>'
```

The script prints JSON records for `init` and each `step`, including parsed action metadata, reward components, diff stats, observations, and metrics.
