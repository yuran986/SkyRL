# ARG/ARC-AGI-3 + SkyRL

This integration wraps the ARC-AGI-3 Toolkit as a `skyrl-gym` text environment so SkyRL can train with GRPO.

## Start Here

Use this file for the day-to-day workflow. The full environment walkthrough is in
`/home/users/yz1051/SkyRL/ARC_AGI3_ENV_SETUP.md`; the actual launcher is
`examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh`. A Chinese parameter guide is in
`examples/train_integrations/arc_agi3/RUN_ARC_AGI3_GRPO_PARAMS_ZH.md`. Training observations
and reward-setting history are tracked in
`examples/train_integrations/arc_agi3/TRAINING_NOTES_ZH.md`. Rollout viewer usage is documented in
`examples/train_integrations/arc_agi3/ROLLOUT_VIEWER_ZH.md`.

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

The launcher uses the repository-local `.venv/bin/python` by default. Override `PYTHON_BIN`
only if you intentionally want to run with a different environment.

Create or edit `.env`:

```bash
OPERATION_MODE=OFFLINE
ARC_AGI3_ENVIRONMENTS_DIR=/home/users/yz1051/rlm/environment_files
ARC_AGI3_FRAME_OBSERVATION_MODE=initial_full_then_diff
ARC_AGI3_FULL_FRAME_INTERVAL=8
ARC_AGI3_PATCH_RADIUS=4
ARC_AGI3_MAX_DIFF_EXAMPLES=32
ARC_AGI3_INVALID_ACTION_REWARD=-0.1
ARC_AGI3_LEVEL_REWARD=3.0
ARC_AGI3_DONE_REWARD=0.0
ARC_AGI3_MEANINGFUL_DIFF_REWARD=0.005
ARC_AGI3_REPEAT_CLICK_PENALTY=-0.02
ARC_AGI3_REPEAT_CLICK_RADIUS=2
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
By default, observations include a color legend and full frame at initialization, compact diff summaries every turn, local changed patches around the diff bbox, and a full-frame refresh every 8 turns.
Reward settings are also written into the parquet rows. Regenerate the data after changing
`ARC_AGI3_*_REWARD` variables so the run artifact records the exact reward used.

## Train Smoke Run

Recommended first run on one 4-GPU node, with 2 GPUs for FSDP training and 2 GPUs for vLLM:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
DATA_DIR=$HOME/data/arc_agi3 \
CKPT_PATH=/usr/project/xtmp/yz1051/ckpts/arc_agi3_3B_4gpu_split \
COLOCATE_ALL=false \
NUM_GPUS=2 \
INFERENCE_NUM_ENGINES=2 \
LOGGER=console \
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
MAX_TURNS=10 \
MAX_INPUT_LENGTH=6144 \
MAX_GENERATE_LENGTH=128 \
N_SAMPLES_PER_PROMPT=2 \
TRAIN_BATCH_SIZE=2 \
POLICY_MINI_BATCH_SIZE=1 \
CKPT_INTERVAL=1 \
EVAL_INTERVAL=-1 \
bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh \
  trainer.epochs=1 \
  trainer.eval_before_train=false \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.log_path=$HOME/skyrl_logs/arc_agi3
```

This is intentionally conservative. After it completes multiple global steps and writes
checkpoints, increase one knob at a time: `MAX_INPUT_LENGTH=8192`, then
`N_SAMPLES_PER_PROMPT=5`, then `TRAIN_BATCH_SIZE=8`, then `POLICY_MINI_BATCH_SIZE=4`.
Keep `trainer.micro_train_batch_size_per_gpu=1` until memory is clearly stable.
Use `COLOCATE_ALL=true` only on nodes where colocated CUDA IPC weight sync is known to work.
On nodes that reject CUDA IPC with `pidfd_getfd: Operation not permitted`, keep
`COLOCATE_ALL=false` and split the available GPUs between training and inference.

By default, `run_arc_agi3_grpo.sh` creates a unique export directory for each run:
`$HOME/exports/arc_agi3/${RUN_NAME}_YYYYmmdd_HHMMSS`. The script prints the resolved
`ARC-AGI-3 export path` and `ARC-AGI-3 log run id` at startup.
Infrastructure logs are written under `trainer.log_path` using the same run id:
`infra-${RUN_ID}.log` and `router-${RUN_ID}.log`. If `SKYRL_LOG_RUN_ID` is set, that value
is used for infrastructure log names; otherwise `run_arc_agi3_grpo.sh` sets it to `RUN_ID`.
The formal Slurm script also uses this id in `slurm-${RUN_ID}.out`, so stdout, infra,
router, and export artifacts can be matched by the same string. If the smoke run runs out
of memory, first reduce `TRAIN_BATCH_SIZE`, `POLICY_MINI_BATCH_SIZE`, and
`N_SAMPLES_PER_PROMPT`. Structured training rollouts are written to
`$EXPORT_PATH/dumped_rollouts/global_step_*_rollouts.jsonl`.
Each JSONL row contains one trajectory with the initial prompt/messages seen by the model,
per-turn action, observation, reward, reward components, diff stats, and environment state.

## Full Training Run

After the smoke run reaches multiple global steps, use this larger 4-GPU split run. It keeps
thinking enabled and raises the conversation budget enough for 10-turn trajectories. The
previous 8192-token setting was too tight: recent rollouts stopped around 7-8 turns with
`stop_reason=length`.

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
DATA_DIR=$HOME/data/arc_agi3 \
CKPT_PATH=/usr/project/xtmp/yz1051/ckpts/arc_agi3_3B_formal \
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
CKPT_INTERVAL=20 \
EVAL_INTERVAL=20 \
RUN_NAME=arc_agi3_formal \
bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh \
  trainer.epochs=50 \
  trainer.eval_before_train=false \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.max_ckpts_to_keep=3 \
  trainer.log_path=$HOME/skyrl_logs/arc_agi3
```

If `batch_padded_seq_len` approaches 16384 or rollouts still end before 10 turns with
`stop_reason=length`, try the 32k maximum-context profile below before changing reward
or model settings.

For a maximum-context experiment with Qwen2.5-3B-Instruct, use the model's 32k context
window, but reduce the batch first because FSDP training cost also grows with sequence
length:

```bash
MAX_INPUT_LENGTH=28672 \
MAX_MODEL_LEN=32768 \
MAX_GENERATE_LENGTH=256 \
TRAIN_BATCH_SIZE=2 \
POLICY_MINI_BATCH_SIZE=1 \
N_SAMPLES_PER_PROMPT=2 \
bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh ...
```

Do not make the 32k setting the default unless the 16k run is still length-limited; it is
much slower and more likely to run out of memory on a 4-GPU 2+2 split.

Useful rollout checks:

```bash
EXPORT_PATH=$HOME/exports/arc_agi3/arc_agi3_latest_YYYYmmdd_HHMMSS

python examples/train_integrations/arc_agi3/visualize_rollouts.py \
  "$EXPORT_PATH" \
  --latest-files 100 \
  --trajectories-per-file 4 \
  -o "$EXPORT_PATH/rollout_viewer.html"

jq -r '.steps[].model_output' "$EXPORT_PATH/dumped_rollouts/global_step_1_rollouts.jsonl" \
  | sort | uniq -c | sort -nr

jq -r '.initial_messages[-1].content' "$EXPORT_PATH/dumped_rollouts/global_step_1_rollouts.jsonl" \
  | head -n 90

jq '.steps[] | {turn, reward, reward_components: .metadata.reward_components, diff_stats: .metadata.diff_stats}' \
  "$EXPORT_PATH/dumped_rollouts/global_step_1_rollouts.jsonl"
```

Open `rollout_viewer.html` in a browser to inspect each trajectory. The viewer shows rollout-derived
training curves by `global_step`, summary metrics, trajectory filters, per-turn `<think>`,
`<action>`, reward components, diff stats, observations, and raw step JSON.
Training curves scan every rollout file in the export, even when the trajectory detail list is
sampled with `--latest-files`, `--trajectories-per-file`, `--step-from`, or `--step-to`.
For long runs, keep the viewer scoped with `--latest-files`, `--trajectories-per-file`,
`--step-from`, `--step-to`, or `--max-trajectories`. A static self-contained HTML that embeds
every rollout and frame from a multi-GB export can exhaust memory, so the tool refuses inputs
above 128 MiB unless `--allow-large` is passed.
When frame data is present, selecting a sample automatically loads the first available turn
in the frame panel. Use the frame slider to scrub through the sample's action sequence, or
click a per-turn `View frame after action` button to jump to that action's frame.

## Metric Loggers

`LOGGER` is passed to `trainer.logger` and controls the metric tracker. It is separate from
`trainer.log_path`, which stores infrastructure logs such as `infra-${RUN_ID}.log` and
`router-${RUN_ID}.log`.
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
last `<action>...</action>` block; keep `<think>` concise so rollout context stays within
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
