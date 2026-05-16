# ALE-Bench SkyRL Integration

This integration wraps ALE-Bench public evaluation as a SkyRL text environment. Each rollout gives the model one AHC problem statement, extracts one source file from the response, runs `session.public_eval(...)`, and uses the public score as reward.

## Setup

From the SkyRL repository:

```sh
uv sync --extra dev --extra fsdp
uv pip install -e /home/users/yz1051/ALE-Bench python-dotenv datasets
```

Build or pull the ALE-Bench Docker images for the target judge/language before training. For example:

```sh
cd /home/users/yz1051/ALE-Bench
bash ./scripts/docker_build_202301.sh $(id -u) $(id -g)
```

If ALE-Bench is not installed in the SkyRL environment, set `ALE_BENCH_REPO=/home/users/yz1051/ALE-Bench`; the run script adds `ALE_BENCH_REPO/src` to `PYTHONPATH`.

## Prepare Data

```sh
cd /home/users/yz1051/SkyRL
ALE_BENCH_REPO=/home/users/yz1051/ALE-Bench \
PYTHONPATH=/home/users/yz1051/ALE-Bench/src:$PWD \
uv run python examples/train_integrations/ale_bench/prepare_dataset.py \
  --output_dir $HOME/data/ale_bench \
  --problem_ids ahc001 \
  --train_size 8 \
  --val_size 2
```

The parquet rows use `env_class=ale_bench` and pass `problem_id`, `code_language`, `judge_version`, scoring, and feedback settings through `env_extras`.

## Train

```sh
cd /home/users/yz1051/SkyRL
DATA_DIR=$HOME/data/ale_bench \
ALE_BENCH_REPO=/home/users/yz1051/ALE-Bench \
LOGGER=console \
bash examples/train_integrations/ale_bench/run_ale_bench_grpo.sh
```

Useful overrides: `MODEL_PATH`, `NUM_GPUS`, `MAX_INPUT_LENGTH`, `MAX_GENERATE_LENGTH`, `MAX_ENV_WORKERS`, `ALE_BENCH_CODE_LANGUAGE`, and `ALE_BENCH_JUDGE_VERSION`.
