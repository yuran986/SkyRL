# ARG/ARC-AGI-3 + SkyRL 环境配置流程

本文给出从空环境到可运行 SkyRL GRPO 训练的配置步骤。当前 shell 环境中我检查到 `arc_agi`、`arcengine`、`rlm` 都还不能直接 import，因此需要先安装 ARC-AGI-3 Toolkit。

## 0. 前置条件

建议使用 Python 3.12。SkyRL 本体支持 Python 3.11+，但 ARC-AGI-3 Toolkit 当前要求 Python 3.12+，`skyrl-agent` 也固定为 3.12。

检查基础工具：

```bash
cd /home/users/yz1051/SkyRL
uv --version
python3 --version
nvidia-smi
```

如果 `uv` 不存在：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 1. 创建 Python 3.12 虚拟环境

```bash
cd /home/users/yz1051/SkyRL
uv venv --python 3.12 .venv
source .venv/bin/activate
python --version
```

后续命令默认都在 `/home/users/yz1051/SkyRL` 下执行。

## 2. 安装 SkyRL 开发依赖

只做环境 wrapper 和单元测试时，先装轻量开发依赖：

```bash
uv sync --extra dev
```

如果要在本机 GPU 上跑 SkyRL FSDP/vLLM 训练，需要安装训练依赖：

```bash
uv sync --extra dev --extra fsdp
```

这一步会安装较重的 GPU 依赖，建议确认 CUDA、驱动和磁盘空间充足。

## 3. 安装 ARC-AGI-3 Toolkit

临时安装到当前虚拟环境：

```bash
uv pip install arc-agi python-dotenv
```

验证导入和动作枚举：

```bash
python -c "import arc_agi; from arcengine import GameAction; print(list(GameAction.__members__.keys()))"
```

期望能看到 `RESET`、`ACTION1` 到 `ACTION7`。

训练脚本默认复用仓库根目录的 `.venv/bin/python`，不会再通过 `uv run --isolated` 每次创建隔离环境。首次配置或依赖变化时，把依赖安装进当前 `.venv`：

```bash
source .venv/bin/activate
uv sync --extra dev --extra fsdp
uv pip install arc-agi python-dotenv
```

如果需要使用其他 Python，可在启动训练时设置 `PYTHON_BIN=/path/to/python`。

## 4. 配置 `.env`

本地开发优先使用 `.env`，避免每次手动 `export`。我已经在 SkyRL 根目录创建了本机默认配置：

```text
.env
.env.example
```

当前 `.env` 内容是：

```bash
OPERATION_MODE=OFFLINE
ARC_AGI3_ENVIRONMENTS_DIR=/home/users/yz1051/rlm/environment_files
ARC_AGI3_FRAME_OBSERVATION_MODE=initial_full_then_diff
ARC_AGI3_FULL_FRAME_INTERVAL=8
ARC_AGI3_PATCH_RADIUS=4
ARC_AGI3_MAX_DIFF_EXAMPLES=32
```

`.env` 是本机配置，当前 `.gitignore` 已忽略它；`.env.example` 是可提交模板。后续如果要在线提交或拉取远程 game，在 `.env` 中补：

```bash
ARC_API_KEY=<your_arc_api_key>
OPERATION_MODE=ONLINE
```

训练环境 wrapper 中建议显式读取 `operation_mode` 和 `environments_dir`，不要只依赖全局环境变量。

shell 脚本中读取 `.env` 的推荐写法：

```bash
set -a
source .env
set +a
```

Python 入口或 wrapper 中读取 `.env` 的推荐写法：

```python
from dotenv import load_dotenv

load_dotenv()
```

## 5. 验证 ARC 本地 game 可运行

用 `ft09` 做最小验证：

```bash
set -a
source .env
set +a
python -c "import os, arc_agi; from arc_agi import OperationMode; from arcengine import GameAction; mode=getattr(OperationMode, os.environ.get('OPERATION_MODE','OFFLINE')); arc=arc_agi.Arcade(operation_mode=mode, environments_dir=os.environ['ARC_AGI3_ENVIRONMENTS_DIR']); env=arc.make('ft09', renderer=lambda *args, **kwargs: None); obs=env.step(GameAction.ACTION6, data={'x':32,'y':32}); print(obs.state, getattr(obs, 'levels_completed', None), len(obs.frame), len(obs.frame[0]))"
```

如果这里失败，先不要跑 SkyRL。常见问题：

- `ModuleNotFoundError: arc_agi`：没有安装 `arc-agi` 或没有激活 venv。
- 找不到 `ft09`：`environments_dir` 路径不对，或 game 文件结构不符合 toolkit 预期。
- `ACTION6 requires x/y`：坐标动作必须带 `data={"x": int, "y": int}`。

当前默认 observation 策略是 `initial_full_then_diff`：初始给颜色表和完整 frame 的 hex rows；每轮给 diff summary 和 changed patch；每 8 轮刷新一次完整当前 frame。

## 6. 准备 SkyRL integration 数据

当前已经新增了 SkyRL integration 文件：

```text
examples/train_integrations/arc_agi3/
  env.py
  entrypoints/main_arc_agi3.py
  prepare_dataset.py
  run_arc_agi3_grpo.sh
  README.md
```

先跑数据准备：

```bash
set -a
source .env
set +a
python examples/train_integrations/arc_agi3/prepare_dataset.py \
  --output_dir $HOME/data/arc_agi3 \
  --task_ids ft09 \
  --environments_dir "$ARC_AGI3_ENVIRONMENTS_DIR"
```

期望生成：

```text
$HOME/data/arc_agi3/train.parquet
$HOME/data/arc_agi3/validation.parquet
```

## 7. 做 wrapper smoke test

在正式训练前，建议新增一个最小测试或临时脚本，验证：

1. `skyrl_gym.register("arc_agi3", ...)` 成功。
2. `skyrl_gym.make("arc_agi3", extras={...})` 能创建环境。
3. `env.init(prompt)` 返回初始 prompt。
4. `env.step("<action>ACTION1</action>")` 返回 `observations/reward/done/metadata`。
5. `env.step('<action>{"action":"ACTION6","x":32,"y":32}</action>')` 能执行坐标动作。

示例测试命令：

```bash
uv run --extra dev pytest -q examples/train_integrations/arc_agi3/tests
```

如果完整 `uv run --extra dev` 正在下载 GPU 相关依赖，也可以先用本地源码路径只验证动作解析：

```bash
PYTHONPATH=/home/users/yz1051/SkyRL/skyrl-gym:/home/users/yz1051/SkyRL \
python -c "from examples.train_integrations.arc_agi3.env import parse_model_action; print(parse_model_action('<action>ACTION1</action>'))"
```

## 8. 启动一次最小 GRPO 训练

日常开训入口看：

```text
examples/train_integrations/arc_agi3/README.md
examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh
examples/train_integrations/arc_agi3/RUN_ARC_AGI3_GRPO_PARAMS_ZH.md
```

本文件保留完整环境配置、参数解释和排错说明。真正开始训练时，优先从 integration
README 的 `Train` 小节复制命令；需要逐项判断参数是否合理时，看同目录下的中文参数说明。

推荐先用 4 卡保守配置跑通链路。这个配置目标是先稳定完成多个 global step、
写出 checkpoint 和 rollout dump，不追求一开始就把 context 或 batch 拉满：

```bash
set -a
source .env
set +a
PYTORCH_ALLOC_CONF=expandable_segments:True \
DATA_DIR=$HOME/data/arc_agi3 \
CKPT_PATH=/usr/project/xtmp/yz1051/ckpts/arc_agi3_3B_4gpu \
NUM_GPUS=4 \
LOGGER=console \
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
MAX_TURNS=8 \
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

如果这组参数能稳定跑完，再按顺序一次只放大一个参数：

1. `MAX_INPUT_LENGTH=8192`
2. `N_SAMPLES_PER_PROMPT=5`
3. `TRAIN_BATCH_SIZE=8`
4. `POLICY_MINI_BATCH_SIZE=4`

暂时不要急着把 `trainer.micro_train_batch_size_per_gpu` 从 `1` 调回 `4`。此前
2 卡 OOM 发生在 FSDP policy backward 阶段，micro batch 是主要放大器。

检查点：

- generator 是否能完成多轮 rollout。
- reward 是否不是全空或全异常。
- `uids` 是否按 prompt 正确分组。
- `response_ids`、`loss_masks` 长度是否一致。
- checkpoint 和 dump 是否写到预期目录。
- 结构化训练 rollout 会写到 `$EXPORT_PATH/dumped_rollouts/global_step_*_rollouts.jsonl`，每行是一条 trajectory，包含每轮 action、observation、reward、reward_components、diff_stats 和 state。

如果在 `Generating Trajectories` 前后看到
`response_end_idx - initial_prompt_length` 的 `NoneType` 报错，通常是
`prompt + env.init()` 超过了 `generator.max_input_length`。ARC-AGI-3 的初始
observation 会包含完整 frame，训练脚本默认使用 `MAX_INPUT_LENGTH=8192`；
需要更长上下文时可继续调大，例如 `MAX_INPUT_LENGTH=12288`。

长度相关参数分四层：

- `trainer.max_prompt_length`：dataset 读入时过滤初始 prompt 的长度，不包含
  `env.init()` 追加的初始 observation。
- `generator.max_input_length`：rollout 中每轮调用模型前的累计上下文上限，包含
  初始 prompt、初始 frame、历史 action、历史 observation/diff。
- `generator.sampling_params.max_generate_length`：每一轮模型最多生成多少 token。
  ARC-AGI-3 当前要求输出简短 `<think>` 加一个 `<action>`，默认 `256`。
- `generator.inference_engine.engine_init_kwargs.max_model_len`：vLLM 单次请求允许的
  `input tokens + generated tokens` 总窗口。通常应满足
  `max_model_len >= generator.max_input_length + max_generate_length`。

训练脚本用 shell 变量映射这些配置：

```bash
MAX_INPUT_LENGTH=8192        # 同时传给 trainer.max_prompt_length 和 generator.max_input_length
MAX_GENERATE_LENGTH=256      # 传给 generator.sampling_params.max_generate_length
MAX_MODEL_LEN=32768          # 可选；传给 vLLM max_model_len，不设时使用模型/框架默认
```

如果 vLLM 日志里显示 `Using max model len 32768`，而
`MAX_INPUT_LENGTH + MAX_GENERATE_LENGTH` 小于 32768，就不需要额外设置
`MAX_MODEL_LEN`。

`run_arc_agi3_grpo.sh` 的同步配置参考了 `examples/train/search/run_search.sh`：
`generator.batched=false`、conversation multi-turn、`n_samples_per_prompt=5`、
`environment.skyrl_gym.max_env_workers=16`、`gpu_memory_utilization=0.5`、默认关闭
训练前 eval。不同之处是 ARC-AGI-3 的初始 frame 更长，因此默认
`MAX_INPUT_LENGTH=8192`，而每轮需要输出简短 reasoning 加一个 action，所以
`MAX_GENERATE_LENGTH=256`。

默认模型使用 `Qwen/Qwen2.5-3B-Instruct`。冷启动阶段建议优先使用 instruct
模型，因为它更容易遵循 `<think>...</think>`、`<action>...</action>` 和 JSON 坐标格式；base 模型更适合
已有 SFT warm start 或想从更原始策略开始做大规模 RL 的场景。

单卡 smoke run 可用：

```bash
DATA_DIR=$HOME/data/arc_agi3 \
NUM_GPUS=1 \
LOGGER=console \
MAX_TURNS=8 \
TRAIN_BATCH_SIZE=8 \
POLICY_MINI_BATCH_SIZE=8 \
bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh \
  trainer.epochs=1 \
  trainer.eval_before_train=false
```

## 9. 推荐训练配置

初始 GRPO 配置建议：

```bash
trainer.algorithm.advantage_estimator="grpo"
trainer.algorithm.use_kl_loss=true
trainer.algorithm.kl_loss_coef=0.001
trainer.algorithm.grpo_norm_by_std=true
generator.n_samples_per_prompt=8
generator.max_turns=64
generator.batched=false
generator.use_conversation_multi_turn=true
generator.sampling_params.temperature=0.7
generator.sampling_params.top_p=0.95
generator.sampling_params.max_generate_length=128
generator.sampling_params.stop='["</action>"]'
generator.eval_sampling_params.stop='["</action>"]'
```

如果 reward 方差低：

```bash
generator.n_samples_per_prompt=16
trainer.algorithm.dynamic_sampling.type="filter"
```

同时优先检查 reward shaping，而不是只调训练超参。

## 10. 日志与调试

训练脚本默认开启：

```bash
trainer.dump_data_batch=true
trainer.dump_rollout_logs=true
```

主要输出位置：

```text
$HOME/skyrl_logs/arc_agi3/infra-*.log
$HOME/skyrl_logs/arc_agi3/router-*.log
$EXPORT_PATH/dumped_data/global_step_*_training_input.pkl
$EXPORT_PATH/dumped_rollouts/global_step_*_rollouts.jsonl
$EXPORT_PATH/dumped_evals/global_step_*_evals/*.jsonl
```

`run_arc_agi3_grpo.sh` 默认会给每次训练生成唯一的 `EXPORT_PATH`：

```text
$HOME/exports/arc_agi3/${RUN_NAME}_YYYYmmdd_HHMMSS
```

脚本启动时会打印 `ARC-AGI-3 export path: ...`。如果你显式传入
`EXPORT_PATH=/some/path`，脚本会使用你给的固定路径；这种情况下同名
`global_step_*` 文件仍可能被覆盖。

`dumped_rollouts` 每行是一条 trajectory，包含模型第一轮生成前看到的
`initial_messages`，以及每轮：

```text
turn, model_output, reward, done, observations, parsed_action,
reward_components, diff_stats, state
```

常用检查：

```bash
EXPORT_PATH=$HOME/exports/arc_agi3/arc_agi3_latest_YYYYmmdd_HHMMSS

python examples/train_integrations/arc_agi3/visualize_rollouts.py \
  "$EXPORT_PATH" \
  -o "$EXPORT_PATH/rollout_viewer.html"

jq -r '.steps[].model_output' "$EXPORT_PATH/dumped_rollouts/global_step_1_rollouts.jsonl" \
  | sort | uniq -c | sort -nr

jq '{sample_index, uid, total_reward, num_steps}' \
  "$EXPORT_PATH/dumped_rollouts/global_step_1_rollouts.jsonl"

jq -r '.initial_messages[-1].content' "$EXPORT_PATH/dumped_rollouts/global_step_1_rollouts.jsonl" \
  | head -n 90

jq '.steps[] | {turn, reward, parsed_action: .metadata.parsed_action, reward_components: .metadata.reward_components, diff_stats: .metadata.diff_stats}' \
  "$EXPORT_PATH/dumped_rollouts/global_step_1_rollouts.jsonl"
```

`rollout_viewer.html` 是静态页面，不需要启动服务。它会按 `global_step` 展示从
rollout 聚合出来的曲线，例如 avg reward、positive trajectory rate、invalid step
rate、avg turns、avg levels completed 和 success rate。它也支持按 reward、invalid
action、positive step 和关键字筛选 trajectory，并展示每轮 `<think>`、`<action>`、
observation、reward components、diff stats 和原始 step JSON。

SkyRL 原生 metric tracker 通过 `trainer.logger` / 脚本里的 `LOGGER` 控制，支持
`wandb`、`mlflow`、`swanlab`、`tensorboard` 和 `console`。`LOGGER` 只控制训练指标上报；
`trainer.log_path` 仍然控制 infra/router 日志文件位置，两者不是同一个概念。

日常选择：

- `console`：无需配置，指标直接打印到终端，适合 smoke run 和排错。
- `tensorboard`：本地曲线，适合 SSH 服务器长期观察；事件文件写到 `TENSORBOARD_DIR`
  或默认 `tensorboard_log`，不是 `trainer.log_path`。
- `wandb`：云端实验管理，适合多组实验对比；SkyRL 会要求 `WANDB_API_KEY`。
- `swanlab`：类似 wandb，可用 `SWANLAB_API_KEY`、`SWANLAB_LOG_DIR`、`SWANLAB_MODE`
  控制登录、目录和 cloud/local 模式。
- `mlflow`：适合已有 MLflow tracking server 的团队；常用 `MLFLOW_TRACKING_URI`。

示例：

```bash
LOGGER=console bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh

TENSORBOARD_DIR=$HOME/skyrl_logs/arc_agi3/tensorboard \
LOGGER=tensorboard \
bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh

WANDB_API_KEY=... LOGGER=wandb bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh
SWANLAB_API_KEY=... LOGGER=swanlab bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh
MLFLOW_TRACKING_URI=http://host:5000 LOGGER=mlflow bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh
```

这些 tracker 适合看 reward、KL、entropy、loss、response length、timing 和 eval pass
rate；静态 rollout viewer 主要用于排查模型具体行为、动作合法性、reward 是否被刷和
observation 是否足够。

## 11. 参考资料

- 本地 RLM 版本：`/home/users/yz1051/rlm/arg-agi/benchmark_adapter.py`
- 本地任务说明：`/home/users/yz1051/rlm/arg-agi/arg-agi_bg.md`
- 官方 Toolkit quickstart：https://docs.arcprize.org/toolkit/overview
- 官方 Arcade API：https://docs.arcprize.org/toolkit/arc_agi
- 官方 action 提交说明：https://docs.arcprize.org/toolkit/submit-action
