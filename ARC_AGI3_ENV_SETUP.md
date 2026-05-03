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

如果后面希望训练脚本自带依赖，而不是依赖当前 venv，可以在脚本里使用：

```bash
uv run --isolated --extra fsdp --with arc-agi --with python-dotenv -m ...
```

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

当前默认 observation 策略是 `initial_full_then_diff`：初始给完整 frame 的 hex rows；每轮给 diff summary 和 changed patch；每 8 轮刷新一次完整当前 frame。

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

先用小 batch、小模型、小步数跑通链路：

```bash
set -a
source .env
set +a
DATA_DIR=$HOME/data/arc_agi3 \
NUM_GPUS=1 \
LOGGER=console \
MAX_TURNS=16 \
bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh \
  trainer.epochs=1 \
  trainer.train_batch_size=2 \
  trainer.policy_mini_batch_size=2 \
  generator.n_samples_per_prompt=4 \
  trainer.eval_before_train=false \
  trainer.dump_data_batch=true
```

检查点：

- generator 是否能完成多轮 rollout。
- reward 是否不是全空或全异常。
- `uids` 是否按 prompt 正确分组。
- `response_ids`、`loss_masks` 长度是否一致。
- checkpoint 和 dump 是否写到预期目录。

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

建议设置：

```bash
export ARC_AGI3_ROLLOUT_LOG_DIR=/tmp/skyrl-arc-agi3-rollouts
```

wrapper 每条轨迹写 JSONL，记录：

```text
task_id, seed, step, action, x, y, valid, reward, state, score, levels_completed, done
```

完整 frame 可以只在 debug 开关打开时写入，避免日志过大。

## 11. 参考资料

- 本地 RLM 版本：`/home/users/yz1051/rlm/arg-agi/benchmark_adapter.py`
- 本地任务说明：`/home/users/yz1051/rlm/arg-agi/arg-agi_bg.md`
- 官方 Toolkit quickstart：https://docs.arcprize.org/toolkit/overview
- 官方 Arcade API：https://docs.arcprize.org/toolkit/arc_agi
- 官方 action 提交说明：https://docs.arcprize.org/toolkit/submit-action
