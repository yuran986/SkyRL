# SkyRL 训练框架梳理与 ARG/ARC-AGI-3 适配方案

本文基于 SkyRL 当前代码和 `/home/users/yz1051/rlm/arg-agi` 中的 ARG/ARC-AGI-3 用法，梳理训练链路，并说明为了用 GRPO 训练该任务需要新增或改动哪些地方。

## 1. SkyRL 训练链路

SkyRL 的 RL 训练主链路是：

```text
parquet/json/jsonl 数据
  -> PromptDataset
  -> prepare_generator_input
  -> SkyRLGymGenerator.agent_loop
  -> skyrl_gym.make(env_class).init/step
  -> GeneratorOutput(response_ids, rewards, loss_masks, metrics)
  -> RayPPOTrainer
  -> GRPO advantage
  -> policy update
  -> sync weights back to inference engine
```

关键文件：

- `skyrl/train/entrypoints/main_base.py`：训练入口。`BasePPOExp` 构建 tokenizer、dataset、inference client、generator、trainer。
- `skyrl/train/dataset/dataset.py`：`PromptDataset` 读取数据。每条样本需要 `prompt`，可选 `env_class`，其它列会作为 `env_extras` 传给环境。
- `skyrl/train/generators/skyrl_gym_generator.py`：多轮 rollout 核心。每轮让模型生成动作，调用环境 `step(action)`，再把 observation 拼回上下文。
- `skyrl-gym/skyrl_gym/envs/base_text_env.py`：环境接口。任务环境实现 `init()`、`step()`、`close()`、`get_metrics()`。
- `skyrl/train/trainer.py`：采样、后处理、算 logprob/value/KL、算 advantage/return、训练 policy、保存 checkpoint、同步权重。
- `skyrl/backends/skyrl_train/utils/ppo_utils.py`：GRPO 实现。按同一 prompt 的 `n_samples_per_prompt` 条轨迹分组，用 outcome reward 做组内相对优势。

结论：SkyRL 已经具备 GRPO、多轮环境交互、可插拔环境、vLLM/HTTP 推理、token/loss mask bookkeeping。我们主要要补的是 ARG/ARC-AGI-3 环境 wrapper 和训练脚本。

## 2. ARG/ARC-AGI-3 当前用法

`/home/users/yz1051/rlm/arg-agi/benchmark_adapter.py` 展示了现有交互方式：

- 导入 `arc_agi` 和 `arcengine.GameAction`。
- 使用 `arc = arc_agi.Arcade()` 创建客户端。
- 使用 `env = arc.make(task_id, renderer=...)` 创建游戏环境；训练时传 no-op renderer，避免 terminal 输出污染日志。
- 通过 `env.action_space` 获取可用动作。
- 通过 `env.observation_space` 读取当前状态和 frame。
- 通过 `env.step(GameAction.ACTION1)` 或 `env.step(GameAction.ACTION6, data={"x": 32, "y": 32})` 执行动作。
- 每次 `step()` 返回 observation，常见字段包括 `done`、`state`、`levels_completed`、`score`、`frame`。
- frame 通常是嵌套 list，形状类似 `[1, 64, 64]`，实际 2D 网格是 `frame[0]`。

现有 RLM 版本给模型暴露的是 REPL 工具：`get_action_space()`、`get_game_state()`、`get_current_frame()`、`make_diffs()`、`make_action()`。SkyRL 训练不建议保留 REPL 工具形态，而应该把这些工具能力折叠进 `BaseTextEnv` 的 observation 和 `step()` 里，让 policy 直接输出一个动作。

## 3. 不建议直接用 mini-swe-agent

mini-swe-agent 的设计中心是 terminal/bash/repo patch，适合 SWE-Bench 类任务。ARG/ARC-AGI-3 是离散动作加 64x64 frame 的交互环境，直接套 mini-swe-agent 会引入不必要的 shell scaffold，并且会干扰 SkyRL 对生成 token、logprob、loss mask、reward 的训练闭环。

建议只借鉴它的“线性 history + 简单动作协议”思想，不把它放进训练主链路。

## 4. 建议新增目录

先做成 integration，稳定后再迁移到 `skyrl-gym`：

```text
examples/train_integrations/arc_agi3/
  env.py
  entrypoints/main_arc_agi3.py
  prepare_dataset.py
  run_arc_agi3_grpo.sh
  README.md
  tests/
```

## 5. 必须新增或修改的地方

### 5.1 新增 `ArcAgi3Env`

在 `examples/train_integrations/arc_agi3/env.py` 新增：

```python
class ArcAgi3Env(BaseTextEnv):
    def __init__(self, env_config, extras):
        ...

    def init(self, prompt):
        ...

    def step(self, action: str):
        ...

    def close(self):
        ...

    def get_metrics(self):
        ...
```

建议 `__init__` 读取：

- `task_id`：例如 `ft09`、`ls20`。
- `seed`：可选，用于可复现实验。
- `renderer`：训练时使用空 renderer，不向 terminal 打印帧。
- `operation_mode`：本地开发建议 `OFFLINE`。
- `environments_dir`：本地环境文件目录，例如 `/home/users/yz1051/rlm/environment_files`。
- `max_turns`：由 SkyRL generator 注入。

环境内部使用：

```python
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, GameState

arc = arc_agi.Arcade(
    operation_mode=OperationMode.OFFLINE,
    environments_dir=environments_dir,
)
env = arc.make(task_id, seed=seed, renderer=noop_renderer)
```

### 5.2 动作协议

模型输出强制为单步动作：

```text
<action>{"action":"ACTION6","x":32,"y":32}</action>
```

也支持简单动作：

```text
<action>ACTION1</action>
```

环境解析规则：

- `ACTION1` 到 `ACTION5`、`ACTION7`：调用 `env.step(GameAction.ACTION*)`。
- `ACTION6`：必须提供 `x`、`y`，调用 `env.step(GameAction.ACTION6, data={"x": x, "y": y})`。
- 非法 JSON、未知动作、缺坐标、坐标越界：不给环境提交，返回短 observation，并给负 reward 或 0 reward。
- 如果当前动作不在 `env.action_space`，按非法动作处理。

训练脚本应设置：

```bash
generator.sampling_params.stop='["</action>"]'
generator.eval_sampling_params.stop='["</action>"]'
```

### 5.3 Observation 序列化

第一版建议返回紧凑文本，不把完整 frame 每轮塞进上下文。可包含：

- 当前 `state`、`score`、`levels_completed`。
- 当前可用动作列表。
- 上一步动作和是否有效。
- 与上一帧的 diff 摘要：变化 cell 数量、bounding box、颜色变化统计、若干代表坐标。
- 必要时给局部 patch，而不是完整 64x64 frame。

示例 observation：

```text
state=NOT_FINISHED score=12 levels_completed=2
available_actions=[ACTION1,ACTION2,ACTION6]
last_action=ACTION6 x=32 y=32 valid=true
frame_diff: num_changes=14 bbox=(20,18)-(35,33) colors 9->0:8, 0->9:6
Choose exactly one next action inside <action>...</action>.
```

完整 frame 应保存在环境内部，用于计算 diff 和 metrics；debug 时再写 JSONL，不默认放进模型上下文。

### 5.4 Reward 设计

当前 integration 采用两层 reward：

- 大奖励：`levels_completed` 增长，或底层 ARC 环境返回 `done=True`。
- 中等奖励：产生了非空且有解释价值的相邻帧 diff。
- 非法动作：小负奖励，默认 `-0.05`。

“有解释价值的 diff”第一版用启发式判断：相邻帧变化数在 `[min_meaningful_diff_changes, max_meaningful_diff_changes]` 之间，默认是 `[1, 512]`。这样避免奖励完全无变化的动作，也避免把整屏大面积重绘当成高质量探索信号。

注意：`step_wise_trajectories=True` 当前主要用最后一步轨迹 advantage 广播到每步。第一版建议保持：

```bash
generator.step_wise_trajectories=false
```

如果 dense reward 效果明显，再单独评估是否改 step-wise 逻辑。

### 5.5 Metrics 与 debug

`get_metrics()` 至少返回：

- `success`
- `final_score`
- `levels_completed`
- `steps`
- `invalid_actions`
- `game_over`
- `win`
- `task_id`

建议额外写一个 rollout JSONL：

```json
{"step": 3, "action": "ACTION6", "x": 32, "y": 32, "reward": 0.1, "state": "NOT_FINISHED", "levels_completed": 2}
```

这和 RLM 版本里的 `FrameStreamWriter` 作用类似，方便分析失败轨迹。

### 5.6 数据准备脚本

新增 `prepare_dataset.py`，输出 parquet。建议 schema：

```text
prompt: list[{"role": "user", "content": "..."}]
env_class: "arc_agi3"
task_id: str
seed: int | null
split: str
max_steps: int
operation_mode: str
environments_dir: str | null
```

`PromptDataset` 会自动把 `task_id`、`seed`、`max_steps` 等列放进 `env_extras`。

prompt 建议短而稳定：

```text
You are playing one ARC-AGI-3 game. Use the observations to infer the rule.
Return exactly one action in <action>...</action>.
For ACTION6, use JSON: {"action":"ACTION6","x":32,"y":32}.
```

不要在 prompt 里泄露特定 game 的解法。

### 5.7 环境注册入口

新增 `entrypoints/main_arc_agi3.py`：

```python
from skyrl_gym.envs import register

register(
    id="arc_agi3",
    entry_point="examples.train_integrations.arc_agi3.env:ArcAgi3Env",
)
exp = BasePPOExp(cfg)
exp.run()
```

训练配置使用：

```bash
environment.env_class=arc_agi3
```

### 5.8 训练脚本

新增 `run_arc_agi3_grpo.sh`，初始建议：

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
trainer.train_batch_size=8
trainer.policy_mini_batch_size=8
```

如果组内 reward 方差长期为 0，优先改 reward shaping 和增大 `n_samples_per_prompt`，再考虑动态采样：

```bash
trainer.algorithm.dynamic_sampling.type="filter"
```

## 6. 可能需要改 SkyRL 主框架的地方

第一阶段尽量不改 trainer，只新增 integration。后续可能需要：

1. 在 `SkyRLGymConfig` 增加 `arc_agi3` 配置 dataclass，集中放 `environments_dir`、`operation_mode`、`timeout`。
2. 增强 `SkyRLGymGenerator` 的 observation/debug hooks，方便保存完整 frame，但不污染模型上下文。
3. 如果要直接训练 VLM，接入 `SkyRLVLMGymGenerator`，让环境返回图像特征或渲染帧。
4. 如果 ARC 环境初始化慢，实现 env pool 或调小 `environment.skyrl_gym.max_env_workers`。
5. 增加官方 evaluation adapter，严格按 ARC-AGI-3 online/offline 约束复现评测。

## 7. 推荐实施顺序

### M1：最小闭环

1. 写 `ArcAgi3Env`，先用 `ft09` 和本地 `environment_files`。
2. 写 `prepare_dataset.py`，生成 16 到 64 条 parquet。
3. 写 `main_arc_agi3.py` 注册环境。
4. 写 `run_arc_agi3_grpo.sh`，小模型、小 batch 跑 1 到 2 个训练 step。
5. 开 `trainer.dump_data_batch=true`，检查 reward、loss mask、uids 分组。

### M2：真实任务训练

1. 扩展多个 `task_id` 和 seed。
2. 完善 frame diff 摘要。
3. 加非法动作、GAME_OVER、超时处理。
4. 加 unit tests：动作解析、env reset/step、dataset schema、mock rollout。
5. 对比 `n_samples_per_prompt=4/8/16` 的 reward 方差。

### M3：效果提升

1. 细化 reward shaping。
2. 增加 curriculum 或 dynamic sampling。
3. 评估是否需要 VLM 输入。
4. 做固定 validation split 和官方风格 evaluation。

## 8. 结论

SkyRL 适合做 ARG/ARC-AGI-3 的 GRPO 训练。当前最小改动路径是新增 `examples/train_integrations/arc_agi3/`，把 `arc_agi.Arcade().make(...).step(...)` 包成 `BaseTextEnv`，让模型直接输出 `<action>...</action>`。不要先改 trainer，也不要把 mini-swe-agent 放进主链路；先跑通环境、数据、reward 和 GRPO 分组，再逐步增强 observation、debug 和 evaluation。
