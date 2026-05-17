# ARC-AGI-3 ft09 Oracle Warm-up 修改方案

本文只规划改造方案，不直接启动训练。目标是使用
`examples/train_integrations/arc_agi3/ft09_oracle_solution.py` 产生的绝对正确
ft09 action / oracle-distance 信息做 warm-up，然后再把 warm-up checkpoint 交给后续
GRPO 或评估流程。

结论先行：不要把单条 oracle trajectory cloning 当成主 warm-up。更合适的结构是：

1. 默认跳过 tiny SFT，直接做 oracle-distance RL warm-up。
2. tiny SFT 只作为 fallback：当换 base model 或 prompt 后 action 格式明显不稳时再启用。
3. 主 warm-up 使用 oracle-distance RL，让模型可以探索、换 action 顺序、从非 oracle prefix
   的状态恢复。

## 目标

- 新增一条与现有正式 GRPO 清晰分离的 ft09 oracle warm-up 路径。
- oracle-distance RL 不要求 action 顺序等于 oracle trajectory，只奖励让 oracle plan 变短、
  level advance、通关和保持合法动作。
- tiny SFT 作为可选 fallback，label 来自 ft09 oracle 的可执行 `<action>...</action>`，只用于
  格式 warm-up。oracle 不提供可监督的 thinking，因此启用 SFT 时反传只训练 action token。
- 生成的数据、启动脚本、测试和说明都放在 `examples/train_integrations/arc_agi3/sft_warmup/`
  下，避免污染现有 `run_arc_agi3_grpo.sh` 和 GRPO 数据准备逻辑。

## 当前日志判断

已有 ARC-AGI-3 rollout 说明，当前主要问题不是 `<think>/<action>` 格式，而是策略没有学到
ft09 规则、容易坍缩到固定点击序列。

抽查结果：

| run / step | valid action | 格式观察 |
| --- | ---: | --- |
| `arc_agi3_formal_20260504_005642` step 1-12 | 多数 step 在 90%+ | 基本都有 `<think>` 和 `<action>`；step 12 有少量 `<Action>` 大写、JSON/坐标错误 |
| `arc_agi3_structured_diff_11636395` step 1 | `136/157` | 初期仍有 JSON 坏格式、缺 x/y、`NO_OP` |
| `arc_agi3_structured_diff_11636395` step 50/100/150/200 | `160/160` | action 格式稳定，坐标合法；很多输出直接是 `<action>...</action>` |

因此 tiny SFT 不是默认必要步骤。它能解决的是格式和坐标先验，但现有 RL 已经能较快学稳这些。
默认应把工程投入放在 oracle-distance RL reward 上。

## 可选 SFT 代码约束

如果后续决定启用 tiny SFT，需要注意现有通用 SFT 入口是 `python -m skyrl.train.main_sft`，核心
实现位于 `skyrl/train/sft_trainer.py`。它当前支持 Alpaca 格式和 chat `messages` 格式，但
tokenization 只返回 `num_actions`，collate 后的 `loss_mask` 默认覆盖最后 `num_actions` 个
token。

这个机制适合“训练完整 assistant 回复”，但不适合直接表达：

```text
<think></think><action>{"action":"ACTION6","x":32,"y":32}</action>
```

中只训练 `<action>...</action>`、不训练 `<think>...</think>` 的需求。虽然可以把
`num_actions` 人为设为 action span 长度，让最后 N 个 token 参与 loss，但这种做法依赖
chat template 结尾 token 的位置，容易把 `eos` / assistant 结束 token 算进去，或者漏掉
`<action>` 开头 token。

因此如果启用 tiny SFT，需要显式 token-level action mask。

## 推荐目录结构

```text
examples/train_integrations/arc_agi3/sft_warmup/
  WARMUP_SFT_PLAN_ZH.md
  oracle_distance_reward.py
  run_ft09_oracle_distance_grpo.sh
  run_ft09_oracle_distance_grpo.slurm
  prepare_ft09_oracle_sft.py           # optional fallback
  run_ft09_oracle_sft_fsdp.sh          # optional fallback
  run_ft09_oracle_sft.slurm            # optional fallback
  README_ZH.md
```

后续如果不想把 action-only SFT 能力放进通用 SFT trainer，可以再新增：

```text
examples/train_integrations/arc_agi3/sft_warmup/entrypoints/
  main_ft09_oracle_sft.py
```

SFT fallback 若要实现，建议复用 `skyrl.train.main_sft`，只对通用 SFT tokenization 增加一个
向后兼容的 `loss_mask` 输入能力；ARC-AGI-3 相关逻辑仍然留在 `sft_warmup/`。

## 为什么不做粗暴 trajectory SFT

按 `ft09_oracle_solution.py` 的 solution trajectory 逐步监督 next action，本质是 behavior
cloning 一个 canonical path。这个做法有两个问题：

- 同一个状态下可能有多个可行 action 或可交换点击顺序，CE 会把 oracle 选中的那个 action
  当成唯一正确答案，压低其他可通关路径。
- 真实 rollout 很容易进入 oracle trace 没覆盖的状态，例如先探索边界、点击顺序不同、或者
  产生部分完成的中间状态。单条 trajectory SFT 对这些 off-trajectory 状态没有 recovery
  先验。

因此 SFT 最多只能定位为“格式和动作协议 fallback”，策略学习交给 RL。若后续仍想扩大 SFT
数据，也应生成多 prefix、多随机点击顺序、多恢复状态的 oracle-guided state-action dataset，
而不是只复制一条固定轨迹。

## Tiny SFT Fallback 条件

默认不跑 tiny SFT。只有满足以下任一条件时再启用：

- 换了 base model 后，invalid action 长期高于 5%-10%。
- 大量输出缺 `<action>`、裸 JSON、大小写错误标签或非法 JSON。
- 坐标边界明显不稳，例如频繁输出 `x=64`、`y=64` 或缺 x/y。
- 需要快速验证 action-only loss mask 管线，而不是为了提升策略能力。

如果启用，建议最多 `num_steps=50`；训练目标仍然只是格式，不用它学习 ft09 解法。

## Tiny SFT Fallback 数据生成

可选新增 `prepare_ft09_oracle_sft.py`，职责是把 oracle trace 转为少量 SFT JSONL/Parquet：

1. 调用 `run_ft09_solution(..., trace=[])`，复用 oracle 的 step-by-step trace。
2. 从 trace 中抽取 `event == "click"` 的记录。
3. 对每个点击构造一个单步 imitation example：
   - `messages[:-1]` 是当前 turn 之前的对话上下文。
   - 最后一条 assistant message 是协议化输出：
     `<think></think><action>{"action":"ACTION6","x":X,"y":Y}</action>`
   - 额外字段记录 `task_id=ft09`、`level_index`、`display` 坐标、`oracle_reason`、
     `advanced_level` 等，方便审计。
4. 每一步点击之后，把 environment observation 追加到下一条样本的上下文里，让 SFT 看到与
   GRPO 一致的多轮状态转移。
5. fallback 第一版只保留 canonical trace 即可，因为它不是主策略训练数据；后续可增加
   `--shuffle-commutable-clicks`、`--recovery-prefixes` 和 `--max-context-turns`，但不是启动
   oracle-distance RL 的 blocker。

数据格式建议采用 chat 格式 JSONL：

```json
{
  "messages": [
    {"role": "user", "content": "...initial observation..."},
    {"role": "assistant", "content": "<think></think><action>{\"action\":\"ACTION6\",\"x\":18,\"y\":42}</action>"},
    {"role": "user", "content": "...observation after click..."},
    {"role": "assistant", "content": "<think></think><action>{\"action\":\"ACTION6\",\"x\":26,\"y\":42}</action>"}
  ],
  "loss_policy": "assistant_action_only",
  "metadata": {"task_id": "ft09", "oracle": "ft09_oracle_solution"}
}
```

样本粒度建议先用“每条样本训练最后一个 assistant action”。也就是说一条样本可以包含历史
turn，但只有最后一条 assistant message 的 action span 有 loss。这样正好贴合当前
`SFTTrainer` 的“最后 assistant 回复”训练假设。

## Tiny SFT Fallback 训练规模

如果启用 tiny SFT fallback，建议 first pass 使用 `num_steps=50`。这个规模的目的不是让模型记住 ft09 解法，而是快速压低
格式错误和非法坐标：

- 学会稳定输出 `<action>{"action":"ACTION6","x":...,"y":...}</action>`。
- 学会 x/y 使用显示坐标范围内的整数。
- 保持 `<think>...</think>` 协议，但 thinking token 不参与 loss。

epoch 计算公式：

```text
effective_epochs = num_steps * global_batch_size / num_train_examples
```

本地未跟踪的 `ft09_oracle_solution.html` 中 canonical oracle trace 有 75 个 click event。
如果 `prepare_ft09_oracle_sft.py` 按每个 click 生成 1 条样本，那么 50 step 对应：

| global batch size | 50 step 约等于 |
| --- | --- |
| 2 | 1.33 epoch |
| 4 | 2.67 epoch |
| 8 | 5.33 epoch |
| 16 | 10.67 epoch |

fallback 推荐先用 `batch_size=4` 或 `batch_size=8`。如果只生成 canonical 75 条样本，50 step 已经是
多 epoch 训练；这正符合 tiny SFT 的定位，但不应继续加大步数。若后续通过随机顺序和 recovery
prefix 把数据扩到 300 条样本，则 `batch_size=8, num_steps=50` 约为 1.33 epoch。

## Action-only loss mask 方案

仅在启用 tiny SFT fallback 时需要新增或扩展一个 tokenization helper，例如：

```python
tokenize_chat_example_with_action_loss(
    example,
    tokenizer,
    max_length,
    messages_key="messages",
    action_start="<action>",
    action_end="</action>",
)
```

核心逻辑：

1. 用 `tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)` 得到 prompt ids。
2. 用 `tokenizer.apply_chat_template(messages, add_generation_prompt=False)` 得到 full ids。
3. response ids 是 `full_ids[len(prompt_ids):]`。
4. 对最后一条 assistant content 里的 `<action>...</action>` 做字符级 span 定位。
5. 用 tokenizer offset mapping 或“前缀重 tokenize”方法，将 action 字符 span 映射到
   response token 下标。
6. 返回：
   - `input_ids`
   - `attention_mask`
   - `response_loss_mask`: 长度等于 response token 数；thinking token、assistant wrapper token、
     template 结束 token 都为 0；action span token 为 1。
   - `num_actions`: response token 总长度，而不是 action token 数。

collate 需要支持两种模式：

- 旧样本没有 `response_loss_mask`：保持现在 `[1] * num_actions` 的行为。
- 新样本有 `response_loss_mask`：按 response 长度右对齐 pad 这个 mask。

这个设计的关键点是：`metadata["response_length"]` 仍然表示 backend 需要从序列末尾切出的
完整 response 长度；`loss_mask` 才决定哪些 response token 参与 cross entropy。这样和
FSDP/Megatron worker 当前的 `log_probs[:, -num_actions - 1 : -1]` / `log_probs[:, -num_actions:]`
切片方式兼容。

## Tiny SFT Fallback 入口方案

可选新增 `run_ft09_oracle_sft_fsdp.sh`，默认走单/少 GPU FSDP fallback：

```bash
DATA_PATH=$HOME/data/arc_agi3_sft/ft09_oracle_train.jsonl
CKPT_PATH=/usr/project/xtmp/yz1051/ckpts/arc_agi3_ft09_oracle_sft
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct

bash examples/train_integrations/arc_agi3/sft_warmup/run_ft09_oracle_sft_fsdp.sh \
  dataset_name=json \
  dataset_data_files="$DATA_PATH" \
  dataset_split=train \
  ckpt_path="$CKPT_PATH"
```

为支持本地 JSONL，SFT config 需要增加一个向后兼容字段，例如：

```python
dataset_data_files: str | list[str] | None = None
```

加载逻辑：

- `dataset_data_files` 为空：保持现有 `load_dataset(dataset_name, split=dataset_split)`。
- `dataset_data_files` 非空：调用 `load_dataset(dataset_name, data_files=dataset_data_files, split=dataset_split)`。

不建议把 SFT fallback 挂到 `main_arc_agi3.py`，因为那条入口是 PPO/GRPO experiment，包含
environment、generator、ref model 和 rollout 相关配置；SFT 不需要这些组件。

## Oracle-distance RL Warm-up

默认直接从 base model 或当前最佳 checkpoint 启动 oracle-distance RL warm-up。核心是给当前
game state 定义一个 oracle potential：

```text
V_oracle(state) = 当前状态下 oracle 到通关还需要的最少/计划点击数
```

每一步 reward 使用 potential difference：

```text
reward_oracle_progress = V_oracle(before) - V_oracle(after)
```

这样模型只要让 oracle plan 变短就拿正 reward，不要求它复制某条固定 action 顺序。建议 reward
组成：

- `valid_action_reward`: 合法 action 小正，非法 action 明确负。
- `oracle_progress_reward`: `V_before - V_after`，主 shaping 信号。
- `level_reward`: level advance 大正。
- `done_reward`: 通关大正。
- `repeat_noop_penalty`: 重复点击、无 frame diff、plan length 不变时负。
- `unrecoverable_penalty`: 如果当前状态 oracle 已无法求解，给明显负 reward 并终止或强惩罚。

实现上新增 `oracle_distance_reward.py`，复用 `ft09_oracle_solution.py` 里的
`solve_click_plan(game)`，但不要执行 oracle click。它只读取当前 env/game state，返回：

```python
{
    "oracle_plan_len": int,
    "oracle_solvable": bool,
    "oracle_next_actions": [...],
}
```

然后在 `ArcAgi3Env.step()` 的 reward 里可选启用该 shaping，或新增一个
`ArcAgi3OracleDistanceEnv` 包装类，避免影响正式 GRPO 默认环境。

## 与正式 GRPO 的衔接

默认路径是从 base model 或已有 checkpoint 直接启动 oracle-distance GRPO：

```bash
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
bash examples/train_integrations/arc_agi3/sft_warmup/run_ft09_oracle_distance_grpo.sh ...
```

如果触发了 tiny SFT fallback，再把 `MODEL_PATH` 指向 tiny SFT checkpoint：

```bash
MODEL_PATH=/usr/project/xtmp/yz1051/ckpts/arc_agi3_ft09_oracle_sft/global_step_N/policy \
bash examples/train_integrations/arc_agi3/sft_warmup/run_ft09_oracle_distance_grpo.sh ...
```

oracle-distance RL 到达稳定合法动作和较低 oracle plan length 后，再切回原有正式 GRPO：

```bash
MODEL_PATH=/usr/project/xtmp/yz1051/ckpts/arc_agi3_ft09_oracle_distance_grpo/global_step_N/policy \
bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh ...
```

如果 checkpoint 保存格式需要 export 成 HuggingFace 目录，再补一个小脚本放在
`sft_warmup/` 下，明确从 SkyRL policy checkpoint 导出到后续脚本可直接加载的模型路径。

## 测试计划

新增或扩展 CPU 单测：

- `tests/train/test_sft_tokenization.py`（仅 SFT fallback 需要）
  - 构造 `<think>bad label</think><action>...</action>`，断言 think token 的 loss mask 全为 0。
  - 断言 `<action>`、JSON payload、`</action>` 对应 token 的 loss mask 为 1。
  - 断言 `num_actions == len(response_ids)`，不是 action-only token 数。
  - 断言旧 Alpaca/chat SFT 样本不受影响。
- `examples/train_integrations/arc_agi3/tests/test_ft09_oracle_sft_data.py`（仅 SFT fallback 需要）
  - 用一小段 fake trace 测试 JSON action 序列化。
  - 断言每条样本最后 assistant message 只有一个 `<action>...</action>`。
  - 断言 metadata 包含 level、display 坐标和 oracle 来源。
- `examples/train_integrations/arc_agi3/tests/test_oracle_distance_reward.py`
  - 断言合法 oracle click 后 `oracle_plan_len` 下降或 level advance。
  - 断言可交换点击顺序不会被强制成唯一 next action。
  - 断言不可解析/非法 action 不会产生 oracle progress 正 reward。

可选 smoke verification：

```bash
uv run --extra dev --extra fsdp pytest -v \
  tests/train/test_sft_tokenization.py \
  examples/train_integrations/arc_agi3/tests/test_ft09_oracle_sft_data.py
```

## 风险与处理

- 数据覆盖单一：当前 oracle 只覆盖 ft09，warm-up 可能让模型过拟合 ft09 action 模式。先把
  run name、checkpoint path 和 README 都标成 `ft09_oracle_*`，不要暗示泛化到全部 ARC-AGI-3。
- 不必要 SFT 干扰：当前日志显示格式已经能学稳，默认跳过 SFT，避免用 canonical trace 给策略
  注入不必要 bias。
- 空 thinking 分布偏移：若启用 SFT，`<think></think>` 被 mask，不会直接惩罚模型生成其他
  thinking；后续 RL 仍可通过 rollout 学习 reasoning。
- 单轨迹 SFT 过拟合：若启用 fallback，50 step 已经可能覆盖 canonical trace 多个 epoch，因此
  不要把 SFT step 继续放大；策略学习用 oracle-distance RL。
- oracle reward 泄漏：oracle 使用源码级信息，只应作为 ft09 warm-up shaping，不应混进最终
  评测或宣称泛化能力。
- token span 对齐：不要用字符串 split 后简单数 token 作为最终实现，必须用单测覆盖 Qwen chat
  template 下的 action span 对齐。
- 上下文长度：多轮历史样本可能很长。数据生成时需要支持 `--max-turns`、`--max-context-turns`
  或按 tokenizer 长度过滤，避免 SFT 阶段大量样本被截断到 action 之外。

## 建议实施顺序

1. 新增 oracle-distance reward helper 和测试。
2. 新增 `run_ft09_oracle_distance_grpo.sh`、Slurm 脚本和 `README_ZH.md`。
3. 直接从 base model 或已有 checkpoint 做 oracle-distance GRPO smoke run，观察 invalid action
   rate、`oracle_plan_len`、level completion。
4. 若 invalid action 长期高于 5%-10%，再实现 tiny SFT fallback：
   扩展 SFT tokenizer/collate、本地 JSONL 加载、`prepare_ft09_oracle_sft.py`、action-only mask
   单测和 `run_ft09_oracle_sft_fsdp.sh`。
5. 若启用 fallback，先做 1-2 step SFT smoke run，再做最多 `num_steps=50`，记录数据条数、
   batch size 和 effective epoch。
6. 用 oracle-distance warm-up checkpoint 启动原有正式 GRPO 脚本，比较 invalid action rate、
   ft09 level completion 和是否出现固定轨迹过拟合。
