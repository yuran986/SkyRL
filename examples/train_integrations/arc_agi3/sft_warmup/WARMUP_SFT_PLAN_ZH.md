# ARC-AGI-3 ft09 Oracle SFT Warm-up 修改方案

本文只规划改造方案，不直接启动训练。目标是使用
`examples/train_integrations/arc_agi3/ft09_oracle_solution.py` 产生的绝对正确
ft09 action 轨迹做 SFT warm-up，然后再把 warm-up checkpoint 交给后续 GRPO 或评估流程。

## 目标

- 新增一条与现有 GRPO 清晰分离的 SFT warm-up 路径。
- SFT label 来自 ft09 oracle 的可执行 `<action>...</action>`。
- oracle 不提供可监督的 thinking，因此训练时保留输出协议，但反传只训练 action token。
- 生成的数据、启动脚本、测试和说明都放在 `examples/train_integrations/arc_agi3/sft_warmup/`
  下，避免污染现有 `run_arc_agi3_grpo.sh` 和 GRPO 数据准备逻辑。

## 当前代码约束

现有通用 SFT 入口是 `python -m skyrl.train.main_sft`，核心实现位于
`skyrl/train/sft_trainer.py`。它当前支持 Alpaca 格式和 chat `messages` 格式，但 tokenization
只返回 `num_actions`，collate 后的 `loss_mask` 默认覆盖最后 `num_actions` 个 token。

这个机制适合“训练完整 assistant 回复”，但不适合直接表达：

```text
<think></think><action>{"action":"ACTION6","x":32,"y":32}</action>
```

中只训练 `<action>...</action>`、不训练 `<think>...</think>` 的需求。虽然可以把
`num_actions` 人为设为 action span 长度，让最后 N 个 token 参与 loss，但这种做法依赖
chat template 结尾 token 的位置，容易把 `eos` / assistant 结束 token 算进去，或者漏掉
`<action>` 开头 token。

因此 warm-up 需要显式 token-level action mask。

## 推荐目录结构

```text
examples/train_integrations/arc_agi3/sft_warmup/
  WARMUP_SFT_PLAN_ZH.md
  prepare_ft09_oracle_sft.py
  run_ft09_oracle_sft_fsdp.sh
  run_ft09_oracle_sft.slurm
  README_ZH.md
```

后续如果不想把 action-only SFT 能力放进通用 SFT trainer，可以再新增：

```text
examples/train_integrations/arc_agi3/sft_warmup/entrypoints/
  main_ft09_oracle_sft.py
```

我的建议是优先复用 `skyrl.train.main_sft`，只对通用 SFT tokenization 增加一个向后兼容的
`loss_mask` 输入能力；ARC-AGI-3 相关逻辑仍然留在 `sft_warmup/`。

## 数据生成方案

新增 `prepare_ft09_oracle_sft.py`，职责是把 oracle trace 转为 SFT JSONL/Parquet：

1. 调用 `run_ft09_solution(..., trace=[])`，复用 oracle 的 step-by-step trace。
2. 从 trace 中抽取 `event == "click"` 的记录。
3. 对每个点击构造一个单步 imitation example：
   - `messages[:-1]` 是当前 turn 之前的对话上下文。
   - 最后一条 assistant message 是协议化输出：
     `<think></think><action>{"action":"ACTION6","x":X,"y":Y}</action>`
   - 额外字段记录 `task_id=ft09`、`level_index`、`display` 坐标、`oracle_reason`、
     `advanced_level` 等，方便审计。
4. 每一步点击之后，把环境 observation 追加到下一条样本的上下文里，让 SFT 看到与 GRPO
   一致的多轮状态转移。

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

## Action-only loss mask 方案

新增或扩展一个 tokenization helper，例如：

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

## 训练入口方案

新增 `run_ft09_oracle_sft_fsdp.sh`，默认走单/少 GPU FSDP warm-up：

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

不建议把 warm-up 挂到 `main_arc_agi3.py`，因为那条入口是 PPO/GRPO experiment，包含
environment、generator、ref model 和 rollout 相关配置；SFT 不需要这些组件。

## 与 GRPO 的衔接

SFT warm-up 结束后，GRPO 只需要把 `MODEL_PATH` 指向 warm-up checkpoint：

```bash
MODEL_PATH=/usr/project/xtmp/yz1051/ckpts/arc_agi3_ft09_oracle_sft/global_step_N/policy \
bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh ...
```

如果 checkpoint 保存格式需要 export 成 HuggingFace 目录，再补一个小脚本放在
`sft_warmup/` 下，明确从 SkyRL policy checkpoint 导出到 GRPO 可直接加载的模型路径。

## 测试计划

新增或扩展 CPU 单测：

- `tests/train/test_sft_tokenization.py`
  - 构造 `<think>bad label</think><action>...</action>`，断言 think token 的 loss mask 全为 0。
  - 断言 `<action>`、JSON payload、`</action>` 对应 token 的 loss mask 为 1。
  - 断言 `num_actions == len(response_ids)`，不是 action-only token 数。
  - 断言旧 Alpaca/chat SFT 样本不受影响。
- `examples/train_integrations/arc_agi3/tests/test_ft09_oracle_sft_data.py`
  - 用一小段 fake trace 测试 JSON action 序列化。
  - 断言每条样本最后 assistant message 只有一个 `<action>...</action>`。
  - 断言 metadata 包含 level、display 坐标和 oracle 来源。

可选 smoke verification：

```bash
uv run --extra dev --extra fsdp pytest -v \
  tests/train/test_sft_tokenization.py \
  examples/train_integrations/arc_agi3/tests/test_ft09_oracle_sft_data.py
```

## 风险与处理

- 数据覆盖单一：当前 oracle 只覆盖 ft09，warm-up 可能让模型过拟合 ft09 action 模式。先把
  run name、checkpoint path 和 README 都标成 `ft09_oracle_sft`，不要暗示泛化到全部 ARC-AGI-3。
- 空 thinking 分布偏移：训练时 `<think></think>` 被 mask，不会直接惩罚模型生成其他 thinking。
  后续 GRPO 仍可通过 rollout 学习 reasoning；SFT warm-up 只负责把 action 协议和合法点击坐标教稳。
- token span 对齐：不要用字符串 split 后简单数 token 作为最终实现，必须用单测覆盖 Qwen chat
  template 下的 action span 对齐。
- 上下文长度：多轮历史样本可能很长。数据生成时需要支持 `--max-turns`、`--max-context-turns`
  或按 tokenizer 长度过滤，避免 SFT 阶段大量样本被截断到 action 之外。

## 建议实施顺序

1. 扩展 SFT tokenizer/collate，支持可选 `response_loss_mask`，保持旧数据格式兼容。
2. 增加本地 JSONL `dataset_data_files` 加载能力。
3. 新增 `sft_warmup/prepare_ft09_oracle_sft.py`，先生成可审计 JSONL。
4. 新增 action-only mask 单测和 oracle SFT 数据单测。
5. 新增 `run_ft09_oracle_sft_fsdp.sh` 和 `README_ZH.md`。
6. 先做 1-2 step smoke run，确认 loss 非零、checkpoint 正常保存、样本里的 think token 不参与 loss。
7. 用 warm-up checkpoint 启动原有 GRPO 脚本，比较 invalid action rate 和 ft09 level completion。
