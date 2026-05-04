# ARC-AGI-3 GRPO 脚本参数说明

本文解释 `run_arc_agi3_grpo.sh` 中的参数如何影响 ARC-AGI-3 训练，并给出当前阶段的推荐设置。脚本最后保留 `"$@"`，所以命令行末尾追加的 Hydra override 会覆盖脚本内同名配置。

## 推荐起步配置

当前更稳的起步方式是在 4 张 48GB 级别 GPU 上分离训练和推理：训练侧 2 张，vLLM 推理侧 2 张。

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

这组参数偏保守，目标是先确认 rollout、policy update、checkpoint、结构化日志都稳定。稳定后按顺序放大：`N_SAMPLES_PER_PROMPT=4`，`MAX_INPUT_LENGTH=8192`，`TRAIN_BATCH_SIZE=4`，`POLICY_MINI_BATCH_SIZE=2`。暂时不要先把 `trainer.micro_train_batch_size_per_gpu` 调回 4。

## 正式训练配置

smoke run 连续多个 global step 正常后，可以用下面这组参数开始正式训练。它仍然是 4 卡总量的 2+2 分离模式，但把上下文、采样数和 batch 拉高，并降低 checkpoint 频率，避免每一步都写 35GB 级别 checkpoint：

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
MAX_INPUT_LENGTH=8192 \
MAX_MODEL_LEN=9216 \
MAX_GENERATE_LENGTH=192 \
N_SAMPLES_PER_PROMPT=4 \
TRAIN_BATCH_SIZE=4 \
POLICY_MINI_BATCH_SIZE=2 \
CKPT_INTERVAL=10 \
EVAL_INTERVAL=20 \
RUN_NAME=arc_agi3_formal \
bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh \
  trainer.epochs=3 \
  trainer.eval_before_train=false \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.max_ckpts_to_keep=3 \
  trainer.log_path=$HOME/skyrl_logs/arc_agi3
```

如果 rollout 仍然只有 5-6 turn 且 `stop_reason=length`，先把 `MAX_INPUT_LENGTH=10240`、`MAX_MODEL_LEN=11264`，再考虑改 prompt 或 reward。`MAX_TURNS=10` 是交互轮数上限，不保证一定跑满；真正提前截断的常见原因是 conversation token 数达到 `generator.max_input_length`。

## Shell 环境变量

`DATA_DIR`  
训练/验证 parquet 所在目录。脚本读取 `$DATA_DIR/train.parquet` 和 `$DATA_DIR/validation.parquet`。默认 `$HOME/data/arc_agi3`。

`CKPT_PATH`  
SkyRL checkpoint 输出目录。建议放到 `/usr/project/xtmp/...`，不要放 home quota。4 卡分离实验建议类似 `/usr/project/xtmp/yz1051/ckpts/arc_agi3_3B_4gpu_split`。

`NUM_GPUS`  
训练侧使用的 GPU 数，传给 policy/ref/critic placement。`COLOCATE_ALL=true` 时通常也让 vLLM 使用同样数量的 GPU；`COLOCATE_ALL=false` 时，它只表示训练侧 GPU 数。

`INFERENCE_NUM_ENGINES`  
vLLM engine 数，默认等于 `NUM_GPUS`。`COLOCATE_ALL=false` 时，这些 engine 会申请独立 GPU。例如 `NUM_GPUS=2 INFERENCE_NUM_ENGINES=2 COLOCATE_ALL=false` 通常需要 4 张 GPU。

`COLOCATE_ALL`  
是否让训练和 vLLM 共用同一批 GPU。`true` 省 GPU，但依赖 CUDA IPC 权重同步；如果节点报 `pidfd_getfd: Operation not permitted`，说明该节点不适合 colocated CUDA IPC。此时改用 `false`，让训练和推理分卡运行。

`LOGGER`  
训练指标 tracker，传给 `trainer.logger`。`console` 适合排错；`tensorboard` 适合本地曲线；`wandb`、`swanlab`、`mlflow` 适合实验管理。它不控制 infra 日志位置，infra 日志由 `trainer.log_path` 控制。

`MODEL_PATH`  
policy/ref/vLLM 使用的模型。当前推荐 `Qwen/Qwen2.5-3B-Instruct`，因为它更容易遵循 `<think>...</think>` 和 `<action>...</action>` 格式。base 模型更适合已有 SFT warm start 后再做 RL。

`INFERENCE_BACKEND`  
生成后端，默认 `vllm`。当前不要改。

`MAX_TURNS`  
每条 trajectory 最多环境交互轮数，传给 `generator.max_turns`。值越大，探索更长，但上下文和显存压力更高。当前建议先用 8。

`N_SAMPLES_PER_PROMPT`  
GRPO 每个 prompt 采几条 rollout。越大，组内相对优势更稳定，但生成量和训练 batch 都变大。当前建议 4；稳定后可以试 5 或 8。

`TRAIN_BATCH_SIZE`  
每个训练 step 使用多少 prompt。实际 trajectory 数大约是 `TRAIN_BATCH_SIZE * N_SAMPLES_PER_PROMPT`。增大它会提高吞吐和统计稳定性，也会增加显存/时间。当前建议 4，稳定后试 8。

`POLICY_MINI_BATCH_SIZE`  
policy update 的 mini-batch 大小。越大吞吐更好但显存更高。当前建议 2，稳定后试 4。不要超过 `TRAIN_BATCH_SIZE`。

`MAX_INPUT_LENGTH`  
同时传给 `trainer.max_prompt_length` 和 `generator.max_input_length`。对 ARC-AGI-3 来说，真正关键的是 `generator.max_input_length`：它限制多轮 conversation 的累计上下文，包括初始 prompt、初始 frame、历史 `<think>/<action>` 和 observation/diff。`MAX_TURNS` 只是轮数上限；如果 rollout 的 `stop_reason=length`，会在达到 10 轮前提前停止。正式训练建议 8192 起步。

`MAX_GENERATE_LENGTH`  
每轮 action 生成上限，传给 train/eval sampling params。当前输出是 `<think>` 加一个 `<action>`；正式训练可以用 192，给模型保留一定 reasoning 空间。过大则会让单轮输出变长并更快触发 `stop_reason=length`。

`GPU_MEMORY_UTILIZATION`  
vLLM KV cache 使用显存比例。默认 0.5。colocated 训练时 vLLM 和 FSDP 共用 GPU，数值太高会挤训练显存；太低会限制生成并发。当前 0.45-0.5 都合理。

`CKPT_INTERVAL`  
每多少个 global step 保存 checkpoint。debug 建议 1，正式训练可调到 10 或 20，避免频繁保存拖慢训练和占磁盘。

`EVAL_INTERVAL`  
每多少 step 做 eval。debug 可设 -1 关闭；正式训练可以 20、50 或更大。eval 会额外启动 rollout，增加时间。

`EVAL_BEFORE_TRAIN`  
训练前是否先 eval。debug 建议 false，避免等待。

`RUN_NAME`、`RUN_ID`、`EXPORT_ROOT`、`EXPORT_PATH`  
控制 rollout dump、data dump、viewer 相关产物目录。默认每次创建 `$HOME/exports/arc_agi3/${RUN_NAME}_YYYYmmdd_HHMMSS`，避免覆盖旧 rollout。

`MAX_MODEL_LEN`  
可选 vLLM 最大窗口，传给 `generator.inference_engine.engine_init_kwargs.max_model_len`。通常只要 vLLM 日志显示的 `max_seq_len` 大于 `MAX_INPUT_LENGTH + MAX_GENERATE_LENGTH` 就不用设置。

`MAX_ENV_WORKERS`  
skyrl-gym 环境并发 worker 数。默认 16。环境本身很轻时可以增加；如果日志太乱或环境初始化压力大，可以降到 8。

`PYTHON_BIN`  
训练脚本使用的 Python。默认是仓库根目录的 `.venv/bin/python`，因此不会每次通过 `uv run --isolated` 重建隔离环境。只有要切到其他已配置好的虚拟环境时才覆盖。

`SKYRL_FORCE_BROADCAST_WEIGHT_SYNC`  
设置为 `1` 时，强制绕开 CUDA IPC，改走 broadcast/NCCL 权重同步。它保留为排错开关，不建议作为 4 卡 colocated 默认方案；在当前节点上，colocated + broadcast/NCCL 也可能触发 vLLM 的 `NCCL error: invalid usage`。更稳的修复是 `COLOCATE_ALL=false` 并给 vLLM 单独 GPU。

## 算法参数

`trainer.algorithm.advantage_estimator=grpo`  
使用 GRPO。每个 prompt 的多条 rollout 会先采样完成，再按组内 reward 计算相对 advantage，然后一起做 policy update，不是一条 rollout 更新一次。

`trainer.algorithm.use_kl_loss=true`、`kl_loss_coef=0.001`  
启用对 reference model 的 KL 约束，防止策略过快偏离初始模型。若模型输出格式明显崩坏，可考虑增大；如果学习太慢且 KL 很低，可考虑减小。

`trainer.algorithm.grpo_norm_by_std=true`  
按组内标准差归一化 advantage。GRPO 常用设置。若 reward 方差长期接近 0，先检查 reward 和任务难度，不要只调这个。

`trainer.policy.optimizer_config.lr=1.0e-6`  
policy 学习率。RL 微调一般要小。若 reward 曲线剧烈震荡或格式退化，先降学习率；若长期无变化且 KL 很低，可小幅增加。

`trainer.update_epochs_per_batch=1`  
每批 rollout 只更新一轮。对在线 RL 比较稳。调大可能提高样本利用率，也更容易过拟合当前 rollout。

## 模型与分布式训练参数

`trainer.strategy=fsdp2`  
使用 PyTorch FSDP2 训练。当前 3B 模型和多 GPU 训练应保持这个设置。

`trainer.placement.colocate_all=true`  
训练和生成共用同一批 GPU。好处是部署简单；坏处是 vLLM 和 FSDP 竞争显存。当前 OOM 主要来自 policy backward，增加 GPU 或减小 micro batch 能缓解。

`trainer.policy.fsdp_config.cpu_offload=false`  
policy 不 offload 到 CPU，速度更快但更吃 GPU 显存。若显存仍然不够，可以尝试改 true，但会明显变慢。

`trainer.ref.fsdp_config.cpu_offload=true`  
reference model 允许 CPU offload，节省显存。保持 true。

`trainer.placement.policy_num_gpus_per_node=$NUM_GPUS`  
policy FSDP 使用的 GPU 数。越大，每卡参数分片更小，但通信更多。

`trainer.placement.critic_num_gpus_per_node=$NUM_GPUS`、`ref_num_gpus_per_node=$NUM_GPUS`  
critic/ref 的 placement 设置。当前 GRPO 主要训练 policy，但框架配置仍保留这些 placement。

`trainer.micro_train_batch_size_per_gpu`  
每卡 backward micro batch。这个参数非常影响显存。此前 2 卡 OOM 时脚本默认 4 过大；当前建议命令行覆盖为 1。

`trainer.micro_forward_batch_size_per_gpu`  
forward/logprob 阶段每卡 micro batch。也影响显存，但通常比 backward 稍轻。当前建议先覆盖为 1。

`trainer.train_batch_size`、`trainer.policy_mini_batch_size`  
由 shell 变量控制。先保证能跑，再放大。

## 生成与 vLLM 参数

`generator.inference_engine.num_engines=$INFERENCE_NUM_ENGINES`  
vLLM engine 数。`COLOCATE_ALL=false` 时它会申请独立 GPU，日志里应看到多个 `EngineCore` 和 router worker。

`generator.inference_engine.tensor_parallel_size=1`  
每个 vLLM engine 不做 TP。当前 3B 模型单卡可放下，TP=1 简单稳定。

`generator.inference_engine.run_engines_locally=true`  
在本 Ray job 内启动本地 inference engines。当前保持。

`generator.inference_engine.weight_sync_backend=nccl`  
训练后同步 policy 权重到 vLLM engine。多 GPU 本地训练推荐保持 nccl。

`generator.inference_engine.async_engine=true`  
使用 vLLM async engine。当前保持。

`generator.batched=false`  
使用 skyrl-gym 的多轮环境循环。ARC-AGI-3 是 multi-turn 环境，保持 false。

`generator.use_conversation_multi_turn=true`  
把历史 assistant action 和环境 observation 保留在 conversation 中。ARC-AGI-3 必须保持 true，否则模型看不到历史交互。

`generator.append_eos_token_after_stop_str_in_multi_turn=true`  
在 stop string 后补 EOS，使 chat template 的多轮拼接更一致。保持 true。

`generator.sampling_params.temperature=1.0`、`top_p=1.0`  
训练采样保持较高探索。若非法动作过多，可先优化 prompt/action parser；必要时把 temperature 降到 0.7。

`generator.sampling_params.stop=["</action>"]`  
生成到 `</action>` 停止。环境会解析最后一个 `<action>...</action>`，因此 stop string 必须保留。

`generator.eval_sampling_params.temperature=0`  
eval 使用确定性输出，便于比较。

## 环境与数据参数

`data.train_data`、`data.val_data`  
由 `DATA_DIR` 拼出 parquet 路径。修改 prompt 或 action 协议后必须重新运行 `prepare_dataset.py`，否则 parquet 里仍是旧 prompt。

`environment.env_class=arc_agi3`  
注册并创建当前 ARC-AGI-3 gym 环境。不要改。

`environment.skyrl_gym.max_env_workers=$MAX_ENV_WORKERS`  
环境并发数。增大能提高 rollout 并发，但可能让日志和资源调度更复杂。

## 日志、checkpoint 与 dump

`trainer.logger=$LOGGER`  
训练指标 tracker。和 `trainer.log_path` 不同。

`trainer.project_name=arc_agi3`、`trainer.run_name=$RUN_NAME`  
用于 wandb/swanlab/mlflow/tensorboard 等 tracker 的项目和 run 名。

`trainer.log_path`  
infra/router 日志目录，建议显式传 `$HOME/skyrl_logs/arc_agi3`。

`trainer.ckpt_path=$CKPT_PATH`  
checkpoint 根目录。之前 home quota 会爆，建议始终放 `/usr/project/xtmp`。

`trainer.resume_mode=null`  
默认不自动 resume，避免调参时误接旧 checkpoint。正式续训时再改为合适的 resume 模式。

`trainer.export_path=$EXPORT_PATH`  
dump 和 eval artifact 输出根目录。脚本默认每次唯一目录，避免覆盖。

`trainer.dump_data_batch=true`  
保存训练输入 pickle，便于查 token、loss mask、advantage 等。会占少量磁盘。

`trainer.dump_rollout_logs=true`  
保存结构化 rollout JSONL。建议保持 true，这是当前判断训练行为最有用的 artifact。

## 判断参数是否合理

优先看这些日志信号：

- `batch_padded_seq_len`：接近或超过 `MAX_INPUT_LENGTH + MAX_GENERATE_LENGTH` 时，说明上下文很满。
- `avg_response_length`：过大说明 `<think>` 太长或 stop 没正常生效；如果 rollout 只有 5-6 轮且 `stop_reason=length`，优先提高 `MAX_INPUT_LENGTH`。
- `environment/invalid_actions`：高说明动作格式、可用动作或坐标范围没学好。
- `reward/avg_raw_reward` 和 `reward/mean_positive_reward`：长期全 0 或负数时，先抽查 rollout，不要只调训练超参。
- `policy_kl`：太高说明策略偏离过快；长期 0 附近且 reward 不动，可能学习率/有效 reward 太弱。
- OOM 发生在 `FSDPPolicyWorkerBase.forward_backward`：优先降 `micro_train_batch_size_per_gpu`、`POLICY_MINI_BATCH_SIZE`、`TRAIN_BATCH_SIZE`、`MAX_INPUT_LENGTH`，或增加 GPU。
- OOM 发生在 vLLM 初始化/生成：优先降 `GPU_MEMORY_UTILIZATION`、`MAX_MODEL_LEN`、`MAX_INPUT_LENGTH` 或生成并发。

当前阶段最重要的目标是：连续跑完多个 global step，能保存 checkpoint，rollout viewer 里能看到合法动作比例提高，并逐步出现非 shaped reward 的 `level_delta` 或 `done` 奖励。
