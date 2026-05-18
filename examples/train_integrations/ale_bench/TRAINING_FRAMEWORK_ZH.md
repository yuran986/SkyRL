# ALE-Bench 训练框架说明

本文说明 `examples/train_integrations/ale_bench/` 这套 SkyRL 集成的设计。它的目标是把 ALE-Bench 的“读题、生成完整程序、调用 public judge 得分”流程包装成 SkyRL 可训练的多轮文本环境，从而可以用 GRPO 对代码生成模型做强化学习。

## 总体架构

这套集成由四层组成：

```text
prepare_dataset.py
  -> train.parquet / validation.parquet
  -> main_ale_bench.py 注册 skyrl_gym 环境
  -> AleBenchEnv 初始化 ALE-Bench session
  -> 模型输出完整代码
  -> session.public_eval(code, language, judge_version)
  -> public score 转成 reward
  -> SkyRL GRPO 更新 policy
```

核心文件如下：

- `prepare_dataset.py`：生成 SkyRL 训练数据，每一行代表一个 ALE-Bench problem episode。
- `env.py`：实现 `AleBenchEnv(BaseTextEnv)`，负责拼接题面、抽取代码、调用 ALE-Bench judge、计算 reward。
- `entrypoints/main_ale_bench.py`：启动 SkyRL 前注册 `env_class=ale_bench`。
- `run_ale_bench_grpo.sh`：参考 `arc_agi3` 的 colocated GRPO 启动脚本，配置 vLLM/FSDP/SkyRL 参数。
- `README.md`：最短安装和运行命令。

## 数据集设计

SkyRL 的 dataset loader 会读取 parquet 行，并把除 `prompt` 和 `env_class` 之外的列都作为 `env_extras` 传给环境。因此 `prepare_dataset.py` 生成的行大致是：

```python
{
    "prompt": [{"role": "user", "content": DEFAULT_PROMPT}],
    "env_class": "ale_bench",
    "problem_id": "ahc001",
    "lite_version": True,
    "max_steps": 1,
    "code_language": "cpp20",
    "judge_version": "202301",
    "score_scale": 1e9,
    "invalid_reward": -1.0,
    "valid_reward_margin": 0.01,
}
```

`problem_id` 决定训练题目，`max_steps` 决定每个 episode 允许提交几次代码。默认是 `1`，也就是单轮“读题后直接提交”。如果设置为大于 `1`，环境会把 public eval 的反馈作为下一轮 observation，让模型继续提交改进版完整代码。

## 环境生命周期

`AleBenchEnv` 的生命周期和 `arc_agi3` 一样遵循 `BaseTextEnv`：

1. `__init__` 从 `env_extras` 读取 `problem_id`、语言、judge 版本、reward 参数等。
2. `_init_session()` 调用 `ale_bench.start(...)` 创建独立 session。
3. `init(prompt)` 加载题面、样例、工具 README，并追加到初始对话。
4. `step(action)` 解析模型输出中的代码，调用 `session.public_eval(...)`。
5. public eval 结果被转换为 reward 和 metadata。
6. episode 结束时 `close()` 清理 ALE-Bench session 的临时目录和资源。

环境不会调用 `private_eval`，训练信号只来自 public cases。这样更适合 RL 训练，也避免在 rollout 中消耗最终评测接口。

## Prompt 与代码抽取

`init()` 会把 ALE-Bench 题目信息组织成一个文本 observation，包括：

- `problem_id`、标题、分数类型、题型。
- 时间和内存限制。
- 默认代码语言。
- 英文题面 `problem.statement`。
- 样例输入输出。
- `tool_readme`，其中通常包含本地工具、可视化器或 tester 的说明。

模型应返回一个完整单文件程序。环境支持三种抽取方式：

```text
```cpp20
// code
```
```

```xml
<code language="cpp20">
// code
</code>
```

或者直接把整个输出当作源代码。推荐使用 fenced code block，因为语言标签可以覆盖默认 `code_language`。目前常用别名包括 `cpp`、`c++17`、`c++20`、`python`、`pypy`、`rust`。

## Reward 设计

环境先调用 ALE-Bench 的 `Result`：

```python
result = session.public_eval(
    code=code,
    code_language=language,
    judge_version=judge_version,
    skip_local_visualization=True,
)
```

然后根据问题的 `score_type` 计算 signed score：

- `maximize`：`signed_score = overall_absolute_score`
- `minimize`：`signed_score = -overall_absolute_score`

默认 `reward_mode=score`，reward 为：

```text
reward = signed_score / score_scale
```

默认 `score_scale=1e9`，目的是把 AHC 原始大整数分数压到更适合 GRPO 的量级。若模型输出无法解析、编译失败且没有可计分结果、judge 调用异常，则返回 `invalid_reward`，默认 `-1.0`。

为了避免 minimize 题里“很差但可计分的提交”因为取负后低于编译失败，环境会对所有可计分提交加一个 reward floor：

```text
reward = max(raw_reward, invalid_reward + valid_reward_margin)
```

默认 `valid_reward_margin=0.01`。这不是常数偏移，而是一个有效性约束：只要 ALE-Bench 能给出可计分结果，它就应该比无法评测的输出更好。GRPO 主要使用同一题多样本之间的相对优势，所以通常不需要为了正负号再额外加整体 offset。

当 `max_steps > 1` 时可以使用 `reward_mode=improvement`。这时每轮 reward 只计算相对当前 best signed score 的正向改进：

```text
reward = max(0, signed_score - best_signed_score) / score_scale
```

这更适合“提交、看 public 反馈、再改”的多轮训练。

## 多轮反馈

如果 `max_steps` 大于当前 turn，环境会把 public eval 摘要返回给模型：

```text
Public evaluation result:
judge=ACCEPTED
absolute_score=...
relative_score=...
case[0] judge=ACCEPTED score=... time=... memory=...
best_absolute_score=...
You may submit an improved complete solution.
```

`case_feedback_limit` 控制最多展示多少个 public case 摘要。当前没有把完整输入、输出、stderr 全部回传，是为了控制上下文长度和避免 rollout 过慢。后续如果要做更强的 self-refine，可以增加更细粒度的 case feedback，但要谨慎处理 token 长度。

## 训练脚本配置

`run_ale_bench_grpo.sh` 默认使用 colocated GRPO：

- `trainer.algorithm.advantage_estimator=grpo`
- `generator.use_conversation_multi_turn=true`
- `generator.batched=false`
- `environment.env_class=ale_bench`
- `generator.n_samples_per_prompt=4`
- `MAX_TURNS=1`
- `MAX_INPUT_LENGTH=24576`
- `MAX_GENERATE_LENGTH=8192`
- `MAX_ENV_WORKERS=4`

ALE-Bench judge 会启动容器并编译/运行代码，开销比普通文本 reward 大很多，所以默认 `MAX_ENV_WORKERS` 比 `arc_agi3` 更保守。多 GPU 训练时可以逐步增大，但建议先观察 CPU、容器运行时、磁盘和内存压力。

## 推荐运行流程

先在 SkyRL 环境安装 ALE-Bench：

```sh
cd /home/users/yz1051/SkyRL
uv sync --extra dev --extra fsdp
uv pip install -e /home/users/yz1051/ALE-Bench python-dotenv datasets
```

准备 ALE-Bench 容器镜像。如果机器有 Docker：

```sh
cd /home/users/yz1051/ALE-Bench
bash ./scripts/docker_build_202301.sh $(id -u) $(id -g)
```

如果集群没有 Docker 但有 Apptainer/Singularity：

```sh
cd /home/users/yz1051/ALE-Bench
bash ./scripts/apptainer_pull_202301.sh yimjk/ale-bench $HOME/ale-bench-sif
export ALE_BENCH_CONTAINER_BACKEND=apptainer
export ALE_BENCH_APPTAINER_IMAGE_DIR=$HOME/ale-bench-sif
```

ALE-Bench 原生会在 `start()` 时编译 Rust tools，并在 `public_eval()` 里编译/运行提交代码。现在容器后端由 `ALE_BENCH_CONTAINER_BACKEND` 控制；默认是 `docker`，在没有 Docker 的 Slurm 集群上应设为 `apptainer`。
pull 脚本会优先生成 `.sif` 文件；如果集群限制导致 `mksquashfs`/SIF 创建失败，会自动退回到 Apptainer sandbox 目录。运行时会在 `ALE_BENCH_APPTAINER_IMAGE_DIR` 下同时识别这两种形式。
Apptainer 后端默认启用 `--writable-tmpfs`，用于创建 `/workdir`、`/judge` 等 bind mount point；如果集群不允许该参数，可以设置 `ALE_BENCH_APPTAINER_WRITABLE_TMPFS=0` 后再测试。

生成训练数据：

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

启动训练：

```sh
DATA_DIR=$HOME/data/ale_bench \
ALE_BENCH_REPO=/home/users/yz1051/ALE-Bench \
ALE_BENCH_CONTAINER_BACKEND=apptainer \
ALE_BENCH_APPTAINER_IMAGE_DIR=$HOME/ale-bench-sif \
LOGGER=console \
bash examples/train_integrations/ale_bench/run_ale_bench_grpo.sh
```

## 和 ARC-AGI3 集成的区别

两者都走 `BaseTextEnv`、parquet dataset、entrypoint 注册、GRPO 脚本这一套 SkyRL 机制，但任务形态不同：

- `arc_agi3` 是交互式离散动作环境，每轮模型输出 `<action>`。
- `ale_bench` 是代码提交环境，每轮模型输出完整源代码。
- `arc_agi3` reward 来自游戏状态变化和通关进度。
- `ale_bench` reward 来自 public judge 分数。
- `arc_agi3` observation 主要是 frame/diff。
- `ale_bench` observation 主要是题面、样例、工具说明和 public eval 摘要。

因此 ALE-Bench 的主要瓶颈不是动作解析，而是上下文长度、代码生成长度、Docker judge 吞吐和 reward 方差。

## 当前限制与后续方向

当前实现是最小可训练版本，主要限制如下：

- 只用 `public_eval`，不做 `private_eval`。
- 默认只做单轮提交，self-refine 需要显式增加 `MAX_TURNS` 和 `reward_mode=improvement`。
- feedback 比较粗，只返回 case 级摘要。
- reward scale 需要按题目调参，不同 AHC 问题原始分数范围差异很大。
- 默认 prompt 没有加入 few-shot，也没有固定代码模板。
- 容器 judge 依赖本机 image、运行时权限和 ALE-Bench 数据缓存。
- Apptainer 后端需要先把 Docker Hub 预构建镜像拉成本地 `.sif` 或 sandbox 目录，并确保 `ALE_BENCH_APPTAINER_IMAGE_DIR` 指向该目录。

后续可以增强的方向：

- 按题目维护 `score_scale` 和默认语言。
- 加入 baseline solution 或 problem-specific scaffold。
- 多轮时回传更有用的 public case stderr、局部可视化或统计特征。
- 增加离线 smoke rollout，自动验证一个 trivial solution 能被 judge 处理。
- 将训练日志中的 `result` metadata 做专门可视化，方便比较每轮 public score。
