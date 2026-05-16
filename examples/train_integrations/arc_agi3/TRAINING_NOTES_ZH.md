# ARC-AGI-3 训练记录

本文记录每次 ARC-AGI-3 GRPO 训练的关键设置、表现和下一次计划。重点关注 rollout 行为是否真的学习规则，而不是只利用 shaping reward。

## 2026-05-04 formal run: `arc_agi3_formal_20260504_005642`

产物路径：

- rollout/eval export: `/home/users/yz1051/exports/arc_agi3/arc_agi3_formal_20260504_005642`
- checkpoint: `/usr/project/xtmp/yz1051/ckpts/arc_agi3_3B_formal`

主要训练设置：

| 参数 | 值 |
| --- | --- |
| model | `Qwen/Qwen2.5-3B-Instruct` |
| placement | `COLOCATE_ALL=false`, 训练 2 GPU + vLLM 2 GPU |
| FSDP | `trainer.strategy=fsdp2` |
| task / seed | 当前数据主要是 `ft09`，少量 seed |
| max turns | `MAX_TURNS=10` |
| context | `MAX_INPUT_LENGTH=16384`, `MAX_MODEL_LEN=18432` |
| generation | `MAX_GENERATE_LENGTH=192`, `temperature=1.0`, `top_p=1.0`, `stop=["</action>"]` |
| GRPO samples | `N_SAMPLES_PER_PROMPT=4` |
| batch | `TRAIN_BATCH_SIZE=4`, `POLICY_MINI_BATCH_SIZE=2` |
| micro batch | `trainer.micro_train_batch_size_per_gpu=1`, `trainer.micro_forward_batch_size_per_gpu=1` |
| eval/checkpoint | `EVAL_INTERVAL=20`, `CKPT_INTERVAL=20`, `trainer.max_ckpts_to_keep=3` |

当时 reward 设置：

| reward component | 值 | 说明 |
| --- | --- | --- |
| `invalid_action_reward` | `-0.05` | 非法动作惩罚偏轻 |
| `level_reward` | `1.0` | 每完成一个 level 的奖励 |
| `done_reward` | `1.0` | 环境 `done=True` 时奖励 |
| `meaningful_diff_reward` | `0.05` | 只要 frame diff 在 `[1, 512]` 内就给 |
| `min_meaningful_diff_changes` | `1` | meaningful diff 下限 |
| `max_meaningful_diff_changes` | `512` | meaningful diff 上限 |

step 12 关键指标：

| 指标 | 值 |
| --- | --- |
| `environment/steps` | `7.1875` |
| `environment/invalid_actions` | `0.6875` |
| `environment/levels_completed` | `0.0000` |
| `environment/success` | `0.0000` |
| `reward/avg_pass_at_4` | `0.5000` |
| `reward/avg_raw_reward` | `-0.0125` |
| `reward/mean_positive_reward` | `0.0219` |
| `eval/all/avg_score` | `0.0500` |
| `eval/all/pass_at_1` | `0.5000` |
| `generate/avg_num_tokens` | `2380.9375` |
| `generate/max_num_tokens` | `3550` |
| `policy/policy_kl` | `0.0113` |
| `policy/policy_entropy` | `0.7344` |

人工观察：

- agent 倾向于反复点击中心或附近一小片区域，后续也会沿相邻坐标尝试。
- 仍有 `x=64` 或 `y=64` 这类越界动作，说明坐标边界没有完全学稳。
- 一些 trajectory 能拿到 positive shaping reward，但几乎没有 `levels_completed` 或 `success`。
- 这更像利用 `meaningful_diff_reward=0.05` 的局部最优，而不是学到了游戏规则。

结论：

当前 reward 对“局部 frame diff”奖励太强，且缺少重复点击和无变化惩罚。下一次训练先调整 reward，再考虑 SFT warm-up。SFT 只有在有高质量示范轨迹时才值得做；用当前局部策略 trajectory 直接 SFT 可能会固化坏行为。

## 下一次训练计划：minimal reward v2

代码默认 reward 已改为以下设置。旧 parquet 若没有显式 reward 字段，会自动使用这些新默认值；重新运行 `prepare_dataset.py` 会把这些字段写入 parquet，方便以后追踪。

| reward component | 新默认值 | 目的 |
| --- | --- | --- |
| `invalid_action_reward` | `-0.1` | 加重非法动作惩罚，特别是越界坐标 |
| `level_reward` | `3.0` | 让真实 level 进展压过 diff shaping |
| `done_reward` | `0.0` | 暂不奖励任意 episode 结束，避免把成功、失败、game over 混成同一个正信号 |
| `meaningful_diff_reward` | `0.005` | diff 只作为弱探索信号 |
| `repeat_click_penalty` | `-0.02` | 惩罚连续点击上一次 ACTION6 坐标附近 |
| `repeat_click_radius` | `2` | 重复点击判定半径，覆盖小范围局部抖动 |
| `min_meaningful_diff_changes` | `1` | meaningful diff 下限 |
| `max_meaningful_diff_changes` | `512` | meaningful diff 上限 |

这版刻意不加入 `success_reward`、`score_delta`、novelty reward 或无变化惩罚。原因是先控制变量：下一次只判断“把 diff shaping 从 `0.05` 降到 `0.005`、把 level progress 从 `1.0` 提到 `3.0`、把非法动作从 `-0.05` 加重到 `-0.1`、关闭 ambiguous done reward、加入局部重复点击惩罚”是否能缓解局部区域过拟合。

`done_reward` 和 `success_reward` 的区别：`done` 是环境终止信号，可能代表成功、失败、game over 或其他终止；`success` 应该只代表明确完成目标。为了不增加新变量，这次两个都不单独加，先只依赖 `level_delta` 作为主要真实进展信号。

下一次短测建议：

```bash
python examples/train_integrations/arc_agi3/prepare_dataset.py \
  --output_dir $HOME/data/arc_agi3 \
  --task_ids ft09 \
  --train_size 32 \
  --val_size 8 \
  --max_steps 10
```

然后用正式配置先跑较短版本：

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
DATA_DIR=$HOME/data/arc_agi3 \
CKPT_PATH=/usr/project/xtmp/yz1051/ckpts/arc_agi3_3B_reward_v2 \
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
CKPT_INTERVAL=10 \
EVAL_INTERVAL=10 \
RUN_NAME=arc_agi3_reward_v2 \
bash examples/train_integrations/arc_agi3/run_arc_agi3_grpo.sh \
  trainer.epochs=25 \
  trainer.eval_before_train=false \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.max_ckpts_to_keep=3 \
  trainer.log_path=$HOME/skyrl_logs/arc_agi3
```

短测重点看：

- `reward_components` 里 `meaningful_diff` 是否不再主导总 reward。
- `reward_components.repeat_click` 出现时，是否压低固定小区域循环的总收益。
- `levels_completed` 是否开始出现，或者至少局部小区域循环是否减少。
- `invalid_actions` 是否因为更强惩罚下降，特别是 `x=64`、`y=64` 这类越界动作。
- rollout viewer 里点击轨迹是否从固定小区域扩展到有策略的搜索。

若这版 reward 仍然只学到探索不学规则，再考虑加入小规模 SFT warm-up。SFT 数据必须是人工筛选或脚本生成的高质量轨迹，至少要覆盖合法坐标、观察 diff 后改变策略、避免重复无效点击这些行为。

## Observation v3：structured diff

`reward_v2` 跑完后发现模型主要学会制造固定局部 diff，而不是理解 diff。尤其后期 trajectory 会坍缩到固定坐标序列，`levels_completed/success/final_score` 仍为 0。这说明仅给整体 `frame_diff`、若干 changed examples 和 current-only `changed_patch` 不够清晰，模型容易把 `num_changes > 0` 当成奖励提示。

当前 observation 已改成 structured diff：

| 字段 | 含义 |
| --- | --- |
| `frame_diff.num_changes` | 上一步 action 后变化的 cell 数 |
| `frame_diff.bbox` | 所有变化 cell 的整体 bbox，仅做粗定位 |
| `frame_diff.colors` | 按颜色变化聚合的统计 |
| `components` | 按“4 邻接连续区域 + 相同 before->after 颜色变化”拆分出的变化片段 |
| `changed_patch_before` | changed bbox 附近的上一帧局部 patch |
| `changed_patch` | changed bbox 附近的当前帧局部 patch |
| `changed_patch_delta` | changed bbox 附近的 delta patch，`.` 是未变，hex 字符是变化后的颜色 |

为了减少上下文噪声，`frame_diff` 文本不再输出全局 `examples=[...]`。这些 examples 仍保留在 rollout metadata 里，方便 viewer/debug，但不作为模型主 observation。默认 `full_frame_interval` 也从 `8` 改为 `0`：模型 initial 仍能看到一次完整 frame 和颜色 legend，后续 turn 默认只看 structured diff 与局部 patch，不周期性塞整张 64x64 frame。

注意：`prepare_dataset.py` 会把 prompt 和 `full_frame_interval` 写进 parquet。若继续使用旧 `$HOME/data/arc_agi3/*.parquet`，可能仍然是旧 prompt 和 `full_frame_interval=8`。下一次训练前需要重新生成数据。
