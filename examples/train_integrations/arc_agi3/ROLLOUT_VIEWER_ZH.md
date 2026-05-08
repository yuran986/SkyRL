# ARC-AGI-3 Rollout Viewer 用法

这个工具把 `dumped_rollouts/*.jsonl` 生成一个静态 HTML，方便查看训练曲线、每条 trajectory 的 `<think>` / `<action>`、reward 明细、frame diff 和 action 后的 frame。

脚本路径：

```bash
examples/train_integrations/arc_agi3/visualize_rollouts.py
```

## 推荐命令

训练结束后先设置本次 run 的 export 目录：

```bash
export EXPORT_PATH=/home/users/yz1051/exports/arc_agi3/arc_agi3_reward_v2_20260507_165418
```

生成 viewer：

```bash
python examples/train_integrations/arc_agi3/visualize_rollouts.py \
  "$EXPORT_PATH" \
  --latest-files 100 \
  --trajectories-per-file 4 \
  -o "$EXPORT_PATH/rollout_viewer.html"
```

含义：

- `--latest-files 100`：详情列表从最近 100 个 `global_step_*_rollouts.jsonl` 文件里取样。
- `--trajectories-per-file 4`：每个 rollout 文件只展示前 4 条 trajectory，避免 HTML 太大。
- `-o "$EXPORT_PATH/rollout_viewer.html"`：输出 HTML 到当前 run 目录。

注意：Training Curves 会扫描 export 里的全部 rollout 文件来算曲线，不受 `--latest-files`、`--trajectories-per-file`、`--max-trajectories`、`--step-from` 或 `--step-to` 影响。也就是说，详情可以抽样或分段查看，曲线始终展示完整训练过程。

## 常用变体

只看某段 global step 的 trajectory 详情：

```bash
python examples/train_integrations/arc_agi3/visualize_rollouts.py \
  "$EXPORT_PATH" \
  --step-from 100 \
  --step-to 110 \
  --trajectories-per-file 16 \
  -o "$EXPORT_PATH/rollout_viewer.html"
```

只看最近少量文件，适合快速检查：

```bash
python examples/train_integrations/arc_agi3/visualize_rollouts.py \
  "$EXPORT_PATH" \
  --latest-files 20 \
  --trajectories-per-file 4 \
  -o "$EXPORT_PATH/rollout_viewer_latest20.html"
```

限制总展示 trajectory 数：

```bash
python examples/train_integrations/arc_agi3/visualize_rollouts.py \
  "$EXPORT_PATH" \
  --max-trajectories 100 \
  -o "$EXPORT_PATH/rollout_viewer_100.html"
```

如果你真的想把所有 rollout 都塞进一个静态 HTML，可以加：

```bash
--allow-large
```

但不建议对多 GB export 使用。JSONL 读进 Python 对象再嵌入 HTML 后会膨胀很多，容易被系统 OOM kill。

## 页面怎么看

页面上方：

- `Training Curves From Rollouts`：按 `global_step` 展示 reward、正样本率、invalid 率、平均 turn、level 完成情况、success rate。
- Summary cards：当前 HTML 中展示的 trajectory 统计。

左侧列表：

- 每行是一个 trajectory。
- 标题里的 `step X / sample Y` 对应 `global_step_X_rollouts.jsonl` 里的 `sample_index=Y`。
- `sample_index` 是一个 rollout batch 里的扁平索引。比如 `TRAIN_BATCH_SIZE=4` 且 `N_SAMPLES_PER_PROMPT=4` 时，每个文件有 16 条 trajectory，`sample_index` 是 `0..15`，不是 `0..3`。

右侧详情：

- `Trajectory Reward`：整条 trajectory 的 total reward 和 reward component 汇总。
- 每个 turn 有 `<think>`、`<action>`、observation、reward components、diff stats 和 raw JSON。
- `Frame Review` 面板会在打开 sample 时自动加载第一个可用 turn 的 frame。
- 可以拖动 frame slider 快速预览整条 trajectory 的 action 后 frame。
- 也可以点击每个 turn 里的 `View frame after action`，跳到该 turn 的 frame。

## 从本地浏览器打开

如果你在 SSH 远程服务器上，可以用本地浏览器打开生成的 HTML：

```bash
scp user@server:$EXPORT_PATH/rollout_viewer.html .
```

或者使用 VS Code Remote / SSHFS 直接打开远程文件。这个 viewer 是静态 HTML，不需要启动服务器。

## 常见问题

### `$EXPORT_PATH` 是空的

如果命令变成类似：

```bash
python .../visualize_rollouts.py -o /rollout_viewer.html
```

说明当前 shell 里没有设置 `EXPORT_PATH`。先运行：

```bash
export EXPORT_PATH=/path/to/your/export/run
```

### 进程显示 `killed`

通常是全量 JSONL 太大，被系统 OOM kill。改用：

```bash
--latest-files 100 --trajectories-per-file 4
```

或者用 `--step-from/--step-to` 分段生成多个 viewer。注意这只影响详情列表，不影响 Training Curves。

### 曲线和左侧列表数量不一致

这是预期行为。曲线用于看完整训练流程，会扫描 export 里的全部 rollout 文件；左侧列表用于人工检查具体策略，可以抽样或分段展示。
