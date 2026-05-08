from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
GLOBAL_STEP_RE = re.compile(r"global_step_(\d+)")
DEFAULT_MAX_INPUT_MB = 128


def _global_step_from_path(path: Path) -> int | None:
    match = GLOBAL_STEP_RE.search(path.name)
    return int(match.group(1)) if match else None


def _global_step_value(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rollout_file_sort_key(path: Path) -> tuple[int, int, str]:
    global_step = _global_step_from_path(path)
    return (0, global_step, path.name) if global_step is not None else (1, 0, path.name)


def _trajectory_sort_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    global_step = _global_step_value(row.get("global_step"))
    if global_step is None:
        source_file = row.get("_source_file")
        if source_file:
            global_step = _global_step_from_path(Path(source_file))
    step_group = global_step if global_step is not None else 0

    sample_index = _global_step_value(row.get("sample_index"))
    row_index = _global_step_value(row.get("_row_index"))
    return (
        0 if global_step is not None else 1,
        step_group,
        sample_index if sample_index is not None else 0,
        row_index if row_index is not None else 0,
    )


def _iter_jsonl(path: Path, limit: int | None = None):
    rows_read = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            row["_source_file"] = str(path)
            yield row
            rows_read += 1
            if limit is not None and rows_read >= limit:
                break


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for row in _iter_jsonl(path, limit=limit):
        rows.append(row)
    return rows


def _find_rollout_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_file():
            files.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)

        dumped_rollouts = path / "dumped_rollouts"
        search_dir = dumped_rollouts if dumped_rollouts.is_dir() else path
        files.extend(sorted(search_dir.glob("*_rollouts.jsonl"), key=_rollout_file_sort_key))

    unique = []
    seen = set()
    for file_path in files:
        resolved = file_path.resolve()
        if resolved not in seen:
            unique.append(file_path)
            seen.add(resolved)
    if not unique:
        raise FileNotFoundError("no *_rollouts.jsonl files found")
    return sorted(unique, key=_rollout_file_sort_key)


def _filter_rollout_files(
    files: list[Path],
    latest_files: int | None,
    step_from: int | None,
    step_to: int | None,
) -> list[Path]:
    filtered = []
    for file_path in files:
        global_step = _global_step_from_path(file_path)
        if step_from is not None and (global_step is None or global_step < step_from):
            continue
        if step_to is not None and (global_step is None or global_step > step_to):
            continue
        filtered.append(file_path)

    if latest_files is not None:
        if latest_files <= 0:
            raise ValueError("--latest-files must be positive")
        filtered = filtered[-latest_files:]

    if not filtered:
        raise FileNotFoundError("no rollout files remain after filtering")
    return filtered


def _total_size_mb(files: list[Path]) -> float:
    total_bytes = sum(file_path.stat().st_size for file_path in files)
    return total_bytes / (1024 * 1024)


def _last_tag(pattern: re.Pattern[str], text: str) -> str:
    matches = pattern.findall(text or "")
    return matches[-1].strip() if matches else ""


def _reward_breakdown(row: dict[str, Any]) -> dict[str, Any]:
    steps = row.get("steps") or []
    components: dict[str, float] = {}
    step_reward_sum = 0.0

    for step in steps:
        reward = float(step.get("reward") or 0.0)
        step_reward_sum += reward
        metadata = step.get("metadata") or {}
        reward_components = metadata.get("reward_components") or {}

        for key, value in reward_components.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            components[key] = components.get(key, 0.0) + numeric_value

    total_reward = float(row["total_reward"]) if row.get("total_reward") is not None else step_reward_sum
    return {
        "total_reward": total_reward,
        "step_reward_sum": step_reward_sum,
        "component_sums": components,
    }


def _trajectory_summary(row: dict[str, Any], index: int) -> dict[str, Any]:
    steps = row.get("steps") or []
    rewards = [float(step.get("reward") or 0.0) for step in steps]
    invalid_steps = 0
    positive_steps = 0
    actions: list[str] = []
    for step in steps:
        if float(step.get("reward") or 0.0) > 0:
            positive_steps += 1
        metadata = step.get("metadata") or {}
        if metadata.get("error") or (metadata.get("valid_action") is False):
            invalid_steps += 1
        actions.append(_last_tag(ACTION_RE, step.get("model_output") or ""))

    total_reward = float(row["total_reward"]) if row.get("total_reward") is not None else sum(rewards)
    env_metrics = row.get("env_metrics") or {}
    return {
        "index": index,
        "uid": row.get("uid"),
        "sample_index": row.get("sample_index"),
        "global_step": row.get("global_step"),
        "source_file": row.get("_source_file"),
        "total_reward": total_reward,
        "num_steps": row.get("num_steps") or len(steps),
        "stop_reason": row.get("stop_reason"),
        "invalid_steps": invalid_steps,
        "positive_steps": positive_steps,
        "reward_breakdown": _reward_breakdown(row),
        "success": env_metrics.get("success"),
        "levels_completed": env_metrics.get("levels_completed"),
        "final_score": env_metrics.get("final_score"),
        "actions": actions,
    }


def _build_report_data(
    rows: list[dict[str, Any]],
    max_trajectories: int | None,
    curve_step_summaries: list[dict[str, Any]] | None = None,
    curve_num_trajectories: int | None = None,
    curve_num_files: int | None = None,
) -> dict[str, Any]:
    rows = sorted(rows, key=_trajectory_sort_key)
    if max_trajectories is not None:
        rows = rows[:max_trajectories]

    summaries = [_trajectory_summary(row, index) for index, row in enumerate(rows)]
    step_summaries = curve_step_summaries if curve_step_summaries is not None else _step_summaries(summaries)
    rewards = [summary["total_reward"] for summary in summaries]
    all_steps = [step for row in rows for step in (row.get("steps") or [])]
    invalid_steps = [
        step
        for step in all_steps
        if (step.get("metadata") or {}).get("error") or (step.get("metadata") or {}).get("valid_action") is False
    ]
    positive_steps = [step for step in all_steps if float(step.get("reward") or 0.0) > 0]

    action_counts: dict[str, int] = {}
    for step in all_steps:
        action = _last_tag(ACTION_RE, step.get("model_output") or "") or "(missing action)"
        action_counts[action] = action_counts.get(action, 0) + 1

    return {
        "summary": {
            "num_trajectories": len(rows),
            "num_steps": len(all_steps),
            "avg_reward": mean(rewards) if rewards else 0.0,
            "min_reward": min(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
            "positive_trajectories": sum(1 for reward in rewards if reward > 0),
            "zero_trajectories": sum(1 for reward in rewards if reward == 0),
            "negative_trajectories": sum(1 for reward in rewards if reward < 0),
            "invalid_steps": len(invalid_steps),
            "positive_steps": len(positive_steps),
            "top_actions": sorted(action_counts.items(), key=lambda item: item[1], reverse=True)[:20],
        },
        "curve_summary": {
            "num_trajectories": curve_num_trajectories if curve_num_trajectories is not None else len(rows),
            "num_files": curve_num_files,
            "sampled_for_detail": curve_step_summaries is not None,
        },
        "step_summaries": step_summaries,
        "trajectories": rows,
        "trajectory_summaries": summaries,
    }


def _step_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for summary in summaries:
        step = summary.get("global_step")
        if step is None:
            continue
        grouped.setdefault(int(step), []).append(summary)

    results = []
    for step in sorted(grouped):
        group = grouped[step]
        total_steps = sum(int(item.get("num_steps") or 0) for item in group)
        invalid_steps = sum(int(item.get("invalid_steps") or 0) for item in group)
        positive_steps = sum(int(item.get("positive_steps") or 0) for item in group)
        rewards = [float(item.get("total_reward") or 0.0) for item in group]
        levels = [float(item.get("levels_completed") or 0.0) for item in group]
        successes = [float(item.get("success") or 0.0) for item in group]
        results.append(
            {
                "global_step": step,
                "num_trajectories": len(group),
                "num_steps": total_steps,
                "avg_reward": mean(rewards) if rewards else 0.0,
                "positive_trajectory_rate": sum(1 for reward in rewards if reward > 0) / len(group),
                "negative_trajectory_rate": sum(1 for reward in rewards if reward < 0) / len(group),
                "invalid_step_rate": invalid_steps / total_steps if total_steps else 0.0,
                "positive_step_rate": positive_steps / total_steps if total_steps else 0.0,
                "avg_num_steps": total_steps / len(group),
                "avg_levels_completed": mean(levels) if levels else 0.0,
                "success_rate": mean(successes) if successes else 0.0,
            }
        )
    return results


def _read_curve_step_summaries(files: list[Path]) -> tuple[list[dict[str, Any]], int]:
    summaries = []
    count = 0
    for file_path in files:
        for row in _iter_jsonl(file_path):
            summaries.append(_trajectory_summary(row, count))
            count += 1
    return _step_summaries(summaries), count


def _default_output_path(input_paths: list[Path]) -> Path:
    first = input_paths[0].expanduser()
    if first.is_file():
        return first.with_suffix(".html")
    if first.name == "dumped_rollouts":
        return first.parent / "rollout_viewer.html"
    if (first / "dumped_rollouts").is_dir():
        return first / "rollout_viewer.html"
    return Path("arc_agi3_rollouts.html")


def _html_template(title: str, data_json: str) -> str:
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --line: #d9dee7;
      --muted: #667085;
      --text: #101828;
      --good: #067647;
      --bad: #b42318;
      --warn: #b54708;
      --accent: #175cd3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 16px 20px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 3;
    }}
    h1 {{ margin: 0 0 10px; font-size: 20px; }}
    .stats {{ display: grid; grid-template-columns: repeat(8, minmax(100px, 1fr)); gap: 8px; }}
    .stat {{ border: 1px solid var(--line); border-radius: 6px; padding: 8px; background: #fbfcfe; }}
    .stat b {{ display: block; font-size: 17px; }}
    .stat span {{ color: var(--muted); font-size: 12px; }}
    .toolbar {{ margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
    input, select, button {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      padding: 7px 9px;
      font: inherit;
    }}
    button {{ cursor: pointer; }}
    main {{ display: grid; grid-template-columns: 380px 1fr; min-height: calc(100vh - 130px); }}
    aside {{
      border-right: 1px solid var(--line);
      background: var(--panel);
      overflow: auto;
      max-height: calc(100vh - 132px);
    }}
    .traj {{
      width: 100%;
      text-align: left;
      border: 0;
      border-bottom: 1px solid var(--line);
      border-radius: 0;
      padding: 10px 12px;
      background: #fff;
    }}
    .traj:hover, .traj.active {{ background: #eef4ff; }}
    .traj-top {{ display: flex; justify-content: space-between; gap: 10px; font-weight: 650; }}
    .traj-meta {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
    .pill {{ display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 1px 7px; margin-right: 4px; font-size: 12px; }}
    .good {{ color: var(--good); }}
    .bad {{ color: var(--bad); }}
    .warn {{ color: var(--warn); }}
    .content {{ padding: 16px; overflow: auto; max-height: calc(100vh - 132px); }}
    .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 12px;
      overflow: hidden;
    }}
    .section h2, .section h3 {{
      margin: 0;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      font-size: 15px;
      background: #fbfcfe;
    }}
    .section-body {{ padding: 12px; }}
    .step {{ border-top: 1px solid var(--line); }}
    .step:first-child {{ border-top: 0; }}
    .step-head {{ display: flex; justify-content: space-between; gap: 10px; align-items: center; }}
    .step-head h3 {{ border: 0; background: transparent; padding: 0; }}
    .charts {{ display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 12px; }}
    .chart {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
    }}
    .chart-title {{ display: flex; justify-content: space-between; gap: 8px; margin-bottom: 6px; font-weight: 650; }}
    .chart-title span {{ color: var(--muted); font-weight: 400; font-size: 12px; }}
    .chart svg {{ display: block; width: 100%; height: 160px; }}
    .chart text {{ fill: var(--muted); font-size: 11px; }}
    .chart .grid {{ stroke: #e4e7ec; stroke-width: 1; }}
    .chart .line {{ fill: none; stroke: var(--accent); stroke-width: 2.5; }}
    .chart .dot {{ fill: var(--accent); }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      margin: 8px 0 0;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f8fafc;
      max-height: 300px;
      overflow: auto;
    }}
    details {{ margin-top: 8px; }}
    summary {{ cursor: pointer; color: var(--accent); }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .empty {{ color: var(--muted); padding: 24px; }}
    .action-viewer-row {{ display: flex; justify-content: space-between; gap: 8px; align-items: center; margin-top: 8px; }}
    .action-viewer-row strong {{ flex: 1; }}
    .frame-panel {{
      position: fixed;
      right: 16px;
      bottom: 16px;
      z-index: 5;
      width: 292px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: 0 10px 30px rgba(16, 24, 40, 0.16);
      overflow: hidden;
    }}
    .frame-panel h2 {{
      margin: 0;
      padding: 9px 11px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
      background: #fbfcfe;
    }}
    .frame-panel-body {{ padding: 10px; }}
    .frame-meta {{ color: var(--muted); font-size: 12px; margin-bottom: 8px; }}
    .frame-controls {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 6px 8px;
      align-items: center;
      margin-bottom: 8px;
    }}
    .frame-controls input[type="range"] {{
      width: 100%;
      padding: 0;
    }}
    .frame-turn-label {{
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    #frame-canvas {{
      width: 256px;
      height: 256px;
      image-rendering: pixelated;
      border: 1px solid var(--line);
      background: #fff;
    }}
    @media (max-width: 900px) {{
      .stats {{ grid-template-columns: repeat(2, 1fr); }}
      main {{ grid-template-columns: 1fr; }}
      aside {{ max-height: 300px; border-right: 0; border-bottom: 1px solid var(--line); }}
      .content {{ max-height: none; }}
      .two-col {{ grid-template-columns: 1fr; }}
      .charts {{ grid-template-columns: 1fr; }}
      .frame-panel {{ position: static; width: auto; margin: 0 12px 12px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{safe_title}</h1>
    <div class="stats" id="stats"></div>
    <div class="toolbar">
      <input id="search" type="search" placeholder="Search think/action/observation">
      <select id="reward-filter">
        <option value="all">All rewards</option>
        <option value="positive">Positive trajectories</option>
        <option value="zero">Zero trajectories</option>
        <option value="negative">Negative trajectories</option>
      </select>
      <label><input id="invalid-only" type="checkbox"> invalid only</label>
      <label><input id="positive-step-only" type="checkbox"> has positive step</label>
      <button id="reset">Reset</button>
    </div>
  </header>
  <main>
    <aside id="trajectory-list"></aside>
    <section class="content" id="detail"></section>
  </main>
  <div class="frame-panel" id="frame-panel">
    <h2>Frame After Action</h2>
    <div class="frame-panel-body">
      <div class="frame-controls">
        <input id="frame-scrubber" type="range" min="0" max="0" value="0" disabled>
        <div class="frame-turn-label" id="frame-turn-label">0 / 0</div>
      </div>
      <div class="frame-meta" id="frame-meta">Select an action to view its frame.</div>
      <canvas id="frame-canvas" width="256" height="256"></canvas>
    </div>
  </div>
  <script type="application/json" id="rollout-data">{data_json}</script>
  <script>
    const data = JSON.parse(document.getElementById('rollout-data').textContent);
    const summaries = data.trajectory_summaries;
    const trajectories = data.trajectories;
    const stepSummaries = data.step_summaries || [];
    const curveSummary = data.curve_summary || {{}};
    let activeIndex = 0;
    let currentFrameItems = [];

    const fmt = (value, digits = 4) => {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '';
      return Number(value).toFixed(digits);
    }};

    function text(value) {{
      if (value === null || value === undefined) return '';
      if (typeof value === 'string') return value;
      return JSON.stringify(value, null, 2);
    }}

    const framePalette = [
      '#ffffff', '#f2f4f7', '#cfd4dc', '#85888f',
      '#202124', '#000000', '#d946ef', '#f0abfc',
      '#ef4444', '#2563eb', '#60a5fa', '#facc15',
      '#f97316', '#7f1d1d', '#22c55e', '#7c3aed',
    ];

    function normalizeFrame(value) {{
      if (!value) return null;
      if (value.frame !== undefined) value = value.frame;
      if (Array.isArray(value) && value.length && Array.isArray(value[0]) && Array.isArray(value[0][0])) {{
        value = value[0];
      }}
      if (!Array.isArray(value) || !value.length || !Array.isArray(value[0])) return null;
      return value.map(row => row.map(cell => {{
        const numberValue = typeof cell === 'string' ? parseInt(cell, 16) : Number(cell);
        return Number.isFinite(numberValue) && numberValue >= 0 && numberValue <= 15 ? numberValue : null;
      }}));
    }}

    function parseFrameFromObservationText(source) {{
      const match = String(source || '').match(/current_frame:\\s*\\nshape=\\d+x\\d+ encoding=hex_0_to_f\\n([\\s\\S]*?)(?:\\n\\n|\\nlast_model_output=|\\nRespond with|$)/);
      if (!match) return null;
      const rows = [];
      match[1].split('\\n').forEach(line => {{
        const rowMatch = line.match(/^y\\d+:\\s*([0-9a-f?]+)/i);
        if (!rowMatch) return;
        rows.push([...rowMatch[1]].map(char => char === '?' ? null : parseInt(char, 16)));
      }});
      return rows.length ? rows : null;
    }}

    function frameForStep(step) {{
      const metadata = step.metadata || {{}};
      const directFrame = normalizeFrame(metadata.frame);
      if (directFrame) return directFrame;
      const observationText = (step.observations || []).map(item => item.content || '').join('\\n');
      return parseFrameFromObservationText(observationText);
    }}

    function parsedActionForStep(step, actionText) {{
      const metadata = step.metadata || {{}};
      if (metadata.parsed_action) return metadata.parsed_action;
      try {{
        const parsed = JSON.parse(actionText);
        return parsed && typeof parsed === 'object' ? parsed : null;
      }} catch {{
        return null;
      }}
    }}

    function setFrameItems(items) {{
      currentFrameItems = items;
      const scrubber = document.getElementById('frame-scrubber');
      const label = document.getElementById('frame-turn-label');
      scrubber.disabled = items.length === 0;
      scrubber.min = '0';
      scrubber.max = String(Math.max(0, items.length - 1));
      scrubber.value = '0';
      label.textContent = items.length ? `1 / ${{items.length}}` : '0 / 0';
    }}

    function selectFrame(index) {{
      const item = currentFrameItems[index];
      if (!item) {{
        renderFrame(null, 'No frame available for this trajectory.', null, null);
        return;
      }}
      renderFrame(item.frame, item.label, item.parsedAction, index);
    }}

    function renderFrame(frame, label, parsedAction, frameIndex = null) {{
      const canvas = document.getElementById('frame-canvas');
      const meta = document.getElementById('frame-meta');
      const scrubber = document.getElementById('frame-scrubber');
      const turnLabel = document.getElementById('frame-turn-label');
      if (frameIndex !== null && frameIndex !== undefined && currentFrameItems[frameIndex]) {{
        scrubber.value = String(frameIndex);
        turnLabel.textContent = `${{frameIndex + 1}} / ${{currentFrameItems.length}}`;
      }}
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!frame) {{
        meta.textContent = `${{label}}: frame unavailable in this rollout log.`;
        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        return;
      }}

      const height = frame.length;
      const width = Math.max(...frame.map(row => row.length));
      const cell = Math.min(canvas.width / width, canvas.height / height);
      const xOffset = (canvas.width - width * cell) / 2;
      const yOffset = (canvas.height - height * cell) / 2;
      ctx.fillStyle = '#f8fafc';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      frame.forEach((row, y) => {{
        row.forEach((value, x) => {{
          ctx.fillStyle = value === null ? '#e5e7eb' : framePalette[value] || '#e5e7eb';
          ctx.fillRect(xOffset + x * cell, yOffset + y * cell, Math.ceil(cell), Math.ceil(cell));
        }});
      }});

      if (parsedAction && parsedAction.x !== undefined && parsedAction.y !== undefined) {{
        const x = Number(parsedAction.x);
        const y = Number(parsedAction.y);
        if (Number.isFinite(x) && Number.isFinite(y)) {{
          ctx.strokeStyle = '#111827';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(xOffset + (x + 0.5) * cell, yOffset + (y + 0.5) * cell, Math.max(4, cell * 1.4), 0, Math.PI * 2);
          ctx.stroke();
        }}
      }}
      const actionLabel = parsedAction ? ` action=${{text(parsedAction)}}` : '';
      meta.textContent = `${{label}} · ${{width}}x${{height}}${{actionLabel}}`;
    }}

    function extractLast(re, source) {{
      const matches = [...String(source || '').matchAll(re)];
      return matches.length ? matches[matches.length - 1][1].trim() : '';
    }}

    function rewardClass(value) {{
      value = Number(value || 0);
      if (value > 0) return 'good';
      if (value < 0) return 'bad';
      return '';
    }}

    function renderStats() {{
      const s = data.summary;
      const items = [
        ['Trajectories', s.num_trajectories],
        ['Steps', s.num_steps],
        ['Avg Reward', fmt(s.avg_reward)],
        ['Min / Max', `${{fmt(s.min_reward)}} / ${{fmt(s.max_reward)}}`],
        ['Positive Traj', s.positive_trajectories],
        ['Negative Traj', s.negative_trajectories],
        ['Positive Steps', s.positive_steps],
        ['Invalid Steps', s.invalid_steps],
      ];
      document.getElementById('stats').replaceChildren(...items.map(([label, value]) => {{
        const div = document.createElement('div');
        div.className = 'stat';
        const b = document.createElement('b');
        b.textContent = value;
        const span = document.createElement('span');
        span.textContent = label;
        div.append(b, span);
        return div;
      }}));
    }}

    function matchesFilters(summary) {{
      const q = document.getElementById('search').value.trim().toLowerCase();
      const rewardFilter = document.getElementById('reward-filter').value;
      const invalidOnly = document.getElementById('invalid-only').checked;
      const positiveStepOnly = document.getElementById('positive-step-only').checked;
      if (rewardFilter === 'positive' && summary.total_reward <= 0) return false;
      if (rewardFilter === 'zero' && summary.total_reward !== 0) return false;
      if (rewardFilter === 'negative' && summary.total_reward >= 0) return false;
      if (invalidOnly && summary.invalid_steps === 0) return false;
      if (positiveStepOnly && summary.positive_steps === 0) return false;
      if (!q) return true;
      const row = trajectories[summary.index];
      return JSON.stringify(row).toLowerCase().includes(q);
    }}

    function renderList() {{
      const list = document.getElementById('trajectory-list');
      const visible = summaries.filter(matchesFilters);
      if (!visible.length) {{
        list.innerHTML = '<div class="empty">No trajectories match the filters.</div>';
        document.getElementById('detail').innerHTML = '';
        return;
      }}
      if (!visible.some(s => s.index === activeIndex)) activeIndex = visible[0].index;
      list.replaceChildren(...visible.map(summary => {{
        const button = document.createElement('button');
        button.className = 'traj' + (summary.index === activeIndex ? ' active' : '');
        button.dataset.index = String(summary.index);
        button.onclick = () => {{
          activeIndex = summary.index;
          updateActiveListSelection();
          renderDetail();
          document.getElementById('detail').scrollTo({{ top: 0 }});
        }};
        const reward = document.createElement('span');
        reward.className = rewardClass(summary.total_reward);
        reward.textContent = fmt(summary.total_reward);
        const top = document.createElement('div');
        top.className = 'traj-top';
        top.append(`step ${{summary.global_step ?? '?'}} / sample ${{summary.sample_index ?? summary.index}}`, reward);
        const meta = document.createElement('div');
        meta.className = 'traj-meta';
        meta.textContent = `${{summary.num_steps}} turns, invalid=${{summary.invalid_steps}}, positive_steps=${{summary.positive_steps}}, stop=${{summary.stop_reason || ''}}`;
        button.append(top, meta);
        return button;
      }}));
    }}

    function updateActiveListSelection() {{
      document.querySelectorAll('#trajectory-list .traj').forEach(button => {{
        button.classList.toggle('active', Number(button.dataset.index) === activeIndex);
      }});
    }}

    function renderDetail() {{
      const row = trajectories[activeIndex];
      const summary = summaries.find(s => s.index === activeIndex);
      const detail = document.getElementById('detail');
      if (!row || !summary) {{
        detail.innerHTML = '<div class="empty">Select a trajectory.</div>';
        return;
      }}
      detail.replaceChildren();

      const rewardBreakdown = section('Reward Breakdown');
      const rewardBody = rewardBreakdown.querySelector('.section-body');
      rewardBody.append(kvPre(summary.reward_breakdown || {{}}));
      detail.append(rewardBreakdown);

      const overview = section('Trajectory Overview');
      const overviewBody = overview.querySelector('.section-body');
      overviewBody.append(kvPre({{
        uid: summary.uid,
        source_file: summary.source_file,
        global_step: summary.global_step,
        sample_index: summary.sample_index,
        total_reward: summary.total_reward,
        num_steps: summary.num_steps,
        stop_reason: summary.stop_reason,
        env_metrics: row.env_metrics,
      }}));
      detail.append(overview);

      const curves = renderCurveSection();
      if (curves) detail.append(curves);

      const frameItems = [];
      (row.steps || []).forEach((step, idx) => {{
        const metadata = step.metadata || {{}};
        const modelOutput = step.model_output || '';
        const think = extractLast(/<think>([\\s\\S]*?)<\\/think>/g, modelOutput);
        const action = extractLast(/<action>([\\s\\S]*?)<\\/action>/g, modelOutput);
        const frame = frameForStep(step);
        const parsedAction = parsedActionForStep(step, action);
        let frameIndex = null;
        if (frame) {{
          frameIndex = frameItems.length;
          frameItems.push({{
            frame,
            label: `Turn ${{step.turn ?? idx + 1}}`,
            parsedAction,
          }});
        }}
        const stepSection = section('');
        const body = stepSection.querySelector('.section-body');
        const head = document.createElement('div');
        head.className = 'step-head';
        const title = document.createElement('h3');
        title.textContent = `Turn ${{step.turn ?? idx + 1}}`;
        const reward = document.createElement('span');
        reward.className = 'pill ' + rewardClass(step.reward);
        reward.textContent = `reward ${{fmt(step.reward)}}`;
        head.append(title, reward);
        body.append(head);

        const meta = document.createElement('div');
        meta.innerHTML = `
          <span class="pill">done=${{Boolean(step.done)}}</span>
          <span class="pill">valid=${{metadata.valid_action === undefined ? '' : metadata.valid_action}}</span>
          <span class="pill">parsed=${{htmlEscape(text(metadata.parsed_action))}}</span>
        `;
        body.append(meta);
        if (metadata.error) {{
          const error = document.createElement('pre');
          error.className = 'bad';
          error.textContent = metadata.error;
          body.append(error);
        }}
        body.append(labelPre('Think', think || '(missing <think>)'));

        const actionBlock = labelPre('Action', action || '(missing <action>)');
        const actionHeader = actionBlock.querySelector('strong');
        const actionRow = document.createElement('div');
        actionRow.className = 'action-viewer-row';
        const actionButton = document.createElement('button');
        actionButton.type = 'button';
        actionButton.textContent = frame ? 'View frame after action' : 'Frame unavailable';
        actionButton.disabled = !frame;
        actionButton.onclick = () => renderFrame(frame, `Turn ${{step.turn ?? idx + 1}}`, parsedAction, frameIndex);
        actionRow.append(actionHeader, actionButton);
        actionBlock.prepend(actionRow);
        body.append(actionBlock);

        body.append(labelPre('Observation', text(step.observations)));

        const two = document.createElement('div');
        two.className = 'two-col';
        two.append(labelPre('Reward Components', text(metadata.reward_components)));
        two.append(labelPre('Diff Stats', text(metadata.diff_stats)));
        body.append(two);

        const raw = document.createElement('details');
        const summaryEl = document.createElement('summary');
        summaryEl.textContent = 'Raw step JSON';
        raw.append(summaryEl, kvPre(step));
        body.append(raw);
        detail.append(stepSection);
      }});
      setFrameItems(frameItems);
      if (frameItems.length) {{
        selectFrame(0);
      }} else {{
        renderFrame(null, `step ${{summary.global_step ?? '?'}} / sample ${{summary.sample_index ?? summary.index}}`, null, null);
      }}
    }}

    function renderCurveSection() {{
      if (!stepSummaries.length) return null;
      const sec = section('Training Curves From Rollouts');
      const body = sec.querySelector('.section-body');
      const note = document.createElement('div');
      note.className = 'muted';
      note.textContent = `Curves use ${{curveSummary.num_trajectories ?? stepSummaries.length}} trajectories from ${{curveSummary.num_files ?? '?'}} rollout file(s), independent of the sampled trajectory list below.`;
      body.append(note);
      const charts = document.createElement('div');
      charts.className = 'charts';
      charts.append(
        lineChart('Avg Reward', 'avg_reward', 'trajectory reward'),
        lineChart('Positive Trajectory Rate', 'positive_trajectory_rate', 'fraction'),
        lineChart('Invalid Step Rate', 'invalid_step_rate', 'fraction'),
        lineChart('Avg Turns', 'avg_num_steps', 'turns'),
        lineChart('Avg Levels Completed', 'avg_levels_completed', 'levels'),
        lineChart('Success Rate', 'success_rate', 'fraction'),
      );
      body.append(charts);
      return sec;
    }}

    function lineChart(title, key, unit) {{
      const width = 360;
      const height = 160;
      const pad = 28;
      const values = stepSummaries.map(d => Number(d[key] || 0));
      const steps = stepSummaries.map(d => Number(d.global_step || 0));
      let minY = Math.min(...values);
      let maxY = Math.max(...values);
      if (minY === maxY) {{
        minY -= 0.05;
        maxY += 0.05;
      }}
      const minX = Math.min(...steps);
      const maxX = Math.max(...steps);
      const sx = x => maxX === minX ? width / 2 : pad + ((x - minX) / (maxX - minX)) * (width - pad * 2);
      const sy = y => pad + (1 - ((y - minY) / (maxY - minY))) * (height - pad * 2);
      const points = stepSummaries.map(d => [sx(Number(d.global_step || 0)), sy(Number(d[key] || 0))]);
      const path = points.map((point, i) => `${{i ? 'L' : 'M'}}${{point[0].toFixed(1)}},${{point[1].toFixed(1)}}`).join(' ');
      const latest = values.length ? values[values.length - 1] : 0;

      const div = document.createElement('div');
      div.className = 'chart';
      const titleEl = document.createElement('div');
      titleEl.className = 'chart-title';
      titleEl.innerHTML = `${{htmlEscape(title)}} <span>latest ${{fmt(latest)}} ${{htmlEscape(unit)}}</span>`;
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
      svg.innerHTML = `
        <line class="grid" x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height - pad}}"></line>
        <line class="grid" x1="${{pad}}" y1="${{height - pad}}" x2="${{width - pad}}" y2="${{height - pad}}"></line>
        <text x="${{pad}}" y="16">${{fmt(maxY)}}</text>
        <text x="${{pad}}" y="${{height - 6}}">${{fmt(minY)}}</text>
        <text x="${{pad}}" y="${{height - 10}}" text-anchor="middle">s${{minX}}</text>
        <text x="${{width - pad}}" y="${{height - 10}}" text-anchor="middle">s${{maxX}}</text>
        <path class="line" d="${{path}}"></path>
        ${{points.map(point => `<circle class="dot" cx="${{point[0].toFixed(1)}}" cy="${{point[1].toFixed(1)}}" r="3"></circle>`).join('')}}
      `;
      div.append(titleEl, svg);
      return div;
    }}

    function section(title) {{
      const div = document.createElement('div');
      div.className = 'section';
      if (title) {{
        const h = document.createElement('h2');
        h.textContent = title;
        div.append(h);
      }}
      const body = document.createElement('div');
      body.className = 'section-body';
      div.append(body);
      return div;
    }}

    function labelPre(label, value) {{
      const wrapper = document.createElement('div');
      const strong = document.createElement('strong');
      strong.textContent = label;
      const pre = document.createElement('pre');
      pre.textContent = value;
      wrapper.append(strong, pre);
      return wrapper;
    }}

    function kvPre(value) {{
      const pre = document.createElement('pre');
      pre.textContent = JSON.stringify(value, null, 2);
      return pre;
    }}

    function htmlEscape(value) {{
      return String(value).replace(/[&<>"']/g, c => ({{
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
      }}[c]));
    }}

    ['search', 'reward-filter', 'invalid-only', 'positive-step-only'].forEach(id => {{
      document.getElementById(id).addEventListener('input', () => {{ renderList(); renderDetail(); }});
    }});
    document.getElementById('frame-scrubber').addEventListener('input', event => {{
      selectFrame(Number(event.target.value));
    }});
    document.getElementById('reset').onclick = () => {{
      document.getElementById('search').value = '';
      document.getElementById('reward-filter').value = 'all';
      document.getElementById('invalid-only').checked = false;
      document.getElementById('positive-step-only').checked = false;
      renderList();
      renderDetail();
    }};

    renderStats();
    renderList();
    renderDetail();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static HTML viewer for ARC-AGI-3 rollout JSONL dumps.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Rollout JSONL files, dumped_rollouts directories, or export directories containing dumped_rollouts/.",
    )
    parser.add_argument("--output", "-o", default=None, help="Output HTML path.")
    parser.add_argument("--title", default="ARC-AGI-3 Rollout Viewer")
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument(
        "--trajectories-per-file",
        type=int,
        default=None,
        help="Only load the first N trajectories from each selected rollout file.",
    )
    parser.add_argument("--latest-files", type=int, default=None, help="Only load the latest N rollout JSONL files.")
    parser.add_argument("--step-from", type=int, default=None, help="Only load rollout files at or after this global step.")
    parser.add_argument("--step-to", type=int, default=None, help="Only load rollout files at or before this global step.")
    parser.add_argument(
        "--max-input-mb",
        type=int,
        default=DEFAULT_MAX_INPUT_MB,
        help="Refuse to build a static viewer when selected JSONL input is larger than this many MiB.",
    )
    parser.add_argument("--allow-large", action="store_true", help="Disable the static viewer input size guard.")
    args = parser.parse_args()
    if args.max_trajectories is not None and args.max_trajectories <= 0:
        raise SystemExit("--max-trajectories must be positive")
    if args.trajectories_per_file is not None and args.trajectories_per_file <= 0:
        raise SystemExit("--trajectories-per-file must be positive")

    input_paths = [Path(path) for path in args.paths]
    all_rollout_files = _find_rollout_files(input_paths)
    curve_files = all_rollout_files
    rollout_files = _filter_rollout_files(
        all_rollout_files,
        latest_files=args.latest_files,
        step_from=args.step_from,
        step_to=args.step_to,
    )
    input_size_mb = _total_size_mb(rollout_files)
    if (
        not args.allow_large
        and args.max_trajectories is None
        and args.trajectories_per_file is None
        and input_size_mb > args.max_input_mb
    ):
        raise SystemExit(
            f"Selected rollout JSONL is {input_size_mb:.1f} MiB across {len(rollout_files)} file(s), "
            "which is too large for one self-contained static HTML viewer. "
            "Use --latest-files 100 --trajectories-per-file 4, --step-from/--step-to, "
            "or --max-trajectories, "
            "or pass --allow-large if you really want to embed everything."
        )

    curve_step_summaries, curve_num_trajectories = _read_curve_step_summaries(curve_files)
    rows = []
    for file_path in rollout_files:
        for row in _read_jsonl(file_path, limit=args.trajectories_per_file):
            row["_row_index"] = len(rows)
            rows.append(row)
            if args.max_trajectories is not None and len(rows) >= args.max_trajectories:
                break
        if args.max_trajectories is not None and len(rows) >= args.max_trajectories:
            break
    report_data = _build_report_data(
        rows,
        args.max_trajectories,
        curve_step_summaries=curve_step_summaries,
        curve_num_trajectories=curve_num_trajectories,
        curve_num_files=len(curve_files),
    )

    data_json = json.dumps(report_data, ensure_ascii=False).replace("</", "<\\/")
    output = Path(args.output).expanduser() if args.output else _default_output_path(input_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_html_template(args.title, data_json), encoding="utf-8")

    print(f"Loaded {len(rows)} display trajectories from {len(rollout_files)} file(s), selected input {input_size_mb:.1f} MiB.")
    print(f"Built curves from {curve_num_trajectories} trajectories across {len(curve_files)} file(s).")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
