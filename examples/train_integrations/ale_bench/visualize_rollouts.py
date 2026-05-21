from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


GLOBAL_STEP_RE = re.compile(r"global_step_(\d+)")
CODE_FENCE_RE = re.compile(r"```([A-Za-z0-9_+#.-]*)\s*\n(.*?)```", re.DOTALL)
TAG_RE = re.compile(
    r"<(?:solution|code)(?:\s+language=[\"']?([^\"'>\s]+)[\"']?)?\s*>(.*?)</(?:solution|code)>",
    re.DOTALL | re.IGNORECASE,
)
DEFAULT_MAX_INPUT_MB = 256


def global_step_from_path(path: Path) -> int | None:
    match = GLOBAL_STEP_RE.search(path.name) or GLOBAL_STEP_RE.search(str(path.parent))
    return int(match.group(1)) if match else None


def rollout_sort_key(path: Path) -> tuple[int, int, str]:
    step = global_step_from_path(path)
    return (0, step, path.name) if step is not None else (1, 0, path.name)


def iter_jsonl(path: Path):
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


def find_rollout_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_file():
            files.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        dumped = path / "dumped_rollouts"
        search_dir = dumped if dumped.is_dir() else path
        files.extend(search_dir.glob("*_rollouts.jsonl"))

    unique: list[Path] = []
    seen: set[Path] = set()
    for file_path in sorted(files, key=rollout_sort_key):
        resolved = file_path.resolve()
        if resolved not in seen:
            unique.append(file_path)
            seen.add(resolved)
    if not unique:
        raise FileNotFoundError("no *_rollouts.jsonl files found")
    return unique


def filter_files(
    files: list[Path],
    latest_files: int | None,
    step_from: int | None,
    step_to: int | None,
) -> list[Path]:
    filtered = []
    for file_path in files:
        step = global_step_from_path(file_path)
        if step_from is not None and (step is None or step < step_from):
            continue
        if step_to is not None and (step is None or step > step_to):
            continue
        filtered.append(file_path)
    if latest_files is not None:
        if latest_files <= 0:
            raise ValueError("--latest-files must be positive")
        filtered = filtered[-latest_files:]
    if not filtered:
        raise FileNotFoundError("no rollout files remain after filtering")
    return filtered


def total_size_mb(files: list[Path]) -> float:
    return sum(file_path.stat().st_size for file_path in files) / (1024 * 1024)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def truncate(text: Any, max_chars: int) -> str:
    value = "" if text is None else str(text)
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars] + f"\n... truncated {len(value) - max_chars} chars"


def normalize_language(language: str | None, default: str) -> str:
    if not language:
        return default
    language = language.strip().lower()
    aliases = {
        "c++": "cpp20",
        "c++17": "cpp17",
        "c++20": "cpp20",
        "c++23": "cpp23",
        "cpp": "cpp20",
        "py": "python",
        "rs": "rust",
    }
    return aliases.get(language, language)


def extract_code(output: str, default_language: str) -> tuple[str, str, str]:
    tag_match = TAG_RE.search(output or "")
    if tag_match:
        return html.unescape(tag_match.group(2)).strip(), normalize_language(tag_match.group(1), default_language), "tag"

    fence_matches = CODE_FENCE_RE.findall(output or "")
    if fence_matches:
        language, code = fence_matches[-1]
        return code.strip(), normalize_language(language, default_language), "fence"

    stripped = (output or "").strip()
    return stripped, default_language, "raw" if stripped else "empty"


def result_metadata(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    steps = row.get("steps") or []
    if not steps:
        return {}, {}
    metadata = steps[-1].get("metadata") or {}
    return metadata, metadata.get("result") or {}


def first_case_message(result: dict[str, Any]) -> str:
    for case in result.get("case_results") or []:
        message = case.get("message")
        if message:
            return str(message)
    return ""


def failure_bucket(judge: str, message: str, error: str | None) -> str:
    if error:
        return "ENV_ERROR"
    if judge == "COMPILATION_ERROR":
        if "missing terminating" in message:
            return "CE: missing quote"
        if "not declared" in message:
            return "CE: not declared"
        if "expected" in message:
            return "CE: syntax"
        if "unknown start of token" in message:
            return "CE: bad token"
        return "CE: other"
    if judge == "WRONG_ANSWER":
        return "WA"
    if judge == "RUNTIME_ERROR":
        return "RE"
    if judge == "TIME_LIMIT_EXCEEDED":
        return "TLE"
    if judge == "ACCEPTED":
        return "AC"
    return judge or "NONE"


def summarize_row(row: dict[str, Any], detail_index: int | None, args: argparse.Namespace) -> dict[str, Any]:
    metadata, result = result_metadata(row)
    env_metrics = row.get("env_metrics") or {}
    extras = metadata.get("env_extras") or row.get("env_extras") or {}
    model_output = ""
    steps = row.get("steps") or []
    if steps:
        model_output = steps[-1].get("model_output") or ""
    default_language = str(metadata.get("code_language") or extras.get("code_language") or "cpp20")
    code, extracted_language, extraction = extract_code(model_output, default_language)
    judge = str(result.get("overall_judge_result") or metadata.get("best_judge_result") or "NONE")
    case_message = first_case_message(result)
    error = metadata.get("error")
    reward = as_float(row.get("total_reward"), as_float(steps[-1].get("reward") if steps else None))
    absolute_score = result.get("overall_absolute_score")

    summary = {
        "detail_index": detail_index,
        "global_step": row.get("global_step") or global_step_from_path(Path(row.get("_source_file", ""))),
        "sample_index": row.get("sample_index"),
        "uid": row.get("uid"),
        "source_file": row.get("_source_file"),
        "reward": reward,
        "stop_reason": row.get("stop_reason"),
        "num_steps": row.get("num_steps") or len(steps),
        "problem_id": metadata.get("problem_id") or extras.get("problem_id"),
        "language": metadata.get("code_language") or default_language,
        "extracted_language": extracted_language,
        "extraction": extraction,
        "judge": judge,
        "failure_bucket": failure_bucket(judge, case_message, error),
        "absolute_score": absolute_score,
        "relative_score": result.get("overall_relative_score"),
        "best_absolute_score": metadata.get("best_absolute_score") or env_metrics.get("best_absolute_score"),
        "best_signed_score": metadata.get("best_signed_score") or env_metrics.get("best_signed_score"),
        "valid_submission": metadata.get("valid_submission"),
        "error": error,
        "code_chars": metadata.get("code_chars") if metadata.get("code_chars") is not None else len(code),
        "output_chars": len(model_output),
        "has_explanation": bool(model_output and extraction != "raw" and model_output.strip().find("```") > 0),
    }

    if detail_index is not None:
        case_results = []
        for idx, case in enumerate((result.get("case_results") or [])[: args.case_limit]):
            case_results.append(
                {
                    "idx": idx,
                    "judge": case.get("judge_result"),
                    "score": case.get("absolute_score"),
                    "time": case.get("execution_time"),
                    "memory": case.get("memory_usage"),
                    "message": truncate(case.get("message"), args.max_message_chars),
                }
            )
        summary.update(
            {
                "model_output": truncate(model_output, args.max_output_chars),
                "code": truncate(code, args.max_code_chars),
                "case_message": truncate(case_message, args.max_message_chars),
                "case_results": case_results,
                "initial_prompt_tail": truncate((row.get("initial_messages") or [{}])[-1].get("content", ""), 4000),
            }
        )
    return summary


def step_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        step = row.get("global_step")
        if step is None:
            step = global_step_from_path(Path(row.get("_source_file", "")))
        if step is not None:
            grouped[int(step)].append(row)

    summaries = []
    for step in sorted(grouped):
        rows_for_step = grouped[step]
        rewards = [as_float(row["reward"]) for row in rows_for_step]
        judges = Counter(row["judge"] for row in rows_for_step)
        buckets = Counter(row["failure_bucket"] for row in rows_for_step)
        summaries.append(
            {
                "global_step": step,
                "count": len(rows_for_step),
                "avg_reward": mean(rewards) if rewards else 0.0,
                "max_reward": max(rewards) if rewards else 0.0,
                "min_reward": min(rewards) if rewards else 0.0,
                "accepted": judges.get("ACCEPTED", 0),
                "positive": sum(1 for reward in rewards if reward > 0),
                "compile_error": judges.get("COMPILATION_ERROR", 0),
                "runtime_error": judges.get("RUNTIME_ERROR", 0),
                "wrong_answer": judges.get("WRONG_ANSWER", 0),
                "time_limit": judges.get("TIME_LIMIT_EXCEEDED", 0),
                "judges": dict(judges),
                "buckets": dict(buckets),
            }
        )
    return summaries


def read_eval_summaries(paths: list[Path]) -> list[dict[str, Any]]:
    evals: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        root = path if path.is_dir() else path.parent.parent
        eval_dir = root / "dumped_evals"
        if not eval_dir.is_dir():
            continue
        for result_path in sorted(eval_dir.glob("global_step_*_evals/aggregated_results.jsonl"), key=rollout_sort_key):
            resolved = result_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            step = global_step_from_path(result_path)
            for row in iter_jsonl(result_path):
                row.pop("_source_file", None)
                row["global_step"] = step
                evals.append(row)
    return evals


def build_data(files: list[Path], detail_files: list[Path], args: argparse.Namespace) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    for file_path in files:
        for row in iter_jsonl(file_path):
            all_rows.append(summarize_row(row, None, args))

    detail_rows: list[dict[str, Any]] = []
    detail_count = 0
    for file_path in detail_files:
        rows_from_file = 0
        for raw in iter_jsonl(file_path):
            if args.trajectories_per_file is not None and rows_from_file >= args.trajectories_per_file:
                break
            if args.max_trajectories is not None and detail_count >= args.max_trajectories:
                break
            detail_rows.append(summarize_row(raw, detail_count, args))
            detail_count += 1
            rows_from_file += 1

    rewards = [row["reward"] for row in all_rows]
    judges = Counter(row["judge"] for row in all_rows)
    buckets = Counter(row["failure_bucket"] for row in all_rows)
    return {
        "meta": {
            "num_files": len(files),
            "num_detail_files": len(detail_files),
            "num_rollouts": len(all_rows),
            "num_details": len(detail_rows),
            "input_files": [str(path) for path in files],
        },
        "overview": {
            "avg_reward": mean(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
            "min_reward": min(rewards) if rewards else 0.0,
            "positive": sum(1 for reward in rewards if reward > 0),
            "accepted": judges.get("ACCEPTED", 0),
            "judges": dict(judges),
            "buckets": dict(buckets),
        },
        "steps": step_summaries(all_rows),
        "details": detail_rows,
        "evals": read_eval_summaries(args.paths),
    }


def render_html(data: dict[str, Any]) -> str:
    payload = (
        json.dumps(data, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ALE-Bench Rollout Viewer</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d8dee8;
      --accent: #1264a3;
      --bad: #b42318;
      --warn: #b54708;
      --good: #027a48;
      --code: #111827;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font: 14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; background: var(--bg); color: var(--ink); }}
    header {{ padding: 18px 24px 12px; border-bottom: 1px solid var(--line); background: var(--panel); position: sticky; top: 0; z-index: 2; }}
    h1 {{ margin: 0 0 10px; font-size: 24px; }}
    main {{ padding: 18px 24px 40px; max-width: 1440px; margin: 0 auto; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
    .toolbar input, .toolbar select {{ height: 34px; border: 1px solid var(--line); border-radius: 6px; padding: 0 10px; background: #fff; }}
    .cards {{ display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 10px; margin: 14px 0; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }}
    .label {{ color: var(--muted); font-size: 12px; }}
    .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
    .grid {{ display: grid; grid-template-columns: minmax(300px, 420px) minmax(0, 1fr); gap: 14px; align-items: start; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    .panel h2 {{ margin: 0; padding: 12px 14px; font-size: 16px; border-bottom: 1px solid var(--line); }}
    .panel-body {{ padding: 12px 14px; }}
    canvas {{ width: 100%; height: 220px; display: block; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ font-size: 12px; color: var(--muted); background: #f8fafc; position: sticky; top: 80px; }}
    tr[data-detail-index] {{ cursor: pointer; }}
    tr[data-detail-index]:hover {{ background: #f4f7fb; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; border: 1px solid var(--line); background: #fff; }}
    .bad {{ color: var(--bad); }}
    .warn {{ color: var(--warn); }}
    .good {{ color: var(--good); }}
    .muted {{ color: var(--muted); }}
    pre {{ margin: 0; padding: 12px; overflow: auto; border-radius: 6px; background: var(--code); color: #e5e7eb; max-height: 520px; white-space: pre-wrap; }}
    .split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .kv {{ display: grid; grid-template-columns: 150px 1fr; gap: 4px 10px; margin-bottom: 10px; }}
    .hidden {{ display: none; }}
    @media (max-width: 1000px) {{ .cards {{ grid-template-columns: repeat(2, 1fr); }} .grid, .split {{ grid-template-columns: 1fr; }} th {{ position: static; }} }}
  </style>
</head>
<body>
<header>
  <h1>ALE-Bench Rollout Viewer</h1>
  <div class="toolbar">
    <input id="search" type="search" placeholder="Search output, error, uid">
    <select id="judgeFilter"><option value="">All judge results</option></select>
    <select id="bucketFilter"><option value="">All failure buckets</option></select>
    <select id="stepFilter"><option value="">All steps</option></select>
  </div>
</header>
<main>
  <section class="cards" id="cards"></section>
  <section class="grid">
    <div class="panel">
      <h2>Training Curves</h2>
      <div class="panel-body">
        <canvas id="rewardChart" width="760" height="260"></canvas>
        <div class="muted">Blue: avg reward. Red bars: compilation errors. Green bars: accepted.</div>
      </div>
    </div>
    <div class="panel">
      <h2>Step Summary</h2>
      <div class="panel-body" style="max-height: 320px; overflow:auto;">
        <table id="stepTable"></table>
      </div>
    </div>
  </section>
  <section class="panel" style="margin-top:14px;">
    <h2>Trajectory Details</h2>
    <div class="panel-body" style="max-height: 520px; overflow:auto;">
      <table id="detailTable"></table>
    </div>
  </section>
  <section class="panel" style="margin-top:14px;">
    <h2 id="selectedTitle">Selected Trajectory</h2>
    <div class="panel-body" id="selected"></div>
  </section>
</main>
<script id="viewer-data" type="application/json">{payload}</script>
<script>
const data = JSON.parse(document.getElementById('viewer-data').textContent);
const details = data.details || [];
const steps = data.steps || [];

function fmt(x, digits = 4) {{
  if (x === null || x === undefined || Number.isNaN(Number(x))) return 'n/a';
  return Number(x).toFixed(digits);
}}
function esc(s) {{
  return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}
function clsJudge(j) {{
  if (j === 'ACCEPTED') return 'good';
  if (j === 'WRONG_ANSWER' || j === 'TIME_LIMIT_EXCEEDED') return 'warn';
  if (j && j !== 'NONE') return 'bad';
  return 'muted';
}}
function cards() {{
  const ov = data.overview || {{}};
  const items = [
    ['Rollouts', data.meta.num_rollouts],
    ['Avg Reward', fmt(ov.avg_reward)],
    ['Max Reward', fmt(ov.max_reward)],
    ['Positive', ov.positive || 0],
    ['Accepted', ov.accepted || 0],
    ['Files', data.meta.num_files],
  ];
  document.getElementById('cards').innerHTML = items.map(([k,v]) => `<div class="card"><div class="label">${{esc(k)}}</div><div class="value">${{esc(v)}}</div></div>`).join('');
}}
function populateFilters() {{
  const judges = [...new Set(details.map(d => d.judge).filter(Boolean))].sort();
  const buckets = [...new Set(details.map(d => d.failure_bucket).filter(Boolean))].sort();
  const stepValues = [...new Set(details.map(d => d.global_step).filter(x => x !== null && x !== undefined))].sort((a,b)=>a-b);
  document.getElementById('judgeFilter').innerHTML += judges.map(j => `<option value="${{esc(j)}}">${{esc(j)}}</option>`).join('');
  document.getElementById('bucketFilter').innerHTML += buckets.map(b => `<option value="${{esc(b)}}">${{esc(b)}}</option>`).join('');
  document.getElementById('stepFilter').innerHTML += stepValues.map(s => `<option value="${{s}}">step ${{s}}</option>`).join('');
}}
function filteredDetails() {{
  const q = document.getElementById('search').value.toLowerCase();
  const jf = document.getElementById('judgeFilter').value;
  const bf = document.getElementById('bucketFilter').value;
  const sf = document.getElementById('stepFilter').value;
  return details.filter(d => {{
    if (jf && d.judge !== jf) return false;
    if (bf && d.failure_bucket !== bf) return false;
    if (sf && String(d.global_step) !== sf) return false;
    if (!q) return true;
    return [d.uid, d.problem_id, d.judge, d.failure_bucket, d.model_output, d.case_message, d.error]
      .some(v => String(v ?? '').toLowerCase().includes(q));
  }});
}}
function renderStepTable() {{
  document.getElementById('stepTable').innerHTML = `<thead><tr><th>Step</th><th>N</th><th>Avg</th><th>Max</th><th>AC</th><th>CE</th><th>RE</th><th>WA</th><th>TLE</th></tr></thead><tbody>` +
    steps.map(s => `<tr><td>${{s.global_step}}</td><td>${{s.count}}</td><td>${{fmt(s.avg_reward)}}</td><td>${{fmt(s.max_reward)}}</td><td class="good">${{s.accepted}}</td><td class="bad">${{s.compile_error}}</td><td class="bad">${{s.runtime_error}}</td><td class="warn">${{s.wrong_answer}}</td><td class="warn">${{s.time_limit}}</td></tr>`).join('') +
    `</tbody>`;
}}
function renderDetailTable() {{
  const rows = filteredDetails();
  document.getElementById('detailTable').innerHTML = `<thead><tr><th>Step</th><th>Sample</th><th>Reward</th><th>Judge</th><th>Bucket</th><th>Score</th><th>Lang</th><th>Chars</th></tr></thead><tbody>` +
    rows.map(d => `<tr data-detail-index="${{d.detail_index}}"><td>${{d.global_step ?? ''}}</td><td>${{d.sample_index ?? ''}}</td><td>${{fmt(d.reward)}}</td><td class="${{clsJudge(d.judge)}}">${{esc(d.judge)}}</td><td><span class="badge">${{esc(d.failure_bucket)}}</span></td><td>${{d.absolute_score ?? ''}}</td><td>${{esc(d.extracted_language)}}/${{esc(d.extraction)}}</td><td>${{d.code_chars ?? ''}}</td></tr>`).join('') +
    `</tbody>`;
  document.querySelectorAll('tr[data-detail-index]').forEach(row => row.addEventListener('click', () => selectDetail(Number(row.dataset.detailIndex))));
}}
function selectDetail(index) {{
  const d = details.find(x => x.detail_index === index);
  if (!d) return;
  document.getElementById('selectedTitle').textContent = `Selected Trajectory: step ${{d.global_step}}, sample ${{d.sample_index}}`;
  const cases = (d.case_results || []).map(c => `<tr><td>${{c.idx}}</td><td class="${{clsJudge(c.judge)}}">${{esc(c.judge)}}</td><td>${{c.score ?? ''}}</td><td>${{fmt(c.time, 3)}}</td><td>${{c.memory ?? ''}}</td><td><pre>${{esc(c.message)}}</pre></td></tr>`).join('');
  document.getElementById('selected').innerHTML = `
    <div class="kv">
      <div class="label">UID</div><div>${{esc(d.uid)}}</div>
      <div class="label">Problem</div><div>${{esc(d.problem_id)}}</div>
      <div class="label">Reward</div><div>${{fmt(d.reward)}}</div>
      <div class="label">Judge</div><div class="${{clsJudge(d.judge)}}">${{esc(d.judge)}} / ${{esc(d.failure_bucket)}}</div>
      <div class="label">Source</div><div>${{esc(d.source_file)}}</div>
    </div>
    <div class="split">
      <div><h3>Extracted Code</h3><pre>${{esc(d.code)}}</pre></div>
      <div><h3>Model Output</h3><pre>${{esc(d.model_output)}}</pre></div>
    </div>
    <h3>Case Results</h3>
    <table><thead><tr><th>#</th><th>Judge</th><th>Score</th><th>Time</th><th>Memory</th><th>Message</th></tr></thead><tbody>${{cases}}</tbody></table>
    <h3>Prompt Tail</h3><pre>${{esc(d.initial_prompt_tail)}}</pre>`;
}}
function renderChart() {{
  const canvas = document.getElementById('rewardChart');
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = '#fff'; ctx.fillRect(0,0,w,h);
  if (!steps.length) return;
  const pad = 34, innerW = w - pad * 2, innerH = h - pad * 2;
  const minReward = Math.min(...steps.map(s => s.avg_reward), -1);
  const maxReward = Math.max(...steps.map(s => s.avg_reward), 1);
  const y = v => pad + (maxReward - v) / Math.max(1e-9, maxReward - minReward) * innerH;
  const x = i => pad + (steps.length === 1 ? 0 : i / (steps.length - 1) * innerW);
  ctx.strokeStyle = '#d8dee8'; ctx.lineWidth = 1;
  for (let i=0;i<=4;i++) {{ const yy = pad + i/4*innerH; ctx.beginPath(); ctx.moveTo(pad, yy); ctx.lineTo(w-pad, yy); ctx.stroke(); }}
  const maxCount = Math.max(...steps.map(s => s.count || 1));
  steps.forEach((s,i) => {{
    const barW = Math.max(2, innerW / steps.length * 0.35);
    const ceH = (s.compile_error || 0) / maxCount * innerH * 0.45;
    const acH = (s.accepted || 0) / maxCount * innerH * 0.45;
    ctx.fillStyle = 'rgba(180,35,24,0.35)'; ctx.fillRect(x(i)-barW, h-pad-ceH, barW, ceH);
    ctx.fillStyle = 'rgba(2,122,72,0.45)'; ctx.fillRect(x(i), h-pad-acH, barW, acH);
  }});
  ctx.strokeStyle = '#1264a3'; ctx.lineWidth = 2; ctx.beginPath();
  steps.forEach((s,i) => {{ if (i===0) ctx.moveTo(x(i), y(s.avg_reward)); else ctx.lineTo(x(i), y(s.avg_reward)); }});
  ctx.stroke();
  ctx.fillStyle = '#667085'; ctx.font = '12px system-ui';
  ctx.fillText(`reward ${{minReward.toFixed(2)}}..${{maxReward.toFixed(2)}}`, pad, 18);
}}
function render() {{ cards(); populateFilters(); renderStepTable(); renderDetailTable(); renderChart(); if (details.length) selectDetail(details[0].detail_index); }}
['search','judgeFilter','bucketFilter','stepFilter'].forEach(id => document.getElementById(id).addEventListener('input', renderDetailTable));
render();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a static HTML viewer for ALE-Bench SkyRL rollouts.")
    parser.add_argument("paths", nargs="+", type=Path, help="Export directory or *_rollouts.jsonl file.")
    parser.add_argument("-o", "--output", type=Path, help="Output HTML path.")
    parser.add_argument("--latest-files", type=int, default=20, help="Detail rows use only the latest N rollout files.")
    parser.add_argument("--trajectories-per-file", type=int, default=4, help="Detail rows sampled per rollout file.")
    parser.add_argument("--max-trajectories", type=int, default=160, help="Maximum detail trajectories embedded in HTML.")
    parser.add_argument("--step-from", type=int)
    parser.add_argument("--step-to", type=int)
    parser.add_argument("--max-input-mb", type=float, default=DEFAULT_MAX_INPUT_MB)
    parser.add_argument("--max-output-chars", type=int, default=12000)
    parser.add_argument("--max-code-chars", type=int, default=12000)
    parser.add_argument("--max-message-chars", type=int, default=6000)
    parser.add_argument("--case-limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = find_rollout_files(args.paths)
    selected_files = filter_files(files, latest_files=None, step_from=args.step_from, step_to=args.step_to)
    detail_files = filter_files(files, args.latest_files, args.step_from, args.step_to)
    input_mb = total_size_mb(selected_files)
    if input_mb > args.max_input_mb:
        raise SystemExit(
            f"Refusing to scan {input_mb:.1f} MiB of rollout JSONL. "
            "Use --step-from/--step-to or raise --max-input-mb."
        )

    data = build_data(selected_files, detail_files, args)
    output = args.output
    if output is None:
        first = args.paths[0].expanduser()
        output = (first if first.is_dir() else first.parent).resolve() / "ale_rollout_viewer.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(data), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
