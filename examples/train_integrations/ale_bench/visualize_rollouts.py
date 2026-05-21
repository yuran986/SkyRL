from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


GLOBAL_STEP_RE = re.compile(r"global_step_(\d+)")
CODE_FENCE_RE = re.compile(r"```([A-Za-z0-9_+#.-]*)\s*\n(.*?)```", re.DOTALL)
TAG_RE = re.compile(
    r"<(?:solution|code)(?:\s+language=[\"']?([^\"'>\s]+)[\"']?)?\s*>(.*?)</(?:solution|code)>",
    re.DOTALL | re.IGNORECASE,
)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
DEFAULT_MAX_INPUT_MB = 128


def _global_step_from_path(path: Path) -> int | None:
    match = GLOBAL_STEP_RE.search(path.name) or GLOBAL_STEP_RE.search(str(path.parent))
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
    if global_step is None and row.get("_source_file"):
        global_step = _global_step_from_path(Path(row["_source_file"]))
    sample_index = _global_step_value(row.get("sample_index"))
    row_index = _global_step_value(row.get("_row_index"))
    return (
        0 if global_step is not None else 1,
        global_step if global_step is not None else 0,
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
    return list(_iter_jsonl(path, limit=limit))


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

    unique: list[Path] = []
    seen: set[Path] = set()
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
    return sum(file_path.stat().st_size for file_path in files) / (1024 * 1024)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truncate(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... truncated {len(text) - max_chars} chars"


def _last_match(pattern: re.Pattern[str], text: str) -> str:
    matches = pattern.findall(text or "")
    if not matches:
        return ""
    value = matches[-1]
    if isinstance(value, tuple):
        value = value[-1]
    return str(value).strip()


def _normalize_language(language: str | None, default: str) -> str:
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


def _extract_code(output: str, default_language: str) -> tuple[str, str, str]:
    tag_match = TAG_RE.search(output or "")
    if tag_match:
        return html.unescape(tag_match.group(2)).strip(), _normalize_language(tag_match.group(1), default_language), "tag"
    fence_matches = CODE_FENCE_RE.findall(output or "")
    if fence_matches:
        language, code = fence_matches[-1]
        return code.strip(), _normalize_language(language, default_language), "fence"
    stripped = (output or "").strip()
    return stripped, default_language, "raw" if stripped else "empty"


def _step_result(step: dict[str, Any]) -> dict[str, Any]:
    metadata = step.get("metadata") or {}
    return metadata.get("result") or {}


def _step_judge(step: dict[str, Any]) -> str:
    metadata = step.get("metadata") or {}
    result = _step_result(step)
    return str(result.get("overall_judge_result") or metadata.get("best_judge_result") or "NONE")


def _first_case_message(result: dict[str, Any]) -> str:
    for case in result.get("case_results") or []:
        if case.get("message"):
            return str(case["message"])
    return ""


def _failure_bucket(judge: str, message: str, error: str | None) -> str:
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


def _is_invalid_judge(judge: str, error: Any) -> bool:
    return bool(error) or judge not in {"ACCEPTED"}


def _reward_breakdown(row: dict[str, Any]) -> dict[str, Any]:
    steps = row.get("steps") or []
    step_reward_sum = sum(_as_float(step.get("reward")) for step in steps)
    last_step = steps[-1] if steps else {}
    metadata = last_step.get("metadata") or {}
    result = _step_result(last_step)
    return {
        "total_reward": _as_float(row.get("total_reward"), step_reward_sum),
        "step_reward_sum": step_reward_sum,
        "overall_judge_result": result.get("overall_judge_result"),
        "overall_absolute_score": result.get("overall_absolute_score"),
        "overall_relative_score": result.get("overall_relative_score"),
        "best_absolute_score": metadata.get("best_absolute_score") or (row.get("env_metrics") or {}).get("best_absolute_score"),
        "best_signed_score": metadata.get("best_signed_score") or (row.get("env_metrics") or {}).get("best_signed_score"),
        "valid_submission": metadata.get("valid_submission"),
    }


def _trajectory_summary(row: dict[str, Any], index: int) -> dict[str, Any]:
    steps = row.get("steps") or []
    rewards = [_as_float(step.get("reward")) for step in steps]
    total_reward = _as_float(row.get("total_reward"), sum(rewards))
    env_metrics = row.get("env_metrics") or {}
    last_step = steps[-1] if steps else {}
    last_metadata = last_step.get("metadata") or {}
    last_result = _step_result(last_step)
    case_message = _first_case_message(last_result)
    judge = _step_judge(last_step) if steps else "NONE"
    error = last_metadata.get("error")
    invalid_steps = 0
    positive_steps = 0
    judges = []
    for step in steps:
        step_judge = _step_judge(step)
        judges.append(step_judge)
        step_error = (step.get("metadata") or {}).get("error")
        if _is_invalid_judge(step_judge, step_error):
            invalid_steps += 1
        if _as_float(step.get("reward")) > 0:
            positive_steps += 1

    return {
        "index": index,
        "uid": row.get("uid"),
        "sample_index": row.get("sample_index"),
        "global_step": row.get("global_step") or _global_step_from_path(Path(row.get("_source_file", ""))),
        "source_file": row.get("_source_file"),
        "total_reward": total_reward,
        "num_steps": row.get("num_steps") or len(steps),
        "stop_reason": row.get("stop_reason"),
        "invalid_steps": invalid_steps,
        "positive_steps": positive_steps,
        "reward_breakdown": _reward_breakdown(row),
        "problem_id": last_metadata.get("problem_id") or (last_metadata.get("env_extras") or {}).get("problem_id"),
        "judge": judge,
        "failure_bucket": _failure_bucket(judge, case_message, error),
        "accepted": judge == "ACCEPTED",
        "absolute_score": last_result.get("overall_absolute_score"),
        "relative_score": last_result.get("overall_relative_score"),
        "valid_submission": last_metadata.get("valid_submission"),
        "code_language": last_metadata.get("code_language"),
        "has_error": bool(error),
        "judges": judges,
    }


def _case_result_view(case: dict[str, Any], idx: int, max_message_chars: int) -> dict[str, Any]:
    return {
        "idx": idx,
        "judge": case.get("judge_result"),
        "score": case.get("absolute_score"),
        "time": case.get("execution_time"),
        "memory": case.get("memory_usage"),
        "message": _truncate(case.get("message"), max_message_chars),
    }


def _display_step(step: dict[str, Any], idx: int, args: argparse.Namespace) -> dict[str, Any]:
    metadata = step.get("metadata") or {}
    extras = metadata.get("env_extras") or {}
    result = _step_result(step)
    model_output = step.get("model_output") or ""
    default_language = str(metadata.get("code_language") or extras.get("code_language") or "cpp20")
    code, extracted_language, extraction = _extract_code(model_output, default_language)
    judge = _step_judge(step)
    case_message = _first_case_message(result)
    error = metadata.get("error")
    case_results = [
        _case_result_view(case, case_idx, args.max_message_chars)
        for case_idx, case in enumerate((result.get("case_results") or [])[: args.case_limit])
    ]
    return {
        "turn": step.get("turn") or idx + 1,
        "reward": _as_float(step.get("reward")),
        "done": step.get("done"),
        "model_output": _truncate(model_output, args.max_output_chars),
        "think": _truncate(_last_match(THINK_RE, model_output), args.max_output_chars // 2),
        "code": _truncate(code, args.max_code_chars),
        "code_language": metadata.get("code_language") or default_language,
        "extracted_language": extracted_language,
        "extraction": extraction,
        "judge": judge,
        "failure_bucket": _failure_bucket(judge, case_message, error),
        "error": _truncate(error, args.max_message_chars),
        "case_message": _truncate(case_message, args.max_message_chars),
        "case_results": case_results,
        "result_summary": {
            "overall_judge_result": result.get("overall_judge_result"),
            "overall_absolute_score": result.get("overall_absolute_score"),
            "overall_relative_score": result.get("overall_relative_score"),
            "num_cases": len(result.get("case_results") or []),
        },
        "metadata_summary": {
            "problem_id": metadata.get("problem_id") or extras.get("problem_id"),
            "valid_submission": metadata.get("valid_submission"),
            "best_absolute_score": metadata.get("best_absolute_score"),
            "best_signed_score": metadata.get("best_signed_score"),
            "code_chars": metadata.get("code_chars") if metadata.get("code_chars") is not None else len(code),
        },
        "observations": _truncate(text_join_observations(step.get("observations")), args.max_observation_chars),
    }


def text_join_observations(observations: Any) -> str:
    if not observations:
        return ""
    if not isinstance(observations, list):
        return str(observations)
    parts = []
    for item in observations:
        if isinstance(item, dict):
            parts.append(str(item.get("content", item)))
        else:
            parts.append(str(item))
    return "\n\n".join(parts)


def _display_row(row: dict[str, Any], index: int, args: argparse.Namespace) -> dict[str, Any]:
    steps = row.get("steps") or []
    initial_messages = row.get("initial_messages") or []
    initial_tail = ""
    if initial_messages:
        last = initial_messages[-1]
        initial_tail = last.get("content", "") if isinstance(last, dict) else str(last)
    return {
        "index": index,
        "uid": row.get("uid"),
        "sample_index": row.get("sample_index"),
        "global_step": row.get("global_step") or _global_step_from_path(Path(row.get("_source_file", ""))),
        "source_file": row.get("_source_file"),
        "total_reward": _as_float(row.get("total_reward")),
        "num_steps": row.get("num_steps") or len(steps),
        "stop_reason": row.get("stop_reason"),
        "env_metrics": row.get("env_metrics") or {},
        "initial_messages": _truncate(initial_tail, args.max_initial_chars),
        "steps": [_display_step(step, idx, args) for idx, step in enumerate(steps)],
    }


def _step_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for summary in summaries:
        step = summary.get("global_step")
        if step is not None:
            grouped.setdefault(int(step), []).append(summary)

    results = []
    for step in sorted(grouped):
        group = grouped[step]
        total_steps = sum(int(item.get("num_steps") or 0) for item in group)
        invalid_steps = sum(int(item.get("invalid_steps") or 0) for item in group)
        positive_steps = sum(int(item.get("positive_steps") or 0) for item in group)
        rewards = [_as_float(item.get("total_reward")) for item in group]
        judges = Counter(item.get("judge") or "NONE" for item in group)
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
                "accepted_rate": judges.get("ACCEPTED", 0) / len(group),
                "compile_error_rate": judges.get("COMPILATION_ERROR", 0) / len(group),
                "runtime_error_rate": judges.get("RUNTIME_ERROR", 0) / len(group),
                "wrong_answer_rate": judges.get("WRONG_ANSWER", 0) / len(group),
                "time_limit_rate": judges.get("TIME_LIMIT_EXCEEDED", 0) / len(group),
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


def _read_eval_summaries(paths: list[Path]) -> list[dict[str, Any]]:
    evals: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        root = path.expanduser()
        if root.is_file():
            root = root.parent.parent if root.parent.name == "dumped_rollouts" else root.parent
        eval_dir = root / "dumped_evals"
        if not eval_dir.is_dir():
            continue
        for result_path in sorted(eval_dir.glob("global_step_*_evals/aggregated_results.jsonl"), key=_rollout_file_sort_key):
            resolved = result_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            step = _global_step_from_path(result_path)
            for row in _iter_jsonl(result_path):
                row.pop("_source_file", None)
                row["global_step"] = step
                evals.append(row)
    return evals


def _build_report_data(
    raw_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    curve_step_summaries: list[dict[str, Any]],
    curve_num_trajectories: int,
    curve_num_files: int,
) -> dict[str, Any]:
    raw_rows = sorted(raw_rows, key=_trajectory_sort_key)
    if args.max_trajectories is not None:
        raw_rows = raw_rows[: args.max_trajectories]

    summaries = [_trajectory_summary(row, index) for index, row in enumerate(raw_rows)]
    trajectories = [_display_row(row, index, args) for index, row in enumerate(raw_rows)]
    rewards = [summary["total_reward"] for summary in summaries]
    invalid_steps = sum(int(summary.get("invalid_steps") or 0) for summary in summaries)
    positive_steps = sum(int(summary.get("positive_steps") or 0) for summary in summaries)
    judges = Counter(summary.get("judge") or "NONE" for summary in summaries)
    buckets = Counter(summary.get("failure_bucket") or "NONE" for summary in summaries)
    return {
        "summary": {
            "num_trajectories": len(summaries),
            "num_steps": sum(int(summary.get("num_steps") or 0) for summary in summaries),
            "avg_reward": mean(rewards) if rewards else 0.0,
            "min_reward": min(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
            "positive_trajectories": sum(1 for reward in rewards if reward > 0),
            "zero_trajectories": sum(1 for reward in rewards if reward == 0),
            "negative_trajectories": sum(1 for reward in rewards if reward < 0),
            "invalid_steps": invalid_steps,
            "positive_steps": positive_steps,
            "accepted": judges.get("ACCEPTED", 0),
            "judges": dict(judges),
            "failure_buckets": dict(buckets),
        },
        "curve_summary": {
            "num_trajectories": curve_num_trajectories,
            "num_files": curve_num_files,
            "sampled_for_detail": True,
        },
        "step_summaries": curve_step_summaries,
        "evals": _read_eval_summaries(args.paths),
        "trajectories": trajectories,
        "trajectory_summaries": summaries,
    }


def _default_output_path(input_paths: list[Path]) -> Path:
    first = input_paths[0].expanduser()
    if first.is_file():
        return first.with_suffix(".html")
    if first.name == "dumped_rollouts":
        return first.parent / "rollout_viewer.html"
    if (first / "dumped_rollouts").is_dir():
        return first / "rollout_viewer.html"
    return Path("ale_bench_rollouts.html")


def _script_safe_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("</", "<\\/")
    )


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
    .pill {{ display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 1px 7px; margin: 2px 4px 2px 0; font-size: 12px; }}
    .good {{ color: var(--good); }}
    .bad {{ color: var(--bad); }}
    .warn {{ color: var(--warn); }}
    .muted {{ color: var(--muted); }}
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
      max-height: 340px;
      overflow: auto;
    }}
    details {{ margin-top: 8px; }}
    summary {{ cursor: pointer; color: var(--accent); }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .empty {{ color: var(--muted); padding: 24px; }}
    .action-viewer-row {{ display: flex; justify-content: space-between; gap: 8px; align-items: center; margin-top: 8px; }}
    .action-viewer-row strong {{ flex: 1; }}
    .case-panel {{
      position: fixed;
      right: 16px;
      bottom: 16px;
      z-index: 5;
      width: 420px;
      max-width: calc(100vw - 32px);
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: 0 10px 30px rgba(16, 24, 40, 0.16);
      overflow: hidden;
    }}
    .case-panel h2 {{
      margin: 0;
      padding: 9px 11px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
      background: #fbfcfe;
    }}
    .case-panel-body {{ padding: 10px; }}
    .case-meta {{ color: var(--muted); font-size: 12px; margin-bottom: 8px; }}
    .case-controls {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 6px 8px;
      align-items: center;
      margin-bottom: 8px;
    }}
    .case-controls input[type="range"] {{ width: 100%; padding: 0; }}
    .case-turn-label {{ color: var(--muted); font-size: 12px; white-space: nowrap; }}
    #case-message {{ max-height: 220px; margin: 0; }}
    @media (max-width: 900px) {{
      .stats {{ grid-template-columns: repeat(2, 1fr); }}
      main {{ grid-template-columns: 1fr; }}
      aside {{ max-height: 300px; border-right: 0; border-bottom: 1px solid var(--line); }}
      .content {{ max-height: none; }}
      .two-col {{ grid-template-columns: 1fr; }}
      .charts {{ grid-template-columns: 1fr; }}
      .case-panel {{ position: static; width: auto; margin: 0 12px 12px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{safe_title}</h1>
    <div class="stats" id="stats"></div>
    <div class="toolbar">
      <input id="search" type="search" placeholder="Search output/code/error/case">
      <select id="reward-filter">
        <option value="all">All rewards</option>
        <option value="positive">Positive trajectories</option>
        <option value="zero">Zero trajectories</option>
        <option value="negative">Negative trajectories</option>
      </select>
      <select id="judge-filter">
        <option value="all">All judge results</option>
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
  <div class="case-panel" id="case-panel">
    <h2>Public Case Result</h2>
    <div class="case-panel-body">
      <div class="case-controls">
        <input id="case-scrubber" type="range" min="0" max="0" value="0" disabled>
        <div class="case-turn-label" id="case-turn-label">0 / 0</div>
      </div>
      <div class="case-meta" id="case-meta">Select a case to view judge feedback.</div>
      <pre id="case-message"></pre>
    </div>
  </div>
  <script type="application/json" id="rollout-data">{data_json}</script>
  <script>
    const data = JSON.parse(document.getElementById('rollout-data').textContent);
    const summaries = data.trajectory_summaries || [];
    const trajectories = data.trajectories || [];
    const stepSummaries = data.step_summaries || [];
    const curveSummary = data.curve_summary || {{}};
    let activeIndex = 0;
    let currentCases = [];

    const fmt = (value, digits = 4) => {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '';
      return Number(value).toFixed(digits);
    }};

    function text(value) {{
      if (value === null || value === undefined) return '';
      if (typeof value === 'string') return value;
      return JSON.stringify(value, null, 2);
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

    function rewardClass(value) {{
      value = Number(value || 0);
      if (value > 0) return 'good';
      if (value < 0) return 'bad';
      return '';
    }}

    function judgeClass(judge) {{
      if (judge === 'ACCEPTED') return 'good';
      if (judge === 'WRONG_ANSWER' || judge === 'TIME_LIMIT_EXCEEDED') return 'warn';
      if (judge && judge !== 'NONE') return 'bad';
      return 'muted';
    }}

    function renderStats() {{
      const s = data.summary || {{}};
      const items = [
        ['Trajectories', s.num_trajectories],
        ['Steps', s.num_steps],
        ['Avg Reward', fmt(s.avg_reward)],
        ['Min / Max', `${{fmt(s.min_reward)}} / ${{fmt(s.max_reward)}}`],
        ['Positive Traj', s.positive_trajectories],
        ['Accepted', s.accepted],
        ['Positive Steps', s.positive_steps],
        ['Invalid Steps', s.invalid_steps],
      ];
      document.getElementById('stats').replaceChildren(...items.map(([label, value]) => {{
        const div = document.createElement('div');
        div.className = 'stat';
        const b = document.createElement('b');
        b.textContent = value ?? '';
        const span = document.createElement('span');
        span.textContent = label;
        div.append(b, span);
        return div;
      }}));
    }}

    function populateJudgeFilter() {{
      const select = document.getElementById('judge-filter');
      const judges = [...new Set(summaries.map(s => s.judge).filter(Boolean))].sort();
      select.replaceChildren(new Option('All judge results', 'all'), ...judges.map(j => new Option(j, j)));
    }}

    function matchesFilters(summary) {{
      const q = document.getElementById('search').value.trim().toLowerCase();
      const rewardFilter = document.getElementById('reward-filter').value;
      const judgeFilter = document.getElementById('judge-filter').value;
      const invalidOnly = document.getElementById('invalid-only').checked;
      const positiveStepOnly = document.getElementById('positive-step-only').checked;
      if (rewardFilter === 'positive' && summary.total_reward <= 0) return false;
      if (rewardFilter === 'zero' && summary.total_reward !== 0) return false;
      if (rewardFilter === 'negative' && summary.total_reward >= 0) return false;
      if (judgeFilter !== 'all' && summary.judge !== judgeFilter) return false;
      if (invalidOnly && summary.invalid_steps === 0) return false;
      if (positiveStepOnly && summary.positive_steps === 0) return false;
      if (!q) return true;
      const row = trajectories[summary.index] || {{}};
      return JSON.stringify({{ summary, row }}).toLowerCase().includes(q);
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
        meta.textContent = `${{summary.problem_id || ''}} · ${{summary.judge || 'NONE'}} · ${{summary.failure_bucket || ''}} · turns=${{summary.num_steps}} · stop=${{summary.stop_reason || ''}}`;
        button.append(top, meta);
        return button;
      }}));
    }}

    function updateActiveListSelection() {{
      document.querySelectorAll('#trajectory-list .traj').forEach(button => {{
        button.classList.toggle('active', Number(button.dataset.index) === activeIndex);
      }});
    }}

    function setCases(cases) {{
      currentCases = cases || [];
      const scrubber = document.getElementById('case-scrubber');
      const label = document.getElementById('case-turn-label');
      scrubber.disabled = currentCases.length === 0;
      scrubber.min = '0';
      scrubber.max = String(Math.max(0, currentCases.length - 1));
      scrubber.value = '0';
      label.textContent = currentCases.length ? `1 / ${{currentCases.length}}` : '0 / 0';
      selectCase(0);
    }}

    function selectCase(index) {{
      const item = currentCases[index];
      const meta = document.getElementById('case-meta');
      const message = document.getElementById('case-message');
      const label = document.getElementById('case-turn-label');
      if (!item) {{
        meta.textContent = 'No public case result embedded for this trajectory.';
        message.textContent = '';
        label.textContent = '0 / 0';
        return;
      }}
      document.getElementById('case-scrubber').value = String(index);
      label.textContent = `${{index + 1}} / ${{currentCases.length}}`;
      meta.textContent = `case ${{item.idx}} · ${{item.judge || 'NONE'}} · score=${{item.score ?? ''}} · time=${{fmt(item.time, 3)}} · memory=${{item.memory ?? ''}}`;
      message.textContent = item.message || '';
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
      rewardBreakdown.querySelector('.section-body').append(kvPre(summary.reward_breakdown || {{}}));
      detail.append(rewardBreakdown);

      const overview = section('Trajectory Overview');
      overview.querySelector('.section-body').append(kvPre({{
        uid: summary.uid,
        source_file: summary.source_file,
        global_step: summary.global_step,
        sample_index: summary.sample_index,
        problem_id: summary.problem_id,
        total_reward: summary.total_reward,
        judge: summary.judge,
        failure_bucket: summary.failure_bucket,
        absolute_score: summary.absolute_score,
        relative_score: summary.relative_score,
        valid_submission: summary.valid_submission,
        num_steps: summary.num_steps,
        stop_reason: summary.stop_reason,
        env_metrics: row.env_metrics,
      }}));
      detail.append(overview);

      if (row.initial_messages) {{
        const initialMessages = section('Initial Messages');
        initialMessages.querySelector('.section-body').append(labelPre('Initial input before turn 1', row.initial_messages));
        detail.append(initialMessages);
      }}

      const curves = renderCurveSection();
      if (curves) detail.append(curves);

      const allCases = [];
      (row.steps || []).forEach((step, idx) => {{
        const stepCases = (step.case_results || []).map(c => ({{ ...c, turn: step.turn ?? idx + 1 }}));
        allCases.push(...stepCases);
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
          <span class="pill ${{judgeClass(step.judge)}}">judge=${{htmlEscape(step.judge || 'NONE')}}</span>
          <span class="pill">bucket=${{htmlEscape(step.failure_bucket || '')}}</span>
          <span class="pill">lang=${{htmlEscape(step.extracted_language || step.code_language || '')}}/${{htmlEscape(step.extraction || '')}}</span>
          <span class="pill">done=${{Boolean(step.done)}}</span>
        `;
        body.append(meta);
        if (step.error) {{
          const error = document.createElement('pre');
          error.className = 'bad';
          error.textContent = step.error;
          body.append(error);
        }}
        body.append(labelPre('Think', step.think || '(missing <think>)'));

        const codeBlock = labelPre('Extracted Source', step.code || '(no code extracted)');
        const codeHeader = codeBlock.querySelector('strong');
        const codeRow = document.createElement('div');
        codeRow.className = 'action-viewer-row';
        const caseButton = document.createElement('button');
        caseButton.type = 'button';
        caseButton.textContent = stepCases.length ? 'View public cases' : 'Cases unavailable';
        caseButton.disabled = !stepCases.length;
        caseButton.onclick = () => setCases(stepCases);
        codeRow.append(codeHeader, caseButton);
        codeBlock.prepend(codeRow);
        body.append(codeBlock);

        body.append(labelPre('Model Output', step.model_output));

        const two = document.createElement('div');
        two.className = 'two-col';
        two.append(labelPre('Judge Result', text(step.result_summary)));
        two.append(labelPre('Metadata', text(step.metadata_summary)));
        body.append(two);

        if (step.case_message) body.append(labelPre('First Case Message', step.case_message));
        if (step.observations) body.append(labelPre('Observation', step.observations));
        const raw = document.createElement('details');
        const summaryEl = document.createElement('summary');
        summaryEl.textContent = 'Display step JSON';
        raw.append(summaryEl, kvPre(step));
        body.append(raw);
        detail.append(stepSection);
      }});
      setCases(allCases);
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
        lineChart('Accepted Rate', 'accepted_rate', 'fraction'),
        lineChart('Compile Error Rate', 'compile_error_rate', 'fraction'),
        lineChart('Avg Turns', 'avg_num_steps', 'turns'),
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

    ['search', 'reward-filter', 'judge-filter', 'invalid-only', 'positive-step-only'].forEach(id => {{
      document.getElementById(id).addEventListener('input', () => {{ renderList(); renderDetail(); }});
    }});
    document.getElementById('case-scrubber').addEventListener('input', event => {{
      selectCase(Number(event.target.value));
    }});
    document.getElementById('reset').onclick = () => {{
      document.getElementById('search').value = '';
      document.getElementById('reward-filter').value = 'all';
      document.getElementById('judge-filter').value = 'all';
      document.getElementById('invalid-only').checked = false;
      document.getElementById('positive-step-only').checked = false;
      renderList();
      renderDetail();
    }};

    renderStats();
    populateJudgeFilter();
    renderList();
    renderDetail();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ARC-style static HTML viewer for ALE-Bench rollout JSONL dumps.")
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Rollout JSONL files, dumped_rollouts directories, or export directories containing dumped_rollouts/.",
    )
    parser.add_argument("--output", "-o", default=None, type=Path, help="Output HTML path.")
    parser.add_argument("--title", default="ALE-Bench Rollout Viewer")
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
    parser.add_argument("--max-output-chars", type=int, default=12000)
    parser.add_argument("--max-code-chars", type=int, default=12000)
    parser.add_argument("--max-message-chars", type=int, default=6000)
    parser.add_argument("--max-observation-chars", type=int, default=6000)
    parser.add_argument("--max-initial-chars", type=int, default=6000)
    parser.add_argument("--case-limit", type=int, default=5)
    args = parser.parse_args()

    if args.max_trajectories is not None and args.max_trajectories <= 0:
        raise SystemExit("--max-trajectories must be positive")
    if args.trajectories_per_file is not None and args.trajectories_per_file <= 0:
        raise SystemExit("--trajectories-per-file must be positive")

    all_rollout_files = _find_rollout_files(args.paths)
    curve_step_summaries, curve_num_trajectories = _read_curve_step_summaries(all_rollout_files)
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
            "or --max-trajectories, or pass --allow-large if you really want to embed everything."
        )

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
        args,
        curve_step_summaries=curve_step_summaries,
        curve_num_trajectories=curve_num_trajectories,
        curve_num_files=len(all_rollout_files),
    )
    output = args.output.expanduser() if args.output else _default_output_path(args.paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_html_template(args.title, _script_safe_json(report_data)), encoding="utf-8")

    print(f"Loaded {len(rows)} display trajectories from {len(rollout_files)} file(s), selected input {input_size_mb:.1f} MiB.")
    print(f"Built curves from {curve_num_trajectories} trajectories across {len(all_rollout_files)} file(s).")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
