from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv


DEFAULT_PROMPT = (
    "You are solving one ALE-Bench AtCoder Heuristic Contest task. "
    "Read the problem statement and submit exactly one complete source file. "
    "Use a fenced code block with the language tag, for example ```cpp20 ... ```. "
    "Do not include multiple alternative files."
)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_rows(args: argparse.Namespace, split: str, target_size: int) -> list[dict]:
    problem_ids = parse_csv(args.problem_ids)
    if not problem_ids:
        raise ValueError("at least one problem id is required")

    repeat_factor = math.ceil(target_size / len(problem_ids))
    expanded = (problem_ids * repeat_factor)[:target_size]
    rows = []
    for idx, problem_id in enumerate(expanded):
        rows.append(
            {
                "data_source": "ale_bench",
                "prompt": [{"role": "user", "content": args.prompt}],
                "env_class": "ale_bench",
                "uid": f"{split}-{idx}-{problem_id}",
                "split": split,
                "problem_id": problem_id,
                "lite_version": args.lite_version,
                "max_steps": args.max_steps,
                "code_language": args.code_language,
                "judge_version": args.judge_version,
                "num_workers": args.num_workers,
                "score_scale": args.score_scale,
                "invalid_reward": args.invalid_reward,
                "reward_mode": args.reward_mode,
                "statement_max_chars": args.statement_max_chars,
                "tool_readme_max_chars": args.tool_readme_max_chars,
                "case_feedback_limit": args.case_feedback_limit,
            }
        )
    return rows


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Prepare ALE-Bench parquet data for SkyRL.")
    parser.add_argument("--output_dir", default=os.getenv("ALE_BENCH_DATA_DIR", "~/data/ale_bench"))
    parser.add_argument("--problem_ids", default=os.getenv("ALE_BENCH_PROBLEM_IDS", "ahc001"))
    parser.add_argument("--train_size", type=int, default=8)
    parser.add_argument("--val_size", type=int, default=2)
    parser.add_argument("--lite_version", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_steps", type=int, default=int(os.getenv("ALE_BENCH_MAX_STEPS", "1")))
    parser.add_argument("--code_language", default=os.getenv("ALE_BENCH_CODE_LANGUAGE", "cpp20"))
    parser.add_argument("--judge_version", default=os.getenv("ALE_BENCH_JUDGE_VERSION", "202301"))
    parser.add_argument("--num_workers", type=int, default=int(os.getenv("ALE_BENCH_NUM_WORKERS", "1")))
    parser.add_argument("--score_scale", type=float, default=float(os.getenv("ALE_BENCH_SCORE_SCALE", "1000000000")))
    parser.add_argument("--invalid_reward", type=float, default=float(os.getenv("ALE_BENCH_INVALID_REWARD", "-1.0")))
    parser.add_argument("--reward_mode", default=os.getenv("ALE_BENCH_REWARD_MODE", "score"))
    parser.add_argument(
        "--statement_max_chars", type=int, default=int(os.getenv("ALE_BENCH_STATEMENT_MAX_CHARS", "12000"))
    )
    parser.add_argument(
        "--tool_readme_max_chars", type=int, default=int(os.getenv("ALE_BENCH_TOOL_README_MAX_CHARS", "5000"))
    )
    parser.add_argument("--case_feedback_limit", type=int, default=int(os.getenv("ALE_BENCH_CASE_FEEDBACK_LIMIT", "5")))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = build_rows(args, split="train", target_size=args.train_size)
    val_rows = build_rows(args, split="validation", target_size=args.val_size)
    Dataset.from_list(train_rows).to_parquet(str(output_dir / "train.parquet"))
    Dataset.from_list(val_rows).to_parquet(str(output_dir / "validation.parquet"))
    print(f"Saved {len(train_rows)} train and {len(val_rows)} validation examples to {output_dir}")


if __name__ == "__main__":
    main()
