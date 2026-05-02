import argparse
import math
import os
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv


DEFAULT_PROMPT = (
    "You are playing one ARC-AGI-3 game. Infer the rule from observations and choose one action at a time. "
    "Return exactly one action in <action>...</action>. Simple actions look like <action>ACTION1</action>. "
    'For coordinate clicks use JSON, for example <action>{"action":"ACTION6","x":32,"y":32}</action>.'
)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_rows(
    task_ids: list[str],
    seeds: list[int | None],
    split: str,
    target_size: int,
    environments_dir: str | None,
    operation_mode: str,
    render_mode: str,
    max_steps: int,
    prompt: str,
) -> list[dict]:
    combos = [(task_id, seed) for task_id in task_ids for seed in seeds]
    if not combos:
        raise ValueError("at least one task_id is required")
    repeat_factor = math.ceil(target_size / len(combos))
    expanded = (combos * repeat_factor)[:target_size]
    rows = []
    for idx, (task_id, seed) in enumerate(expanded):
        rows.append(
            {
                "data_source": "arc_agi3",
                "prompt": [{"role": "user", "content": prompt}],
                "env_class": "arc_agi3",
                "task_id": task_id,
                "seed": seed if seed is not None else idx,
                "split": split,
                "max_steps": max_steps,
                "render_mode": render_mode,
                "operation_mode": operation_mode,
                "environments_dir": environments_dir,
            }
        )
    return rows


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Prepare ARC-AGI-3 parquet data for SkyRL.")
    parser.add_argument("--output_dir", default="~/data/arc_agi3")
    parser.add_argument("--task_ids", default=os.getenv("ARC_AGI3_TASK_IDS", "ft09"))
    parser.add_argument("--seeds", default=os.getenv("ARC_AGI3_SEEDS", "0,1,2,3"))
    parser.add_argument("--train_size", type=int, default=32)
    parser.add_argument("--val_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=int(os.getenv("ARC_AGI3_MAX_STEPS", "64")))
    parser.add_argument("--render_mode", default=os.getenv("ARC_AGI3_RENDER_MODE", "terminal-fast"))
    parser.add_argument("--operation_mode", default=os.getenv("OPERATION_MODE", "OFFLINE"))
    parser.add_argument("--environments_dir", default=os.getenv("ARC_AGI3_ENVIRONMENTS_DIR"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_ids = parse_csv(args.task_ids)
    seeds: list[int | None] = [int(seed) for seed in parse_csv(args.seeds)] if args.seeds else [None]

    train_rows = build_rows(
        task_ids=task_ids,
        seeds=seeds,
        split="train",
        target_size=args.train_size,
        environments_dir=args.environments_dir,
        operation_mode=args.operation_mode,
        render_mode=args.render_mode,
        max_steps=args.max_steps,
        prompt=args.prompt,
    )
    val_rows = build_rows(
        task_ids=task_ids,
        seeds=seeds,
        split="validation",
        target_size=args.val_size,
        environments_dir=args.environments_dir,
        operation_mode=args.operation_mode,
        render_mode=args.render_mode,
        max_steps=args.max_steps,
        prompt=args.prompt,
    )

    Dataset.from_list(train_rows).to_parquet(str(output_dir / "train.parquet"))
    Dataset.from_list(val_rows).to_parquet(str(output_dir / "validation.parquet"))
    print(f"Saved {len(train_rows)} train and {len(val_rows)} validation examples to {output_dir}")


if __name__ == "__main__":
    main()

