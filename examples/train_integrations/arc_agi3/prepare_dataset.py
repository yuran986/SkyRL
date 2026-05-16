import argparse
import math
import os
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv


DEFAULT_PROMPT = (
    "You are playing one ARC-AGI-3 game. Infer the rule from observations and choose one action at a time. "
    "Respond with brief reasoning in <think>...</think>, then exactly one executable action in "
    "<action>...</action>. Keep <think> concise. The environment executes only the <action> tag. "
    "Observation fields: frame_diff compares the frame after your last action with the previous frame; "
    "components split changed cells by connected region and before->after color change; "
    "changed_patch_before/changed_patch/changed_patch_delta show local before/current/delta crops; use these to infer which clicks changed the game. "
    "Simple actions look like <action>ACTION1</action>. "
    'For coordinate clicks use JSON inside <action>, for example <action>{"action":"ACTION6","x":32,"y":32}</action>. '
    "Never output bare JSON outside <action>."
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
    max_steps: int,
    prompt: str,
    frame_observation_mode: str,
    full_frame_interval: int,
    patch_radius: int,
    max_diff_examples: int,
    invalid_action_reward: float,
    level_reward: float,
    done_reward: float,
    meaningful_diff_reward: float,
    min_meaningful_diff_changes: int,
    max_meaningful_diff_changes: int,
    repeat_click_penalty: float,
    repeat_click_radius: int,
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
                "operation_mode": operation_mode,
                "environments_dir": environments_dir,
                "frame_observation_mode": frame_observation_mode,
                "full_frame_interval": full_frame_interval,
                "patch_radius": patch_radius,
                "max_diff_examples": max_diff_examples,
                "invalid_action_reward": invalid_action_reward,
                "level_reward": level_reward,
                "done_reward": done_reward,
                "meaningful_diff_reward": meaningful_diff_reward,
                "min_meaningful_diff_changes": min_meaningful_diff_changes,
                "max_meaningful_diff_changes": max_meaningful_diff_changes,
                "repeat_click_penalty": repeat_click_penalty,
                "repeat_click_radius": repeat_click_radius,
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
    parser.add_argument("--operation_mode", default=os.getenv("OPERATION_MODE", "OFFLINE"))
    parser.add_argument("--environments_dir", default=os.getenv("ARC_AGI3_ENVIRONMENTS_DIR"))
    parser.add_argument("--frame_observation_mode", default=os.getenv("ARC_AGI3_FRAME_OBSERVATION_MODE", "initial_full_then_diff"))
    parser.add_argument("--full_frame_interval", type=int, default=int(os.getenv("ARC_AGI3_FULL_FRAME_INTERVAL", "0")))
    parser.add_argument("--patch_radius", type=int, default=int(os.getenv("ARC_AGI3_PATCH_RADIUS", "4")))
    parser.add_argument("--max_diff_examples", type=int, default=int(os.getenv("ARC_AGI3_MAX_DIFF_EXAMPLES", "32")))
    parser.add_argument("--invalid_action_reward", type=float, default=float(os.getenv("ARC_AGI3_INVALID_ACTION_REWARD", "-0.1")))
    parser.add_argument("--level_reward", type=float, default=float(os.getenv("ARC_AGI3_LEVEL_REWARD", "3.0")))
    parser.add_argument("--done_reward", type=float, default=float(os.getenv("ARC_AGI3_DONE_REWARD", "0.0")))
    parser.add_argument("--meaningful_diff_reward", type=float, default=float(os.getenv("ARC_AGI3_MEANINGFUL_DIFF_REWARD", "0.005")))
    parser.add_argument("--min_meaningful_diff_changes", type=int, default=int(os.getenv("ARC_AGI3_MIN_MEANINGFUL_DIFF_CHANGES", "1")))
    parser.add_argument("--max_meaningful_diff_changes", type=int, default=int(os.getenv("ARC_AGI3_MAX_MEANINGFUL_DIFF_CHANGES", "512")))
    parser.add_argument("--repeat_click_penalty", type=float, default=float(os.getenv("ARC_AGI3_REPEAT_CLICK_PENALTY", "-0.02")))
    parser.add_argument("--repeat_click_radius", type=int, default=int(os.getenv("ARC_AGI3_REPEAT_CLICK_RADIUS", "2")))
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
        max_steps=args.max_steps,
        prompt=args.prompt,
        frame_observation_mode=args.frame_observation_mode,
        full_frame_interval=args.full_frame_interval,
        patch_radius=args.patch_radius,
        max_diff_examples=args.max_diff_examples,
        invalid_action_reward=args.invalid_action_reward,
        level_reward=args.level_reward,
        done_reward=args.done_reward,
        meaningful_diff_reward=args.meaningful_diff_reward,
        min_meaningful_diff_changes=args.min_meaningful_diff_changes,
        max_meaningful_diff_changes=args.max_meaningful_diff_changes,
        repeat_click_penalty=args.repeat_click_penalty,
        repeat_click_radius=args.repeat_click_radius,
    )
    val_rows = build_rows(
        task_ids=task_ids,
        seeds=seeds,
        split="validation",
        target_size=args.val_size,
        environments_dir=args.environments_dir,
        operation_mode=args.operation_mode,
        max_steps=args.max_steps,
        prompt=args.prompt,
        frame_observation_mode=args.frame_observation_mode,
        full_frame_interval=args.full_frame_interval,
        patch_radius=args.patch_radius,
        max_diff_examples=args.max_diff_examples,
        invalid_action_reward=args.invalid_action_reward,
        level_reward=args.level_reward,
        done_reward=args.done_reward,
        meaningful_diff_reward=args.meaningful_diff_reward,
        min_meaningful_diff_changes=args.min_meaningful_diff_changes,
        max_meaningful_diff_changes=args.max_meaningful_diff_changes,
        repeat_click_penalty=args.repeat_click_penalty,
        repeat_click_radius=args.repeat_click_radius,
    )

    Dataset.from_list(train_rows).to_parquet(str(output_dir / "train.parquet"))
    Dataset.from_list(val_rows).to_parquet(str(output_dir / "validation.parquet"))
    print(f"Saved {len(train_rows)} train and {len(val_rows)} validation examples to {output_dir}")


if __name__ == "__main__":
    main()
