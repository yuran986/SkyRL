from __future__ import annotations

import argparse
import json
import os
from typing import Any

from dotenv import load_dotenv

from examples.train_integrations.arc_agi3.env import ArcAgi3Env


DEFAULT_PROMPT = (
    "You are playing one ARC-AGI-3 game. Infer the rule from observations and choose one action at a time. "
    "Respond with brief reasoning in <think>...</think>, then exactly one executable action in "
    "<action>...</action>. Keep <think> concise. The environment executes only the <action> tag. "
    "Observations only report new feedback from your last action; when there is a visible diff, "
    "changed_patch_delta shows the local changed crop. Use these changes to infer which clicks move the game forward. "
    "Simple actions look like <action>ACTION1</action>. "
    'For coordinate clicks use JSON inside <action>, for example <action>{"action":"ACTION6","x":32,"y":32}</action>. '
    "Never output bare JSON outside <action>."
)


DEFAULT_ACTIONS = [
    "<action>ACTION1</action>",
    "<action>ACTION2</action>",
    '<action>{"action":"ACTION6","x":32,"y":32}</action>',
]


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run one local ARC-AGI-3 environment rollout without SkyRL training.")
    parser.add_argument("--task_id", default=os.getenv("ARC_AGI3_TASK_ID", "ft09"))
    parser.add_argument("--seed", type=int, default=int(os.getenv("ARC_AGI3_DEBUG_SEED", "0")))
    parser.add_argument("--max_steps", type=int, default=int(os.getenv("ARC_AGI3_MAX_STEPS", "16")))
    parser.add_argument("--operation_mode", default=os.getenv("OPERATION_MODE", "OFFLINE"))
    parser.add_argument("--environments_dir", default=os.getenv("ARC_AGI3_ENVIRONMENTS_DIR"))
    parser.add_argument("--frame_observation_mode", default=os.getenv("ARC_AGI3_FRAME_OBSERVATION_MODE", "initial_full_then_diff"))
    parser.add_argument("--full_frame_interval", type=int, default=int(os.getenv("ARC_AGI3_FULL_FRAME_INTERVAL", "8")))
    parser.add_argument("--patch_radius", type=int, default=int(os.getenv("ARC_AGI3_PATCH_RADIUS", "4")))
    parser.add_argument("--max_diff_examples", type=int, default=int(os.getenv("ARC_AGI3_MAX_DIFF_EXAMPLES", "32")))
    parser.add_argument("--action", action="append", dest="actions", help="Action string. May be passed repeatedly.")
    args = parser.parse_args()

    actions = args.actions or DEFAULT_ACTIONS
    env = ArcAgi3Env(
        env_config={},
        extras={
            "task_id": args.task_id,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "operation_mode": args.operation_mode,
            "environments_dir": args.environments_dir,
            "frame_observation_mode": args.frame_observation_mode,
            "full_frame_interval": args.full_frame_interval,
            "patch_radius": args.patch_radius,
            "max_diff_examples": args.max_diff_examples,
        },
    )

    prompt = [{"role": "user", "content": DEFAULT_PROMPT}]
    try:
        initialized_prompt, init_metadata = env.init(prompt)
        print(
            json.dumps(
                {
                    "type": "init",
                    "input_prompt": prompt,
                    "initialized_prompt": initialized_prompt,
                    "init_metadata": init_metadata,
                    "prompt_source": "examples/train_integrations/arc_agi3/prepare_dataset.py::DEFAULT_PROMPT",
                    "env_observation_source": "examples/train_integrations/arc_agi3/env.py::ArcAgi3Env.init",
                },
                ensure_ascii=False,
            )
        )

        for index, action in enumerate(actions, start=1):
            output = env.step(action)
            print(
                json.dumps(
                    {
                        "type": "step",
                        "index": index,
                        "step_input_model_output": action,
                        "step_output": _jsonable(output),
                        "metrics_after_step": _jsonable(env.get_metrics()),
                    },
                    ensure_ascii=False,
                )
            )
            if output["done"]:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
