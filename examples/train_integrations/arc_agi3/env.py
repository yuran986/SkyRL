from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType

from examples.train_integrations.arc_agi3.sft_warmup.oracle_distance_reward import (
    OracleDistanceInfo,
    inspect_oracle_distance,
    matches_oracle_next_action,
    oracle_progress_delta,
    should_penalize_oracle_no_progress,
)


ACTION_PATTERN = re.compile(r"\b(RESET|ACTION[1-7])\b", re.IGNORECASE)
COORD_PATTERN = re.compile(r"\b([xy])\s*[:=]\s*(-?\d+)\b", re.IGNORECASE)
FRAME_COLOR_NAMES = {
    0: "white",
    1: "off-white",
    2: "light gray",
    3: "gray",
    4: "off-black",
    5: "black",
    6: "magenta",
    7: "light magenta",
    8: "red",
    9: "blue",
    10: "light blue",
    11: "yellow",
    12: "orange",
    13: "maroon",
    14: "green",
    15: "purple",
}

OBSERVATION_GUIDE = (
    "observation_guide: frame_diff compares the current frame after your last action with the previous frame; "
    "num_changes is the number of changed cells; bbox is the changed rectangle; examples list changed cells as "
    "{x,y,before,after}; components split contiguous cells with the same before->after color change; "
    "changed_patch_before/changed_patch/changed_patch_delta show local before/current/delta crops; coordinates are 0..63."
)


def _format_color_legend() -> str:
    parts = []
    for value, name in sorted(FRAME_COLOR_NAMES.items()):
        normalized_name = name.replace(" ", "_").replace("-", "_")
        parts.append(f"{format(value, 'x')}={normalized_name}")
    return "color_legend: " + " ".join(parts)


def noop_renderer(*args: Any, **kwargs: Any) -> None:
    return None


@dataclass
class ParsedAction:
    name: str
    x: int | None = None
    y: int | None = None


@dataclass
class FrameDiffStats:
    num_changes: int
    bbox: tuple[int, int, int, int] | None
    color_changes: list[tuple[Any, Any, int]]
    examples: list[dict[str, Any]]
    components: list[dict[str, Any]] = field(default_factory=list)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain(v) for v in value]
    if hasattr(value, "value"):
        try:
            return _to_plain(value.value)
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _to_plain(tolist())
        except Exception:
            pass
    return str(value)


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "name", None) or getattr(value, "value", None) or value)


def _extract_action_payload(output: str) -> str:
    if "<action>" not in output:
        return output.strip()
    payload = output.split("<action>")[-1]
    if "</action>" in payload:
        payload = payload.split("</action>", 1)[0]
    return payload.strip().strip("`").strip()


def parse_model_action(output: str) -> ParsedAction:
    """Parse a model response into one ARC action.

    Supported forms:
    - <action>ACTION1</action>
    - <action>{"action":"ACTION6","x":32,"y":32}</action>
    - <action>ACTION6 x=32 y=32</action>
    """
    payload = _extract_action_payload(output)
    if not payload:
        raise ValueError("empty action payload")

    parsed_json: Any | None = None
    if payload.startswith("{"):
        try:
            parsed_json = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON action: {exc}") from exc

    if isinstance(parsed_json, dict):
        action_name = parsed_json.get("action") or parsed_json.get("name") or parsed_json.get("id")
        if not action_name:
            raise ValueError("JSON action must contain an action/name/id field")
        name = str(action_name).upper()
        x = parsed_json.get("x")
        y = parsed_json.get("y")
    else:
        match = ACTION_PATTERN.search(payload)
        if not match:
            raise ValueError(f"could not find action name in payload: {payload!r}")
        name = match.group(1).upper()
        coords = {m.group(1).lower(): int(m.group(2)) for m in COORD_PATTERN.finditer(payload)}
        x = coords.get("x")
        y = coords.get("y")

    if name == "ACTION6":
        if x is None or y is None:
            raise ValueError("ACTION6 requires integer x and y coordinates")
        x = int(x)
        y = int(y)
        if not (0 <= x <= 63 and 0 <= y <= 63):
            raise ValueError(f"ACTION6 coordinates must be in [0, 63], got x={x}, y={y}")
        return ParsedAction(name=name, x=x, y=y)

    return ParsedAction(name=name)


def _grid_from_frame(frame_like: Any) -> list[list[Any]] | None:
    frame = _to_plain(frame_like)
    if isinstance(frame, dict) and "frame" in frame:
        frame = frame["frame"]
    if isinstance(frame, list) and frame and isinstance(frame[0], list):
        if frame[0] and isinstance(frame[0][0], list):
            frame = frame[0]
        return frame
    return None


def _frame_metadata(frame_like: Any) -> list[list[int | None]] | None:
    grid = _grid_from_frame(frame_like)
    if grid is None:
        return None
    normalized: list[list[int | None]] = []
    for row in grid:
        normalized_row: list[int | None] = []
        for cell in row:
            try:
                value = int(cell)
            except (TypeError, ValueError):
                normalized_row.append(None)
                continue
            normalized_row.append(value if 0 <= value <= 15 else None)
        normalized.append(normalized_row)
    return normalized


def _format_cell(value: Any) -> str:
    if isinstance(value, int) and 0 <= value <= 15:
        return format(value, "x")
    try:
        intval = int(value)
    except (TypeError, ValueError):
        return "?"
    if 0 <= intval <= 15:
        return format(intval, "x")
    return "?"


def _format_hex_rows(grid: list[list[Any]], y_offset: int = 0, max_rows: int | None = None) -> str:
    rows = grid if max_rows is None else grid[:max_rows]
    width = max((len(row) for row in rows), default=0)
    lines = []
    for idx, row in enumerate(rows):
        encoded = "".join(_format_cell(cell) for cell in row)
        lines.append(f"y{y_offset + idx:02d}: {encoded}")
    if max_rows is not None and len(grid) > max_rows:
        lines.append(f"... truncated {len(grid) - max_rows} rows")
    return f"shape={len(grid)}x{width} encoding=hex_0_to_f\n" + "\n".join(lines)


def _format_delta_rows(
    prev_grid: list[list[Any]], cur_grid: list[list[Any]], y_offset: int = 0, max_rows: int | None = None
) -> str:
    height = max(len(prev_grid), len(cur_grid))
    width = max(
        max((len(row) for row in prev_grid), default=0),
        max((len(row) for row in cur_grid), default=0),
    )
    shown_height = height if max_rows is None else min(height, max_rows)
    lines = []
    for y in range(shown_height):
        prev_row = prev_grid[y] if y < len(prev_grid) else []
        cur_row = cur_grid[y] if y < len(cur_grid) else []
        encoded = []
        for x in range(width):
            before = prev_row[x] if x < len(prev_row) else None
            after = cur_row[x] if x < len(cur_row) else None
            encoded.append("." if before == after else _format_cell(after))
        lines.append(f"y{y_offset + y:02d}: {''.join(encoded)}")
    if max_rows is not None and height > max_rows:
        lines.append(f"... truncated {height - max_rows} rows")
    return f"shape={height}x{width} encoding=changed_after_hex_unchanged_dot\n" + "\n".join(lines)


def _format_full_frame(frame_like: Any, max_rows: int | None = None) -> str:
    grid = _grid_from_frame(frame_like)
    if grid is None:
        return "current_frame: unavailable"
    return "current_frame:\n" + _format_hex_rows(grid, max_rows=max_rows)


def _crop_grid(
    grid: list[list[Any]], x1: int, y1: int, x2: int, y2: int
) -> tuple[list[list[Any]], tuple[int, int, int, int]]:
    if not grid:
        return [], (0, 0, 0, 0)
    height = len(grid)
    width = max((len(row) for row in grid), default=0)
    x1 = max(0, min(x1, max(0, width - 1)))
    x2 = max(0, min(x2, max(0, width - 1)))
    y1 = max(0, min(y1, max(0, height - 1)))
    y2 = max(0, min(y2, max(0, height - 1)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    cropped = []
    for row in grid[y1 : y2 + 1]:
        cropped.append(row[x1 : x2 + 1])
    return cropped, (x1, y1, x2, y2)


def _format_diff_patch(
    prev_frame_like: Any, cur_frame_like: Any, diff_stats: FrameDiffStats | None, radius: int
) -> str:
    if diff_stats is None or diff_stats.bbox is None or diff_stats.num_changes == 0:
        return "changed_patch: none"
    prev_grid = _grid_from_frame(prev_frame_like)
    cur_grid = _grid_from_frame(cur_frame_like)
    if prev_grid is None or cur_grid is None:
        return "changed_patch: unavailable"
    x1, y1, x2, y2 = diff_stats.bbox
    cur_patch, (cx1, cy1, cx2, cy2) = _crop_grid(cur_grid, x1 - radius, y1 - radius, x2 + radius, y2 + radius)
    prev_patch, _ = _crop_grid(prev_grid, cx1, cy1, cx2, cy2)
    return "\n".join(
        [
            f"changed_patch_before: x={cx1}..{cx2} y={cy1}..{cy2}",
            _format_hex_rows(prev_patch, y_offset=cy1),
            f"changed_patch: x={cx1}..{cx2} y={cy1}..{cy2}",
            _format_hex_rows(cur_patch, y_offset=cy1),
            f"changed_patch_delta: x={cx1}..{cx2} y={cy1}..{cy2}",
            _format_delta_rows(prev_patch, cur_patch, y_offset=cy1),
        ]
    )


def _diff_stats(prev_frame: Any, cur_frame: Any, max_examples: int = 8) -> FrameDiffStats | None:
    prev_grid = _grid_from_frame(prev_frame)
    cur_grid = _grid_from_frame(cur_frame)
    if prev_grid is None or cur_grid is None:
        return None

    changes: list[tuple[int, int, Any, Any]] = []
    h = max(len(prev_grid), len(cur_grid))
    w = max(
        max((len(row) for row in prev_grid), default=0),
        max((len(row) for row in cur_grid), default=0),
    )
    for y in range(h):
        prev_row = prev_grid[y] if y < len(prev_grid) else []
        cur_row = cur_grid[y] if y < len(cur_grid) else []
        for x in range(w):
            before = prev_row[x] if x < len(prev_row) else None
            after = cur_row[x] if x < len(cur_row) else None
            if before != after:
                changes.append((x, y, before, after))

    if not changes:
        return FrameDiffStats(num_changes=0, bbox=None, color_changes=[], examples=[])

    xs = [x for x, _, _, _ in changes]
    ys = [y for _, y, _, _ in changes]
    color_changes = Counter((before, after) for _, _, before, after in changes)
    examples = [{"x": x, "y": y, "before": before, "after": after} for x, y, before, after in changes[:max_examples]]
    components = _changed_components(changes, max_examples=max(1, min(max_examples, 8)))
    return FrameDiffStats(
        num_changes=len(changes),
        bbox=(min(xs), min(ys), max(xs), max(ys)),
        color_changes=[(before, after, count) for (before, after), count in color_changes.most_common(8)],
        examples=examples,
        components=components,
    )


def _changed_components(
    changes: list[tuple[int, int, Any, Any]], max_examples: int = 8
) -> list[dict[str, Any]]:
    change_by_coord = {(x, y): (before, after) for x, y, before, after in changes}
    visited: set[tuple[int, int]] = set()
    components: list[dict[str, Any]] = []

    for x, y, before, after in changes:
        start = (x, y)
        if start in visited:
            continue

        stack = [start]
        visited.add(start)
        cells: list[tuple[int, int]] = []
        while stack:
            cx, cy = stack.pop()
            cells.append((cx, cy))
            for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                if (nx, ny) in visited:
                    continue
                if change_by_coord.get((nx, ny)) != (before, after):
                    continue
                visited.add((nx, ny))
                stack.append((nx, ny))

        xs = [cell_x for cell_x, _ in cells]
        ys = [cell_y for _, cell_y in cells]
        component_examples = [
            {"x": cell_x, "y": cell_y, "before": before, "after": after}
            for cell_x, cell_y in sorted(cells, key=lambda cell: (cell[1], cell[0]))[:max_examples]
        ]
        components.append(
            {
                "size": len(cells),
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
                "center": (round(sum(xs) / len(xs), 2), round(sum(ys) / len(ys), 2)),
                "before": before,
                "after": after,
                "examples": component_examples,
            }
        )

    components.sort(key=lambda item: (-int(item["size"]), item["bbox"]))
    for idx, component in enumerate(components):
        component["id"] = idx
    return components


def _component_summary(components: list[dict[str, Any]], max_components: int = 6) -> str:
    if not components:
        return "components=[]"

    compact_components = []
    for component in components[:max_components]:
        x1, y1, x2, y2 = component["bbox"]
        before = _color_name(component["before"])
        after = _color_name(component["after"])
        compact_components.append(
            {
                "id": component["id"],
                "size": component["size"],
                "bbox": f"({x1},{y1})-({x2},{y2})",
                "center": component["center"],
                "change": f"{before}->{after}",
            }
        )
    suffix = "" if len(components) <= max_components else f" more={len(components) - max_components}"
    return f"components={compact_components}{suffix}"


def _diff_summary(diff_stats: FrameDiffStats | None) -> str:
    if diff_stats is None:
        return "frame_diff: unavailable"
    if diff_stats.num_changes == 0:
        return "frame_diff: num_changes=0"
    assert diff_stats.bbox is not None
    color_text = ", ".join(
        f"{_color_name(before)}->{_color_name(after)}:{count}"
        for before, after, count in diff_stats.color_changes
    )
    x1, y1, x2, y2 = diff_stats.bbox
    return (
        f"frame_diff: num_changes={diff_stats.num_changes} "
        f"bbox=({x1},{y1})-({x2},{y2}) "
        f"colors={color_text} {_component_summary(diff_stats.components)}"
    )


def _action_effect_summary(diff_stats: FrameDiffStats | None) -> str:
    if diff_stats is None:
        return "last_action_effect=UNKNOWN"
    if diff_stats.num_changes == 0:
        return "last_action_effect=NO_CHANGE"
    assert diff_stats.bbox is not None
    x1, y1, x2, y2 = diff_stats.bbox
    return f"last_action_effect=CHANGED num_changes={diff_stats.num_changes} bbox=({x1},{y1})-({x2},{y2})"


def _color_name(value: Any) -> str:
    return FRAME_COLOR_NAMES.get(value, str(value))


def _parsed_action_metadata(parsed: ParsedAction | None) -> dict[str, Any] | None:
    if parsed is None:
        return None
    return {"name": parsed.name, "x": parsed.x, "y": parsed.y}


def _last_action_summary(parsed: ParsedAction | None) -> str:
    metadata = _parsed_action_metadata(parsed)
    if metadata is None:
        return "last_action=unparsed"
    return f"last_action={json.dumps(metadata, separators=(',', ':'))}"


def _diff_metadata(diff_stats: FrameDiffStats | None) -> dict[str, Any] | None:
    if diff_stats is None:
        return None
    return {
        "num_changes": diff_stats.num_changes,
        "bbox": diff_stats.bbox,
        "color_changes": diff_stats.color_changes,
        "examples": diff_stats.examples,
        "components": diff_stats.components,
        "meaningful": diff_stats.num_changes > 0,
    }


class ArcAgi3Env(BaseTextEnv):
    """SkyRL text environment wrapper around the ARC-AGI-3 Toolkit."""

    def __init__(self, env_config: Any, extras: dict[str, Any] | None = None):
        super().__init__()
        self.env_config = env_config or {}
        self.extras = extras or {}
        self.max_turns = int(self.extras.get("max_turns", self.extras.get("max_steps", 64)))
        self.task_id = str(self.extras.get("task_id", os.getenv("ARC_AGI3_TASK_ID", "ft09")))
        self.seed = self.extras.get("seed")
        self.operation_mode = str(
            self.extras.get("operation_mode", os.getenv("OPERATION_MODE", "OFFLINE"))
        ).upper()
        self.environments_dir = self.extras.get("environments_dir", os.getenv("ARC_AGI3_ENVIRONMENTS_DIR"))
        self.levels_to_complete = int(self.extras.get("levels_to_complete", 6))
        self.invalid_action_reward = float(self.extras.get("invalid_action_reward", -0.1))
        self.level_reward = float(self.extras.get("level_reward", 3.0))
        self.done_reward = float(self.extras.get("done_reward", 0.0))
        self.meaningful_diff_reward = float(self.extras.get("meaningful_diff_reward", 0.005))
        self.min_meaningful_diff_changes = int(self.extras.get("min_meaningful_diff_changes", 1))
        self.max_meaningful_diff_changes = int(self.extras.get("max_meaningful_diff_changes", 512))
        self.repeat_click_penalty = float(self.extras.get("repeat_click_penalty", -0.02))
        self.repeat_click_radius = int(self.extras.get("repeat_click_radius", 2))
        self.oracle_distance_reward_enabled = _truthy(
            self.extras.get(
                "oracle_distance_reward_enabled",
                os.getenv("ARC_AGI3_ORACLE_DISTANCE_REWARD_ENABLED", "false"),
            )
        )
        self.oracle_distance_reward = float(
            self.extras.get("oracle_distance_reward", os.getenv("ARC_AGI3_ORACLE_DISTANCE_REWARD", "0.05"))
        )
        self.oracle_distance_valid_action_reward = float(
            self.extras.get(
                "oracle_distance_valid_action_reward",
                os.getenv("ARC_AGI3_ORACLE_DISTANCE_VALID_ACTION_REWARD", "0.0"),
            )
        )
        self.oracle_action_match_reward = float(
            self.extras.get(
                "oracle_action_match_reward",
                os.getenv("ARC_AGI3_ORACLE_ACTION_MATCH_REWARD", "0.0"),
            )
        )
        self.oracle_action_match_radius = int(
            self.extras.get(
                "oracle_action_match_radius",
                os.getenv("ARC_AGI3_ORACLE_ACTION_MATCH_RADIUS", "0"),
            )
        )
        self.oracle_no_progress_penalty = float(
            self.extras.get(
                "oracle_no_progress_penalty",
                os.getenv("ARC_AGI3_ORACLE_NO_PROGRESS_PENALTY", "0.0"),
            )
        )
        self.oracle_distance_unrecoverable_penalty = float(
            self.extras.get(
                "oracle_distance_unrecoverable_penalty",
                os.getenv("ARC_AGI3_ORACLE_DISTANCE_UNRECOVERABLE_PENALTY", "-1.0"),
            )
        )
        self.oracle_distance_max_next_actions = int(
            self.extras.get(
                "oracle_distance_max_next_actions",
                os.getenv("ARC_AGI3_ORACLE_DISTANCE_MAX_NEXT_ACTIONS", "8"),
            )
        )
        self.frame_observation_mode = str(self.extras.get("frame_observation_mode", "initial_full_then_diff"))
        self.full_frame_interval = int(self.extras.get("full_frame_interval", 0))
        self.patch_radius = int(self.extras.get("patch_radius", 4))
        self.max_diff_examples = int(self.extras.get("max_diff_examples", 32))
        self.max_full_frame_rows = self.extras.get("max_full_frame_rows")
        self.max_full_frame_rows = (
            None if self.max_full_frame_rows in (None, "", "none", "None") else int(self.max_full_frame_rows)
        )

        self.arc = None
        self.env = None
        self.GameAction = None
        self.OperationMode = None
        self.turns = 0
        self.invalid_actions = 0
        self.last_frame = None
        self.last_score = 0.0
        self.last_levels_completed = 0
        self.last_diff_stats: FrameDiffStats | None = None
        self.last_click: tuple[int, int] | None = None
        self.last_oracle_distance: OracleDistanceInfo | None = None
        self.done = False
        self.success = False
        self.game_over = False

        self._init_arc_env()

    def _init_arc_env(self) -> None:
        try:
            import arc_agi
            from arc_agi import OperationMode
            from arcengine import GameAction
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "arc_agi/arcengine is not installed. Install it with `uv pip install arc-agi` "
                "inside the repository .venv, then rerun the training script."
            ) from exc

        self.OperationMode = OperationMode
        self.GameAction = GameAction
        mode = getattr(OperationMode, self.operation_mode, OperationMode.OFFLINE)
        arcade_kwargs: dict[str, Any] = {"operation_mode": mode}
        if self.environments_dir:
            arcade_kwargs["environments_dir"] = str(Path(self.environments_dir).expanduser())
        self.arc = arc_agi.Arcade(**arcade_kwargs)

        make_kwargs: dict[str, Any] = {"renderer": noop_renderer}
        if self.seed not in (None, ""):
            make_kwargs["seed"] = int(self.seed)
        self.env = self.arc.make(self.task_id, **make_kwargs)

        obs_space = getattr(self.env, "observation_space", None)
        self.last_frame = _to_plain(getattr(obs_space, "frame", None))
        self.last_score = self._read_score(obs_space)
        self.last_levels_completed = self._read_levels_completed(obs_space)
        self.last_oracle_distance = self._oracle_distance_info()

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        prompt = list(prompt)
        prompt.append({"role": "user", "content": self._build_observation_text(initial=True)})
        return prompt, {"task_id": self.task_id}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1

        parsed: ParsedAction | None = None
        reward_components: dict[str, float] = {}
        diff_metadata = None
        current_frame = None
        oracle_before = self._oracle_distance_info()
        try:
            parsed = parse_model_action(action)
            step_output = self._apply_action(parsed)
            valid_action = True
            error = None
        except Exception as exc:
            step_output = None
            valid_action = False
            error = str(exc)
            self.invalid_actions += 1

        if step_output is not None:
            current_frame = _to_plain(getattr(step_output, "frame", None))
            diff_stats = _diff_stats(self.last_frame, current_frame, max_examples=self.max_diff_examples)
            oracle_after = self._oracle_distance_info()
            reward, reward_components = self._compute_reward(
                step_output,
                diff_stats,
                parsed,
                valid_action=valid_action,
                oracle_before=oracle_before,
                oracle_after=oracle_after,
            )
            self.done = bool(getattr(step_output, "done", False)) or self.turns >= self.max_turns
            self.success = self._is_success(step_output)
            self.game_over = self._is_game_over(step_output)
        else:
            diff_stats = None
            oracle_after = oracle_before
            reward = self.invalid_action_reward
            reward_components = {"invalid_action": self.invalid_action_reward}
            self.done = self.turns >= self.max_turns

        observation_text = self._build_observation_text(
            parsed_action=parsed,
            valid_action=valid_action,
            error=error,
            step_output=step_output,
            diff_stats=diff_stats,
        )
        diff_metadata = _diff_metadata(diff_stats)
        if diff_metadata is not None:
            diff_metadata["meaningful"] = self._has_meaningful_diff(diff_stats)
        if self.done:
            observations: ConversationType = []
        else:
            observations = [{"role": "user", "content": observation_text}]

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=float(reward),
            done=self.done,
            metadata={
                "task_id": self.task_id,
                "turns": self.turns,
                "valid_action": valid_action,
                "invalid_actions": self.invalid_actions,
                "success": self.success,
                "game_over": self.game_over,
                "error": error,
                "model_output": action,
                "parsed_action": _parsed_action_metadata(parsed),
                "reward_components": reward_components,
                "oracle_distance": self._oracle_distance_metadata(oracle_before, oracle_after, reward_components),
                "diff_stats": diff_metadata,
                "frame": _frame_metadata(current_frame if current_frame is not None else self.last_frame),
                "state": self._state_metadata(step_output),
            },
        )

    def _apply_action(self, parsed: ParsedAction) -> Any:
        action_members = getattr(self.GameAction, "__members__", {})
        if parsed.name not in action_members:
            raise ValueError(f"unknown action {parsed.name}; available={sorted(action_members.keys())}")
        allowed_actions = self._allowed_action_names()
        if allowed_actions and parsed.name not in allowed_actions:
            raise ValueError(f"action {parsed.name} is not currently available; available={sorted(allowed_actions)}")
        action_enum = action_members[parsed.name]
        if parsed.name == "ACTION6":
            return self.env.step(action_enum, data={"x": int(parsed.x), "y": int(parsed.y)})
        return self.env.step(action_enum)

    def _compute_reward(
        self,
        observation: Any,
        diff_stats: FrameDiffStats | None,
        parsed: ParsedAction | None,
        *,
        valid_action: bool = True,
        oracle_before: OracleDistanceInfo | None = None,
        oracle_after: OracleDistanceInfo | None = None,
    ) -> tuple[float, dict[str, float]]:
        levels_completed = self._read_levels_completed(observation)
        level_delta = max(0, levels_completed - self.last_levels_completed)
        done = bool(getattr(observation, "done", False))
        success = self._is_success(observation)
        click = self._click_tuple(parsed)
        repeated_click = self._is_repeated_click(click)

        components = {
            "level_delta": level_delta * self.level_reward,
            "done": self.done_reward if done else 0.0,
            "meaningful_diff": self.meaningful_diff_reward if self._has_meaningful_diff(diff_stats) else 0.0,
            "repeat_click": self.repeat_click_penalty if repeated_click else 0.0,
        }
        if self.oracle_distance_reward_enabled:
            progress = oracle_progress_delta(
                oracle_before,
                oracle_after,
                level_delta=level_delta,
                success=success,
            )
            components["oracle_progress"] = progress * self.oracle_distance_reward
            components["oracle_valid_action"] = self.oracle_distance_valid_action_reward if valid_action else 0.0
            components["oracle_action_match"] = (
                self.oracle_action_match_reward
                if matches_oracle_next_action(
                    click,
                    oracle_before,
                    radius=self.oracle_action_match_radius,
                )
                else 0.0
            )
            components["oracle_no_progress"] = (
                self.oracle_no_progress_penalty
                if should_penalize_oracle_no_progress(
                    oracle_before,
                    oracle_after,
                    progress_delta=progress,
                    level_delta=level_delta,
                    success=success,
                    valid_action=valid_action,
                )
                else 0.0
            )
            components["oracle_unrecoverable"] = (
                self.oracle_distance_unrecoverable_penalty
                if self._is_oracle_unrecoverable(oracle_before, oracle_after, success)
                else 0.0
            )
        reward = sum(components.values())

        self.last_score = self._read_score(observation)
        self.last_levels_completed = levels_completed
        self.last_diff_stats = diff_stats
        self.last_oracle_distance = oracle_after
        if click is not None:
            self.last_click = click
        return reward, components

    def _oracle_distance_info(self) -> OracleDistanceInfo | None:
        if not self.oracle_distance_reward_enabled or self.task_id != "ft09" or self.env is None:
            return None
        game = getattr(self.env, "_game", None)
        if game is None:
            return None
        return inspect_oracle_distance(game, max_next_actions=self.oracle_distance_max_next_actions)

    def _is_oracle_unrecoverable(
        self,
        before: OracleDistanceInfo | None,
        after: OracleDistanceInfo | None,
        success: bool,
    ) -> bool:
        if success:
            return False
        return bool(before and before.solvable and after and not after.solvable)

    def _oracle_distance_metadata(
        self,
        before: OracleDistanceInfo | None,
        after: OracleDistanceInfo | None,
        reward_components: dict[str, float],
    ) -> dict[str, Any] | None:
        if not self.oracle_distance_reward_enabled:
            return None
        if self.oracle_distance_reward:
            progress = reward_components.get("oracle_progress", 0.0) / self.oracle_distance_reward
        else:
            progress = 0.0
        return {
            "before": before.to_metadata() if before else None,
            "after": after.to_metadata() if after else None,
            "progress_delta": progress,
        }

    def _has_meaningful_diff(self, diff_stats: FrameDiffStats | None) -> bool:
        if diff_stats is None:
            return False
        return self.min_meaningful_diff_changes <= diff_stats.num_changes <= self.max_meaningful_diff_changes

    def _click_tuple(self, parsed: ParsedAction | None) -> tuple[int, int] | None:
        if parsed is None or parsed.name != "ACTION6" or parsed.x is None or parsed.y is None:
            return None
        return (int(parsed.x), int(parsed.y))

    def _is_repeated_click(self, click: tuple[int, int] | None) -> bool:
        if click is None or self.last_click is None:
            return False
        return (
            abs(click[0] - self.last_click[0]) <= self.repeat_click_radius
            and abs(click[1] - self.last_click[1]) <= self.repeat_click_radius
        )

    def _build_observation_text(
        self,
        initial: bool = False,
        parsed_action: ParsedAction | None = None,
        valid_action: bool | None = None,
        error: str | None = None,
        step_output: Any | None = None,
        diff_stats: FrameDiffStats | None = None,
    ) -> str:
        source = step_output if step_output is not None else getattr(self.env, "observation_space", None)
        state = _enum_name(getattr(source, "state", None))
        levels_completed = self._read_levels_completed(source)
        score = self._read_score(source)
        current_frame = _to_plain(getattr(source, "frame", None))
        action_effect = None
        if initial:
            diff = "frame_diff: initial"
            frame_lines = self._frame_observation_lines(
                initial=initial,
                previous_frame=None,
                current_frame=current_frame,
                diff_stats=None,
                levels_completed=levels_completed,
            )
        else:
            diff_stats = diff_stats if diff_stats is not None else _diff_stats(
                self.last_frame, current_frame, max_examples=self.max_diff_examples
            )
            diff = _diff_summary(diff_stats)
            action_effect = _action_effect_summary(diff_stats)
            frame_lines = self._frame_observation_lines(
                initial=initial,
                previous_frame=self.last_frame,
                current_frame=current_frame,
                diff_stats=diff_stats,
                levels_completed=levels_completed,
            )
        if current_frame is not None:
            self.last_frame = current_frame

        action_space = self._action_space_text()
        lines = [
            f"task_id={self.task_id}",
            f"turn={self.turns}/{self.max_turns}",
            f"state={state}",
            f"score={score}",
            f"levels_completed={levels_completed}/{self.levels_to_complete}",
            f"available_actions={action_space}",
        ]
        if initial and self.frame_observation_mode != "none":
            lines.append(OBSERVATION_GUIDE)
            lines.append(_format_color_legend())
        lines.append(diff)
        if action_effect is not None:
            lines.append(action_effect)
        lines.extend(frame_lines)
        if valid_action is not None:
            lines.append(_last_action_summary(parsed_action))
            lines.append(f"last_action_valid={valid_action}")
        if error:
            lines.append(f"action_error={error}")
        lines.append(
            'Respond with brief reasoning in <think>...</think>, then exactly one next action in <action>...</action>. '
            'Use {"action":"ACTION6","x":32,"y":32} inside <action> for coordinate clicks.'
        )
        return "\n".join(lines)

    def _frame_observation_lines(
        self,
        initial: bool,
        previous_frame: Any,
        current_frame: Any,
        diff_stats: FrameDiffStats | None,
        levels_completed: int,
    ) -> list[str]:
        mode = self.frame_observation_mode
        if mode == "none":
            return []

        include_full_frame = False
        if mode == "full_every_turn":
            include_full_frame = True
        elif mode == "initial_full_then_diff":
            include_full_frame = initial
            include_full_frame = include_full_frame or (
                self.full_frame_interval > 0 and self.turns > 0 and self.turns % self.full_frame_interval == 0
            )
            include_full_frame = include_full_frame or levels_completed > self.last_levels_completed
            include_full_frame = include_full_frame or self.done
        else:
            include_full_frame = initial

        frame_lines = []
        if include_full_frame:
            frame_lines.append(_format_full_frame(current_frame, max_rows=self.max_full_frame_rows))
        if mode == "full_every_turn":
            return frame_lines
        if diff_stats is not None and diff_stats.num_changes > 0:
            frame_lines.append(_format_diff_patch(previous_frame, current_frame, diff_stats, radius=self.patch_radius))
        return frame_lines

    def _action_space_text(self) -> str:
        available_actions = sorted(self._allowed_action_names())
        raw_action_space = repr(getattr(self.env, "action_space", None))
        return f"{available_actions}; raw={raw_action_space}"

    def _allowed_action_names(self) -> set[str]:
        raw_action_space = getattr(self.env, "action_space", None)
        if raw_action_space is None:
            return set()
        if isinstance(raw_action_space, (list, tuple, set)):
            items = raw_action_space
        else:
            items = getattr(raw_action_space, "actions", None) or getattr(raw_action_space, "values", None)
            if items is None:
                try:
                    items = list(raw_action_space)
                except TypeError:
                    items = []
        names = {_enum_name(item) for item in items}
        return {name for name in names if name}

    def _read_levels_completed(self, source: Any) -> int:
        value = getattr(source, "levels_completed", None)
        if value is None:
            value = getattr(source, "completed", None)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _read_score(self, source: Any) -> float:
        value = getattr(source, "score", None)
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _state_metadata(self, source: Any) -> dict[str, Any]:
        return {
            "state": _enum_name(getattr(source, "state", None)),
            "score": self._read_score(source),
            "levels_completed": self._read_levels_completed(source),
            "done": bool(getattr(source, "done", False)) if source is not None else self.done,
        }

    def _is_success(self, observation: Any) -> bool:
        if (
            bool(getattr(observation, "done", False))
            and self._read_levels_completed(observation) >= self.levels_to_complete
        ):
            return True
        state_name = (_enum_name(getattr(observation, "state", None)) or "").upper()
        return state_name in {"WIN", "WON", "SUCCESS", "COMPLETE", "COMPLETED"}

    def _is_game_over(self, observation: Any) -> bool:
        state_name = (_enum_name(getattr(observation, "state", None)) or "").upper()
        return state_name in {"GAME_OVER", "LOSE", "LOST", "FAILED"}

    def close(self):
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def get_metrics(self) -> dict[str, Any]:
        obs_space = getattr(self.env, "observation_space", None)
        return {
            "task_id": self.task_id,
            "success": float(self.success),
            "game_over": float(self.game_over),
            "final_score": self._read_score(obs_space),
            "levels_completed": self._read_levels_completed(obs_space),
            "steps": self.turns,
            "invalid_actions": self.invalid_actions,
        }
