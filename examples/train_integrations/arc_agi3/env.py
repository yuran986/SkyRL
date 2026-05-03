from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType


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


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def _format_diff_patch(frame_like: Any, diff_stats: FrameDiffStats | None, radius: int) -> str:
    if diff_stats is None or diff_stats.bbox is None or diff_stats.num_changes == 0:
        return "changed_patch: none"
    grid = _grid_from_frame(frame_like)
    if grid is None:
        return "changed_patch: unavailable"
    x1, y1, x2, y2 = diff_stats.bbox
    patch, (cx1, cy1, cx2, cy2) = _crop_grid(grid, x1 - radius, y1 - radius, x2 + radius, y2 + radius)
    return f"changed_patch: x={cx1}..{cx2} y={cy1}..{cy2}\n" + _format_hex_rows(patch, y_offset=cy1)


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
    return FrameDiffStats(
        num_changes=len(changes),
        bbox=(min(xs), min(ys), max(xs), max(ys)),
        color_changes=[(before, after, count) for (before, after), count in color_changes.most_common(8)],
        examples=examples,
    )


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
        f"colors={color_text} examples={diff_stats.examples}"
    )


def _color_name(value: Any) -> str:
    return FRAME_COLOR_NAMES.get(value, str(value))


def _parsed_action_metadata(parsed: ParsedAction | None) -> dict[str, Any] | None:
    if parsed is None:
        return None
    return {"name": parsed.name, "x": parsed.x, "y": parsed.y}


def _diff_metadata(diff_stats: FrameDiffStats | None) -> dict[str, Any] | None:
    if diff_stats is None:
        return None
    return {
        "num_changes": diff_stats.num_changes,
        "bbox": diff_stats.bbox,
        "color_changes": diff_stats.color_changes,
        "examples": diff_stats.examples,
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
        self.invalid_action_reward = float(self.extras.get("invalid_action_reward", -0.05))
        self.level_reward = float(self.extras.get("level_reward", 1.0))
        self.done_reward = float(self.extras.get("done_reward", 1.0))
        self.meaningful_diff_reward = float(self.extras.get("meaningful_diff_reward", 0.05))
        self.min_meaningful_diff_changes = int(self.extras.get("min_meaningful_diff_changes", 1))
        self.max_meaningful_diff_changes = int(self.extras.get("max_meaningful_diff_changes", 512))
        self.frame_observation_mode = str(self.extras.get("frame_observation_mode", "initial_full_then_diff"))
        self.full_frame_interval = int(self.extras.get("full_frame_interval", 8))
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
                "or run via `uv run --with arc-agi ...`."
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

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        prompt = list(prompt)
        prompt.append({"role": "user", "content": self._build_observation_text(initial=True)})
        return prompt, {"task_id": self.task_id}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1

        parsed: ParsedAction | None = None
        reward_components: dict[str, float] = {}
        diff_metadata = None
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
            reward, reward_components = self._compute_reward(step_output, diff_stats)
            self.done = bool(getattr(step_output, "done", False)) or self.turns >= self.max_turns
            self.success = self._is_success(step_output)
            self.game_over = self._is_game_over(step_output)
        else:
            diff_stats = None
            reward = self.invalid_action_reward
            reward_components = {"invalid_action": self.invalid_action_reward}
            self.done = self.turns >= self.max_turns

        observation_text = self._build_observation_text(
            action=action,
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
                "diff_stats": diff_metadata,
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

    def _compute_reward(self, observation: Any, diff_stats: FrameDiffStats | None) -> tuple[float, dict[str, float]]:
        levels_completed = self._read_levels_completed(observation)
        level_delta = max(0, levels_completed - self.last_levels_completed)
        done = bool(getattr(observation, "done", False))

        components = {
            "level_delta": level_delta * self.level_reward,
            "done": self.done_reward if done else 0.0,
            "meaningful_diff": self.meaningful_diff_reward if self._has_meaningful_diff(diff_stats) else 0.0,
        }
        reward = 0.0
        if level_delta > 0:
            reward += components["level_delta"]
        if done:
            reward += components["done"]
        if self._has_meaningful_diff(diff_stats):
            reward += components["meaningful_diff"]

        self.last_score = self._read_score(observation)
        self.last_levels_completed = levels_completed
        self.last_diff_stats = diff_stats
        return reward, components

    def _has_meaningful_diff(self, diff_stats: FrameDiffStats | None) -> bool:
        if diff_stats is None:
            return False
        return self.min_meaningful_diff_changes <= diff_stats.num_changes <= self.max_meaningful_diff_changes

    def _build_observation_text(
        self,
        initial: bool = False,
        action: str | None = None,
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
        if initial:
            diff = "frame_diff: initial"
            frame_lines = self._frame_observation_lines(
                initial=initial,
                current_frame=current_frame,
                diff_stats=None,
                levels_completed=levels_completed,
            )
        else:
            diff_stats = diff_stats if diff_stats is not None else _diff_stats(
                self.last_frame, current_frame, max_examples=self.max_diff_examples
            )
            diff = _diff_summary(diff_stats)
            frame_lines = self._frame_observation_lines(
                initial=initial,
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
            diff,
        ]
        lines.extend(frame_lines)
        if action is not None:
            lines.append(f"last_model_output={action.strip()[:500]}")
            lines.append(f"last_action_valid={valid_action}")
        if error:
            lines.append(f"action_error={error}")
        lines.append(
            'Return exactly one next action in <action>...</action>. '
            'Use {"action":"ACTION6","x":32,"y":32} for coordinate clicks.'
        )
        return "\n".join(lines)

    def _frame_observation_lines(
        self,
        initial: bool,
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
            frame_lines.append(_format_diff_patch(current_frame, diff_stats, radius=self.patch_radius))
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
