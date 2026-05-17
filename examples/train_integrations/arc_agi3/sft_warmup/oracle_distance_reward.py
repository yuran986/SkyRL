"""Oracle-distance helpers for ARC-AGI-3 ft09 warm-up.

The helper in this module is intentionally read-only: it inspects the current
game object, asks the ft09 oracle for a click plan, and summarizes that plan for
reward shaping. It must not execute oracle actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


OraclePlanSolver = Callable[[Any], list[tuple[Any, int]]]


@dataclass(frozen=True)
class OracleDistanceInfo:
    solvable: bool
    plan_len: int | None
    level_index: int | None
    level_name: str | None
    next_actions: list[dict[str, Any]]
    error: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "solvable": self.solvable,
            "plan_len": self.plan_len,
            "level_index": self.level_index,
            "level_name": self.level_name,
            "next_actions": self.next_actions,
            "error": self.error,
        }


def _default_solve_click_plan(game: Any) -> list[tuple[Any, int]]:
    from examples.train_integrations.arc_agi3.ft09_oracle_solution import solve_click_plan

    return solve_click_plan(game)


def _level_index(game: Any) -> int | None:
    value = getattr(game, "level_index", getattr(game, "_current_level_index", None))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _level_name(game: Any) -> str | None:
    return getattr(getattr(game, "current_level", None), "name", None)


def _display_center(sprite: Any) -> tuple[int, int]:
    return 2 * (int(getattr(sprite, "x")) + 1), 2 * (int(getattr(sprite, "y")) + 1)


def inspect_oracle_distance(
    game: Any,
    *,
    solver: OraclePlanSolver | None = None,
    max_next_actions: int = 8,
) -> OracleDistanceInfo:
    """Return the current ft09 oracle plan length and representative next actions."""
    solve_click_plan = solver or _default_solve_click_plan
    try:
        plan = solve_click_plan(game)
    except Exception as exc:
        return OracleDistanceInfo(
            solvable=False,
            plan_len=None,
            level_index=_level_index(game),
            level_name=_level_name(game),
            next_actions=[],
            error=str(exc),
        )

    next_actions: list[dict[str, Any]] = []
    for sprite, count in plan[:max_next_actions]:
        x, y = _display_center(sprite)
        next_actions.append(
            {
                "action": "ACTION6",
                "x": x,
                "y": y,
                "count": int(count),
                "sprite_name": getattr(sprite, "name", None),
                "grid": {"x": int(getattr(sprite, "x")), "y": int(getattr(sprite, "y"))},
            }
        )

    return OracleDistanceInfo(
        solvable=True,
        plan_len=sum(max(0, int(count)) for _, count in plan),
        level_index=_level_index(game),
        level_name=_level_name(game),
        next_actions=next_actions,
    )


def oracle_progress_delta(
    before: OracleDistanceInfo | None,
    after: OracleDistanceInfo | None,
    *,
    level_delta: int,
    success: bool,
) -> float:
    """Compute potential progress from two oracle inspections.

    Within a level, this is `before.plan_len - after.plan_len`. On level
    advance, the next level may have a larger fresh plan, so reward consuming
    the remaining current-level oracle plan instead of comparing across levels.
    """
    if before is None or not before.solvable or before.plan_len is None:
        return 0.0
    if success or level_delta > 0:
        return float(max(1, before.plan_len))
    if after is None or not after.solvable or after.plan_len is None:
        return 0.0
    return float(before.plan_len - after.plan_len)
