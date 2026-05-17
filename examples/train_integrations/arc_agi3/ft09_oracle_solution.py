"""Pure-action oracle attempt for the local ft09 environment.

The solver is allowed to inspect the source-level rules, but it only changes the game by
issuing public ACTION6 clicks. It models each level as a modular linear system over block
colors and fails explicitly if the current local source has no legal action plan.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction


CHECKS = (
    (0, 0, -1, -1),
    (0, 1, 0, -1),
    (0, 2, 1, -1),
    (1, 0, -1, 0),
    (1, 2, 1, 0),
    (2, 0, -1, 1),
    (2, 1, 0, 1),
    (2, 2, 1, 1),
)

NEIGHBOR_OFFSETS = (
    ((-1, -1), (0, -1), (1, -1)),
    ((-1, 0), (0, 0), (1, 0)),
    ((-1, 1), (0, 1), (1, 1)),
)


@dataclass(frozen=True)
class PlannedClick:
    reason: str
    sprite_name: str
    grid_x: int
    grid_y: int
    display_x: int
    display_y: int


def noop_renderer(*args: Any, **kwargs: Any) -> None:
    return None


def center_color(sprite: Any) -> int:
    return int(sprite.pixels[1][1])


def sprite_key(sprite: Any) -> tuple[int, int]:
    return int(sprite.x), int(sprite.y)


def display_center(sprite: Any) -> tuple[int, int]:
    # ft09 uses a 32x32 internal grid rendered to a 64x64 display.
    return 2 * (int(sprite.x) + 1), 2 * (int(sprite.y) + 1)


def has_tag(sprite: Any, tag: str) -> bool:
    return tag in getattr(sprite, "tags", [])


def get_blocks(game: Any) -> dict[tuple[int, int], Any]:
    blocks: dict[tuple[int, int], Any] = {}
    for sprite in game.current_level.get_sprites_by_tag("Hkx"):
        blocks[sprite_key(sprite)] = sprite
    for sprite in game.current_level.get_sprites_by_tag("NTi"):
        blocks[sprite_key(sprite)] = sprite
    return blocks


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "name"):
        return value.name
    return str(value)


def normalize_frame(frame: Any) -> list[list[int]] | None:
    plain = to_jsonable(frame)
    if isinstance(plain, dict) and "frame" in plain:
        plain = plain["frame"]
    if (
        isinstance(plain, list)
        and len(plain) == 1
        and isinstance(plain[0], list)
        and plain[0]
        and isinstance(plain[0][0], list)
    ):
        plain = plain[0]
    if not (isinstance(plain, list) and plain and isinstance(plain[0], list)):
        return None
    return [[int(value) for value in row] for row in plain]


def current_display_frame(env: Any) -> list[list[int]] | None:
    """Return the current 64x64 display frame, including internal-only mutations."""
    try:
        internal = env._game.get_pixels(0, 0, 64, 64)
        internal_frame = normalize_frame(internal)
        if internal_frame:
            return [
                [int(value) for value in row for _ in range(2)]
                for row in internal_frame
                for _ in range(2)
            ]
    except Exception:
        pass
    return normalize_frame(getattr(env._last_response, "frame", None))


def record_trace_event(
    trace: list[dict[str, Any]] | None,
    env: Any,
    event: str,
    **fields: Any,
) -> None:
    if trace is None:
        return
    game = env._game
    response = getattr(env, "_last_response", None)
    level_index = int(getattr(game, "level_index", getattr(game, "_current_level_index", 0)))
    state = getattr(game, "_state", None)
    completed = int(getattr(response, "levels_completed", 0)) if response is not None else 0
    if getattr(state, "name", None) == "WIN":
        completed = max(completed, 6)

    trace.append(
        {
            "index": len(trace),
            "event": event,
            "level_index": level_index,
            "level_name": getattr(getattr(game, "current_level", None), "name", None),
            "game_state": to_jsonable(state),
            "levels_completed": completed,
            "frame": current_display_frame(env),
            **to_jsonable(fields),
        }
    )


def write_trace_jsonl(trace: list[dict[str, Any]], path: str | os.PathLike[str]) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for event in trace:
            handle.write(json.dumps(to_jsonable(event), ensure_ascii=False) + "\n")


def allowed_colors(game: Any) -> dict[tuple[int, int], set[int]]:
    blocks = get_blocks(game)
    colors = set(int(color) for color in game.gqb)
    allowed: dict[tuple[int, int], set[int]] = {}

    for target in game.current_level.get_sprites_by_tag("bsT"):
        target_color = center_color(target)
        for row, col, dx, dy in CHECKS:
            coord = (int(target.x) + dx * 4, int(target.y) + dy * 4)
            if coord not in blocks:
                continue

            cell_is_zero = int(target.pixels[row][col]) == 0
            candidate = {target_color} if cell_is_zero else colors - {target_color}
            allowed[coord] = candidate if coord not in allowed else allowed[coord] & candidate
            if not allowed[coord]:
                raise RuntimeError(
                    f"Conflicting ft09 constraints at level={game.current_level.name} coord={coord}"
                )

    return allowed


def choose_desired_colors(game: Any) -> dict[tuple[int, int], int]:
    blocks = get_blocks(game)
    desired: dict[tuple[int, int], int] = {}
    for coord, options in allowed_colors(game).items():
        current = center_color(blocks[coord])
        desired[coord] = current if current in options else min(options)
    return desired


def click_effects(
    game: Any,
    sprite: Any,
    blocks: dict[tuple[int, int], Any],
) -> list[tuple[int, int]]:
    if has_tag(sprite, "NTi"):
        effect_pattern = [
            [1 if int(sprite.pixels[row][col]) == 6 else 0 for col in range(3)]
            for row in range(3)
        ]
        effect_pattern[1][1] = 1
    else:
        effect_pattern = game.irw

    effects: list[tuple[int, int]] = []
    for row in range(3):
        for col in range(3):
            if int(effect_pattern[row][col]) != 1:
                continue
            dx, dy = NEIGHBOR_OFFSETS[row][col]
            coord = (int(sprite.x) + dx * 4, int(sprite.y) + dy * 4)
            if coord in blocks:
                effects.append(coord)
    return effects


def inverse_mod(value: int, modulus: int) -> int:
    value %= modulus
    for candidate in range(modulus):
        if (value * candidate) % modulus == 1:
            return candidate
    raise ValueError(f"{value} has no inverse modulo {modulus}")


def solve_modular_linear(
    rows: list[list[int]],
    rhs: list[int],
    modulus: int,
) -> list[int] | None:
    if not rows:
        return []

    num_vars = len(rows[0])
    matrix = [
        [value % modulus for value in rows[index]] + [rhs[index] % modulus]
        for index in range(len(rows))
    ]
    pivots: list[tuple[int, int]] = []
    pivot_row = 0

    for col in range(num_vars):
        selected = next(
            (
                row
                for row in range(pivot_row, len(matrix))
                if matrix[row][col] % modulus
                and math.gcd(matrix[row][col] % modulus, modulus) == 1
            ),
            None,
        )
        if selected is None:
            continue
        matrix[pivot_row], matrix[selected] = matrix[selected], matrix[pivot_row]
        inverse = inverse_mod(matrix[pivot_row][col], modulus)
        matrix[pivot_row] = [(value * inverse) % modulus for value in matrix[pivot_row]]

        for row in range(len(matrix)):
            factor = matrix[row][col] % modulus
            if row != pivot_row and factor:
                matrix[row] = [
                    (matrix[row][index] - factor * matrix[pivot_row][index]) % modulus
                    for index in range(num_vars + 1)
                ]
        pivots.append((pivot_row, col))
        pivot_row += 1

    if any(
        all(value % modulus == 0 for value in row[:num_vars]) and row[num_vars]
        for row in matrix
    ):
        return None

    solution = [0] * num_vars
    for row, col in pivots:
        solution[col] = matrix[row][num_vars] % modulus
    return solution


def solve_click_plan(game: Any) -> list[tuple[Any, int]]:
    blocks = get_blocks(game)
    variables = list(blocks.values())
    colors = [int(color) for color in game.gqb]
    color_index = {color: index for index, color in enumerate(colors)}
    constraints = allowed_colors(game)
    desired = choose_desired_colors(game)

    rows: list[list[int]] = []
    rhs: list[int] = []
    for coord in constraints:
        rows.append(
            [
                1 if coord in click_effects(game, sprite, blocks) else 0
                for sprite in variables
            ]
        )
        rhs.append(
            (color_index[desired[coord]] - color_index[center_color(blocks[coord])])
            % len(colors)
        )

    solution = solve_modular_linear(rows, rhs, len(colors))
    if solution is None:
        raise RuntimeError(
            f"Level {game.current_level.name} has no legal ACTION6 plan under the "
            "source-defined click effects and target constraints"
        )

    plan = [
        (sprite, count % len(colors))
        for sprite, count in zip(variables, solution)
        if count % len(colors)
    ]
    if not plan and game.cgj() and variables:
        plan = [(variables[0], len(colors))]
    return plan


def execute_click(
    env: Any,
    sprite: Any,
    reason: str,
    verbose: bool,
    trace: list[dict[str, Any]] | None = None,
) -> bool:
    before_level = int(env._game.level_index)
    x, y = display_center(sprite)
    click = PlannedClick(reason, sprite.name, int(sprite.x), int(sprite.y), x, y)
    observation = env.step(GameAction.ACTION6, data={"x": x, "y": y})
    advanced = int(env._game.level_index) != before_level
    record_trace_event(
        trace,
        env,
        "click",
        reason=reason,
        sprite_name=click.sprite_name,
        grid={"x": click.grid_x, "y": click.grid_y},
        display={"x": click.display_x, "y": click.display_y},
        advanced_level=advanced,
        observation_state=getattr(observation, "state", None),
        observation_levels_completed=getattr(observation, "levels_completed", None),
    )
    if verbose:
        print(
            f"{click.reason}: {click.sprite_name}@grid=({click.grid_x},{click.grid_y}) "
            f"display=({click.display_x},{click.display_y}) -> "
            f"state={observation.state} levels_completed={observation.levels_completed}"
        )
    return advanced


def solve_current_level(
    env: Any,
    *,
    verbose: bool,
    trace: list[dict[str, Any]] | None = None,
) -> None:
    game = env._game
    before_level = int(game.level_index)
    plan = solve_click_plan(game)

    if verbose:
        planned_clicks = sum(count for _, count in plan)
        print(
            f"level={game.level_index} name={game.current_level.name} "
            f"colors={[int(color) for color in game.gqb]} planned_clicks={planned_clicks}"
        )

    for sprite, count in plan:
        for _ in range(count):
            if execute_click(env, sprite, "planned_legal_click", verbose, trace):
                return

    if int(env._game.level_index) == before_level and env._game._state.name != "WIN":
        raise RuntimeError(
            f"Legal click plan for level {game.current_level.name} did not advance the level"
        )


def run_ft09_solution(
    *,
    environments_dir: str,
    operation_mode: str,
    verbose: bool,
    max_level_attempts: int,
    trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    mode = getattr(OperationMode, operation_mode)
    arcade = arc_agi.Arcade(operation_mode=mode, environments_dir=environments_dir)
    env = arcade.make("ft09", renderer=noop_renderer)
    record_trace_event(trace, env, "init", reason="initial_frame")

    for _ in range(max_level_attempts):
        response = env._last_response
        if getattr(response, "levels_completed", 0) >= 6 or env._game._state.name == "WIN":
            record_trace_event(trace, env, "final", reason="game_finished")
            return {
                "state": env._game._state,
                "levels_completed": max(int(getattr(response, "levels_completed", 0)), 6),
                "last_response": response,
            }
        try:
            solve_current_level(env, verbose=verbose, trace=trace)
        except Exception as exc:
            record_trace_event(trace, env, "failure", reason=str(exc))
            raise

    if env._game._state.name == "WIN":
        record_trace_event(trace, env, "final", reason="game_finished")
        return {
            "state": env._game._state,
            "levels_completed": max(int(getattr(env._last_response, "levels_completed", 0)), 6),
            "last_response": env._last_response,
        }

    raise RuntimeError(
        f"ft09 did not finish within {max_level_attempts} level attempts; "
        f"last_response={env._last_response}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pure-action oracle attempt for ARC-AGI-3 ft09."
    )
    parser.add_argument(
        "--environments_dir",
        default=os.getenv("ARC_AGI3_ENVIRONMENTS_DIR", "/home/users/yz1051/rlm/environment_files"),
    )
    parser.add_argument("--operation_mode", default=os.getenv("OPERATION_MODE", "OFFLINE"))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max-level-attempts", type=int, default=16)
    parser.add_argument("--trace-jsonl", help="Optional path for a step-by-step oracle trace.")
    args = parser.parse_args()

    trace: list[dict[str, Any]] | None = [] if args.trace_jsonl else None
    try:
        final_result = run_ft09_solution(
            environments_dir=args.environments_dir,
            operation_mode=args.operation_mode,
            verbose=not args.quiet,
            max_level_attempts=args.max_level_attempts,
            trace=trace,
        )
    finally:
        if args.trace_jsonl and trace is not None:
            write_trace_jsonl(trace, args.trace_jsonl)
    print(
        f"final_state={final_result['state']} "
        f"levels_completed={final_result['levels_completed']}"
    )


if __name__ == "__main__":
    main()
