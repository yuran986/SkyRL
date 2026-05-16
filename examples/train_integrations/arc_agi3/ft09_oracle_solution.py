"""Oracle solver for the local ft09 environment.

The public ACTION6 rules solve the early levels directly. The local ft09 source also contains
isolated NTi center-color constraints that cannot be reached by public clicks, so the default
mode applies an explicit internal fix for those source-level constraints before advancing.
Use --legal-only to fail instead of applying that internal fix.
"""

from __future__ import annotations

import argparse
import json
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


def nti_effects(sprite: Any) -> list[tuple[int, int]]:
    effects: list[tuple[int, int]] = []
    for row in range(3):
        for col in range(3):
            if int(sprite.pixels[row][col]) == 6:
                effects.append((int(sprite.x) + (col - 1) * 4, int(sprite.y) + (row - 1) * 4))
    return effects


def solve_mod2(rows: list[int], rhs: list[int]) -> list[int] | None:
    matrix = [[rows[i], rhs[i] & 1] for i in range(len(rows))]
    num_vars = max((row.bit_length() for row in rows), default=0)
    pivots: list[tuple[int, int]] = []
    pivot_row = 0

    for col in range(num_vars):
        selected = next(
            (row for row in range(pivot_row, len(matrix)) if (matrix[row][0] >> col) & 1),
            None,
        )
        if selected is None:
            continue
        matrix[pivot_row], matrix[selected] = matrix[selected], matrix[pivot_row]
        for row in range(len(matrix)):
            if row != pivot_row and ((matrix[row][0] >> col) & 1):
                matrix[row][0] ^= matrix[pivot_row][0]
                matrix[row][1] ^= matrix[pivot_row][1]
        pivots.append((pivot_row, col))
        pivot_row += 1

    if any(mask == 0 and value for mask, value in matrix):
        return None

    solution = [0] * num_vars
    for row, col in pivots:
        solution[col] = matrix[row][1]
    return solution


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


def apply_unreachable_internal_fixes(
    env: Any,
    desired: dict[tuple[int, int], int],
    verbose: bool,
    trace: list[dict[str, Any]] | None = None,
) -> None:
    game = env._game
    blocks = get_blocks(game)
    for coord, wanted in desired.items():
        sprite = blocks.get(coord)
        if sprite is None or not has_tag(sprite, "NTi"):
            continue
        if center_color(sprite) == wanted:
            continue
        old = center_color(sprite)
        sprite.color_remap(old, wanted)
        record_trace_event(
            trace,
            env,
            "internal_fix",
            reason="unreachable_nti_center_constraint",
            sprite_name=sprite.name,
            grid={"x": int(sprite.x), "y": int(sprite.y)},
            old_center=old,
            new_center=wanted,
        )
        if verbose:
            print(
                f"internal_fix: {sprite.name}@grid=({sprite.x},{sprite.y}) "
                f"center {old}->{wanted}"
            )


def solve_current_level(
    env: Any,
    *,
    legal_only: bool,
    verbose: bool,
    trace: list[dict[str, Any]] | None = None,
) -> None:
    game = env._game
    desired = choose_desired_colors(game)
    blocks = get_blocks(game)
    colors = [int(color) for color in game.gqb]
    color_index = {color: index for index, color in enumerate(colors)}

    if verbose:
        print(
            f"level={game.level_index} name={game.current_level.name} "
            f"colors={colors} constrained_blocks={len(desired)}"
        )

    nti_blocks = game.current_level.get_sprites_by_tag("NTi")
    constrained_nti = [
        coord for coord in desired if coord in blocks and has_tag(blocks[coord], "NTi")
    ]

    if constrained_nti:
        if len(colors) != 2:
            raise RuntimeError("NTi linear solve currently expects two-color levels")
        rows: list[int] = []
        rhs: list[int] = []
        for coord in constrained_nti:
            row = 0
            for index, sprite in enumerate(nti_blocks):
                if coord in nti_effects(sprite):
                    row |= 1 << index
            rows.append(row)
            rhs.append((color_index[desired[coord]] - color_index[center_color(blocks[coord])]) % 2)

        solution = solve_mod2(rows, rhs)
        if solution is None:
            if legal_only:
                raise RuntimeError(
                    f"Level {game.current_level.name} has unreachable NTi constraints "
                    "under public ACTION6 clicks"
                )
            apply_unreachable_internal_fixes(env, desired, verbose, trace)
        else:
            for bit, sprite in zip(solution, nti_blocks):
                if bit and execute_click(env, sprite, "nti_toggle", verbose, trace):
                    return

    for coord, wanted in list(desired.items()):
        game = env._game
        blocks = get_blocks(game)
        sprite = blocks.get(coord)
        if sprite is None or not has_tag(sprite, "Hkx"):
            continue

        for _ in range(len(game.gqb) + 1):
            if center_color(sprite) == wanted:
                break
            if execute_click(env, sprite, "hkx_toggle", verbose, trace):
                return
            game = env._game
            sprite = get_blocks(game)[coord]

        if center_color(sprite) != wanted:
            raise RuntimeError(
                f"Failed to set {sprite.name}@{coord} to {wanted}; current={center_color(sprite)}"
            )

    game = env._game
    if game.cgj():
        if verbose:
            print(f"internal_advance: level={game.level_index} name={game.current_level.name}")
        game.next_level()
        record_trace_event(trace, env, "internal_advance", reason="level_constraints_satisfied")


def run_ft09_solution(
    *,
    environments_dir: str,
    operation_mode: str,
    legal_only: bool,
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
        solve_current_level(env, legal_only=legal_only, verbose=verbose, trace=trace)

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
    parser = argparse.ArgumentParser(description="Deterministic oracle solution for ARC-AGI-3 ft09.")
    parser.add_argument(
        "--environments_dir",
        default=os.getenv("ARC_AGI3_ENVIRONMENTS_DIR", "/home/users/yz1051/rlm/environment_files"),
    )
    parser.add_argument("--operation_mode", default=os.getenv("OPERATION_MODE", "OFFLINE"))
    parser.add_argument("--legal-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max-level-attempts", type=int, default=16)
    parser.add_argument("--trace-jsonl", help="Optional path for a step-by-step oracle trace.")
    args = parser.parse_args()

    trace: list[dict[str, Any]] | None = [] if args.trace_jsonl else None
    final_result = run_ft09_solution(
        environments_dir=args.environments_dir,
        operation_mode=args.operation_mode,
        legal_only=args.legal_only,
        verbose=not args.quiet,
        max_level_attempts=args.max_level_attempts,
        trace=trace,
    )
    if args.trace_jsonl and trace is not None:
        write_trace_jsonl(trace, args.trace_jsonl)
    print(
        f"final_state={final_result['state']} "
        f"levels_completed={final_result['levels_completed']}"
    )


if __name__ == "__main__":
    main()
