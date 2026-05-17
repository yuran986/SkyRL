from types import SimpleNamespace

import pytest

from examples.train_integrations.arc_agi3.sft_warmup.oracle_distance_reward import (
    inspect_oracle_distance,
    oracle_progress_delta,
)


def _sprite(name: str, x: int, y: int):
    return SimpleNamespace(name=name, x=x, y=y)


def _game(level_index: int = 0, level_name: str = "L0"):
    return SimpleNamespace(level_index=level_index, current_level=SimpleNamespace(name=level_name))


def test_inspect_oracle_distance_summarizes_plan_without_executing_actions():
    calls = []
    sprite_a = _sprite("A", 3, 4)
    sprite_b = _sprite("B", 10, 11)

    def solver(game):
        calls.append(game)
        return [(sprite_a, 2), (sprite_b, 1)]

    game = _game(level_index=2, level_name="abc")
    info = inspect_oracle_distance(game, solver=solver)

    assert calls == [game]
    assert info.solvable is True
    assert info.plan_len == 3
    assert info.level_index == 2
    assert info.level_name == "abc"
    assert info.next_actions == [
        {
            "action": "ACTION6",
            "x": 8,
            "y": 10,
            "count": 2,
            "sprite_name": "A",
            "grid": {"x": 3, "y": 4},
        },
        {
            "action": "ACTION6",
            "x": 22,
            "y": 24,
            "count": 1,
            "sprite_name": "B",
            "grid": {"x": 10, "y": 11},
        },
    ]


def test_inspect_oracle_distance_reports_unsolvable_solver_errors():
    def solver(_game):
        raise RuntimeError("no legal plan")

    info = inspect_oracle_distance(_game(), solver=solver)

    assert info.solvable is False
    assert info.plan_len is None
    assert info.next_actions == []
    assert info.error == "no legal plan"


def test_oracle_progress_delta_rewards_shorter_same_level_plan():
    before = inspect_oracle_distance(_game(), solver=lambda _game: [(_sprite("A", 0, 0), 4)])
    after = inspect_oracle_distance(_game(), solver=lambda _game: [(_sprite("A", 0, 0), 1)])

    assert oracle_progress_delta(before, after, level_delta=0, success=False) == pytest.approx(3.0)


def test_oracle_progress_delta_allows_negative_same_level_progress():
    before = inspect_oracle_distance(_game(), solver=lambda _game: [(_sprite("A", 0, 0), 1)])
    after = inspect_oracle_distance(_game(), solver=lambda _game: [(_sprite("A", 0, 0), 4)])

    assert oracle_progress_delta(before, after, level_delta=0, success=False) == pytest.approx(-3.0)


def test_oracle_progress_delta_does_not_compare_across_level_advance():
    before = inspect_oracle_distance(_game(0), solver=lambda _game: [(_sprite("A", 0, 0), 2)])
    after = inspect_oracle_distance(_game(1), solver=lambda _game: [(_sprite("B", 1, 1), 10)])

    assert oracle_progress_delta(before, after, level_delta=1, success=False) == pytest.approx(2.0)
