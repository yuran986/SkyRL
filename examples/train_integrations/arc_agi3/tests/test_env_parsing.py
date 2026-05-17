from types import SimpleNamespace

import pytest

from examples.train_integrations.arc_agi3.env import (
    ArcAgi3Env,
    FrameDiffStats,
    ParsedAction,
    _action_effect_summary,
    _diff_stats,
    _diff_summary,
    _format_diff_patch,
    parse_model_action,
)
from examples.train_integrations.arc_agi3.sft_warmup.oracle_distance_reward import OracleDistanceInfo


def test_parse_simple_action():
    parsed = parse_model_action("<action>ACTION1</action>")
    assert parsed.name == "ACTION1"
    assert parsed.x is None
    assert parsed.y is None


def test_parse_json_click_action():
    parsed = parse_model_action('<action>{"action":"ACTION6","x":32,"y":40}</action>')
    assert parsed.name == "ACTION6"
    assert parsed.x == 32
    assert parsed.y == 40


def test_parse_thinking_then_action():
    parsed = parse_model_action(
        '<think>The lower-left area changed last turn, so click nearby.</think>'
        '<action>{"action":"ACTION6","x":8,"y":56}</action>'
    )
    assert parsed.name == "ACTION6"
    assert parsed.x == 8
    assert parsed.y == 56


def test_parse_uses_last_action_block():
    parsed = parse_model_action("<action>ACTION1</action><think>revise</think><action>ACTION2</action>")
    assert parsed.name == "ACTION2"


def test_parse_text_click_action():
    parsed = parse_model_action("<action>ACTION6 x=3 y=4</action>")
    assert parsed.name == "ACTION6"
    assert parsed.x == 3
    assert parsed.y == 4


def test_click_requires_coordinates():
    with pytest.raises(ValueError, match="requires"):
        parse_model_action("<action>ACTION6</action>")


def test_action_effect_summary_marks_no_change():
    diff = FrameDiffStats(num_changes=0, bbox=None, color_changes=[], examples=[])

    assert _action_effect_summary(diff) == "last_action_effect=NO_CHANGE"


def test_action_effect_summary_marks_changed_area():
    diff = FrameDiffStats(num_changes=8, bbox=(1, 2, 3, 4), color_changes=[], examples=[])

    assert _action_effect_summary(diff) == "last_action_effect=CHANGED num_changes=8 bbox=(1,2)-(3,4)"


def test_diff_stats_splits_components_by_region_and_color_change():
    prev = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    cur = [
        [2, 2, 1, 3, 1],
        [2, 2, 1, 3, 1],
        [1, 1, 1, 1, 1],
        [2, 2, 1, 1, 1],
    ]

    diff = _diff_stats(prev, cur)

    assert diff is not None
    assert diff.num_changes == 8
    assert diff.bbox == (0, 0, 3, 3)
    assert [(item["size"], item["bbox"], item["before"], item["after"]) for item in diff.components] == [
        (4, (0, 0, 1, 1), 1, 2),
        (2, (0, 3, 1, 3), 1, 2),
        (2, (3, 0, 3, 1), 1, 3),
    ]


def test_diff_summary_includes_component_summary():
    diff = FrameDiffStats(
        num_changes=2,
        bbox=(0, 0, 1, 0),
        color_changes=[(1, 2, 2)],
        examples=[{"x": 0, "y": 0, "before": 1, "after": 2}],
        components=[
            {
                "id": 0,
                "size": 2,
                "bbox": (0, 0, 1, 0),
                "center": (0.5, 0.0),
                "before": 1,
                "after": 2,
                "examples": [],
            }
        ],
    )

    summary = _diff_summary(diff)

    assert "components=" in summary
    assert "'change': 'off-white->light gray'" in summary


def test_format_diff_patch_includes_before_after_and_delta():
    prev = [
        [1, 1, 1],
        [1, 1, 1],
    ]
    cur = [
        [1, 2, 1],
        [1, 2, 1],
    ]
    diff = _diff_stats(prev, cur)

    patch = _format_diff_patch(prev, cur, diff, radius=0)

    assert "changed_patch_before: x=1..1 y=0..1" in patch
    assert "changed_patch: x=1..1 y=0..1" in patch
    assert "changed_patch_delta: x=1..1 y=0..1" in patch
    assert "encoding=changed_after_hex_unchanged_dot" in patch
    assert "y00: 2" in patch


def _reward_env():
    env = ArcAgi3Env.__new__(ArcAgi3Env)
    env.levels_to_complete = 6
    env.level_reward = 3.0
    env.done_reward = 0.0
    env.meaningful_diff_reward = 0.005
    env.min_meaningful_diff_changes = 1
    env.max_meaningful_diff_changes = 512
    env.repeat_click_penalty = -0.02
    env.repeat_click_radius = 2
    env.oracle_distance_reward_enabled = False
    env.oracle_distance_reward = 0.05
    env.oracle_distance_valid_action_reward = 0.0
    env.oracle_distance_unrecoverable_penalty = -1.0
    env.oracle_distance_max_next_actions = 8
    env.last_score = 0.0
    env.last_levels_completed = 0
    env.last_diff_stats = None
    env.last_click = None
    env.last_oracle_distance = None
    return env


def test_reward_uses_minimal_components():
    env = _reward_env()
    observation = SimpleNamespace(levels_completed=0, score=0.0, done=True, state=None)
    diff = FrameDiffStats(num_changes=8, bbox=(1, 1, 2, 2), color_changes=[], examples=[])

    reward, components = env._compute_reward(observation, diff, ParsedAction(name="ACTION6", x=32, y=32))

    assert components == {
        "level_delta": 0.0,
        "done": 0.0,
        "meaningful_diff": pytest.approx(0.005),
        "repeat_click": 0.0,
    }
    assert reward == pytest.approx(0.005)


def test_reward_keeps_progress_dominant_over_diff_shaping():
    env = _reward_env()
    observation = SimpleNamespace(levels_completed=1, score=0.0, done=False, state=None)
    diff = FrameDiffStats(num_changes=8, bbox=(1, 1, 2, 2), color_changes=[], examples=[])

    reward, components = env._compute_reward(observation, diff, ParsedAction(name="ACTION6", x=10, y=10))

    assert components["level_delta"] == pytest.approx(3.0)
    assert components["meaningful_diff"] == pytest.approx(0.005)
    assert reward == pytest.approx(3.005)


def test_reward_penalizes_nearby_repeated_clicks():
    env = _reward_env()
    observation = SimpleNamespace(levels_completed=0, score=0.0, done=False, state=None)
    diff = FrameDiffStats(num_changes=8, bbox=(1, 1, 2, 2), color_changes=[], examples=[])

    env._compute_reward(observation, diff, ParsedAction(name="ACTION6", x=32, y=32))
    reward, components = env._compute_reward(observation, diff, ParsedAction(name="ACTION6", x=33, y=33))

    assert components["meaningful_diff"] == pytest.approx(0.005)
    assert components["repeat_click"] == pytest.approx(-0.02)
    assert reward == pytest.approx(-0.015)


def test_reward_adds_oracle_distance_progress_when_enabled():
    env = _reward_env()
    env.oracle_distance_reward_enabled = True
    env.oracle_distance_reward = 0.5
    observation = SimpleNamespace(levels_completed=0, score=0.0, done=False, state=None)
    diff = FrameDiffStats(num_changes=0, bbox=None, color_changes=[], examples=[])
    before = OracleDistanceInfo(True, 5, 0, "L0", [])
    after = OracleDistanceInfo(True, 2, 0, "L0", [])

    reward, components = env._compute_reward(
        observation,
        diff,
        ParsedAction(name="ACTION6", x=10, y=10),
        oracle_before=before,
        oracle_after=after,
    )

    assert components["oracle_progress"] == pytest.approx(1.5)
    assert reward == pytest.approx(1.5)


def test_reward_penalizes_oracle_unrecoverable_state_when_enabled():
    env = _reward_env()
    env.oracle_distance_reward_enabled = True
    env.oracle_distance_unrecoverable_penalty = -2.0
    observation = SimpleNamespace(levels_completed=0, score=0.0, done=False, state=None)
    diff = FrameDiffStats(num_changes=0, bbox=None, color_changes=[], examples=[])
    before = OracleDistanceInfo(True, 5, 0, "L0", [])
    after = OracleDistanceInfo(False, None, 0, "L0", [], error="no plan")

    reward, components = env._compute_reward(
        observation,
        diff,
        ParsedAction(name="ACTION6", x=10, y=10),
        oracle_before=before,
        oracle_after=after,
    )

    assert components["oracle_unrecoverable"] == pytest.approx(-2.0)
    assert reward == pytest.approx(-2.0)
