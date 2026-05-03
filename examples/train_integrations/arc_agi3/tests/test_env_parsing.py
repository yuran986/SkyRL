import pytest

from examples.train_integrations.arc_agi3.env import parse_model_action


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
