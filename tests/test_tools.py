import json

import pytest

from dnd_auto_dmg.tools import (
    add,
    divide,
    extract_json,
    fuzzy_match,
    multiply,
    roll_dice,
    subtract,
)


# --------- roll_dice --------- #
def test_roll_dice_fixed_roll_sums_correctly(monkeypatch):
    monkeypatch.setattr("dnd_auto_dmg.tools.random.randint", lambda a, b: 4)
    result = roll_dice.invoke({"num_dice": 3, "num_sides": 6, "modifier": 0})
    assert result == 3 * 4 + 0


def test_roll_dice_applies_modifier(monkeypatch):
    monkeypatch.setattr("dnd_auto_dmg.tools.random.randint", lambda a, b: 5)
    result = roll_dice.invoke({"num_dice": 2, "num_sides": 8, "modifier": 7})
    assert result == 2 * 5 + 7


def test_roll_dice_negative_modifier(monkeypatch):
    monkeypatch.setattr("dnd_auto_dmg.tools.random.randint", lambda a, b: 1)
    result = roll_dice.invoke({"num_dice": 1, "num_sides": 20, "modifier": -3})
    assert result == 1 * 1 - 3


# --------- add / subtract / multiply --------- #
def test_add():
    assert add.invoke({"a": 2, "b": 3}) == 5


def test_add_negative():
    assert add.invoke({"a": -5, "b": 2}) == -3


def test_subtract():
    assert subtract.invoke({"a": 10, "b": 4}) == 6


def test_multiply():
    assert multiply.invoke({"a": 6, "b": 7}) == 42


def test_multiply_by_zero():
    assert multiply.invoke({"a": 0, "b": 100}) == 0


# --------- divide --------- #
def test_divide_normal_case_floors():
    # floor(7 / 2) == 3
    assert divide.invoke({"a": 7, "b": 2}) == 3


def test_divide_clamps_small_positive_result_to_one():
    # floor(1 / 5) == 0 -> clamped to 1
    assert divide.invoke({"a": 1, "b": 5}) == 1


def test_divide_clamps_negative_result_to_one():
    # floor(-4 / 2) == -2 -> clamped to 1
    assert divide.invoke({"a": -4, "b": 2}) == 1


def test_divide_exact():
    assert divide.invoke({"a": 10, "b": 2}) == 5


# --------- extract_json --------- #
def test_extract_json_object_embedded_in_prose():
    text = 'Here is the result: {"total_damage": 12, "type": "fire"} — enjoy!'
    assert extract_json(text) == {"total_damage": 12, "type": "fire"}


def test_extract_json_nested_braces():
    text = 'prefix {"a": 1, "b": {"c": 2, "d": [1, 2, 3]}} suffix'
    result = extract_json(text)
    assert result == {"a": 1, "b": {"c": 2, "d": [1, 2, 3]}}


def test_extract_json_array_case():
    text = "values: [1, 2, 3] end"
    assert extract_json(text) == [1, 2, 3]


def test_extract_json_returns_none_when_absent():
    assert extract_json("no json anywhere here") is None


def test_extract_json_returns_none_on_malformed_json():
    text = '{"a": 1, "b": }'
    assert extract_json(text) is None


def test_extract_json_round_trips_dumped_json():
    payload = {"nested": {"list": [1, 2, {"x": "y"}]}}
    text = f"noise before {json.dumps(payload)} noise after"
    assert extract_json(text) == payload


# --------- fuzzy_match --------- #
def test_fuzzy_match_returns_best_match_above_threshold():
    name, score = fuzzy_match("fireball", ["fireball", "healing word"])
    assert name == "fireball"
    assert score > 40


def test_fuzzy_match_returns_none_tuple_below_threshold():
    result = fuzzy_match("xyz", ["fireball", "healing word"])
    assert result == (None, None)


def test_fuzzy_match_default_threshold_is_40():
    name, score = fuzzy_match("fire", ["fireball", "healing word"])
    assert name == "fireball"
    assert score > 40


def test_fuzzy_match_custom_threshold_rejects_weak_match():
    # "flame" scores ~44 against "fireball" (best of partial/token-set
    # ratio) -- above the default threshold (40) but below a stricter one.
    default_result = fuzzy_match("flame", ["fireball", "healing word"])
    assert default_result == ("fireball", pytest.approx(44.44444444444444))

    strict_result = fuzzy_match("flame", ["fireball", "healing word"], threshold=50)
    assert strict_result == (None, None)


def test_fuzzy_match_exact_match_scores_high():
    name, score = fuzzy_match("healing word", ["fireball", "healing word"])
    assert name == "healing word"
    assert score >= 95
