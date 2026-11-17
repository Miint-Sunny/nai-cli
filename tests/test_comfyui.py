"""ComfyUI multi-character prompts: the three ways of writing them, and the switch that splits them."""

import pytest
from helpers import run

from nai_cli import comfyui, config


def _characters(found):
    return [(c.prompt, c.center) for c in found.characters]


def test_couple_regions_become_centres():
    found = comfyui.split("2girls, best FILL() COUPLE(0 0.5) 1girl, red hair COUPLE(0.5 1, 0 0.5) 1girl, blue hair")
    assert found.base == "2girls, best"
    assert _characters(found) == [("1girl, red hair", (0.25, 0.5)), ("1girl, blue hair", (0.75, 0.25))]


def test_couple_with_masks_and_index_masks():
    found = comfyui.split("park COUPLE MASK(0 0.5, 0 1) 1boy, smile COUPLE IMASK(1) 1girl")
    assert _characters(found) == [("1boy, smile", (0.25, 0.5)), ("1girl", None)]


def test_and_mask_has_no_base_and_drops_weights():
    found = comfyui.split("cat MASK(0 0.5, 0 1) AND dog AREA(0.5 1.4, 0 1):1.2")
    assert found.base == "" and _characters(found) == [("cat", (0.25, 0.5)), ("dog", (0.95, 0.5))]


def test_pipe_lines_give_characters_without_places():
    found = comfyui.split("2girls, masterpiece\n| girl, black hair\n  | girl, white hair")
    assert found.base == "2girls, masterpiece"
    assert _characters(found) == [("girl, black hair", None), ("girl, white hair", None)]


@pytest.mark.parametrize(
    "text",
    ["plain, prompt", "a|b", "||cat|dog||, park", "(red | blue) hair", 'sign, "A COUPLE AND MASK(0 1)"', "x \\| y"],
)
def test_other_prompts_are_left_alone(text):
    assert comfyui.split(text) is None


def test_the_model_decides_how_many_characters_fit(capsys):
    seven = "base" + "\n| girl" * 7
    assert len(comfyui.split(seven).characters) == 7
    code, out = run(capsys, "gen", "--model", "v4.5", "--comfyui-characters", "--prompt", seven, "--dry-run")
    assert (code, out["code"]) == (1, "model_unsupported")
    code, out = run(capsys, "gen", "--model", "v5", "--comfyui-characters", "--prompt", seven, "--dry-run")
    assert code == 0, out


@pytest.mark.parametrize(
    ("text", "base", "characters"),
    [
        ("COUPLE(0 0.5) 1girl COUPLE(0.5 1) 1boy", "", [("1girl", (0.25, 0.5)), ("1boy", (0.75, 0.5))]),
        ("2girls COUPLE (1girl:1.2), blue hair", "2girls", [("(1girl:1.2), blue hair", None)]),
        (
            'sign "HAPPY COUPLE" COUPLE(0 1) 1girl, "BLACK AND WHITE"',
            'sign "HAPPY COUPLE"',
            [('1girl, "BLACK AND WHITE"', (0.5, 0.5))],
        ),
    ],
)
def test_couple_at_the_start_brackets_and_quotes(text, base, characters):
    found = comfyui.split(text)
    assert found.base == base and _characters(found) == characters


def test_apostrophes_are_not_quotes():
    found = comfyui.split("cat's ears MASK(0 0.5, 0 1) AND dog's tail AREA(0.5 1, 0 1)")
    assert _characters(found) == [("cat's ears", (0.25, 0.5)), ("dog's tail", (0.75, 0.5))]
    assert comfyui.split("'a AND b MASK(0 1)'") is None
    curly = comfyui.split("‘90s style, COUPLE(0 0.5) 1girl, girls’ uniform COUPLE(0.5 1) 1boy")
    assert [c[0] for c in _characters(curly)] == ["1girl, girls’ uniform", "1boy"]


def _gen(capsys, *argv):
    return run(capsys, "gen", "--model", "v4.5", "--dry-run", "--show-body", *argv)


def test_switch_splits_the_prompt_into_placed_characters(capsys):
    code, out = _gen(capsys, "--comfyui-characters", "--prompt", "park COUPLE(0 0.5) 1girl COUPLE 1boy")
    assert code == 0, out
    v4 = out["body"]["parameters"]["v4_prompt"]
    assert v4["use_coords"] is True
    assert [c["char_caption"] for c in v4["caption"]["char_captions"]] == ["girl", "boy"]
    assert v4["caption"]["base_caption"].startswith("park")
    centers = [c["centers"][0] for c in v4["caption"]["char_captions"]]
    # The second character has no region and takes the first free place
    assert centers[0] == {"x": 0.3, "y": 0.5} and centers[1] != centers[0]


def test_switch_is_off_by_default_and_from_the_config(capsys, isolated_dirs):
    text = "park\n| 1girl\n| 1boy"
    _, out = _gen(capsys, "--prompt", text)
    assert out["body"]["parameters"]["v4_prompt"]["caption"]["char_captions"] == []
    config.default_path().write_text('novelai_token = ""\n[defaults]\ncomfyui_characters = true\n', encoding="utf-8")
    _, out = _gen(capsys, "--prompt", text)
    v4 = out["body"]["parameters"]["v4_prompt"]
    assert len(v4["caption"]["char_captions"]) == 2 and v4["use_coords"] is False
    _, out = _gen(capsys, "--no-comfyui-characters", "--prompt", text)
    assert out["body"]["parameters"]["v4_prompt"]["caption"]["char_captions"] == []


def test_switch_refuses_characters_given_twice(capsys):
    code, out = _gen(capsys, "--comfyui-characters", "--prompt", "a\n| b", "--char", "c")
    assert code == 1 and out["code"] == "bad_args"
    code, out = _gen(capsys, "--comfyui-characters", "--prompt", "plain", "--char", "c")
    assert code == 0


@pytest.mark.parametrize(
    ("text", "base", "characters"),
    [
        # A | on the first line, with or without spaces before it, leaves no base prompt
        ("| 1girl\n| 1boy", "", [("1girl", None), ("1boy", None)]),
        ("  | 1girl\n  | 1boy", "", [("1girl", None), ("1boy", None)]),
        # Apostrophes in two tags do not hide the COUPLE between them
        (
            "'90s style COUPLE(0 0.5) 1girl, girls' uniform COUPLE(0.5 1) 1boy",
            "'90s style",
            [("1girl, girls' uniform", (0.25, 0.5)), ("1boy", (0.75, 0.5))],
        ),
        ("girl's hat, 'quoted text' COUPLE (0 0.5) 1girl", "girl's hat, 'quoted text'", [("1girl", (0.25, 0.5))]),
        ("base,COUPLE(0 0.5) 1girl,COUPLE(0.5 1) 1boy", "base", [("1girl", (0.25, 0.5)), ("1boy", (0.75, 0.5))]),
        # A region that is not a number places nothing, and is kept as written
        ("x COUPLE (nan 1.2) 1girl", "x", [("(nan 1.2) 1girl", None)]),
        ("x COUPLE MASK(0 inf, 0 1) 1girl", "x", [("1girl", None)]),
    ],
)
def test_edges_of_each_way_of_writing(text, base, characters):
    found = comfyui.split(text)
    assert found is not None and found.base == base and _characters(found) == characters


def test_a_first_line_character_reaches_the_request(capsys):
    code, out = run(capsys, "gen", "--comfyui-characters", "--prompt", "| 1girl\n| 1boy", "--dry-run")
    assert code == 0 and [c["prompt"] for c in out["characters"]] == ["girl", "boy"], out
