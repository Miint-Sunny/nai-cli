import re
from datetime import datetime

import pytest

from nai_cli import naming

META = naming.Meta(
    prompt="1girl, smile, very long hair",
    seed=42,
    model="nai-diffusion-5-full",
    width=832,
    height=1216,
    index=0,
    id="abc",
    date=datetime(2026, 11, 17, 9, 5, 7),
    characters=(
        naming.Character("1girl, red hair"),
        naming.Character("cat ears, girl"),
        naming.Character("boy"),
        naming.Character("dog"),
    ),
)


def name(template: str, meta: naming.Meta = META) -> str:
    return naming.render(naming.parse(template), meta)


@pytest.mark.parametrize(
    "template, expected",
    [
        ("{prompt} s-{seed}", "1girl, smile, very long hair s-42"),
        ("{date} {time}", "2026-11-17 09-05-07"),
        ("{modelname} {model}", "V5 Full nai-diffusion-5-full"),
        ("{resolution} {width} {height}", "832x1216 832 1216"),
        ("{index|pad:3} {id}", "001 abc"),
        ("{{seed}", "{seed}"),
        # Names are trimmed, filter values are not
        ("{ seed | suf:_ }x", "42_ x"),
        ("{prompt|tags:2}", "1girl, smile"),
        ("{prompt|tags:-1}", "very long hair"),
        ("{prompt|words:2}", "1girl, smile,"),
        ("{prompt|first:5|upper}", "1GIRL"),
        ("{prompt|last:4}", "hair"),
        # Empty values stay empty, so separators around them disappear
        ("{type|pre:-}x", "x"),
        # Each character by name, else its first tag, or the second when the first only says who it is
        ("{characters}", "red hair+cat ears+boy"),
        ("{characters|max:4|join: & }", "red hair & cat ears & boy & dog"),
        ("{characters|at:2|full}", "cat ears, girl"),
        ("{characters|at:9}x", "x"),
        ("{characters|each:3|upper}", "RED+CAT+BOY"),
    ],
)
def test_template_like_official(template, expected):
    assert name(template) == expected


def test_named_characters_use_their_names():
    meta = naming.Meta(characters=(naming.Character("1girl, red hair", name="Alice"),))
    assert name("{characters}", meta) == "Alice"


def test_prompt_shortened_unless_a_length_filter_is_given():
    meta = naming.Meta(prompt="a" * 100, seed=1)
    assert name("{prompt}", meta) == "a" * 80
    assert name("{prompt|first:90}", meta) == "a" * 90
    # The whole name stops at 180 characters
    assert name("{prompt|first:300}{prompt|first:300}", meta) == "a" * 180


def test_unsafe_characters_replaced():
    meta = naming.Meta(prompt=' a/b:c?"d*e<f>g|h\\i%j\nk\tl ')
    assert name("{prompt}", meta) == "a_b_c__d_e_f_g_h_i_j_k_l"


def test_empty_name_falls_back_like_official():
    meta = naming.Meta(prompt="cat: tabby", seed=7)
    assert name("{type}", meta) == ""
    assert naming.fallback(meta) == "cat_ tabby s-7"


def test_unknown_model_named_by_its_id():
    assert name("{modelname}", naming.Meta(model="custom-model")) == "custom-model"
    assert name("{modelname}x", naming.Meta()) == "x"


@pytest.mark.parametrize(
    "template, message",
    [
        ("x{prompt", 'Unclosed "{" at position 2'),
        ("{nope}", 'Unknown token "{nope}"'),
        ("{prompt|nope}", 'Unknown filter "nope"'),
        ("{prompt|max:2}", 'Filter "max" only applies to {characters}'),
        ("{characters|first:2|max:2}", 'Filter "max" must come before text filters like "first" or "pre"'),
        ("{prompt|lower:1}", 'Filter "lower" takes no value'),
        ("{prompt|pre}", 'Filter "pre" needs a value, e.g. "pre:-"'),
        ("{prompt|first:0}", 'Filter "first" needs a positive number'),
        ("{prompt|first:x}", 'Filter "first" needs a positive number'),
        ("{prompt|tags:0}", 'Filter "tags" needs a non-zero number'),
    ],
)
def test_template_errors_match_official(template, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        naming.parse(template)


def test_date_and_settings_tokens():
    meta = naming.Meta(seed=1, date=datetime(2026, 1, 2, 3, 4, 5), steps=28, cfg=5.0, sampler="k_euler")
    assert name("{year}-{month}-{day} {steps} {cfg} {sampler}", meta) == "2026-01-02 28 5 k_euler"
    assert name("{cfg}", naming.Meta(cfg=5.5)) == "5.5" and name("{steps|pre:s}", naming.Meta()) == ""


def folders(template: str, meta: naming.Meta = META) -> list[str]:
    return naming.render_folders(naming.parse_folders(template), meta)


def test_folder_rules_split_only_on_written_separators():
    assert folders("{date}") == ["2026-11-17"]
    assert folders(r"{year}/{month}\{modelname}") == ["2026", "11", "V5 Full"]
    # A separator in a value is not a folder
    slashed = naming.Meta(prompt="a/b, c")
    assert folders("{prompt}", slashed) == ["a_b, c"]
    # Empty folders and folders of dots only are left out, so nothing leaves the output folder
    assert folders("x/{id}/y", naming.Meta()) == ["x", "y"]
    assert folders("{prompt|first:2}/z", naming.Meta(prompt="..")) == ["z"]
    assert folders("") == []


@pytest.mark.parametrize("template", ["/abs", r"\abs", "C:/x", "{date}/../x", "./x", "a/ .. /b"])
def test_folder_rules_that_leave_the_output_folder_are_refused(template):
    with pytest.raises(ValueError):
        naming.parse_folders(template)


def test_destination_files_by_the_rule(tmp_path):
    rule = naming.parse_folders("{year}/{model}")
    destination = naming.Destination(tmp_path, rule)
    path = destination.image_path(META, seed=42, index=1)
    assert path == tmp_path / "2026" / "nai-diffusion-5-full" / "090507_42_1.png"
    assert destination.fallback_path(META, seed=42, index=1) == tmp_path / "090507_42_1.png"
    assert naming.Destination(tmp_path).image_path(META, 42, name="n") == tmp_path / "n.png"
