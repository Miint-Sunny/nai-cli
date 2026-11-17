"""Random prompts from the official dice button's recipe, and the character negative V4 fills in."""

import random
import re

import pytest
from helpers import run

from nai_cli import random_prompt, rerun
from nai_cli.random_prompt import Recipe


def _draws(recipe, n=400, seed=7):
    rng = random.Random(seed)
    return [random_prompt.generate(recipe, rng) for _ in range(n)]


def _needing_flags():
    """Tags that only entries needing flags hold."""
    lists = random_prompt.word_lists()

    def tags(needing):
        return {t for entries in lists.values() for e in entries if bool(e[3]) is needing for t in e[0].split(", ")}

    return tags(True) - tags(False)


def _tags(drawn):
    return {t for text in (drawn.prompt, *drawn.characters) for t in text.split(", ")}


def test_the_errata_match_the_official_records():
    lists, corrected = random_prompt.word_lists(), random_prompt.corrected_lists()
    changed = sum(a != b for name in lists for a, b in zip(lists[name], corrected[name], strict=True))
    assert changed == 5 and lists["legwear"][0] == ["socks", 12, [], ["feel"], []]


@pytest.mark.parametrize("recipe", list(Recipe))
def test_counts_match_the_characters(recipe):
    for drawn in _draws(recipe, 200):
        head = drawn.prompt.split(", ")
        if "no humans" in head:
            assert drawn.characters == ()
            continue
        counts = {"girl": 0, "boy": 0, "other": 0}
        for tag in head:
            if m := re.fullmatch(r"(\d)(girl|boy|other)s?", tag):
                counts[m[2]] = int(m[1])
        words = [c.split(", ")[0] for c in drawn.characters]
        # Girls come first among the characters, as the official client lists them
        assert words == ["girl"] * counts["girl"] + ["boy"] * counts["boy"] + ["other"] * counts["other"]
        assert len(head) == len(set(head)) and "" not in head


def test_the_official_recipe_never_draws_entries_that_need_flags():
    needing = _needing_flags()
    assert not set().union(*map(_tags, _draws(Recipe.OFFICIAL))) & needing


def test_the_corrected_recipe_sets_the_flags_entries_declare():
    tags = set().union(*map(_tags, _draws(Recipe.CORRECTED)))
    assert tags & _needing_flags()
    footwear = {e[0] for e in random_prompt.word_lists()["footwear"]}
    assert any(t.split(" ")[-1] in footwear or t in footwear for t in tags)


def test_a_legless_body_wears_nothing_on_its_legs():
    lists = random_prompt.corrected_lists()
    legless = {e[0] for e in lists["animal"] if "nolegs" in e[2]}
    leg_clothes = {e[0] for g in ("bottom", "footwear") for e in lists[g]}
    for drawn in _draws(Recipe.CORRECTED, 3000, seed=3):
        for c in drawn.characters:
            tags = c.split(", ")
            if legless & set(tags):
                assert not any(t in leg_clothes or t.split(" ", 1)[-1] in leg_clothes for t in tags)


def test_gen_sends_random_characters_as_the_official_client_does(capsys):
    random.seed(4)
    seen = 0
    for _ in range(20):
        code, out = run(capsys, "gen", "--random-prompt", "--dry-run", "--show-body")
        assert code == 0, out
        params = out["body"]["parameters"]
        assert params["v4_prompt"]["use_coords"] is False
        for c in params["characterPrompts"]:
            seen += 1
            assert c["uc"] == "lowres, aliasing, " and c["center"] == {"x": 0, "y": 0}
    assert seen


def test_random_prompt_replaces_typed_text(capsys):
    code, out = run(capsys, "gen", "--random-prompt", "--prompt", "cat", "--dry-run")
    assert code == 1 and out["code"] == "bad_args"
    code, out = run(capsys, "gen", "--random-prompt=official", "--char-pos", "0.5,0.5", "--dry-run")
    assert code == 1 and out["code"] == "bad_args"


def test_v4_fills_a_new_characters_negative(capsys):
    code, out = run(capsys, "gen", "--prompt", "x", "--model", "v4", "--char", "girl", "--dry-run")
    assert code == 0 and out["characters"] == [{"prompt": "girl", "negative": "lowres, aliasing, "}]
    code, out = run(capsys, "gen", "--prompt", "x", "--model", "v4", "--char", "girl", "--char-neg", "", "--dry-run")
    assert out["characters"] == [{"prompt": "girl", "negative": ""}]
    _, out = run(capsys, "gen", "--prompt", "x", "--model", "v4.5", "--char", "girl", "--dry-run")
    assert out["characters"] == [{"prompt": "girl", "negative": ""}]


def test_reruns_keep_negatives_that_differ_from_the_models():
    base = {"prompt": "x", "negative": "", "model": "nai-diffusion-4-full"}
    empty = dict(base, characters=[{"prompt": "a", "negative": "", "x": None, "y": None}])
    assert "--char-neg=" in rerun.to_gen_args(empty)
    filled = dict(base, characters=[{"prompt": "a", "negative": "lowres, aliasing, ", "x": None, "y": None}])
    assert not any(a.startswith("--char-neg") for a in rerun.to_gen_args(filled))


def test_v5_medium_draws_without_a_warning_and_char_pos_gets_its_own_message(capsys):
    for seed in range(8):
        code, out = run(capsys, "gen", "--model", "v5-medium", "--random-prompt", "--seed", str(seed), "--dry-run")
        assert code == 0 and not any("negative" in w for w in out["warnings"]), out["warnings"]
    code, out = run(capsys, "gen", "--random-prompt", "--char-pos", "0.5,0.5", "--dry-run")
    assert code == 1 and "places its characters itself" in out["message"]
