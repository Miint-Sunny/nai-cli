"""nai reword: copies whose settings have words replaced, in the PNG text and the stealth data alike, and whose
names have them replaced too."""

import json
from pathlib import Path

import pytest
from helpers import run, run_batch
from PIL import Image, PngImagePlugin
from samples import layered_png, nai_image, settings, with_comment_field

from nai_cli import config, errors, meta, words

PROMPT = "1girl, Lolita, loli"
# A rule that matches part of a text, whatever its case, hits Lolita as well
# The rule hits the prompt in Comment and the Description alike
EVERY = "1girl, 1011ta, 1011"
HITS = {"loli": 4}
CHARACTERS = {
    "v4_prompt": {
        "caption": {
            "base_caption": PROMPT,
            "char_captions": [{"char_caption": "girl, loli", "centers": [{"x": 0.5, "y": 0.5}]}],
        }
    }
}


def _source(path: Path, prompt: str = PROMPT, **fields) -> Path:
    """A PNG with NovelAI's settings in its text and in its alpha channel."""
    text = with_comment_field(settings(prompt=prompt), **fields)
    return layered_png(path, text, text)


def _layers(copy: str) -> tuple[dict, dict]:
    """What a copy's PNG text holds, and what its alpha channel does."""
    data = Path(copy).read_bytes()
    with Image.open(copy) as img:
        img.load()
        return meta.png_text_chunks(data), meta.stealth_text(img)


def test_a_rule_matches_part_of_a_text_whatever_its_case(capsys, tmp_path):
    code, out = run(capsys, "reword", str(_source(tmp_path / "a.png")), "-w", "loli=1011", "-o", str(tmp_path / "out"))
    assert code == 0, out
    [(entry)] = out["files"]
    assert entry["replaced"] == {"loli": 4} and entry["file"] == str(tmp_path / "out" / "a.png")
    assert out["skipped"] == [] and out["warnings"] == []
    text, hidden = _layers(entry["file"])
    assert json.loads(text["Comment"])["prompt"] == EVERY and text["Description"] == EVERY == hidden["Description"]


def test_a_rule_between_slashes_is_a_regular_expression(capsys, tmp_path):
    source = _source(tmp_path / "a.png")
    code, out = run(capsys, "reword", str(source), "-w", r"/\bloli\b/=1011", "-o", str(tmp_path / "out"))
    assert code == 0 and out["files"][0]["replaced"] == {r"/\bloli\b/": 2}, out
    (text, _) = _layers(out["files"][0]["file"])
    assert json.loads(text["Comment"])["prompt"] == "1girl, Lolita, 1011"


def test_a_regular_expression_ends_at_its_first_slash_equals_so_that_it_may_hold_an_equals_sign(capsys, tmp_path):
    # (?= looks at what follows, which leaves the loli of Lolita alone
    rule = "/loli(?=,|$)/=1011"
    folder = config.config_dir() / "words"
    folder.mkdir(parents=True)
    (folder / "lookahead.txt").write_text(f"{rule}\n", encoding="utf-8")
    source = _source(tmp_path / "a.png")
    for i, given in enumerate((rule, "lookahead")):
        code, out = run(capsys, "reword", str(source), "-w", given, "-o", str(tmp_path / f"out{i}"))
        assert code == 0 and [f["replaced"] for f in out["files"]] == [{"/loli(?=,|$)/": 2}], out
        text, hidden = _layers(out["files"][0]["file"])
        assert json.loads(text["Comment"])["prompt"] == "1girl, Lolita, 1011" == hidden["Description"]


def test_a_word_list_is_a_file_of_rules_and_a_list_of_the_user_wins_over_the_built_in(capsys, tmp_path):
    folder = config.config_dir() / "words"
    folder.mkdir(parents=True)
    (folder / "mine.txt").write_text("# a comment\n\nloli = 1011\ncat = dog\n", encoding="utf-8")
    (folder / "discord.txt").write_text("loli=2000\n", encoding="utf-8")
    source = _source(tmp_path / "a.png")
    code, out = run(capsys, "reword", str(source), "-w", "mine", "-o", str(tmp_path / "out"))
    assert code == 0 and out["files"][0]["replaced"] == {"loli": 4}, out
    (text, _) = _layers(out["files"][0]["file"])
    assert json.loads(text["Comment"])["prompt"] == "1girl, 1011ta, 1011"
    code, out = run(capsys, "reword", str(source), "-w", "discord", "-o", str(tmp_path / "out"))
    (text, _) = _layers(out["files"][0]["file"])
    assert code == 0 and json.loads(text["Comment"])["prompt"] == "1girl, 2000ta, 2000", out


def test_the_built_in_list_and_a_missing_one(capsys, tmp_path):
    source = _source(tmp_path / "a.png")
    code, out = run(capsys, "reword", str(source), "-w", "discord", "-o", str(tmp_path / "out"))
    assert code == 0 and out["files"][0]["replaced"] == {"loli": 4}, out
    code, out = run(capsys, "reword", str(source), "-w", "no such list", "-o", str(tmp_path / "out"))
    assert (code, out["code"]) == (1, "bad_args") and "no such list" in out["message"] and "words" in out["message"]


def test_rules_merge_in_the_order_given(capsys, tmp_path):
    source = _source(tmp_path / "a.png")
    code, out = run(capsys, "reword", str(source), "-w", "loli=one", "-w", "loli=two", "-o", str(tmp_path / "out"))
    (text, _) = _layers(out["files"][0]["file"])
    # A later rule for the same text replaces the earlier one
    assert code == 0 and json.loads(text["Comment"])["prompt"] == "1girl, twota, two", out
    code, out = run(capsys, "reword", str(source), "-w", "girl=cat", "-w", "cat=dog", "-o", str(tmp_path / "out"))
    (text, _) = _layers(out["files"][0]["file"])
    # Rules apply in turn, each to what the ones before left
    assert code == 0 and json.loads(text["Comment"])["prompt"] == "1dog, Lolita, loli", out
    assert out["files"][0]["replaced"] == {"girl": 2, "cat": 2}


def test_the_prompts_inside_comment_are_reworded_too(capsys, tmp_path):
    source = _source(tmp_path / "a.png", **CHARACTERS)
    code, out = run(capsys, "reword", str(source), "-w", "loli=1011", "-o", str(tmp_path / "out"))
    (text, _) = _layers(out["files"][0]["file"])
    caption = json.loads(text["Comment"])["v4_prompt"]["caption"]
    assert code == 0 and caption["char_captions"] == [{"char_caption": "girl, 1011", "centers": [{"x": 0.5, "y": 0.5}]}]
    assert caption["base_caption"] == EVERY


def test_the_signature_goes_and_both_layers_read_back_the_same(capsys, tmp_path):
    source = _source(tmp_path / "a.png", **CHARACTERS)
    code, out = run(capsys, "reword", str(source), "-w", "loli=1011", "-o", str(tmp_path / "out"))
    text, hidden = _layers(out["files"][0]["file"])
    assert code == 0 and "signed_hash" not in json.loads(text["Comment"]), out
    assert list(text) == ["Title", "Description", "Software", "Source", "Generation time", "Comment"]
    assert hidden == {key: value for key, value in text.items() if key != "Title"}
    assert meta.parse(out["files"][0]["file"])["seed"] == 5


def test_the_name_of_a_copy_is_reworded_too(capsys, tmp_path):
    source = _source(tmp_path / "in" / "1girl, loli s-1234.png")
    out = tmp_path / "out"
    code, made = run(capsys, "reword", str(source), "-w", "loli=1011", "-o", str(out))
    assert code == 0 and Path(made["files"][0]["file"]).name == "1girl, 1011 s-1234.png", made
    # The source is told of by its own name, and the warning about NovelAI's download names holds here too
    assert made["files"][0]["source"] == str(source)
    assert any("--rename" in warning for warning in made["warnings"])
    code, filed = run(capsys, "reword", str(source), "-w", "loli=1011")
    assert code == 0 and Path(filed["files"][0]["file"]).name == "1girl, 1011 s-1234_reworded.png", filed


def test_copies_that_the_rules_name_alike_hold_their_own_sources_settings(capsys, tmp_path):
    # The rules make the names of two images alike: the second copy is numbered, as any name taken makes the next
    first = _source(tmp_path / "in" / "x solo.png", prompt="solo test A", seed=11)
    second = _source(tmp_path / "in" / "x duo.png", prompt="solo test B", seed=22)
    out = tmp_path / "out"
    code, made = run(capsys, "reword", str(tmp_path / "in"), "-w", "solo=duo", "-o", str(out))
    assert code == 0 and [(Path(f["file"]).name, f["source"]) for f in made["files"]] == [
        ("x duo.png", str(second)),
        ("x duo-1.png", str(first)),
    ], made
    for name, prompt, seed in (("x duo.png", "duo test B", 22), ("x duo-1.png", "duo test A", 11)):
        # In both layers a copy reads as its own source, reworded
        text, hidden = _layers(out / name)
        assert json.loads(text["Comment"])["prompt"] == prompt == text["Description"] == hidden["Description"]
        assert json.loads(hidden["Comment"])["seed"] == seed == json.loads(text["Comment"])["seed"]


def test_a_rule_that_leaves_no_file_name_is_refused(capsys, tmp_path):
    source = _source(tmp_path / "in" / "loli.png")
    code, out = run(capsys, "reword", str(source), "-w", "loli=/", "-o", str(tmp_path / "out"))
    assert (code, out["code"]) == (1, "bad_args") and "loli.png" in out["message"], out
    assert not (tmp_path / "out").exists()


def test_images_without_settings_or_without_a_match_are_skipped(capsys, tmp_path):
    plain = tmp_path / "plain.png"
    Image.new("RGB", (8, 8)).save(plain)
    source = _source(tmp_path / "a.png")
    out = tmp_path / "out"
    code, made = run(capsys, "reword", str(plain), "-w", "loli=1011", "-o", str(out))
    assert code == 0 and made == {
        "ok": True,
        "files": [],
        "skipped": [{"source": str(plain), "reason": "no_settings"}],
        "warnings": [],
    }
    code, made = run(capsys, "reword", str(source), "-w", "zzz=1", "-o", str(out))
    assert code == 0 and made["skipped"] == [{"source": str(source), "reason": "no_match"}] and made["files"] == []
    assert not out.exists()
    code, made = run(capsys, "reword", str(plain), str(source), "-w", "loli=1011", "-o", str(out))
    assert code == 0 and [(f["source"], f["replaced"]) for f in made["files"]] == [(str(source), {"loli": 4})], made
    assert made["skipped"] == [{"source": str(plain), "reason": "no_settings"}]


def test_a_source_without_an_alpha_channel_gets_one(capsys, tmp_path):
    source = tmp_path / "a.png"
    nai_image(source, w=64, h=64, prompt="1girl, loli")
    code, out = run(capsys, "reword", str(source), "-w", "loli=1011", "-o", str(tmp_path / "out"))
    assert code == 0, out
    text, hidden = _layers(out["files"][0]["file"])
    with Image.open(out["files"][0]["file"]) as copy:
        # Opaque: the lowest bits the stealth data takes are no transparency
        assert copy.mode == "RGBA" and copy.getchannel("A").getextrema()[0] >= 254
    assert json.loads(text["Comment"])["prompt"] == "1girl, 1011"
    assert hidden == {"Source": text["Source"], "Comment": text["Comment"]}


def test_an_image_too_small_to_hold_its_settings_is_refused(capsys, tmp_path):
    source = tmp_path / "tiny.png"
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", json.dumps({"prompt": "loli"}))
    Image.new("RGB", (8, 8)).save(source, pnginfo=info)
    code, out = run(capsys, "reword", str(source), "-w", "loli=1011", "-o", str(tmp_path / "out"))
    assert (code, out["code"]) == (1, "bad_args") and "tiny.png" in out["message"], out
    assert not (tmp_path / "out").exists()


def test_a_dry_run_lists_the_copies_and_writes_none(capsys, tmp_path):
    source = _source(tmp_path / "a.png")
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(
        json.dumps(["reword", str(source), "-w", "loli=1011", "-o", str(tmp_path / "out")]), encoding="utf-8"
    )
    code, lines = run_batch(capsys, str(jobs), "--dry-run")
    assert code == 0 and lines[0]["dry_run"] is True and lines[0]["files"][0]["file"] == str(tmp_path / "out" / "a.png")
    assert not (tmp_path / "out").exists()


def test_rules_are_read_from_a_list_file_and_bad_ones_are_refused():
    assert words.parse_list("# a comment\n\n  loli = 1011 \ncat=dog\nno rule here\n") == {"loli": "1011", "cat": "dog"}
    with pytest.raises(errors.NaiError) as refused:
        words.compile({"/[/": "x"})
    assert refused.value.code == errors.BAD_ARGS and "[" in refused.value.message
    with pytest.raises(errors.NaiError):
        words.compile({"": "x"})


def test_rewording_opens_comment_drops_the_signature_and_counts():
    (rule,) = words.compile({"loli": "1011"})
    reworded, found = words.reword(with_comment_field(settings(prompt=PROMPT), **CHARACTERS), (rule,))
    comment = json.loads(reworded["Comment"])
    assert found == {"loli": 7} and "signed_hash" not in comment
    assert reworded["Description"] == EVERY and comment["v4_prompt"]["caption"]["base_caption"] == EVERY
    assert comment["seed"] == 5 and reworded["Title"] == "NovelAI generated image"
    # A text with a backslash in it is not read as a regular expression's replacement
    (rule,) = words.compile({"loli": r"\1"})
    assert words.substitute("Loli", (rule,))[0] == r"\1"
