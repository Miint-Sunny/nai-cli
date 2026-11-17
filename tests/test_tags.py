from dataclasses import replace
from types import SimpleNamespace

import pytest
from helpers import run, write_defaults

from nai_cli import commands, config, errors, models, tags
from nai_cli.commands import tags as command


def test_japanese_by_ratio_like_official():
    assert tags.is_japanese("青い")
    assert tags.is_japanese("青いhair")
    assert not tags.is_japanese("青い hair")
    assert not tags.is_japanese("blue h")
    # Han characters count too, so Chinese is looked up as Japanese
    assert tags.is_japanese("蓝色")
    # Emoji are two code units in JavaScript: 1 of 5, not 1 of 3
    assert not tags.is_japanese("青\U0001f600\U0001f600")


def test_query_cleaned_like_official():
    assert tags.lookup("  {blue hair}  ", models.resolve("v5"), False).text == "{blue hair}"
    assert tags.lookup("{]blue/]", models.resolve("v5"), False).text == "blue"
    with pytest.raises(errors.NaiError) as e:
        tags.lookup(" [] ", models.resolve("v5"), False)
    assert e.value.code == errors.BAD_ARGS


@pytest.mark.parametrize(
    ("name", "furry", "model", "lexicon"),
    [
        ("v5", False, "nai-diffusion-5-full", "animev5"),
        ("v5-curated", True, "nai-diffusion-5-curated", "furryv5"),
        ("v5-medium", False, "nai-diffusion-5-full-medium", "animev5"),
        ("v4.5", False, "nai-diffusion-4-5-full", None),
        ("v4.5-curated", True, "nai-diffusion-furry-3", None),
    ],
)
def test_lexicon_by_model(name, furry, model, lexicon):
    item = tags.lookup("blue h", models.resolve(name), furry)
    assert (item.model, item.lexicon, item.lang) == (model, lexicon, None)


def test_suggest_pauses_between_lookups(monkeypatch):
    class Client:
        def __init__(self):
            self.sent = []

        def suggest_tags(self, *args):
            self.sent.append(args)
            return [{"tag": "blue eyes", "jp": "青い目", "count": 1}] if args[3] else [{"tag": "x", "count": 1}]

    client, sleeps = Client(), []
    v5 = models.resolve("v5")
    monkeypatch.setattr(command.time, "sleep", sleeps.append)
    results = command.run(command.TagsSettings(v5, False, ("blue h", "青い")), SimpleNamespace(client=lambda: client))[
        "results"
    ]
    assert sleeps == [tags.PAUSE]
    assert client.sent == [
        ("blue h", "nai-diffusion-5-full", "animev5", None),
        ("青い", "nai-diffusion-5-full", "animev5", "jp"),
    ]
    assert [(r["query"], r["lang"]) for r in results] == [("blue h", "en"), ("青い", "jp")]


def test_furry_needs_furry_mode():
    medium = models.resolve("v5-medium")
    # V5 Medium takes every V5 Full feature in the official client, furry mode included
    assert tags.lookup("blue h", medium, True).lexicon == "furryv5"
    with pytest.raises(errors.NaiError) as e:
        tags.lookup("blue h", replace(medium, furry_mode=False), True)
    assert e.value.code == errors.MODEL_UNSUPPORTED


def test_tags_without_token_or_query(capsys, monkeypatch):
    code, out = run(capsys, "tags", "blue h")
    assert code == 1 and out["code"] == "config_missing"
    code, out = run(capsys, "tags")
    assert out["code"] == "bad_args"
    code, out = run(capsys, "tags", "blue h", "[]")
    assert out["code"] == "bad_args"


def test_tags_looks_up_each_query(capsys, with_token, monkeypatch):
    sleeps = []
    monkeypatch.setattr(commands.tags.time, "sleep", sleeps.append)
    code, out = run(capsys, "tags", "blue h", "red e", "--model", "v4.5", "--furry")
    assert code == 0 and out["model"] == "nai-diffusion-4-5-full" and out["furry"] is True
    assert [r["tags"][0]["tag"] for r in out["results"]] == [
        "blue h nai-diffusion-furry-3 None",
        "red e nai-diffusion-furry-3 None",
    ]
    assert len(sleeps) == 1


def test_models_lists_every_model(capsys, isolated_dirs):
    code, out = run(capsys, "models")
    assert code == 0 and out["default"] == models.DEFAULT_MODEL
    assert [m["id"] for m in out["models"]] == list(models.MODELS)
    listed = {alias for m in out["models"] for alias in m["aliases"]}
    assert listed == set(models.ALIASES)
    medium = next(m for m in out["models"] if m["id"] == "nai-diffusion-5-full-medium")
    assert medium["fixed"] == {"steps": 14, "sampler": "k_euler_ancestral", "uc_preset": "heavy"}
    assert (medium["furry"], medium["text_limit"], medium["tokenizer"]) == (True, None, "qwen")
    v45 = next(m for m in out["models"] if m["id"] == "nai-diffusion-4-5-full")
    assert (v45["furry"], v45["text_limit"], v45["tokenizer"]) == (True, 118, "t5")
    cfg = config.default_path()
    cfg.write_text(cfg.read_text(encoding="utf-8") + '[defaults]\nmodel = "v4.5-curated"\n', encoding="utf-8")
    _, out = run(capsys, "models")
    assert out["default"] == "nai-diffusion-4-5-curated"


def test_tags_furry_on_medium(capsys, with_token):
    code, out = run(capsys, "tags", "wolf", "--model", "v5-medium", "--furry")
    assert code == 0 and out["furry"] is True


def test_models_reports_configured_v5_features(capsys):
    write_defaults("v5_hidden_options = true\n")
    code, out = run(capsys, "models")
    assert code == 0
    for row in out["models"]:
        if row["family"] == "v5":
            assert row["variety"] == row["noise_schedule"] == (row["effort"] != "medium")
