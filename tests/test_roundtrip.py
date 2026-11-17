"""Round trip on real images: gen_args from nai meta, passed to nai gen, must rebuild the recorded request exactly."""

import json

import pytest
from conftest import tagged, untagged

from nai_cli import cli, config, meta, rerun
from nai_cli.presets import MAX_PIXELS


def _rerun(raw: dict[str, str]) -> tuple[dict, dict]:
    record = meta.parse_raw(raw)
    gen_args = rerun.to_gen_args(record)
    if (record.get("width") or 0) * (record.get("height") or 0) > MAX_PIXELS:
        # Enhance results exceed the normal size limit, so only the prompts are compared
        gen_args = [a for a in gen_args if not a.startswith("--size=")]
    result, _ = cli.run(["gen", *gen_args, "--dry-run"])
    return record, result


def _untagged_comment(row: dict) -> dict:
    chars = row["chars"]
    char_uc = row["char_uc"] or [""] * len(chars)
    return {
        "prompt": row["base"],
        "uc": row["uc"] or "",
        "v4_prompt": {
            "caption": {
                "base_caption": row["base"],
                "char_captions": [{"char_caption": c, "centers": []} for c in chars],
            },
            "use_coords": False,
        },
        "v4_negative_prompt": {
            "caption": {
                "base_caption": row["uc"] or "",
                "char_captions": [{"char_caption": c, "centers": []} for c in char_uc],
            },
        },
    }


TAGGED = [r for r in tagged() if meta.model_from_source(r.get("source"))]


@pytest.mark.skipif(not TAGGED, reason="no samples in tests/private")
@pytest.mark.parametrize("row", TAGGED, ids=[r["file"] for r in TAGGED])
def test_tagged_roundtrip(row):
    cm = row["cm"]
    record, result = _rerun({"Comment": json.dumps(cm), "Source": row["source"]})
    params = result["body"]["parameters"]
    assert result["prompt"] == cm["v4_prompt"]["caption"]["base_caption"]
    assert result["negative"] == cm["v4_negative_prompt"]["caption"]["base_caption"]
    if not record["negative_verbatim"] and "tag_hint_uc_preset" in cm:
        assert params["tag_hint_uc_preset"] == cm["tag_hint_uc_preset"]
    if record["quality_from"] == "tag_hint" and record["quality"] != "none":
        assert params["tag_hint_qt"] == cm["tag_hint_qt"]


UNTAGGED = [r for r in untagged() if meta.model_from_source(r.get("source"))]


@pytest.mark.skipif(not UNTAGGED, reason="no samples in tests/private")
@pytest.mark.parametrize("row", UNTAGGED, ids=[r["file"][:40] for r in UNTAGGED])
def test_untagged_roundtrip(row):
    """Images without tag_hint: quality tags and UC presets are recognized from the text, the rest is sent verbatim."""
    _, result = _rerun({"Comment": json.dumps(_untagged_comment(row)), "Source": row["source"]})
    assert result["prompt"] == row["base"]
    assert result["negative"] == (row["uc"] or "")


TIDY_DEFAULTS = (
    "[defaults]\nformat_prompt = true\nformat_underscores = true\nspace_numeric_close = true\nofficial_nsfw = true\n"
)


@pytest.mark.skipif(not TAGGED, reason="no samples in tests/private")
@pytest.mark.parametrize("row", TAGGED[:8], ids=[r["file"] for r in TAGGED[:8]])
def test_roundtrip_ignores_tidy_defaults(row):
    """Switches written into gen_args override tidying and the official nsfw rule enabled in the config."""
    path = config.default_path()
    path.write_text(path.read_text(encoding="utf-8") + TIDY_DEFAULTS, encoding="utf-8")
    cm = row["cm"]
    _, result = _rerun({"Comment": json.dumps(cm), "Source": row["source"]})
    assert result["prompt"] == cm["v4_prompt"]["caption"]["base_caption"]
    assert result["negative"] == cm["v4_negative_prompt"]["caption"]["base_caption"]
