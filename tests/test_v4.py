"""V4 Full and V4 Curated, which the official client still offers."""

import pytest
from helpers import run

from nai_cli import models, presets


def _dry(capsys, *argv):
    code, out = run(capsys, "gen", "--prompt", "x", "--dry-run", "--show-body", *argv)
    assert code == 0, out
    return out


@pytest.mark.parametrize(
    ("alias", "model_id", "quality"),
    [
        ("v4", "nai-diffusion-4-full", "no text, best quality, very aesthetic, absurdres"),
        ("v4-curated", "nai-diffusion-4-curated-preview", "rating:general, best quality, very aesthetic, absurdres"),
    ],
)
def test_v4_requests_follow_the_official_presets(capsys, alias, model_id, quality):
    out = _dry(capsys, "--model", alias, "--steps", "23", "--cfg", "5.5")
    assert out["body"]["model"] == model_id and out["prompt"].endswith(quality)
    assert out["negative"].startswith(presets.UC_PRESETS[model_id]["heavy"])


def test_v4_has_no_precise_references_or_human_preset(capsys, tmp_path):
    from PIL import Image

    ref = tmp_path / "r.png"
    Image.new("RGB", (64, 64)).save(ref)
    code, out = run(capsys, "gen", "--prompt", "x", "--model", "v4", "--ref", str(ref), "--dry-run")
    assert code == 1 and out["code"] == "model_unsupported"
    code, out = run(capsys, "gen", "--prompt", "x", "--model", "v4", "--uc-preset", "human", "--dry-run")
    assert code == 1 and out["code"] == "bad_args"


def test_v4_variety_uses_its_own_sigma(capsys):
    v4 = _dry(capsys, "--model", "v4", "--variety", "--size", "832x1216")["body"]["parameters"]
    v45 = _dry(capsys, "--model", "v4.5", "--variety", "--size", "832x1216")["body"]["parameters"]
    assert v45["skip_cfg_above_sigma"] / v4["skip_cfg_above_sigma"] == pytest.approx(58 / 19)


def test_legacy_uc_is_a_v4_switch(capsys):
    body = _dry(capsys, "--model", "v4-curated", "--legacy-uc")["body"]
    assert body["parameters"]["legacy_uc"] is True and body["parameters"]["v4_negative_prompt"]["legacy_uc"] is True
    assert _dry(capsys, "--model", "v4")["body"]["parameters"]["legacy_uc"] is False
    code, out = run(capsys, "gen", "--prompt", "x", "--legacy-uc", "--dry-run")
    assert code == 1 and out["code"] == "model_unsupported" and out["feature"] == "legacy_uc"


def test_v4_inpainting_models_lead_back():
    assert models.resolve("nai-diffusion-4-curated-inpainting").id == "nai-diffusion-4-curated-preview"
    assert models.for_action(models.resolve("v4-curated"), "infill").id == "nai-diffusion-4-curated-preview"
    # V5 Curated borrows V4.5 Curated's inpainting model, which still leads to V4.5 Curated
    assert models.resolve("nai-diffusion-4-5-curated-inpainting").id == "nai-diffusion-4-5-curated"
    assert models.for_action(models.resolve("v5-curated"), "infill").id == "nai-diffusion-4-5-curated"
