"""--from-image, enhance and variations bring settings in the way the official import assigns them."""

import io
import json
import sys

import pytest
from helpers import run, write_defaults
from PIL import Image, PngImagePlugin
from samples import nai_image

from nai_cli import config

V5 = "NovelAI Diffusion V5 0ADF9AB7"
V45 = "NovelAI Diffusion V4.5 4BDE2A90"
V4 = "NovelAI Diffusion V4 37442FCA"


def _image(path, source=V5, size=(832, 1216), **comment):
    base = {
        "prompt": "1girl, smile, very aesthetic, masterpiece, no text",
        "uc": "lowres",
        "seed": 5,
        "width": size[0],
        "height": size[1],
        "steps": 23,
        "scale": 6.5,
        "sampler": "k_dpmpp_2m",
        "noise_schedule": "exponential",
        "n_samples": 1,
        "tag_hint_qt": 1,
        "tag_hint_uc_preset": 0,
    }
    info = PngImagePlugin.PngInfo()
    info.add_text("Source", source)
    info.add_text("Comment", json.dumps({**base, **comment}))
    Image.new("RGB", size, "gray").save(path, pnginfo=info)
    return path


def test_default_parts_are_the_local_dialogs(capsys, tmp_path):
    img = _image(tmp_path / "pic.png", size=(1216, 832))
    code, out = run(capsys, "gen", "--from-image", str(img), "--dry-run")
    p = out["body"]["parameters"]
    # Prompt, negative prompt and characters only: the settings and the seed stay as configured
    assert code == 0 and out["prompt"].startswith("1girl, smile")
    assert (p["width"], p["height"], p["steps"], p["sampler"]) == (832, 1216, 28, "k_euler_ancestral")
    code, out = run(capsys, "gen", "--from-image", str(img), "--import", "settings", "--dry-run")
    p = out["body"]["parameters"]
    assert (p["width"], p["height"], p["steps"], p["scale"], p["sampler"]) == (1216, 832, 23, 6.5, "k_dpmpp_2m")


def test_clean_brackets_reaches_every_prompt(capsys, tmp_path):
    img = tmp_path / "pic.png"
    nai_image(img, prompt="{1girl},[smile]", uc="[bad],{worse}", chars=[("{boy},x", "[ugly]")])
    code, out = run(capsys, "gen", "--from-image", str(img), "--clean-brackets", "--neg-verbatim", "--dry-run")
    assert code == 0 and out["prompt"].startswith("1girl, smile")
    assert out["negative"] == "bad, worse"
    assert out["characters"] == [{"prompt": "boy, x", "negative": "ugly"}]


def test_imported_text_never_reads_a_file(capsys, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("SECRET-FILE-CONTENTS", encoding="utf-8")
    img = tmp_path / "pic.png"
    nai_image(img, prompt=f"[@{secret.as_posix()}]", chars=[(f"@{secret.as_posix()}", "")])
    code, out = run(capsys, "gen", "--from-image", str(img), "--clean-brackets", "--dry-run")
    assert code == 0 and out["prompt"].startswith(f"@{secret.as_posix()}")
    assert "SECRET-FILE-CONTENTS" not in json.dumps(out)


def test_vibes_alone_follow_the_image_model(capsys, tmp_path):
    img = _image(
        tmp_path / "pic.png",
        source=V45,
        reference_image_multiple=["E1"],
        reference_strength_multiple=[0.5],
        reference_information_extracted_multiple=[1],
    )
    code, out = run(capsys, "gen", "--from-image", str(img), "--import", "vibes", "--prompt", "x", "--dry-run")
    assert code == 0 and out["model"] == "nai-diffusion-4-5-full"
    assert out["body"]["parameters"]["reference_image_multiple"] == ["<2 base64 characters>"]


def test_settings_bring_the_quality_and_uc_the_preset(capsys, tmp_path):
    img = _image(tmp_path / "pic.png", prompt="1girl", tag_hint_qt=0, tag_hint_uc_preset=1)
    code, out = run(capsys, "gen", "--from-image", str(img), "--import", "settings", "--prompt", "x", "--dry-run")
    # No quality tags in the image, so none are added; the UC preset belongs to the negative prompt's part
    assert code == 0 and out["prompt"] == "x" and out["negative"].startswith("lowres, artistic error")
    code, out = run(capsys, "gen", "--from-image", str(img), "--import", "uc", "--prompt", "x", "--dry-run")
    assert out["prompt"] == "x, very aesthetic, masterpiece, no text"


def test_another_model_keeps_what_it_can(capsys, tmp_path):
    img = _image(tmp_path / "pic.png", source=V45, skip_cfg_above_sigma=58, steps=40)
    code, out = run(capsys, "gen", "--from-image", str(img), "--import", "all", "--model", "v5", "--dry-run")
    # Refused before, as the image's model was forced onto the request; now cut to what V5 has
    assert code == 0 and out["model"] == "nai-diffusion-5-full"
    assert any("Variety+" in w for w in out["warnings"]) and out["body"]["parameters"]["steps"] == 40


def test_switches_turn_off_what_the_image_has_on(capsys, tmp_path):
    img = _image(tmp_path / "pic.png", source=V45, skip_cfg_above_sigma=58)
    code, out = run(capsys, "gen", "--from-image", str(img), "--import", "settings", "--prompt", "x", "--dry-run")
    assert code == 0 and out["body"]["parameters"]["skip_cfg_above_sigma"] is not None
    code, out = run(
        capsys, "gen", "--from-image", str(img), "--import", "settings", "--prompt", "x", "--no-variety", "--dry-run"
    )
    assert code == 0 and out["body"]["parameters"].get("skip_cfg_above_sigma") is None


def test_warnings_only_for_parts_brought_in(capsys, tmp_path):
    img = _image(tmp_path / "pic.png", size=(2048, 2048), director_reference_strengths=[1.0])
    code, out = run(capsys, "gen", "--from-image", str(img), "--dry-run")
    assert code == 0 and out["warnings"] == []
    code, out = run(capsys, "gen", "--from-image", str(img), "--import", "settings", "--dry-run")
    assert any("pixel limit" in w for w in out["warnings"]) and any("precise" in w for w in out["warnings"])
    code, out = run(capsys, "enhance", "--image", str(img), "--dry-run")
    assert not any("pixel limit" in w for w in out.get("warnings", []))


def _drawn_image(path):
    return _image(
        path,
        prompt="||cat|dog||, smile",
        v4_prompt={"caption": {"base_caption": "||cat|dog||, smile", "char_captions": []}, "use_coords": False},
        actual_prompts={
            "prompt": {"base_caption": "dog, smile", "char_captions": []},
            "negative_prompt": {"base_caption": "lowres", "char_captions": []},
        },
    )


def test_actual_prompts_bring_the_draws(capsys, tmp_path):
    img = _drawn_image(tmp_path / "pic.png")
    code, out = run(capsys, "gen", "--from-image", str(img), "--dry-run")
    assert code == 0 and out["prompt"].startswith("||cat|dog||, smile")
    code, out = run(capsys, "gen", "--from-image", str(img), "--actual-prompts", "--dry-run")
    assert out["prompt"].startswith("dog, smile")


def test_enhance_and_variations_send_the_draws_again(capsys, tmp_path):
    img = _drawn_image(tmp_path / "pic.png")
    for command in ("enhance", "variations"):
        code, out = run(capsys, command, "--image", str(img), "--dry-run")
        assert code == 0 and out["prompt"].startswith("dog, smile"), command
        # Typed groups the image drew take its draws too, as the official client fills them
        code, out = run(capsys, command, "--image", str(img), "--prompt", "||cat|dog||, frown", "--dry-run")
        assert out["prompt"].startswith("dog, frown"), command


def test_standard_input_with_an_imported_image(capsys, tmp_path, monkeypatch):
    img = tmp_path / "pic.png"
    nai_image(img)
    monkeypatch.setattr("sys.stdin", _Stdin("piped cat\n"))
    code, out = run(capsys, "gen", "--from-image", str(img), "--prompt", "@-", "--dry-run")
    assert code == 0 and out["prompt"].startswith("piped cat")


class _Stdin:
    def __init__(self, text):
        self.buffer = _Buffer(text.encode())


class _Buffer:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data


def test_fixed_tags_never_join_an_imported_prompt(capsys, tmp_path):
    path = tmp_path / "config" / "nai-cli" / "config.toml"
    path.write_text(path.read_text(encoding="utf-8") + '[defaults]\nprompt_prefix = "artist:x"\n', encoding="utf-8")
    img = tmp_path / "pic.png"
    nai_image(img, prompt="artist:x, cat")
    code, out = run(capsys, "gen", "--from-image", str(img), "--dry-run")
    assert code == 0 and out["prompt"].startswith("artist:x, cat") and "artist:x, artist:x" not in out["prompt"]
    code, out = run(capsys, "gen", "--from-image", str(img), "--prompt", "dog", "--dry-run")
    assert out["prompt"].startswith("artist:x, dog")


def test_actual_prompts_keep_the_switches_beside_them(capsys, tmp_path):
    img = _image(
        tmp_path / "pic.png",
        source=V4,
        prompt="||cat|dog||",
        v4_prompt={"caption": {"base_caption": "||cat|dog||", "char_captions": []}, "use_coords": True},
        v4_negative_prompt={"caption": {"base_caption": "lowres", "char_captions": []}, "legacy_uc": True},
        actual_prompts={
            "prompt": {"base_caption": "dog", "char_captions": []},
            "negative_prompt": {"base_caption": "x"},
        },
    )
    code, out = run(capsys, "meta", str(img))
    assert code == 0 and out["actual_prompts"]["prompt"]["base_caption"] == "dog"
    code, out = run(capsys, "gen", "--from-image", str(img), "--import", "all", "--actual-prompts", "--dry-run")
    assert code == 0 and out["prompt"].startswith("dog") and out["body"]["parameters"]["v4_prompt"]["use_coords"]


def test_importing_keeps_the_current_legacy_uc_and_alpha_and_a_rework_keeps_the_images(capsys, tmp_path):
    old = _image(
        tmp_path / "v4.png",
        source=V4,
        legacy_uc=True,
        v4_negative_prompt={"caption": {"base_caption": "lowres", "char_captions": []}, "legacy_uc": True},
    )
    # The import dialog keeps the editor's legacy UC and the account's alpha setting
    code, out = run(capsys, "gen", "--from-image", str(old), "--import", "settings", "--dry-run")
    assert code == 0 and out["body"]["parameters"]["legacy_uc"] is False, out
    code, out = run(capsys, "variations", "--image", str(old), "--dry-run")
    assert code == 0 and out["body"]["parameters"]["legacy_uc"] is True
    clear = _image(tmp_path / "v5.png", straight_alpha=False, tag_hint_transparent_background=True)
    code, out = run(capsys, "gen", "--from-image", str(clear), "--import", "settings", "--dry-run")
    assert code == 0 and out["body"]["parameters"]["straight_alpha"] is True, out
    code, out = run(capsys, "variations", "--image", str(clear), "--dry-run")
    assert code == 0 and out["body"]["parameters"]["straight_alpha"] is False
    # The settings a rerun needs stay in gen_args
    code, out = run(capsys, "meta", str(old))
    assert "--legacy-uc" in out["gen_args"]


def test_gen_args_hold_only_what_the_images_model_takes(capsys, tmp_path):
    img = _image(tmp_path / "pic.png", sampler="k_dpm_fast", legacy_uc=True)
    code, out = run(capsys, "meta", str(img))
    assert not any(a.startswith(("--sampler", "--legacy-uc")) for a in out["gen_args"]), out["gen_args"]
    code, again = run(capsys, "gen", *out["gen_args"], "--dry-run")
    assert code == 0, again


@pytest.mark.parametrize(("value", "sent"), [(0, 0), (0.25, 0.25), (None, 0.7)])
def test_an_imported_zero_is_a_value_like_any_other(capsys, tmp_path, value, sent):
    path = config.default_path()
    path.write_text(path.read_text(encoding="utf-8") + "[defaults]\ncfg_rescale = 0.7\n", encoding="utf-8")
    img = _image(tmp_path / "a.png", V45, **({} if value is None else {"cfg_rescale": value}))
    commands = [
        ["gen", "--from-image", str(img), "--import", "settings"],
        ["enhance", "--image", str(img)],
        ["variations", "--image", str(img)],
    ]
    for argv in commands:
        code, out = run(capsys, *argv, "--dry-run")
        assert code == 0 and out["body"]["parameters"]["cfg_rescale"] == sent, (argv, out)
    code, out = run(capsys, "meta", str(img))
    code, out = run(capsys, "gen", *out["gen_args"], "--dry-run")
    # gen_args and the import agree, an unset value left to the config by both
    assert code == 0 and out["body"]["parameters"]["cfg_rescale"] == sent


@pytest.mark.parametrize("text", ["a\x00b", "a\ud800b"])
def test_text_no_request_can_carry_is_refused_however_it_comes(capsys, tmp_path, monkeypatch, text):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_bytes(text.encode("utf-8", "surrogatepass"))
    code, out = run(capsys, "gen", "--prompt", f"@{prompt_file}", "--dry-run")
    assert (code, out["code"]) == (1, "bad_args"), out
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(text.encode("utf-8", "surrogatepass"))))
    code, out = run(capsys, "gen", "--prompt", "@-", "--dry-run")
    assert (code, out["code"]) == (1, "bad_args"), out
    img = _image(tmp_path / "a.png", prompt=text)
    code, out = run(capsys, "gen", "--from-image", str(img), "--dry-run")
    assert (code, out["code"]) == (1, "bad_args"), out
    # The settings are still shown as they are, but no command is given that could not run
    code, out = run(capsys, "meta", str(img))
    assert code == 0 and out["gen_args"] is None and any("gen_args is null" in w for w in out["warnings"]), out


@pytest.mark.parametrize(
    ("key", "bad", "flag"),
    [
        ("steps", 51, "--steps"),
        ("scale", -1, "--cfg"),
        ("cfg_rescale", 2, "--cfg-rescale"),
        ("seed", 10_000_000_000, "--seed"),
    ],
)
def test_out_of_range_image_numbers_do_not_break_replay(capsys, tmp_path, key, bad, flag):
    path = _image(tmp_path / "made.png", V45, **{key: bad})
    code, out = run(capsys, "meta", str(path), "--raw")
    assert code == 0 and out[key] == out["raw"][key] == bad
    code, replay = run(capsys, "gen", *out["gen_args"], "--dry-run")
    assert code == 0, replay
    assert not any(arg.startswith(f"{flag}=") for arg in out["gen_args"])
    assert any(key in warning for warning in out["warnings"])


def test_the_default_brings_vibes_only_for_the_model_that_made_them(capsys, tmp_path):
    img = _image(
        tmp_path / "pic.png",
        source=V45,
        reference_image_multiple=["E1"],
        reference_strength_multiple=[0.5],
        reference_information_extracted_multiple=[1],
    )
    code, out = run(capsys, "gen", "--from-image", str(img), "--model", "v4.5", "--dry-run")
    assert code == 0 and out["body"]["parameters"]["reference_image_multiple"] == ["<2 base64 characters>"]
    code, out = run(capsys, "gen", "--from-image", str(img), "--dry-run")
    assert code == 0 and out["model"] != "nai-diffusion-4-5-full"
    assert "reference_image_multiple" not in out["body"]["parameters"]


@pytest.mark.parametrize("mode", ["meta", "settings", "enhance", "variations"])
def test_hidden_variety_survives_v5_image_import(capsys, tmp_path, mode):
    cfg = config.default_path()
    cfg.write_text(cfg.read_text("utf-8") + "[defaults]\nv5_hidden_options = true\n", encoding="utf-8")
    img = _image(tmp_path / "legacy.png", size=(512, 512), model_name="custom", skip_cfg_above_sigma=58)
    if mode == "meta":
        code, out = run(capsys, "meta", str(img))
        assert code == 0 and "--variety" in out["gen_args"]
        args = ["gen", *out["gen_args"]]
    elif mode == "settings":
        args = ["gen", "--from-image", str(img), "--import", "all"]
    else:
        args = [mode, "--image", str(img)]
    code, out = run(capsys, *args, "--dry-run")
    assert code == 0 and out["model"] == "nai-diffusion-5-curated", out
    params = out["body"]["parameters"]
    assert params["noise_schedule"] == "exponential" and params.get("skip_cfg_above_sigma")


@pytest.mark.parametrize("hidden", [False, True])
@pytest.mark.parametrize("schedule", ["native", "exponential"])
def test_legacy_v5_schedule_agrees_in_import_and_rerun(capsys, tmp_path, hidden, schedule):
    cfg = config.default_path()
    cfg.write_text(
        cfg.read_text("utf-8") + f"[defaults]\nv5_hidden_options = {str(hidden).lower()}\n", encoding="utf-8"
    )
    pic = _image(tmp_path / "legacy.png", model_name="custom", noise_schedule=schedule)
    code, record = run(capsys, "meta", str(pic))
    assert code == 0 and record["model"] == "nai-diffusion-5-curated"
    for args in (record["gen_args"], ["--from-image", str(pic), "--import", "all"]):
        code, out = run(capsys, "gen", *args, "--dry-run")
        assert code == 0 and out["body"]["parameters"]["noise_schedule"] == (
            schedule if hidden and schedule != "native" else "karras"
        )
        assert any("karras" in warning for warning in out["warnings"]) == (not hidden and schedule != "native")


def test_i2i_sizes_from_image(capsys, tmp_path):
    img = tmp_path / "src.png"
    Image.new("RGB", (1000, 1500)).save(img)
    _, out = run(capsys, "i2i", "--image", str(img), "--prompt", "x", "--dry-run")
    assert out["size"] == "896x1344" and out["body"]["parameters"]["image"].startswith("<")
    assert any("1000x1500" in w and "896x1344" in w for w in out["warnings"])


def test_options_given_override_the_image(capsys, tmp_path):
    img = tmp_path / "pic.png"
    prompt = "fur dataset, 1girl, smile, very aesthetic, masterpiece, no text"
    nai_image(img, prompt=prompt, uc="ugly face", chars=[("girl", "bad"), ("boy", "worse")])
    code, out = run(capsys, "variations", "--image", str(img), "--dry-run")
    assert code == 0 and out["prompt"].startswith("fur dataset, 1girl, smile") and "ugly face" in out["negative"]
    assert [c["negative"] for c in out["characters"]] == ["bad", "worse"]
    # An empty value given on purpose clears the image's
    code, out = run(capsys, "variations", "--image", str(img), "--prompt", "", "--neg", "", "--dry-run")
    assert out["prompt"].startswith("fur dataset, very aesthetic") and "ugly face" not in out["negative"]
    # New characters replace the image's along with their negatives
    code, out = run(capsys, "variations", "--image", str(img), "--char", "cat", "--dry-run")
    assert code == 0 and out["characters"] == [{"prompt": "cat", "negative": ""}]
    code, out = run(capsys, "enhance", "--image", str(img), "--no-furry", "--dry-run")
    assert code == 0 and out["prompt"].startswith("1girl, smile")


def test_image_settings_never_read_local_files(capsys, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("SECRET-FILE-CONTENTS", encoding="utf-8")
    img = tmp_path / "shared.png"
    nai_image(img, prompt=f"@{secret.as_posix()}, very aesthetic, masterpiece, no text", chars=[("@_@", "@x.txt")])
    for cmd in ("variations", "enhance"):
        code, out = run(capsys, cmd, "--image", str(img), "--dry-run")
        assert code == 0 and out["prompt"].startswith(f"@{secret.as_posix()}, very aesthetic")
        assert "SECRET-FILE-CONTENTS" not in json.dumps(out)
        assert out["characters"] == [{"prompt": "@_@", "negative": "@x.txt"}]
    # gen_args spells a literal @ as @@, so a rerun sends the same text
    code, record = run(capsys, "meta", str(img))
    assert f"--prompt=@@{secret.as_posix()}" in record["gen_args"]
    code, out = run(capsys, "gen", *record["gen_args"], "--dry-run")
    assert (
        code == 0
        and out["prompt"].startswith(f"@{secret.as_posix()}, ")
        and "SECRET-FILE-CONTENTS" not in out["prompt"]
    )


def test_image_texts_kept_while_new_ones_are_tidied(capsys, tmp_path):
    write_defaults("format_prompt = true\n")
    img = tmp_path / "dirty.png"
    nai_image(img, prompt="old_tag,  extra, very aesthetic, masterpiece, no text", uc="bad,  hands")
    code, out = run(capsys, "variations", "--image", str(img), "--prompt", "1girl\uff0c smile", "--dry-run")
    assert code == 0 and out["prompt"].startswith("1girl, smile, ") and out["negative"].endswith("bad,  hands")


def test_switches_given_tidy_the_image_texts(capsys, tmp_path):
    write_defaults("format_prompt = true\n")
    img = tmp_path / "dirty.png"
    nai_image(img, prompt="1girl\uff0c 1.2::year 2025::, very aesthetic, masterpiece, no text")
    code, out = run(capsys, "variations", "--image", str(img), "--dry-run")
    assert out["prompt"].startswith("1girl\uff0c 1.2::year 2025::")
    argv = ["--format-prompt", "--space-numeric-close", "--dry-run"]
    code, out = run(capsys, "variations", "--image", str(img), *argv)
    assert code == 0 and out["prompt"].startswith("1girl, 1.2::year 2025 ::")


def test_from_image_imports_the_official_parts(capsys, tmp_path):
    img = tmp_path / "pic.png"
    nai_image(img, seed=77, prompt="{1girl}, [smile], very aesthetic, masterpiece, no text", chars=[("boy", "bad")])
    code, out = run(capsys, "gen", "--from-image", str(img), "--dry-run")
    p = out["body"]["parameters"]
    # The seed stays out unless asked for, as in the import dialog
    assert code == 0 and out["seed"] != 77 and p["width"] == 832 and p["steps"] == 28
    assert out["prompt"].startswith("{1girl}, [smile]") and [c["prompt"] for c in out["characters"]] == ["boy"]
    code, out = run(capsys, "gen", "--from-image", str(img), "--import", "all", "--clean-brackets", "--dry-run")
    assert out["seed"] == 77 and out["prompt"].startswith("1girl, smile")
    # Options given win, and characters given replace the image's
    code, out = run(capsys, "gen", "--from-image", str(img), "--steps", "20", "--char", "cat", "--dry-run")
    assert out["body"]["parameters"]["steps"] == 20 and [c["prompt"] for c in out["characters"]] == ["cat"]
    code, out = run(capsys, "gen", "--from-image", str(img), "--import", "settings", "--prompt", "dog", "--dry-run")
    assert out["prompt"].startswith("dog") and out["characters"] == []
    code, out = run(capsys, "gen", "--from-image", str(img), "--import", "nope", "--dry-run")
    assert code == 1 and out["code"] == "bad_args"


def test_from_image_keeps_the_inpainting_size(capsys, tmp_path):
    img, other = tmp_path / "a.png", tmp_path / "b.png"
    nai_image(img, w=512, h=512)
    nai_image(other, w=1216, h=832)
    mask = tmp_path / "m.png"
    Image.new("L", (512, 512), 255).save(mask)
    code, out = run(
        capsys, "inpaint", "--image", str(img), "--mask", str(mask), "--from-image", str(other), "--dry-run"
    )
    assert code == 0 and out["size"] == "512x512"
