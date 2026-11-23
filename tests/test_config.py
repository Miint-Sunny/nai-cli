"""The config file: defaults, accounts' tokens hidden, values refused and the folders it lives in."""

import json
import sys
from pathlib import Path

import pytest
from helpers import run, set_filename, write_defaults

from nai_cli import config


def test_config_show_hides_tokens(capsys, with_token, monkeypatch):
    _, out = run(capsys, "config")
    assert out["novelai_token"] is True and out["token_source"] == "config" and "pst-x" not in json.dumps(out)
    monkeypatch.setenv(config.TOKEN_ENV, "pst-env")
    _, out = run(capsys, "config")
    assert out["token_source"] == "env" and "pst-env" not in json.dumps(out)


def test_v5_hidden_options_only_with_the_config_switch(capsys):
    argv = ["gen", "--prompt", "x", "--noise-schedule", "exponential", "--variety", "--dry-run"]
    code, out = run(capsys, *argv)
    assert code == 1 and out["code"] == "model_unsupported", out
    write_defaults("v5_hidden_options = true\n")
    code, out = run(capsys, *argv)
    p = out["body"]["parameters"]
    assert code == 0 and p["noise_schedule"] == "exponential" and p["skip_cfg_above_sigma"] > 0, out
    code, out = run(capsys, *argv, "--effort", "medium")
    assert code == 1 and out["code"] == "model_unsupported", out


def test_config_defaults_fill_missing_params(capsys):
    write_defaults('model = "v4.5"\nsteps = 20\ncfg = 6\nn = 2\nofficial_nsfw = true\nsize = "1024x1024"\n')
    code, out = run(capsys, "gen", "--prompt", "x", "--dry-run")
    p = out["body"]["parameters"]
    assert code == 0 and out["body"]["model"] == "nai-diffusion-4-5-full"
    assert p["steps"] == 20 and p["scale"] == 6.0 and p["n_samples"] == 2 and out["size"] == "1024x1024"
    assert out["negative"].startswith("nsfw, ")
    code, out = run(capsys, "gen", "--prompt", "x", "--steps", "23", "--n", "1", "--no-official-nsfw", "--dry-run")
    p = out["body"]["parameters"]
    assert p["steps"] == 23 and p["n_samples"] == 1 and not out["negative"].startswith("nsfw")
    code, out = run(capsys, "cost")
    assert code == 0 and out["model"] == "nai-diffusion-4-5-full" and out["size"] == "1024x1024"


def test_cli_beats_config_defaults(capsys):
    write_defaults("cfg = 6\nsteps = 20\n")
    code, out = run(capsys, "gen", "--prompt", "x", "--cfg", "5", "--dry-run")
    p = out["body"]["parameters"]
    assert code == 0 and p["scale"] == 5.0 and p["steps"] == 20


def test_a_table_nai_does_not_know_is_ignored(capsys):
    config.default_path().write_text('novelai_token = ""\n[other]\ndir = ""\n', encoding="utf-8")
    code, out = run(capsys, "gen", "--prompt", "x", "--dry-run")
    assert code == 0, out


def test_bad_config_defaults(capsys):
    write_defaults("stepz = 20\n")
    code, out = run(capsys, "gen", "--prompt", "x", "--dry-run")
    assert code == 1 and out["code"] == "config_missing"
    config.default_path().write_text('novelai_token = ""\n[defaults]\nofficial_nsfw = 1\n', encoding="utf-8")
    code, out = run(capsys, "gen", "--prompt", "x", "--dry-run")
    assert code == 1 and out["code"] == "config_missing"


@pytest.mark.parametrize(
    "text",
    [
        "output = 123\n",
        "[output]\ndir = 3\n",
        '[cost]\nmax_per_request = "abc"\n',
        "[cost]\nmax_per_request = true\n",
        "[cost]\nmax_per_request = -1\n",
        "novelai_token = 5\n",
    ],
)
def test_bad_config_values(capsys, text):
    config.default_path().write_text(text, encoding="utf-8")
    code, out = run(capsys, "config")
    assert code == 1 and out["code"] == "config_missing"


@pytest.mark.parametrize(
    "text", ["cfg = nan\n", "cfg = inf\n", "steps = 28.5\n", "steps = inf\n", "cfg_rescale = -inf\n"]
)
def test_config_defaults_must_be_finite_numbers(capsys, text):
    write_defaults(text)
    code, out = run(capsys, "config")
    assert code == 1 and out["code"] == "config_missing"


def test_config_shows_the_filename_template(capsys):
    _, out = run(capsys, "config")
    assert out["filename"] == ""
    set_filename("{prompt} s-{seed}")
    _, out = run(capsys, "config")
    assert out["filename"] == "{prompt} s-{seed}"


def test_no_home_folder_needs_the_folder_variables(capsys, monkeypatch):
    def no_home():
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "home", no_home)
    code, out = run(capsys, "config")
    assert code == 0, out
    # An empty variable is ignored as XDG says, which leaves only the home folder
    for name in ("APPDATA", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(name, "")
    code, out = run(capsys, "config")
    assert code == 1 and out["code"] == "config_missing" and "home folder" in out["message"]


def test_default_image_folder(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("XDG_DATA_HOME")
    monkeypatch.setattr(sys, "platform", "darwin")
    # Finder hides the data directory, so images go to Pictures on macOS
    assert config.default_output_dir() == home / "Pictures" / "nai-cli"
    # Images already in the data directory stay together
    legacy = home / ".local" / "share" / "nai-cli" / "images"
    legacy.mkdir(parents=True)
    assert config.default_output_dir() == legacy
    # A data directory chosen with XDG_DATA_HOME keeps the images too
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert config.default_output_dir() == tmp_path / "xdg" / "nai-cli" / "images"
    monkeypatch.delenv("XDG_DATA_HOME")
    legacy.rmdir()
    monkeypatch.setattr(sys, "platform", "linux")
    assert config.default_output_dir() == legacy
