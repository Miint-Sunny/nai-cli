"""The command line itself: JSON on every path, arguments, standard input and help."""

import io
import json
from datetime import UTC, datetime

import httpx
import pytest
from helpers import run
from PIL import Image
from samples import with_comment

from nai_cli import api, cli, commands, errors, ledger, vibe


def test_dry_run_without_token(capsys):
    code, out = run(capsys, "gen", "--prompt", "1girl", "--seed", "3", "--dry-run")
    assert code == 0 and out["dry_run"] and out["seed"] == 3
    assert out["cost"]["estimate"] == 0 and out["cost"]["assumed_opus"] is True
    assert out["body"]["model"] == "nai-diffusion-5-full"
    assert out["tokens"]["limit"] == 1471 and out["tokens"]["exact"] is False


def test_bad_args_are_json(capsys):
    code, out = run(capsys, "gen", "--size", "100x100", "--dry-run")
    assert code == 1 and out["code"] == "bad_args"
    code, out = run(capsys, "gen", "--nope")
    assert code == 1 and out["code"] == "bad_args"
    code, out = run(capsys, "gen", "--char-neg", "x", "--dry-run")
    assert out["code"] == "bad_args"


def test_an_unknown_option_shows_the_usage_of_its_command(capsys):
    _, out = run(capsys, "gen", "--nope")
    assert out["message"] == "unrecognized arguments: --nope" and out["usage"].startswith("usage: nai gen ")
    _, out = run(capsys, "meta", "a.png", "--nope", "x")
    assert out["message"] == "unrecognized arguments: --nope x" and out["usage"].startswith("usage: nai meta ")


def test_a_command_name_is_never_abbreviated(capsys):
    code, out = run(capsys, "tag", "x.png")
    assert code == 1 and out["code"] == "bad_args"


def test_text_from_a_missing_file_refused(capsys, tmp_path):
    code, out = run(capsys, "gen", "--prompt", f"@{tmp_path / 'nope.txt'}", "--dry-run")
    assert code == 1 and out["code"] == "bad_args"
    code, out = run(capsys, "gen", "--prompt", "@_@, 1girl", "--dry-run")
    assert code == 0 and out["prompt"].startswith("@_@, 1girl, ")
    text = tmp_path / "p.txt"
    text.write_text("cat\n", encoding="utf-8")
    code, out = run(capsys, "gen", "--prompt", f"@{text}", "--dry-run")
    assert code == 0 and out["prompt"].startswith("cat, ")


def test_output_is_strict_json(capsys, monkeypatch):
    nan = commands.Command(commands.models.resolve, lambda s, ctx: {"ok": True, "value": float("nan")})
    monkeypatch.setitem(commands.COMMANDS, "models", nan)
    code, out = run(capsys, "models")
    assert code == 1 and out["code"] == "internal"


def test_text_arg_rules(capsys, tmp_path):
    bom = tmp_path / "bom.txt"
    bom.write_bytes("\ufefffur dataset, wolf".encode())
    code, out = run(capsys, "gen", "--prompt", f"@{bom}", "--furry", "--dry-run")
    assert code == 0 and out["prompt"].startswith("fur dataset, wolf, very aesthetic")
    code, out = run(capsys, "gen", "--prompt", "@@alice.art", "--dry-run")
    assert code == 0 and out["prompt"].startswith("@alice.art, ")
    # @@ is for the command line; text read from a file is sent as it is
    double = tmp_path / "double.txt"
    double.write_text("@@_@, 1girl", encoding="utf-8")
    code, out = run(capsys, "gen", "--prompt", f"@{double}", "--dry-run")
    assert code == 0 and out["prompt"].startswith("@@_@, 1girl, ")
    for missing in ("C:\\work\\a,b\\p.txt", "./a, b.txt", "~/nope, 1.txt", "alice.art"):
        code, out = run(capsys, "gen", "--prompt", f"@{missing}", "--dry-run")
        assert code == 1 and out["code"] == "bad_args" and "@@" in out["message"]


def test_prompt_from_standard_input(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO("﻿cat, smile\n".encode()), encoding="utf-8"))
    code, out = run(capsys, "gen", "--prompt", "@-", "--quality", "none", "--dry-run")
    assert code == 0 and out["prompt"] == "cat, smile"
    code, out = run(capsys, "gen", "--prompt", "@-", "--neg", "@-", "--dry-run")
    assert code == 1 and out["code"] == "bad_args"


def test_prompt_from_standard_input_keeps_a_leading_at(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(b"@_@, cat"), encoding="utf-8"))
    code, out = run(capsys, "gen", "--prompt", "@-", "--quality", "none", "--dry-run")
    assert code == 0 and out["prompt"] == "@_@, cat"


def test_text_that_is_not_unicode_never_breaks_the_json_of_any_output(capsys, tmp_path, monkeypatch):
    # A JSON escape in an image's settings, or a POSIX file name that is not UTF-8, gives a lone surrogate
    odd = with_comment(tmp_path / "odd.png", r'{"prompt": "1girl \ud800", "uc": ""}')
    code, out = run(capsys, "meta", str(odd))
    assert code == 0 and out["prompt"] == "1girl \ud800"
    code = cli.main(["meta", str(odd), str(odd)])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 0 and lines[0]["prompt"] == "1girl \ud800" and lines[-1]["type"] == "summary"
    ledger.record({"command": "gen", "argv": ["gen", "--image", "pic\udc80.png"]}, datetime.now(UTC))
    assert next(ledger.entries())["argv"][-1] == "pic\udc80.png"
    sent = []
    transport = httpx.MockTransport(lambda r: sent.append(r.content) or httpx.Response(500))
    with pytest.raises(errors.NaiError), api.Client("t", sleep=lambda s: None, transport=transport) as client:
        client.generate({"input": "1girl \ud800", "parameters": {"n_samples": 1}})
    assert json.loads(sent[0])["input"] == "1girl \ud800"
    assert json.loads(vibe.dumps({"name": "pic\udc80"}))["name"] == "pic\udc80"
    # Typed text has to be sent, so text that is not UTF-8 is refused there
    code, out = run(capsys, "gen", "--prompt", "cat \udc80", "--dry-run")
    assert code == 1 and out["code"] == "bad_args" and "UTF-8" in out["message"]


def test_numbers_and_sizes_at_the_edges_are_refused_not_crashed(capsys, tmp_path, monkeypatch):
    src = tmp_path / "src.png"
    Image.new("RGB", (8, 8)).save(src)
    code, out = run(capsys, "tool", "pixel-snap", str(src), "--colors", "²", "--dry-run")
    assert code == 1 and out["code"] == "bad_args"
    # Within the area limit, but scaled to the request size a side would round to nothing
    tall = tmp_path / "tall.png"
    Image.new("L", (1, 3_145_000)).save(tall)
    code, out = run(capsys, "tool", "lineart", str(tall), "--dry-run")
    assert code == 0 and out["sent_size"].startswith("1x"), out
