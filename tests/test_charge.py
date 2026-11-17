"""Paid requests: the spending cap, the balance read under the lock, and what is settled when a request fails."""

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pytest
from helpers import FakeClient, fake_clients, ledger_entry, run, set_cap
from PIL import Image
from samples import nai_image, png_bytes

from nai_cli import api, cli, config, errors, fs, ledger
from nai_cli.commands import generate


def test_need_confirm_over_cap(capsys, with_token):
    code, out = run(capsys, "gen", "--prompt", "x", "--size", "1472x1472", "--n", "4", "--steps", "28")
    assert code == 1 and out["code"] == "need_confirm" and out["cost"]["estimate"] > 100


def test_generate_writes_file_and_ledger(capsys, with_token):
    code, out = run(capsys, "gen", "--prompt", "x", "--seed", "9")
    assert code == 0 and out["cost"]["spent"] == 30 and out["cost"]["anlas_left"] == 970
    saved = out["images"][0]
    assert Path(saved).read_bytes().startswith(b"\x89PNG")
    entry = ledger_entry(saved)
    assert entry["seed"] == 9 and entry["anlas_spent"] == 30
    today = datetime.now().date()
    assert ledger.summary(today, today, "day")["anlas_spent"] == 30


def test_upscale_prices_by_account(capsys, with_token, tmp_path):
    img = tmp_path / "big.png"
    Image.new("RGB", (832, 1216)).save(img)
    code, out = run(capsys, "upscale", str(img), "--dry-run")
    assert out["cost"]["estimate"] == 1 and "assumed_opus" not in out["cost"]
    code, out = run(capsys, "upscale", str(img))
    assert code == 0 and out["size"] == "1664x2432" and out["cost"]["spent"] == 2
    # The main page's tiers up to 3 MP, with no Opus discount
    for size, price in (((1024, 1536), 2), ((1536, 1536), 3), ((1536, 2048), 4), ((640, 640), 1), ((512, 512), 1)):
        Image.new("RGB", size).save(img)
        code, out = run(capsys, "upscale", str(img), "--dry-run")
        assert code == 0 and out["cost"]["estimate"] == price, size
    Image.new("RGB", (1792, 1792)).save(img)
    code, out = run(capsys, "upscale", str(img), "--dry-run")
    assert code == 1 and out["code"] == "bad_args"


def test_tool_resizes_like_official_and_joins_emotion(capsys, with_token, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(FakeClient, "augment", lambda self, *a, **k: calls.append(a[1:]) or [png_bytes(8, 8)])
    img = tmp_path / "small.png"
    Image.new("RGB", (512, 512)).save(img)
    code, out = run(capsys, "tool", "emotion", str(img), "--emotion", "happy", "--prompt", "wink", "--defry", "2")
    assert code == 0 and calls == [("emotion", 1024, 1024, "happy;;wink", 2)]
    code, out = run(capsys, "tool", "lineart", str(img), "--dry-run")
    assert out["size"] == "512x512" and out["sent_size"] == "1024x1024" and out["cost"]["estimate"] == 0
    code, out = run(capsys, "tool", "emotion", str(img), "--defry", "6", "--dry-run")
    assert out["code"] == "bad_args"


def test_generate_reads_the_balance_only_under_the_lock(capsys, with_token, monkeypatch):
    clients = fake_clients(monkeypatch)
    code, _ = run(capsys, "gen", "--prompt", "x")
    assert code == 0 and clients[0].calls == ["account", "generate", "account"]


def test_price_confirmed_against_the_balance_under_the_lock(capsys, with_token, monkeypatch):
    set_cap(0)
    exhausted = []

    def account(self):
        self.calls.append("account")
        return api.Account(3, True, self.anlas, 0, 0.0, bool(exhausted), None)

    monkeypatch.setattr(FakeClient, "account", account)
    real_lock = ledger.generation_lock

    @contextmanager
    def lock():
        # Another process used up the free quota while this one waited
        exhausted.append(True)
        with real_lock():
            yield

    monkeypatch.setattr(ledger, "generation_lock", lock)
    clients = fake_clients(monkeypatch)
    code, out = run(capsys, "gen", "--prompt", "x")
    assert code == 1 and out["code"] == "need_confirm" and out["cost"]["quota_exhausted"] is True
    assert out["cost"]["estimate"] > 0 and "generate" not in clients[0].calls


def test_images_kept_when_the_closing_balance_cannot_be_read(capsys, with_token, monkeypatch):
    def account(self):
        self.calls.append("account")
        if "generate" in self.calls:
            raise errors.NaiError(errors.NETWORK, "cannot reach NovelAI")
        return api.Account(3, True, self.anlas, 0, 50.0, False, None)

    monkeypatch.setattr(FakeClient, "account", account)
    code, out = run(capsys, "gen", "--prompt", "x", "--seed", "4")
    assert code == 0 and Path(out["images"][0]).read_bytes().startswith(b"\x89PNG")
    assert out["cost"]["spent"] is None and "anlas_left" not in out["cost"]
    assert any("spend is unknown" in w for w in out["warnings"])
    assert ledger_entry(out["images"][0])["anlas_spent"] is None
    today = datetime.now().date()
    assert ledger.summary(today, today, "day")["requests_with_unknown_spend"] == 1


def test_unusable_output_folder_fails_before_charging(capsys, with_token, monkeypatch, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a folder", encoding="utf-8")
    path = config.default_path()
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace((tmp_path / "out").as_posix(), (blocker / "out").as_posix()), encoding="utf-8")
    clients = fake_clients(monkeypatch)
    code, out = run(capsys, "gen", "--prompt", "x")
    assert code == 1 and out["code"] == "bad_args" and clients[0].calls == []


def test_failure_after_paying_still_settles(capsys, with_token, monkeypatch, tmp_path):
    def two_images(self, body):
        self.calls.append("generate")
        self.anlas -= 30
        return [png_bytes(), png_bytes()]

    real_create = fs.create
    written = []

    def write_once(path, data, private=False):
        # Every later write fails, so the fallback to the output folder fails too; the folder is tried before
        if path.name.startswith(".nai-"):
            return real_create(path, data, private)
        if written:
            raise OSError("disk full")
        written.append(path)
        return real_create(path, data, private)

    monkeypatch.setattr(FakeClient, "generate", two_images)
    monkeypatch.setattr(fs, "create", write_once)
    code, out = run(capsys, "gen", "--prompt", "x", "--n", "2", "--seed", "4")
    assert code == 1 and out["code"] == "save_failed" and len(out["saved"]) == 1
    assert out["cost"] == {"spent": 30, "anlas_left": 970}
    (entry,) = ledger.entries()
    assert entry["error"] == "save_failed" and entry["anlas_spent"] == 30 and entry["files"] == out["saved"]


def test_nothing_sent_nothing_recorded(capsys, with_token, monkeypatch, tmp_path):
    def poor(self):
        self.calls.append("account")
        return api.Account(0, True, 1, 0, 0.0, False, None)

    monkeypatch.setattr(FakeClient, "account", poor)
    clients = fake_clients(monkeypatch)
    img = tmp_path / "v.png"
    Image.new("RGB", (64, 64), "red").save(img)
    code, out = run(capsys, "gen", "--model", "v4.5", "--prompt", "x", "--vibe", str(img))
    assert code == 1 and out["code"] == "insufficient_anlas" and clients[0].calls == ["account"]
    assert list(ledger.entries()) == [] and "saved" not in out


@pytest.mark.parametrize(
    "error",
    [
        errors.NaiError(errors.NETWORK, "could not connect"),
        errors.NaiError(errors.INSUFFICIENT_ANLAS, "not enough Anlas"),
        errors.NaiError(errors.RATE_LIMITED, "still busy"),
        # Ctrl+C while waiting to resend, which the client reports this way
        errors.NaiError(errors.INTERRUPTED, "interrupted"),
    ],
)
def test_refused_or_unsent_request_costs_nothing(capsys, with_token, monkeypatch, error):
    def refused(self, body):
        self.calls.append("generate")
        raise error

    monkeypatch.setattr(FakeClient, "generate", refused)
    clients = fake_clients(monkeypatch)
    code, out = run(capsys, "gen", "--prompt", "x")
    assert code == (130 if error.code == errors.INTERRUPTED else 1) and out["code"] == error.code
    assert "cost" not in out and clients[0].calls == ["account", "generate"] and list(ledger.entries()) == []


def test_interrupt_after_paying_reports_the_spend(capsys, with_token, monkeypatch):
    def two_images(self, body):
        self.anlas -= 30
        return [png_bytes(), png_bytes()]

    real_create = fs.create

    def interrupt_second(path, data, private=False):
        if list(path.parent.glob("*.png")):
            raise KeyboardInterrupt
        return real_create(path, data, private)

    monkeypatch.setattr(FakeClient, "generate", two_images)
    monkeypatch.setattr(fs, "create", interrupt_second)
    code, out = run(capsys, "gen", "--prompt", "x", "--n", "2")
    assert code == 130 and out["code"] == "interrupted" and out["cost"]["spent"] == 30 and len(out["saved"]) == 1
    (entry,) = ledger.entries()
    assert entry["error"] == "interrupted" and entry["anlas_spent"] == 30


def test_interrupt_while_reading_the_closing_balance(capsys, with_token, monkeypatch):
    def account(self):
        self.calls.append("account")
        if self.calls.count("account") > 1:
            raise KeyboardInterrupt
        return api.Account(3, True, self.anlas, 0, 50.0, False, None)

    monkeypatch.setattr(FakeClient, "account", account)
    code, out = run(capsys, "gen", "--prompt", "x")
    assert code == 130 and out["code"] == "interrupted" and len(out["saved"]) == 1 and out["cost"]["spent"] is None
    assert any("interrupted" in w for w in out["warnings"])
    (entry,) = ledger.entries()
    assert entry["anlas_spent"] is None and entry["files"] == out["saved"]


@pytest.mark.parametrize("finished", [True, False])
def test_interrupt_while_writing_the_ledger_keeps_the_spend(capsys, with_token, monkeypatch, finished):
    def in_flight(self, body):
        self.anlas -= 30
        if not finished:
            raise KeyboardInterrupt
        return [png_bytes()]

    def interrupted(entry, at):
        raise KeyboardInterrupt

    monkeypatch.setattr(FakeClient, "generate", in_flight)
    monkeypatch.setattr(ledger, "record", interrupted)
    code, out = run(capsys, "gen", "--prompt", "x")
    assert code == 130 and out["code"] == "interrupted" and out["cost"]["spent"] == 30
    assert len(out.get("saved", [])) == (1 if finished else 0)
    assert any("ledger" in w for w in out["warnings"])


def test_damaged_image_keeps_the_others(capsys, with_token, monkeypatch):
    def partly_damaged(self, body):
        self.anlas -= 30
        raise api.DamagedImages("1 of 2 images in the response are damaged", [None, png_bytes()])

    monkeypatch.setattr(FakeClient, "generate", partly_damaged)
    code, out = run(capsys, "gen", "--prompt", "x", "--n", "2", "--seed", "9")
    assert code == 1 and out["code"] == "api" and out["maybe_processed"] and out["cost"]["spent"] == 30
    # The second image keeps its own number
    assert len(out["saved"]) == 1 and Path(out["saved"][0]).stem.endswith("_9_1")
    (entry,) = ledger.entries()
    assert entry["files"] == out["saved"] and entry["error"] == "api"


def test_damage_still_reported_when_saving_fails(capsys, with_token, monkeypatch):
    def partly_damaged(self, body):
        self.anlas -= 30
        raise api.DamagedImages("1 of 2 images in the response are damaged", [None, png_bytes()])

    real_create = fs.create

    def full(path, data, private=False):
        # The disk fills after the folder is tried
        if path.name.startswith(".nai-"):
            return real_create(path, data, private)
        raise OSError("disk full")

    monkeypatch.setattr(FakeClient, "generate", partly_damaged)
    monkeypatch.setattr(fs, "create", full)
    code, out = run(capsys, "gen", "--prompt", "x", "--n", "2")
    assert code == 1 and out["code"] == "save_failed" and out["cost"]["spent"] == 30
    assert any("damaged" in w for w in out["warnings"])


def test_background_removal_saves_every_image(capsys, with_token, monkeypatch, tmp_path):
    def three(self, data, req_type, w, h, prompt, defry, expected=1):
        self.anlas -= 65
        return [png_bytes(8, 8), png_bytes(8, 8), png_bytes(8, 8)]

    monkeypatch.setattr(FakeClient, "augment", three)
    img = tmp_path / "cat.png"
    Image.new("RGB", (64, 64)).save(img)
    code, out = run(capsys, "tool", "bg-removal", str(img), "--yes")
    assert code == 0 and [Path(p).stem.rsplit("_", 1)[1] for p in out["images"]] == ["masked", "generated", "blend"]
    (entry,) = ledger.entries()
    assert entry["files"] == out["images"] and entry["anlas_spent"] == 65


def test_tool_keeps_images_of_a_damaged_reply(capsys, with_token, monkeypatch, tmp_path):
    def damaged(self, data, req_type, w, h, prompt, defry, expected=1):
        self.anlas -= 65
        raise api.DamagedImages(
            "1 of 3 images in the Director tool response are damaged", [png_bytes(), None, png_bytes()]
        )

    monkeypatch.setattr(FakeClient, "augment", damaged)
    img = tmp_path / "cat.png"
    Image.new("RGB", (64, 64)).save(img)
    code, out = run(capsys, "tool", "bg-removal", str(img), "--yes")
    assert code == 1 and out["code"] == "api" and out["cost"]["spent"] == 65
    assert [Path(p).stem.rsplit("_", 1)[1] for p in out["saved"]] == ["masked", "blend"]


def test_interrupt_while_waiting_to_reread_the_balance(capsys, with_token, monkeypatch):
    def account(self):
        self.calls.append("account")
        if self.calls.count("account") > 1:
            # What api.Client raises for Ctrl+C between attempts
            raise errors.NaiError(errors.INTERRUPTED, "interrupted")
        return api.Account(3, True, self.anlas, 0, 50.0, False, None)

    monkeypatch.setattr(FakeClient, "account", account)
    code, out = run(capsys, "gen", "--prompt", "x")
    assert code == 130 and out["code"] == "interrupted" and len(out["saved"]) == 1 and out["cost"]["spent"] is None


def test_bug_after_paying_reports_the_spend(capsys, with_token, monkeypatch):
    def broken(*args, **kwargs):
        raise TypeError("bug")

    monkeypatch.setattr(generate, "_save_images", broken)
    code = cli.main(["gen", "--prompt", "x"])
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert code == 1 and out["code"] == "internal" and out["cost"]["spent"] == 30
    assert "TypeError: bug" in captured.err


def test_a_dry_run_reports_the_refusal_of_the_real_run(capsys, with_token):
    code, out = run(capsys, "gen", "--prompt", "x", "--size", "1472x1472", "--n", "4", "--dry-run")
    assert code == 0 and out["refused"]["code"] == "need_confirm"
    code, out = run(capsys, "gen", "--prompt", "x", "--size", "1472x1472", "--n", "4", "--dry-run", "--yes")
    assert code == 0 and "refused" not in out


def test_a_preview_is_refused_where_the_request_it_previews_is(capsys, with_token, tmp_path, monkeypatch):
    clients = fake_clients(monkeypatch)
    img = tmp_path / "big.png"
    Image.new("RGB", (2048, 1536), "red").save(img)
    argv = ["inpaint", "--image", str(img), "--region", "rect:0,0,1,1", "--prompt", "x", "--steps", "50"]
    for extra in (["--dry-run"], ["--preview"], []):
        code, out = run(capsys, *argv, *extra, "-o", str(tmp_path / "out"))
        assert (code, out["code"]) == (1, "bad_args") and "per-image limit" in out["message"], extra
    assert not (tmp_path / "out").exists() and not any("generate" in c.calls for c in clients)


def test_a_balance_that_cannot_be_read_prices_a_dry_run_as_opus(capsys, with_token, tmp_path, monkeypatch):
    def down(self):
        raise errors.NaiError(errors.NETWORK, "connection refused")

    monkeypatch.setattr(FakeClient, "account", down)
    img = tmp_path / "a.png"
    nai_image(img, w=512, h=512)
    for argv in (
        ["gen", "--prompt", "x", "--dry-run"],
        ["upscale", str(img), "--dry-run"],
        ["vibe", str(img), "--dry-run"],
        ["cost"],
    ):
        code, out = run(capsys, *argv)
        assert code == 0 and out["cost"]["assumed_opus"], (argv, out)
        assert any("connection refused" in w for w in out["warnings"]), (argv, out)
