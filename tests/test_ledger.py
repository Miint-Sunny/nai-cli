"""The ledger, stats and batch pacing."""

import json
from datetime import date, datetime, timedelta, timezone

import pytest
from helpers import FakeClient, fake_clients, run, run_batch
from PIL import Image
from samples import png_bytes

from nai_cli import api, config, errors, ledger
from nai_cli.commands import context


def _lines(*lines):
    path = ledger.ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def _set(section, text):
    path = config.default_path()
    path.write_text(path.read_text(encoding="utf-8") + f"[{section}]\n{text}\n", encoding="utf-8")


def _today():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def test_summary_groups_by_day_command_and_model():
    _lines(
        {"time": "2026-11-01T10:00:00+09:00", "command": "generate", "model": "m5", "anlas_spent": 0, "files": ["x"]},
        {"time": "2026-11-02T10:00:00+09:00", "command": "upscale", "anlas_spent": 2, "files": ["y"]},
        {
            "time": "2026-11-02T11:00:00+09:00",
            "command": "generate",
            "model": "m5",
            "anlas_spent": None,
            "anlas_estimate": 7,
            "files": [],
        },
        {"time": "2026-10-30T11:00:00+09:00", "command": "generate", "anlas_spent": 99, "files": []},
    )
    out = ledger.summary(date(2026, 11, 1), date(2026, 11, 30), "command")
    assert (out["requests"], out["images"], out["anlas_spent"], out["free_images"]) == (3, 2, 2, 1)
    assert out["requests_with_unknown_spend"] == 1
    assert sorted(out["groups"]) == ["generate", "upscale"]
    by_day = ledger.summary(date(2026, 11, 1), date(2026, 11, 30), "day")["groups"]
    assert list(by_day) == ["2026-11-01", "2026-11-02"]
    by_model = ledger.summary(date(2026, 11, 1), date(2026, 11, 30), "model")["groups"]
    assert by_model["m5"]["requests"] == 2 and by_model["-"]["requests"] == 1


def test_a_request_is_written_once_it_has_run(capsys, with_token, monkeypatch):
    seen = []

    def generate(self, body):
        seen.append(ledger.ledger_path().exists())
        self.anlas -= 30
        return [png_bytes()]

    monkeypatch.setattr(FakeClient, "generate", generate)
    code, _ = run(capsys, "gen", "--prompt", "x", "--size", "1216x1216", "--yes")
    (entry,) = ledger.entries()
    assert code == 0 and seen == [False] and entry["anlas_spent"] == 30 and len(entry["files"]) == 1


def test_a_refused_request_is_not_written(capsys, with_token, monkeypatch):
    def refused(self, body):
        raise errors.NaiError(errors.INSUFFICIENT_ANLAS, "no")

    monkeypatch.setattr(FakeClient, "generate", refused)
    code, _ = run(capsys, "gen", "--prompt", "x", "--size", "1216x1216", "--yes")
    assert code == 1 and list(ledger.entries()) == []


def test_stats_command_ranges(capsys):
    _lines({"time": _today(), "command": "generate", "anlas_spent": 3, "files": ["a"]})
    code, out = run(capsys, "stats")
    assert code == 0 and out["requests"] == 1 and out["by"] == "day" and len(out["groups"]) == 1
    code, out = run(capsys, "stats", "--days", "1", "--since", "2026-01-01")
    assert code == 1 and out["code"] == "bad_args"
    code, out = run(capsys, "stats", "--since", "2026-12-02", "--until", "2026-12-01")
    assert code == 1 and out["code"] == "bad_args"
    code, out = run(capsys, "stats", "--by", "command", "--since", "2000-01-01")
    assert out["groups"]["generate"]["anlas_spent"] == 3


def test_account_totals_the_month(capsys, with_token):
    _lines({"time": _today(), "command": "generate", "anlas_spent": 3, "files": ["a"]})
    code, out = run(capsys, "account")
    assert code == 0 and out["month"]["anlas_spent"] == 3 and out["month"]["groups"]["generate"]
    code, out = run(capsys, "account", "--month", "1999-01")
    assert out["month"]["requests"] == 0 and out["month"]["month"] == "1999-01"
    assert out["month"]["scope"] == "all_accounts" and "this_month" not in out


def _clock(monkeypatch):
    """A monotonic clock that sleeping moves on, and the waits slept."""
    now, waits = [100.0], []

    def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(context.time, "sleep", sleep)
    monkeypatch.setattr(context.time, "monotonic", lambda: now[0])
    return now, waits


def test_batch_waits_before_a_request_counting_from_the_last_sent(capsys, with_token, tmp_path, monkeypatch):
    _, waits = _clock(monkeypatch)
    jobs = tmp_path / "jobs.jsonl"
    rows = [["gen", "--prompt", "a"], ["models"], ["config"], ["gen", "--prompt", "b"], ["gen", "--prompt", "c"]]
    jobs.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    code, lines = run_batch(capsys, str(jobs), "--interval", "3")
    # Local jobs do not wait, and a job waits once, before its first request, not between its own requests
    assert code == 0 and waits == [3.0, 3.0]
    waits.clear()
    run_batch(capsys, str(jobs), "--interval", "3", "--dry-run")
    assert waits == []
    for bad in ("5-2", "5-", "-", "1e20", "²"):
        code, lines = run_batch(capsys, str(jobs), "--interval", bad)
        assert code == 1 and lines[-1]["code"] == "bad_args" and lines[-1]["type"] == "summary", bad
    code, lines = run_batch(capsys, str(jobs), "--interval", "1e-2")
    assert code == 0 and waits == [0.01, 0.01]
    waits.clear()
    # Each end of a range may be written with an exponent
    monkeypatch.setattr("random.uniform", lambda low, high: high)
    code, lines = run_batch(capsys, str(jobs), "--interval", "1e-2-5e-2")
    assert code == 0 and waits == [0.05, 0.05]


def test_time_already_passed_counts_toward_the_wait(capsys, with_token, tmp_path, monkeypatch):
    now, waits = _clock(monkeypatch)
    real_account = FakeClient.account

    def slow(self):
        # The reply comes two seconds after the request goes out
        out = real_account(self)
        now[0] += 2
        return out

    monkeypatch.setattr(FakeClient, "account", slow)
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text('["gen", "--prompt", "a"]\n["gen", "--prompt", "b"]\n', encoding="utf-8")
    run_batch(capsys, str(jobs), "--interval", "3")
    # The first job ends with a balance read; its two seconds count toward the three
    assert waits == [1.0]


def test_batch_interval_from_the_config(capsys, with_token, tmp_path, monkeypatch):
    _set("batch", "interval = [2, 2]")
    _, waits = _clock(monkeypatch)
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text('["gen", "--prompt", "a"]\n["gen", "--prompt", "b"]\n', encoding="utf-8")
    run_batch(capsys, str(jobs))
    assert waits == [2.0]


@pytest.mark.parametrize("value", ["-1", "[3, 1]", '"x"', "[1, 2, 3]", "1e20", "[0, 86401]"])
def test_bad_batch_interval_refused(capsys, value):
    _set("batch", f"interval = {value}")
    code, out = run(capsys, "config")
    assert code == 1 and out["code"] == "config_missing"


def test_an_unreadable_closing_balance_leaves_the_spend_unknown(capsys, with_token, monkeypatch):
    real = FakeClient.account

    def account(self):
        if "account" in self.calls:
            self.calls.append("account")
            return api.UNMEASURED
        return real(self)

    monkeypatch.setattr(FakeClient, "account", account)
    code, out = run(capsys, "gen", "--prompt", "x", "--size", "1216x1216", "--yes")
    assert code == 0 and out["cost"]["spent"] is None
    assert any("spend is unknown" in w for w in out["warnings"])
    (entry,) = ledger.entries()
    assert entry["anlas_spent"] is None


NOW = datetime.now().astimezone().isoformat(timespec="seconds")


def test_a_lock_that_cannot_be_made_is_reported_before_sending(capsys, with_token, monkeypatch):
    clients = fake_clients(monkeypatch)
    (config.data_dir() / "generate.lock").mkdir(parents=True)
    code, out = run(capsys, "gen", "--prompt", "x", "--yes")
    assert code == 1 and out["code"] == "ledger" and "generation lock" in out["message"]
    assert not any("generate" in c.calls for c in clients)


def test_a_kept_raw_image_is_not_counted_again(capsys, with_token, tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (512, 512)).save(img)
    region = "ellipse:0.4,0.4,0.5,0.5"
    code, _ = run(capsys, "inpaint", "--image", str(img), "--region", region, "--prompt", "hand", "--keep-raw")
    assert code == 0 and ledger.summary(date.today(), date.today(), "command")["images"] == 1


def test_date_edges_and_a_line_of_other_bytes_are_no_crash(capsys, with_token, tmp_path):
    path = ledger.ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'{"tx": "a", "status": "settled", "time": "2026-11-01T00:00:00"}\n\xff\xfe broken\n')
    for argv in (
        ["stats", "--until", "0001-01-01"],
        ["stats", "--days", "999999999"],
        ["account", "show", "--month", "9999-12"],
        ["meta", str(tmp_path)],
    ):
        code, out = run(capsys, *argv)
        assert code == 0, (argv, out)
    # A damaged line never stops a request; only the history leaves it out
    code, out = run(capsys, "gen", "--prompt", "x", "--size", "1216x1216", "--yes")
    assert code == 0, out


def test_days_follow_this_computers_clock_whatever_zone_a_line_was_written_in():
    here = datetime.now().astimezone()
    far = timezone(here.utcoffset() + timedelta(hours=13 if here.utcoffset() <= timedelta(0) else -13))
    _lines(
        {
            "time": here.astimezone(far).isoformat(timespec="seconds"),
            "tx": "a",
            "status": "settled",
            "command": "generate",
            "anlas_spent": 7,
            "files": ["a.png"],
        },
        {"time": "not a time", "tx": "b", "status": "settled", "command": "generate", "anlas_spent": 1, "files": []},
    )
    # The second line belongs to no day, so history leaves it out
    assert ledger.summary(here.date(), here.date(), "day")["requests"] == 1
    assert ledger.summary(here.date(), here.date(), "day")["groups"] == {
        here.date().isoformat(): ledger.summary(here.date(), here.date(), "command")["groups"]["generate"]
    }


def test_unreadable_rows_do_not_hide_later_spend(capsys):
    path = ledger.ledger_path()
    _lines({"time": _today(), "command": "gen", "anlas_spent": 7})
    good = path.read_bytes()
    path.write_bytes(b"[" * 2000 + b"]" * 2000 + b'\n{"time":"0001-01-01T00:00:00+23:59"}\n' + good)
    code, out = run(capsys, "stats", "--days", "1")
    assert code == 0 and out["requests"] == 1 and out["anlas_spent"] == 7, out
