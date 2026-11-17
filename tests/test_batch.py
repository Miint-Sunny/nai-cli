"""Resuming a batch from its own output: run twice, appended logs, a dry run first."""

import json

import pytest
from helpers import FakeClient, fake_clients, run_batch
from PIL import Image
from samples import nai_image, pixel_art

from nai_cli import batch, cli, commands, config, errors, fs


def _jobs(tmp_path, *rows):
    path = tmp_path / "jobs.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _log(path, lines):
    with path.open("a", encoding="utf-8") as f:
        f.write("".join(json.dumps(line) + "\n" for line in lines))


@pytest.mark.parametrize(
    ("line", "done"),
    [
        ({"job": "a", "ok": True}, True),
        ({"job": "a", "skipped": "succeeded"}, True),
        ({"job": "a", "ok": True, "dry_run": True}, False),
        ({"job": "a", "ok": False, "code": "bad_args"}, False),
        ({"job": "a", "ok": False, "code": "save_failed", "cost": {"spent": 20}}, False),
    ],
)
def test_only_a_job_that_succeeded_counts_as_done(line, done):
    lines = [line, {"type": "summary", "ok": True, "batch": {}}]
    assert batch.succeeded("\n".join(json.dumps(item) for item in lines)) == ({"a"} if done else set())


def test_a_job_that_succeeded_in_any_earlier_run_is_done():
    lines = [{"job": "a", "ok": False, "code": "api"}, {"job": "a", "ok": True}, {"job": "b", "ok": False}]
    assert batch.succeeded("\n".join(json.dumps(line) for line in lines)) == {"a"}


def test_resumed_twice_runs_nothing_again(capsys, with_token, tmp_path, monkeypatch):
    clients = fake_clients(monkeypatch)
    jobs = _jobs(tmp_path, ["gen", "--prompt", "a"], ["gen", "--prompt", "b"])
    log = tmp_path / "log.jsonl"
    _, lines = run_batch(capsys, str(jobs))
    _log(log, lines)
    _, lines = run_batch(capsys, str(jobs), "--skip-done", str(log))
    _log(log, lines)
    # The second resume reads the skipped lines of the first
    _, lines = run_batch(capsys, str(jobs), "--skip-done", str(log))
    assert [line.get("skipped") for line in lines[:2]] == ["succeeded", "succeeded"]
    assert sum(c.calls.count("generate") for c in clients) == 2


def test_a_dry_run_log_leaves_every_job_to_run(capsys, with_token, tmp_path, monkeypatch):
    clients = fake_clients(monkeypatch)
    jobs = _jobs(tmp_path, ["gen", "--prompt", "a"], ["gen", "--prompt", "b"])
    log = tmp_path / "log.jsonl"
    _, lines = run_batch(capsys, str(jobs), "--dry-run")
    _log(log, lines)
    _, lines = run_batch(capsys, str(jobs), "--skip-done", str(log))
    assert [line["exit"] for line in lines[:2]] == [0, 0]
    assert sum(c.calls.count("generate") for c in clients) == 2
    _log(log, lines)
    # A dry run after the real one skips the same jobs the real one would
    _, lines = run_batch(capsys, str(jobs), "--skip-done", str(log), "--dry-run")
    assert [line.get("skipped") for line in lines[:2]] == ["succeeded", "succeeded"]


def test_dry_run_reaches_jobs_that_end_their_options(capsys, with_token, tmp_path, monkeypatch):
    clients = fake_clients(monkeypatch)
    # A --dry-run added after "--" would have been read as a file name
    jobs = _jobs(tmp_path, ["gen", "--prompt", "a"], ["strip", "--", str(tmp_path / "pic.png")])
    Image.new("RGB", (8, 8)).save(tmp_path / "pic.png")
    code, lines = run_batch(capsys, str(jobs), "--dry-run")
    assert code == 0 and all(line.get("dry_run") for line in lines[:2])
    assert sum(c.calls.count("generate") for c in clients) == 0
    # The dry run wrote no copy
    assert not list(tmp_path.glob("pic_*"))


def test_every_line_says_what_it_is(capsys, tmp_path):
    jobs = _jobs(tmp_path, ["gen", "--prompt", "a", "--dry-run"])
    log = tmp_path / "log.jsonl"
    _log(log, [{"job": "1", "ok": True}])
    _, lines = run_batch(capsys, str(jobs))
    assert [line["type"] for line in lines] == ["job", "summary"]
    _, lines = run_batch(capsys, str(jobs), "--skip-done", str(log))
    assert [line["type"] for line in lines] == ["job", "summary"]
    _, lines = run_batch(capsys, str(tmp_path / "missing.jsonl"))
    assert [line["type"] for line in lines] == ["summary"]


@pytest.mark.parametrize(
    ("args", "ok"),
    [
        (["--account", "x", "gen"], True),
        (["--account=x", "gen"], True),
        (["--config", "c.toml", "gen"], False),
        (["--account", "x"], False),
        (["--account", "x", "batch", "f"], False),
        (["--yes", "gen"], False),
    ],
)
def test_jobs_may_name_nai_options_before_their_command(args, ok):
    text = json.dumps(args)
    if ok:
        assert batch.read_jobs(text, "jobs")[0].argv == args
    else:
        with pytest.raises(errors.NaiError):
            batch.read_jobs(text, "jobs")


def _files(root):
    # Caches only spare work, so a dry run may fill them
    caches = ("tokenizers", "vibe-cache")
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and not any(part in caches for part in p.parts)
    }


def test_a_dry_run_changes_nothing_in_any_command(capsys, with_token, tmp_path):
    img = tmp_path / "a.png"
    nai_image(img, w=512, h=512)
    art = tmp_path / "art.png"
    Image.fromarray(pixel_art(16, 12, 8)[1], "RGBA").save(art)
    region = "rect:0.1,0.1,0.5,0.5"
    rows = {
        "gen": ["gen", "--prompt", "x"],
        "i2i": ["i2i", "--image", str(img), "--prompt", "x"],
        "inpaint": ["inpaint", "--image", str(img), "--region", region, "--prompt", "x", "--preview"],
        "enhance": ["enhance", "--image", str(img)],
        "variations": ["variations", "--image", str(img)],
        "upscale": ["upscale", str(img)],
        "tool": ["tool", "pixel-snap", str(art)],
        "vibe": ["vibe", str(img)],
        "strip": ["strip", str(img)],
        "redact": ["redact", str(img), "--region", region],
        "watermark": ["watermark", str(img)],
        "meta": ["meta", str(img)],
        "lint": ["lint", "--prompt", "x"],
        "tags": ["tags", "1gi"],
        "models": ["models"],
        "cost": ["cost"],
        "account": ["account", "use", "default"],
        "stats": ["stats"],
        "interrogate": ["interrogate", str(img)],
        "local": ["local", "download", "sam"],
        "config": ["config", "init", "--force"],
    }
    # A batch runs no batch or grid; a grid takes its own --dry-run, below
    assert set(rows) | {"batch", "grid"} == set(commands.COMMANDS)
    jobs_file = _jobs(tmp_path, *rows.values())
    before = _files(tmp_path)
    _, lines = run_batch(capsys, str(jobs_file), "--dry-run")
    jobs = [line for line in lines if line.get("type") == "job"]
    after = _files(tmp_path)
    assert after == before
    # Each command ran, except one with no local model configured
    failed = {name for name, job in zip(rows, jobs, strict=True) if not job["ok"]}
    assert failed <= {"interrogate"}, [j for j in jobs if not j["ok"]]
    cli.main(["grid", "--dry-run", "--x", "steps=20,28", "gen", "--prompt", "x"])
    capsys.readouterr()
    assert _files(tmp_path) == after


def test_a_dry_run_refuses_an_output_folder_the_real_run_refuses(capsys, with_token, tmp_path):
    img = tmp_path / "a.png"
    nai_image(img, w=512, h=512)
    art = tmp_path / "art.png"
    Image.fromarray(pixel_art(16, 12, 8)[1], "RGBA").save(art)
    blocking = tmp_path / "taken"
    blocking.write_text("a file", encoding="utf-8")
    out_dir = str(blocking / "sub")
    rows = [
        ["gen", "--prompt", "x", "-o", out_dir],
        ["i2i", "--image", str(img), "--prompt", "x", "-o", out_dir],
        ["upscale", str(img), "-o", out_dir],
        ["tool", "lineart", str(img), "-o", out_dir],
        ["tool", "pixel-snap", str(art), "-o", out_dir],
        ["strip", str(img), "-o", out_dir],
        ["watermark", str(img), "-o", out_dir],
    ]
    jobs = _jobs(tmp_path, *rows)
    for extra in (["--dry-run"], []):
        _, lines = run_batch(capsys, str(jobs), *extra)
        found = [(line["ok"], line.get("code"), "is a file" in line.get("message", "")) for line in lines[:-1]]
        assert found == [(False, "bad_args", True)] * len(rows), (extra, lines)


def test_a_job_result_cannot_pass_for_another_line(capsys, with_token, tmp_path):
    img = tmp_path / "a.png"
    nai_image(img, w=512, h=512)
    empty = tmp_path / "empty"
    empty.mkdir()
    folder = tmp_path / "pics"
    folder.mkdir()
    nai_image(folder / "b.png", w=512, h=512)
    rows = [
        ["meta", str(empty)],
        ["meta", str(folder)],
        ["meta", str(img)],
        ["models"],
        ["stats"],
        ["gen", "--prompt", "x"],
    ]
    jobs = _jobs(tmp_path, *rows)
    for extra in ([], ["--dry-run"]):
        _, lines = run_batch(capsys, str(jobs), *extra)
        assert [(line["type"], line["job"], line["line"]) for line in lines[:-1]] == [
            ("job", str(n), n) for n in range(1, len(rows) + 1)
        ]
        assert lines[-1]["type"] == "summary"
        # Lines of images would be printed as lines of the batch, so a folder is refused even when empty
        assert [line.get("code") for line in lines[:2]] == ["bad_args", "bad_args"]
        assert all(line.get("dry_run") is True for line in lines[:-1]) is bool(extra)
    # A dry run's results mark nothing as done, so going on from them runs every job
    log = tmp_path / "dry.jsonl"
    _, lines = run_batch(capsys, str(jobs), "--dry-run")
    log.write_text("\n".join(map(json.dumps, lines)), encoding="utf-8")
    _, lines = run_batch(capsys, str(jobs), "--skip-done", str(log), "--dry-run")
    assert not any("skipped" in line for line in lines[:-1])


def test_a_config_written_by_a_job_is_read_by_the_jobs_after_it(capsys, tmp_path):
    jobs = _jobs(tmp_path, ["config"], ["config", "init", "--force"], ["config"])
    code, lines = run_batch(capsys, str(jobs))
    assert code == 0 and lines[0]["output_dir"] == str(tmp_path / "out")
    assert lines[2]["output_dir"] != lines[0]["output_dir"], lines


def _add_config(section, text):
    path = config.default_path()
    path.write_text(path.read_text(encoding="utf-8") + f"[{section}]\n{text}\n", encoding="utf-8")


def test_a_dry_run_refuses_an_account_the_run_would_not_find(capsys, with_token):
    for extra in (["--dry-run"], []):
        code = cli.main(["--account", "absent", "gen", "--prompt", "x", *extra])
        out = json.loads(capsys.readouterr().out)
        assert (code, out["code"]) == (1, "bad_args"), (extra, out)


def test_a_grid_dry_run_reads_the_balance_once(capsys, with_token, monkeypatch):
    clients = fake_clients(monkeypatch)
    code = cli.main(["grid", "--dry-run", "--x", "steps=20,24", "gen", "--prompt", "x"])
    capsys.readouterr()
    assert code == 0 and sum(c.calls.count("account") for c in clients) == 1


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\x85"])
def test_resume_keeps_line_separators_inside_job_ids(capsys, with_token, tmp_path, monkeypatch, separator):
    clients = fake_clients(monkeypatch)
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(
        json.dumps({"id": f"a{separator}b", "args": ["gen", "--prompt", "cat", "--size", "1216x1216"]}) + "\n",
        encoding="utf-8",
    )
    assert cli.main(["batch", str(jobs)]) == 0
    log = tmp_path / "results.jsonl"
    log.write_text(capsys.readouterr().out, encoding="utf-8")
    raw = (
        jobs.read_text(encoding="utf-8")
        .replace("\\u2028", "\u2028")
        .replace("\\u2029", "\u2029")
        .replace("\\u0085", "\x85")
    )
    jobs.write_text(raw, encoding="utf-8")
    code = cli.main(["batch", str(jobs), "--skip-done", str(log)])
    first = json.loads(capsys.readouterr().out.split("\n")[0])
    assert code == 0 and first.get("skipped") == "succeeded", first
    assert sum(c.calls.count("generate") for c in clients) == 1


def test_a_job_that_failed_after_paying_runs_again(capsys, with_token, tmp_path, monkeypatch):
    clients = fake_clients(monkeypatch)
    create = fs.create

    def full(path, data, private=False):
        if path.suffix == ".png" and not path.name.startswith(".nai-"):
            raise OSError("disk full")
        return create(path, data, private)

    jobs = _jobs(tmp_path, ["gen", "--prompt", "cat", "--size", "1216x1216", "--yes"])
    with monkeypatch.context() as first:
        first.setattr(fs, "create", full)
        assert cli.main(["batch", str(jobs)]) == 1
    log = tmp_path / "results.jsonl"
    raw = capsys.readouterr().out
    log.write_text(raw, encoding="utf-8")
    failure = json.loads(raw.splitlines()[0])
    assert failure["code"] == "save_failed" and failure["cost"]["spent"] > 0
    code, lines = run_batch(capsys, str(jobs), "--skip-done", str(log))
    assert code == 0 and "skipped" not in lines[0]
    assert sum(c.calls.count("generate") for c in clients) == 2


def test_dry_run_skips_are_not_proof_of_success(capsys, tmp_path):
    jobs = _jobs(tmp_path, ["models"])
    log = tmp_path / "results.jsonl"
    _log(log, [{"job": "1", "ok": True}])
    code, lines = run_batch(capsys, str(jobs), "--skip-done", str(log), "--dry-run")
    assert code == 0 and lines[0].get("dry_run") is True
    assert batch.succeeded(json.dumps(lines[0])) == set()


def test_budget_refusal_stops_batch_dry_run(capsys, tmp_path, with_token, monkeypatch):
    clients = fake_clients(monkeypatch)
    jobs = _jobs(tmp_path, ["gen", "--prompt", "x", "--size", "1472x1472", "--n", "4"], ["models"])
    code, lines = run_batch(capsys, str(jobs), "--dry-run", "--stop-on-error")
    assert code == 1 and len(lines) == 2 and lines[0]["code"] == errors.NEED_CONFIRM
    assert lines[0]["exit"] == 1 and lines[-1]["batch"]["failed"] == 1
    assert not any("generate" in c.calls for c in clients)


def test_jobs_and_resume_logs_split_only_at_lf(capsys, tmp_path, with_token, monkeypatch):
    clients = fake_clients(monkeypatch)
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_bytes(b'\xef\xbb\xbf# disabled\r["gen", "--prompt", "paid"]\r\n["models"]\n')
    log = tmp_path / "results.jsonl"
    log.write_bytes(b'# ignored\r{"job":"2","ok":true}\r\n')
    code, lines = run_batch(capsys, str(jobs), "--skip-done", str(log))
    assert code == 0 and len(lines) == 2 and lines[0]["line"] == 2 and lines[0]["ok"]
    assert not any("generate" in c.calls for c in clients)


@pytest.mark.parametrize("fails", [False, True])
def test_close_interrupt_preserves_settled_job_and_stops_batch(capsys, with_token, tmp_path, monkeypatch, fails):
    clients = fake_clients(monkeypatch)
    jobs = _jobs(tmp_path, ["gen", "--prompt", "first"], ["gen", "--prompt", "second"])

    def interrupt(self):
        raise KeyboardInterrupt

    if fails:

        def after_send(self, body):
            self._sent("generate")
            self.anlas -= 7
            raise errors.NaiError(errors.SERVER, "lost reply", maybe_processed=True)

        monkeypatch.setattr(FakeClient, "generate", after_send)
    monkeypatch.setattr(FakeClient, "close", interrupt)
    code, lines = run_batch(capsys, str(jobs))
    assert code == 130 and len(lines) == 2 and lines[-1]["code"] == errors.INTERRUPTED
    assert sum(c.calls.count("generate") for c in clients) == 1
    job = lines[0]
    assert job["cost"]["spent"] == (7 if fails else 30)
    if fails:
        assert job["code"] == errors.SERVER and job["maybe_processed"]
    else:
        assert job["ok"] and job["images"] and batch.succeeded(json.dumps(job)) == {"1"}


def test_interrupt_settling_a_failure_stops_before_another_paid_job(capsys, with_token, tmp_path, monkeypatch):
    clients = fake_clients(monkeypatch)
    jobs = _jobs(tmp_path, ["gen", "--prompt", "a"], ["gen", "--prompt", "b"])
    read = FakeClient.account

    def account(self):
        if "generate" in self.calls:
            raise KeyboardInterrupt
        return read(self)

    def generate(self, body):
        self._sent("generate")
        raise errors.NaiError(errors.SERVER, "lost reply", maybe_processed=True)

    monkeypatch.setattr(FakeClient, "account", account)
    monkeypatch.setattr(FakeClient, "generate", generate)
    code, lines = run_batch(capsys, str(jobs))
    assert code == 130 and len(lines) == 2 and lines[0]["code"] == errors.SERVER and lines[0]["maybe_processed"]
    assert sum(c.calls.count("generate") for c in clients) == 1


def test_dry_run_jobs_share_neither_output_files_nor_vibe_encodings(capsys, tmp_path, with_token, monkeypatch):
    clients = fake_clients(monkeypatch)
    source = tmp_path / "input.png"
    result = tmp_path / "made.png"
    Image.new("RGB", (64, 64)).save(source)
    jobs = _jobs(
        tmp_path,
        ["vibe", str(source)],
        ["vibe", str(source)],
        ["gen", "--prompt", "cat", "-o", str(result)],
        ["upscale", str(result)],
    )
    code, lines = run_batch(capsys, str(jobs), "--dry-run")
    assert code == 1 and [line["cost"]["estimate"] for line in lines[:2]] == [2, 2]
    assert not result.exists() and lines[3]["code"] == errors.BAD_ARGS and lines[-1]["batch"]["estimate"] == 4
    assert all(set(c.calls) <= {"account"} for c in clients)


def test_unknown_job_option_keeps_command_usage(tmp_path, capsys):
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text('["gen", "--unknown"]\n["models"]', encoding="utf-8")
    code, (failed, passed, summary) = run_batch(capsys, str(jobs), "--dry-run")
    assert (code, failed["exit"], failed["code"]) == (1, 1, "bad_args")
    assert failed["usage"].startswith("usage: nai gen ") and passed["exit"] == 0
    assert summary["type"] == "summary" and summary["code"] == "batch_failed"


def test_batch_runs_each_job_and_sums_the_spend(capsys, with_token, tmp_path):
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(
        "# comments and blank lines are skipped\n\n"
        + json.dumps(["gen", "--prompt", "a", "--seed", "1", "--size", "1216x1216"])
        + "\n"
        + json.dumps({"id": "bad", "args": ["gen", "--size", "1x1"]})
        + "\n"
        + json.dumps({"id": "b", "args": ["gen", "--prompt", "b", "--size", "1216x1216"]})
        + "\n",
        encoding="utf-8",
    )
    code, lines = run_batch(capsys, str(jobs))
    assert code == 1 and [line["job"] for line in lines[:3]] == ["3", "bad", "b"]
    assert [line["exit"] for line in lines[:3]] == [0, 1, 0] and lines[1]["code"] == "bad_args"
    summary = lines[-1]
    assert summary["code"] == "batch_failed"
    assert summary["batch"] == {"jobs": 3, "succeeded": 2, "failed": 1, "skipped": 0, "spent": 60}
    # Run again from the results: only the failed job runs
    results = tmp_path / "results.jsonl"
    results.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    code, again = run_batch(capsys, str(jobs), "--skip-done", str(results))
    assert [line.get("skipped") for line in again[:3]] == ["succeeded", None, "succeeded"]
    assert again[-1]["batch"]["skipped"] == 2 and again[-1]["batch"]["spent"] == 0


def test_batch_dry_run_estimates_and_stops_on_error(capsys, tmp_path):
    jobs = tmp_path / "jobs.jsonl"
    rows = [
        ["gen", "--prompt", "a", "--size", "1216x1216"],
        ["meta", str(tmp_path / "none.png")],
        ["gen", "--prompt", "b"],
    ]
    jobs.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    code, lines = run_batch(capsys, str(jobs), "--dry-run")
    assert lines[-1]["batch"]["estimate"] == lines[0]["cost"]["estimate"] > 0
    code, lines = run_batch(capsys, str(jobs), "--dry-run", "--stop-on-error")
    assert code == 1 and len(lines) == 3 and lines[-1]["batch"]["failed"] == 1


def test_batch_refuses_bad_jobs(capsys, tmp_path):
    jobs = tmp_path / "jobs.jsonl"
    for text in ('["batch", "x"]', '{"args": "gen"}', "not json", '["--config", "x", "gen"]'):
        jobs.write_text(text, encoding="utf-8")
        code, lines = run_batch(capsys, str(jobs))
        assert code == 1 and lines[-1]["code"] == "bad_args"
    jobs.write_text('["gen", "--prompt", "@-", "--dry-run"]\n["gen", "--help"]', encoding="utf-8")
    code, lines = run_batch(capsys, str(jobs))
    assert [line["code"] for line in lines[:2]] == ["bad_args", "bad_args"]
