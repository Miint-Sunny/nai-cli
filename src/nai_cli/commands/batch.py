"""batch: run the jobs of a JSONL file in process, one result line each as soon as it ends."""

import random
import re
from dataclasses import dataclass, field
from typing import Any

from nai_cli import config, errors, values
from nai_cli.batch import Job, Totals, read_jobs, succeeded
from nai_cli.commands.context import Context

_NUMBER = r"((?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
_INTERVAL = re.compile(rf"{_NUMBER}(?:\s*-\s*{_NUMBER})?")


@dataclass(frozen=True)
class RunOptions:
    stop_on_error: bool
    dry_run: bool
    # Seconds from the last request sent to the first of the next job, drawn between the two for each job
    interval: tuple[float, float]


@dataclass(frozen=True)
class BatchSettings:
    jobs: tuple[Job, ...]
    # Jobs a previous run finished, skipped this time
    done: set[str]
    options: RunOptions


def resolve(opts: dict[str, Any], ctx: Context) -> BatchSettings:
    try:
        source = opts["jobs"]
        jobs = read_jobs(ctx.read_stdin() if source == "-" else _read(source, "jobs"), source)
        done = succeeded(_read(opts["skip_done"], "results")) if opts.get("skip_done") else set()
        options = run_options(opts, ctx)
    except errors.NaiError as e:
        raise as_summary(e) from None
    return BatchSettings(tuple(jobs), done, options)


def run_options(opts: dict[str, Any], ctx: Context) -> RunOptions:
    return RunOptions(
        bool(opts.get("stop_on_error")),
        bool(opts.get("dry_run")),
        _interval(opts.get("interval"), ctx),
    )


def as_summary(e: errors.NaiError) -> errors.NaiError:
    """An error before any job ran, still printed as one summary line, as a run of jobs always prints."""
    e.detail.setdefault("type", "summary")
    e.detail.setdefault("batch", None)
    return e


def _interval(text: str | None, ctx: Context) -> tuple[float, float]:
    if text is None:
        return ctx.config.interval
    # The - of 1e-2 is part of a number, not a range
    found = _INTERVAL.fullmatch(text.strip())
    try:
        if found is None:
            raise ValueError(text)
        low, high = found.group(1), found.group(2) or found.group(1)
        return config.interval([float(low), float(high)])
    except ValueError as e:
        raise errors.NaiError(
            errors.BAD_ARGS, f"--interval takes seconds or MIN-MAX, up to {config.MAX_INTERVAL}: {text!r}"
        ) from e


def _read(path: str, what: str) -> str:
    # Decoded without newline translation, so only line feeds end a line, as on standard input
    try:
        return values.path(path).read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        raise errors.NaiError(errors.BAD_ARGS, f"cannot read the {what} in {path}: {e}") from e


def run(s: BatchSettings, ctx: Context) -> dict[str, Any]:
    return summary(run_jobs(s.jobs, s.done, s.options, ctx), s.options.dry_run)


@dataclass
class Ran:
    totals: Totals
    # The result of each job that ran, by id
    results: dict[str, dict[str, Any]] = field(default_factory=dict)


def run_jobs(jobs: tuple[Job, ...], done: set[str], o: RunOptions, ctx: Context) -> Ran:
    """Each job runs as its own command under the same config, or as its own dry run. Its line is emitted as soon as
    it ends, so a run cut short still leaves the results of the jobs that ran."""
    ran = Ran(Totals(len(jobs)))
    try:
        for job in jobs:
            if job.id in done:
                ran.totals.skipped += 1
                skipped = {"type": "job", "job": job.id, "line": job.line, "skipped": "succeeded"}
                ctx.emit({**skipped, "dry_run": True} if o.dry_run else skipped)
                continue
            # Spent before the job's first request, so a job that sends nothing does not wait
            ctx.traffic.wait = 0.0 if o.dry_run else random.uniform(*o.interval)
            result, code = ctx.run_job(job.argv, ctx.job(job.argv, o.dry_run))
            ctx.emit(_job_line(job, code, result, o.dry_run))
            ran.results[job.id] = result
            ran.totals.add(result, code)
            if ctx.shared.interrupted:
                ran.totals.code = 130
            if ran.totals.code == 130 or (code and o.stop_on_error):
                break
    except KeyboardInterrupt:
        ran.totals.code = 130
    return ran


def _job_line(job: Job, code: int, result: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    """The batch's own keys over the job's result, so a result that has a type or a line of its own cannot pass
    for another line; in a dry run every line says so, whether or not its command has a dry run."""
    line: dict[str, Any] = {"type": "job", "job": job.id, "line": job.line, "exit": code}
    line.update((key, value) for key, value in result.items() if key not in line)
    if dry_run:
        line["dry_run"] = True
    return line


def summary(ran: Ran, dry_run: bool, **extra: Any) -> dict[str, Any]:
    totals = ran.totals
    batch = totals.to_json(dry_run)
    if totals.code == 130:
        raise errors.NaiError(errors.INTERRUPTED, "interrupted", type="summary", batch=batch, **extra)
    if totals.code:
        raise errors.NaiError(
            errors.BATCH_FAILED,
            f"{totals.failed} of {totals.jobs} jobs failed",
            type="summary",
            batch=batch,
            **extra,
        )
    return {"type": "summary", "ok": True, "batch": batch, **extra}
