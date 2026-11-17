"""Job lists and resuming them: which jobs a previous run finished, from every line it printed."""

from dataclasses import dataclass
from typing import Any

from nai_cli import errors, values

# The option of nai itself a job may give before its command; --config is refused, so the batch's limits hold
JOB_OPTION = "--account"


@dataclass(frozen=True)
class Job:
    id: str
    line: int
    argv: list[str]


def read_jobs(text: str, source: str) -> list[Job]:
    jobs: list[Job] = []
    seen: set[str] = set()
    for number, line in enumerate(_records(text), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        where = f"line {number} of {source}"
        try:
            item = values.parse_json(line)
        except ValueError as e:
            raise errors.NaiError(errors.BAD_ARGS, f"{where} is not JSON: {e}") from None
        job_id = str(number)
        if isinstance(item, dict):
            if "id" in item:
                if not (isinstance(item["id"], str) or values.is_whole(item["id"])):
                    raise errors.NaiError(errors.BAD_ARGS, f"{where}: id must be a string or a number")
                job_id = str(item["id"])
            item = item.get("args")
        if not isinstance(item, list) or not item or not all(isinstance(a, str) for a in item):
            raise errors.NaiError(errors.BAD_ARGS, f"{where}: a job is a list of strings, or an object with args")
        command = _command(item)
        # Commands that print lines of their own cannot be one job's single result
        if command is None or command.startswith("-") or command in ("batch", "grid"):
            raise errors.NaiError(
                errors.BAD_ARGS, f"{where}: a job is a command other than batch or grid, after --account if given"
            )
        if job_id in seen:
            raise errors.NaiError(errors.BAD_ARGS, f"{where}: the id {job_id} is used twice")
        seen.add(job_id)
        jobs.append(Job(job_id, number, item))
    return jobs


def succeeded(text: str) -> set[str]:
    """The jobs a previous output shows succeeding, from any of several runs appended one after another; a dry run's
    lines count for nothing."""
    done: set[str] = set()
    for line in _records(text):
        try:
            item = values.parse_json(line)
        except ValueError:
            continue
        if isinstance(item, dict) and "job" in item and _succeeded(item):
            done.add(str(item["job"]))
    return done


def _records(text: str) -> list[str]:
    # str.splitlines also breaks at U+2028 and the like, which JSON strings may hold unescaped
    return text.split("\n")


def _succeeded(line: dict[str, Any]) -> bool:
    return not line.get("dry_run") and (line.get("ok") is True or line.get("skipped") == "succeeded")


@dataclass
class Totals:
    jobs: int
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    spent: int = 0
    unknown: int = 0
    estimate: int = 0
    # The exit code of the last job that failed
    code: int = 0

    def add(self, result: dict[str, Any], code: int) -> None:
        cost_json = result.get("cost") if isinstance(result.get("cost"), dict) else {}
        if "spent" in cost_json:
            if cost_json["spent"] is None:
                self.unknown += 1
            else:
                self.spent += cost_json["spent"]
        # A preview's estimate is the request it shows, which it does not send
        if result.get("dry_run") and "preview" not in result and values.is_whole(cost_json.get("estimate")):
            self.estimate += max(cost_json["estimate"], 0)
        if code == 0:
            self.succeeded += 1
        else:
            self.failed += 1
            self.code = code

    def to_json(self, dry_run: bool) -> dict[str, Any]:
        out: dict[str, Any] = {
            "jobs": self.jobs,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "spent": self.spent,
        }
        if self.unknown:
            out["jobs_with_unknown_spend"] = self.unknown
        if dry_run:
            out["estimate"] = self.estimate
        return out


def _command(argv: list[str]) -> str | None:
    rest = iter(argv)
    for arg in rest:
        if arg == JOB_OPTION:
            next(rest, None)
        elif not arg.startswith(f"{JOB_OPTION}="):
            return arg
    return None
