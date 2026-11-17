"""local: what the optional models run on this computer need, and downloading the SAM model when asked."""

import hashlib
from dataclasses import dataclass
from typing import Any

from nai_cli import api, errors, fs, local_models
from nai_cli.commands.context import Context


@dataclass(frozen=True)
class LocalSettings:
    action: str
    dry_run: bool


def resolve(opts: dict[str, Any], ctx: Context) -> LocalSettings:
    action = opts.get("local_command") or "status"
    if action == "download" and opts.get("model") != "sam":
        raise errors.NaiError(errors.BAD_ARGS, "download takes the model to fetch: nai local download sam")
    if action == "status" and opts.get("model"):
        raise errors.NaiError(errors.BAD_ARGS, f"status takes no model: {opts['model']}")
    return LocalSettings(action, bool(opts.get("dry_run")) or ctx.dry_run)


def run(s: LocalSettings, ctx: Context) -> dict[str, Any]:
    return _download_sam(s, ctx) if s.action == "download" else _status(ctx)


def _status(ctx: Context) -> dict[str, Any]:
    local = ctx.config.local
    out: dict[str, Any] = {
        "ok": True,
        "onnxruntime": local_models.available(),
        "tagger": str(local.tagger) if local.tagger else None,
        "sam": {
            "dir": str(local.sam_dir),
            "missing": [a.name for a in local_models.missing(local.sam_dir, local_models.SAM)],
        },
    }
    if not local_models.available():
        out["install"] = local_models.INSTALL_HINT
    return out


def _download_sam(s: LocalSettings, ctx: Context) -> dict[str, Any]:
    """Files already there are kept only when their hash matches; each download is checked before it replaces
    anything."""
    folder = ctx.config.local.sam_dir
    wanted = local_models.unverified(folder, local_models.SAM)
    plan = {"dir": str(folder), "files": [a.name for a in wanted], "bytes": sum(a.size for a in wanted)}
    if s.dry_run or not wanted:
        return {"ok": True, "dry_run": s.dry_run, **plan}
    for asset in wanted:
        data = api.fetch_file(asset.url, asset.size)
        try:
            if hashlib.sha256(data).hexdigest() != asset.sha256:
                raise errors.NaiError(errors.NETWORK, f"{asset.name} does not match its published hash")
            fs.replace(folder / asset.name, data)
        except OSError as e:
            raise errors.NaiError(errors.SAVE_FAILED, f"cannot save {asset.name}: {e}") from e
    return {"ok": True, **plan}
