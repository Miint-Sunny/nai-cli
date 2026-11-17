"""config: show the settings in use, or write the config template."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nai_cli import auth, config, errors, fs
from nai_cli.commands.context import Context


@dataclass(frozen=True)
class ConfigSettings:
    path: Path
    init: bool
    force: bool
    dry_run: bool


def resolve(opts: dict[str, Any], ctx: Context) -> ConfigSettings:
    if opts.get("force") and opts["action"] != "init":
        raise errors.NaiError(errors.BAD_ARGS, "--force only applies to config init")
    return ConfigSettings(ctx.shared.config_path, opts["action"] == "init", bool(opts.get("force")), ctx.dry_run)


def run(s: ConfigSettings, ctx: Context) -> dict[str, Any]:
    if s.init:
        if s.dry_run:
            return {"ok": True, "dry_run": True, "path": str(s.path), "exists": fs.exists(s.path)}
        try:
            written = config.init(s.path, s.force)
        except OSError as e:
            raise errors.NaiError(errors.BAD_ARGS, f"cannot write {s.path}: {e}") from e
        ctx.forget_config()
        return {"ok": True, "path": str(s.path), "created": written}
    cfg = ctx.config
    return {
        "ok": True,
        "path": str(cfg.path),
        "exists": fs.exists(cfg.path),
        "novelai_token": bool(cfg.novelai_token),
        "token_source": cfg.token_source,
        "output_dir": str(cfg.output_dir),
        "filename": cfg.filename,
        "folder": cfg.folder,
        "max_per_request": cfg.max_per_request,
        "batch_interval": list(cfg.interval),
        "defaults": cfg.defaults,
        "account": ctx.account_name,
        "accounts": auth.listing(cfg),
        "local": {
            "tagger": str(cfg.local.tagger) if cfg.local.tagger else None,
            "tagger_labels": str(cfg.local.tagger_labels) if cfg.local.tagger_labels else None,
            "tagger_kind": cfg.local.tagger_kind or None,
            "sam_dir": str(cfg.local.sam_dir),
        },
        "data_dir": str(config.data_dir()),
    }
