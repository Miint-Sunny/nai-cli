"""lint: check prompts as gen would send them, without sending anything."""

from typing import Any

from nai_cli.commands import generation
from nai_cli.commands.context import Context
from nai_cli.commands.generation import GenSettings
from nai_cli.commands.spec import prepare


def resolve(opts: dict[str, Any], ctx: Context) -> GenSettings:
    return generation.resolve(opts, ctx, "lint")


def run(s: GenSettings, ctx: Context) -> dict[str, Any]:
    prep = prepare(s)
    return {
        "ok": True,
        "issues": [i.to_json() for i in prep.issues],
        "tokens": prep.tokens.to_json(),
        "prompt": prep.built.prompt,
        "negative": prep.built.negative,
        "warnings": prep.warnings,
    }
