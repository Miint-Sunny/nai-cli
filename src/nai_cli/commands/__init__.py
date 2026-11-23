"""Every command as a pair: resolve options into settings, then run the settings."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from nai_cli.commands import (
    account,
    batch,
    config,
    cost,
    fake,
    generate,
    generation,
    grid,
    image_tools,
    interrogate,
    lint,
    local,
    meta,
    models,
    redact,
    reword,
    stats,
    strip,
    tags,
    vibe,
    watermark,
)
from nai_cli.commands.context import Context


@dataclass(frozen=True)
class Command:
    resolve: Callable[[dict[str, Any], Context], Any]
    run: Callable[[Any, Context], dict[str, Any]]


COMMANDS = {
    **{
        name: Command(partial(generation.resolve, command=name), generate.run)
        for name in ("gen", "i2i", "inpaint", "enhance", "variations")
    },
    "upscale": Command(image_tools.resolve_upscale, image_tools.run_upscale),
    "tool": Command(image_tools.resolve_tool, image_tools.run_tool),
    "vibe": Command(vibe.resolve, vibe.run),
    "strip": Command(strip.resolve, strip.run),
    "redact": Command(redact.resolve, redact.run),
    "reword": Command(reword.resolve, reword.run),
    "fake": Command(fake.resolve, fake.run),
    "watermark": Command(watermark.resolve, watermark.run),
    "batch": Command(batch.resolve, batch.run),
    "grid": Command(grid.resolve, grid.run),
    "meta": Command(meta.resolve, meta.run),
    "lint": Command(lint.resolve, lint.run),
    "tags": Command(tags.resolve, tags.run),
    "models": Command(models.resolve, models.run),
    "cost": Command(cost.resolve, cost.run),
    "account": Command(account.resolve, account.run),
    "stats": Command(stats.resolve, stats.run),
    "interrogate": Command(interrogate.resolve, interrogate.run),
    "local": Command(local.resolve, local.run),
    "config": Command(config.resolve, config.run),
}
