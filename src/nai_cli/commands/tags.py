"""tags: tag suggestions the way the official prompt box makes them."""

import time
from dataclasses import dataclass
from typing import Any

from nai_cli import models, tags
from nai_cli.commands.context import Context


@dataclass(frozen=True)
class TagsSettings:
    model: models.Model
    furry: bool
    queries: tuple[str, ...]


def resolve(opts: dict[str, Any], ctx: Context) -> TagsSettings:
    model = models.resolve(opts.get("model") or ctx.config.defaults.get("model"))
    return TagsSettings(model, bool(opts.get("furry")), tuple(opts["query"]))


def run(s: TagsSettings, ctx: Context) -> dict[str, Any]:
    lookups = [tags.lookup(q, s.model, s.furry) for q in s.queries]
    client = ctx.client()
    results = []
    for i, item in enumerate(lookups):
        if i:
            time.sleep(tags.PAUSE)
        found = client.suggest_tags(item.text, item.model, item.lexicon, item.lang)
        results.append({"query": item.query, "lang": item.lang or "en", "tags": found})
    return {"ok": True, "model": s.model.id, "furry": s.furry, "results": results}
