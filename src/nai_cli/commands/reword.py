"""reword: copies of images with words in their settings replaced, for sharing where some words are acted upon."""

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from nai_cli import config, errors, fs, meta, words
from nai_cli.commands import copies
from nai_cli.commands.context import Context
from nai_cli.commands.sources import Source


class Skipped(Enum):
    """Why an image gets no copy."""

    NO_SETTINGS = "no_settings"
    NO_MATCH = "no_match"


@dataclass(frozen=True)
class RewordSettings:
    targets: copies.Targets
    rules: tuple[words.Rule, ...]


def resolve(opts: dict[str, Any], ctx: Context) -> RewordSettings:
    return RewordSettings(copies.resolve(opts, ctx), words.compile(_rules(opts["words"])))


def run(s: RewordSettings, ctx: Context) -> dict[str, Any]:
    warnings: list[str] = []
    prepared, skipped = _prepared(s.targets.sources, s.rules)
    named = [_named(source, s.targets, s.rules) for source, _, _ in prepared]
    # Keyed by the image itself, not by the name of its copy: the rules may give the copies of two images one name
    made = {source.path: settings for source, settings, _ in prepared}

    def data(path: Path) -> bytes:
        return copies.with_settings(path, lambda size: made[path])

    targets = replace(s.targets, sources=tuple(named))
    out = copies.write(targets, "reworded", data, warnings)
    for entry, (_, _, replaced) in zip(out["files"], prepared, strict=True):
        entry["replaced"] = replaced
    return {**out, "skipped": skipped, "warnings": out.get("warnings", [])}


def _rules(given: list[str]) -> dict[str, str]:
    """The rules -w names, a later one replacing an earlier one for the same text: OLD=NEW, /REGEX/=NEW, or the
    name of a word list, the user's in the words folder of the config directory, or else one nai-cli brings."""
    rules: dict[str, str] = {}
    for spec in given:
        if "=" in spec:
            old, new = words.split(spec)
            rules[old.strip()] = new.strip()
            continue
        name = spec.strip()
        path = config.config_dir() / "words" / f"{name}.txt"
        if fs.is_file(path):
            rules.update(words.parse_list(path.read_text(encoding="utf-8")))
        elif name in words.BUILTIN:
            rules.update(words.BUILTIN[name])
        else:
            raise errors.NaiError(
                errors.BAD_ARGS, f"no word list named {name}: a list is a .txt file of rules in {path.parent}"
            )
    return rules


def _prepared(
    sources: tuple[Source, ...], rules: tuple[words.Rule, ...]
) -> tuple[list[tuple[Source, dict[str, str], dict[str, int]]], list[dict[str, str]]]:
    """For each image that gets a copy, the source, its settings with the rules applied and how often each matched;
    and, for each that gets none, the source and why: it holds no NovelAI settings, or no rule matched."""
    prepared, skipped = [], []
    for source in sources:
        try:
            raw = meta.read_raw(source.path)
            meta.parse_raw(raw)
        except errors.NaiError as e:
            if e.code != errors.NO_METADATA:
                raise
            skipped.append({"source": str(source.path), "reason": Skipped.NO_SETTINGS.value})
            continue
        settings, replaced = words.reword(raw, rules)
        if replaced:
            prepared.append((source, settings, replaced))
        else:
            skipped.append({"source": str(source.path), "reason": Skipped.NO_MATCH.value})
    return prepared, skipped


def _named(source: Source, t: copies.Targets, rules: tuple[words.Rule, ...]) -> Source:
    """The source as its copy is named after it: with the rules applied to its name too, as NovelAI's download
    names hold the start of the prompt. With --rename the name is the day's and a number instead."""
    if t.rename:
        return source
    stem = words.substitute(source.path.stem, rules)[0]
    try:
        # with_name refuses what no file name is: a stem that is empty or holds a separator
        source.path.with_name(stem + source.path.suffix)
    except ValueError as e:
        raise errors.NaiError(errors.BAD_ARGS, f"the rules leave {source.path.name} without a file name: {e}") from e
    return replace(source, copy_stem=stem)
