"""Word rules for rewording settings: what -w takes, and the replacing of the texts in the settings themselves."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from nai_cli import errors, stealth, values

# The word lists that come with nai-cli; a list of the user's with the same name is used in its place
BUILTIN = {"discord": {"loli": "1011"}}


class Kind(Enum):
    TEXT = "text"
    REGEX = "regex"


@dataclass(frozen=True)
class Rule:
    """A text to replace and what replaces it."""

    old: str
    new: str
    pattern: re.Pattern[str]
    kind: Kind

    def apply(self, text: str) -> tuple[str, int]:
        """text with the rule applied, and how often it matched. The new text of a plain rule is taken as it is;
        that of a regular expression may name its groups."""
        if self.kind is Kind.REGEX:
            return self.pattern.subn(self.new, text)
        return self.pattern.subn(lambda match: self.new, text)


def parse_list(text: str) -> dict[str, str]:
    """A word list: a rule OLD=NEW on each line, with blank lines and lines starting with # left out."""
    rules: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            old, new = split(line)
            rules[old.strip()] = new.strip()
    return rules


def split(spec: str) -> tuple[str, str]:
    """A rule as it is written, into what it replaces and what it puts in its place: OLD=NEW, or /REGEX/=NEW. A rule
    that starts with a slash is a regular expression, which ends at the first /=, so that it may hold an = itself; one
    that starts with a slash and has no /= is cut at its first = like any other."""
    regex = spec.startswith("/") and "/=" in spec
    old, _, new = spec.partition("/=" if regex else "=")
    return old + "/" if regex else old, new


def compile(rules: Mapping[str, str]) -> tuple[Rule, ...]:
    """The rules in the order given. OLD matches part of a text whatever its case, or is a regular expression
    between slashes; bad_args for an OLD that is empty or does not compile."""
    out = []
    for old, new in rules.items():
        if not old:
            raise errors.NaiError(errors.BAD_ARGS, "a rule needs a text to replace, as in OLD=NEW")
        regex = len(old) > 2 and old.startswith("/") and old.endswith("/")
        try:
            pattern = re.compile(old[1:-1] if regex else re.escape(old), re.IGNORECASE)
        except re.error as e:
            raise errors.NaiError(errors.BAD_ARGS, f"{old} is not a regular expression: {e}") from e
        out.append(Rule(old, new, pattern, Kind.REGEX if regex else Kind.TEXT))
    return tuple(out)


def substitute(value: Any, rules: tuple[Rule, ...]) -> tuple[Any, dict[str, int]]:
    """value with every rule applied in turn to each text in it, inside lists and dicts too, and how often each
    rule's text was replaced."""
    found: dict[str, int] = {}
    return _reworded(value, rules, found), found


def reword(settings: dict[str, str], rules: tuple[Rule, ...]) -> tuple[dict[str, str], dict[str, int]]:
    """NovelAI's settings with the rules applied to every text in them, Comment opened to reach the prompts inside,
    and without their signature, which any change undoes; and how often each rule matched. Only the fields NovelAI
    writes are kept."""
    found: dict[str, int] = {}
    out: dict[str, str] = {}
    for field in ("Title", *stealth.HIDDEN):
        if field not in settings:
            continue
        if field == "Comment":
            comment = _reworded(values.parse_json(settings[field]), rules, found)
            comment.pop("signed_hash", None)
            out[field] = json.dumps(comment, ensure_ascii=False)
        else:
            out[field] = _reworded(settings[field], rules, found)
    return out, found


def _reworded(value: Any, rules: tuple[Rule, ...], found: dict[str, int]) -> Any:
    if isinstance(value, str):
        for rule in rules:
            value, count = rule.apply(value)
            if count:
                found[rule.old] = found.get(rule.old, 0) + count
        return value
    if isinstance(value, dict):
        return {key: _reworded(item, rules, found) for key, item in value.items()}
    if isinstance(value, list):
        return [_reworded(item, rules, found) for item in value]
    return value
