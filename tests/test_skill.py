"""The agent skill must stay valid under the Agent Skills format and in step with the command line."""

import argparse
import re
from pathlib import Path

import pytest

from nai_cli import parser as nai_parser

SKILL = Path(__file__).resolve().parents[1] / "skills" / "nai-cli"
DOCS = sorted(SKILL.rglob("*.md"))


def _frontmatter(text: str) -> dict[str, str]:
    head = text.split("---\n")[1]
    return dict(line.split(": ", 1) for line in head.strip().splitlines())


def _vocabulary() -> tuple[set[str], set[str]]:
    parser = nai_parser.build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    parsers = [parser, *sub.choices.values()]
    return set(sub.choices), {flag for p in parsers for a in p._actions for flag in a.option_strings}


def test_frontmatter():
    meta = _frontmatter((SKILL / "SKILL.md").read_text(encoding="utf-8"))
    assert meta["name"] == SKILL.name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", meta["name"]) and len(meta["name"]) <= 64
    assert 0 < len(meta["description"]) <= 1024
    assert len(meta.get("compatibility", "")) <= 500


@pytest.mark.parametrize("doc", DOCS, ids=[d.name for d in DOCS])
def test_mentions_existing_commands_and_flags(doc):
    commands, flags = _vocabulary()
    text = doc.read_text(encoding="utf-8")
    assert set(re.findall(r"(?:^|`)nai ([a-z0-9]+)", text, re.MULTILINE)) <= commands
    assert set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", text)) <= flags


@pytest.mark.parametrize("doc", DOCS, ids=[d.name for d in DOCS])
def test_links_resolve(doc):
    for target in re.findall(r"\]\(([^)#]+)\)", doc.read_text(encoding="utf-8")):
        if not target.startswith("http"):
            assert (doc.parent / target).exists(), target
