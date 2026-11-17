"""Find common prompt mistakes. Reports only; nothing is changed."""

import re
from dataclasses import dataclass

from nai_cli import presets
from nai_cli import prompt as P
from nai_cli.models import Model
from nai_cli.tokenizer import Tokenizer

_SD_WEIGHT = re.compile(r"[(\[]([^()\[\]:]+):\s*([0-9]*\.?[0-9]+)[)\]]")
_FULLWIDTH = re.compile(r"[，、：；（）]")


@dataclass
class Issue:
    level: str
    code: str
    message: str
    field: str
    snippet: str = ""

    def to_json(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v != ""}


def check_weights(text: str, field: str) -> list[Issue]:
    """Numeric weights n::...:: must pair up, with the number right before the opening :: and none before the closing one."""
    issues = [_weight_issue(m, text, field) for m in P.weight_marks(text) if m.kind not in ("open", "close")]
    depth = {"{": 0, "[": 0}
    pairs = {"}": "{", "]": "["}
    for ch in text:
        if ch in depth:
            depth[ch] += 1
        elif ch in pairs:
            depth[pairs[ch]] -= 1
    for opener, n in depth.items():
        if n:
            closer = "}" if opener == "{" else "]"
            issues.append(
                Issue("warning", "bracket_unbalanced", f"unbalanced {opener}{closer} (off by {abs(n)})", field)
            )
    return issues


def _weight_issue(mark: P.WeightMark, text: str, field: str) -> Issue:
    n = mark.number
    snippet = _context(text, mark.start, mark.end)
    if mark.kind == "numeric_close":
        message = (
            f"{n} right before a closing :: starts a new weight, so every following tag gets {n}x; "
            f"write `{n} ::` or use --space-numeric-close"
        )
        return Issue("warning", "weight_numeric_close", message, field, snippet)
    if mark.kind == "switch":
        return Issue(
            "warning",
            "weight_unclosed",
            f"switched to {n}:: before closing the previous weight with ::",
            field,
            snippet,
        )
    if mark.kind == "unclosed":
        return Issue(
            "warning",
            "weight_unclosed",
            "numeric weight is never closed with ::",
            field,
            _context(text, mark.start, mark.end + 30),
        )
    if mark.kind == "spaced":
        return Issue(
            "warning",
            "weight_spaced",
            f"`{n} ::` has a space before ::, so the weight is ignored; write `{n}::`",
            field,
            snippet,
        )
    if mark.kind == "number_after":
        return Issue(
            "warning", "weight_number_after", f"the number goes before ::, as `{n}::`, not `::{n}`", field, snippet
        )
    extra = f"; the {n} right before it is read as a weight" if n else ""
    return Issue("warning", "weight_stray_close", f"stray ::{extra}", field, snippet)


def check_text(text: str, field: str) -> list[Issue]:
    issues: list[Issue] = []
    if not text:
        return issues
    literal = P.literal_ranges(text)
    for m in _SD_WEIGHT.finditer(text):
        if _inside(literal, m.start()):
            continue
        issues.append(
            Issue(
                "error",
                "sd_weight",
                f"{m.group(0)} is Stable Diffusion syntax that NovelAI ignores; use {m.group(2)}::{m.group(1).strip()}::",
                field,
                m.group(0),
            )
        )
    for m in _FULLWIDTH.finditer(text):
        if _inside(literal, m.start()) or not _beside_latin(text, m.start(), m.end()):
            continue
        issues.append(
            Issue(
                "warning",
                "fullwidth_punct",
                f"full-width {m.group(0)}; separate tags with an ASCII comma",
                field,
                _context(text, m.start(), m.end()),
            )
        )
    issues += check_weights(text, field)
    tags = [t.strip().lower() for t in re.split(r",|\n", text) if t.strip()]
    seen: dict[str, int] = {}
    for t in tags:
        bare = re.sub(r"^-?[0-9.]+::|::$|[{}\[\]]", "", t).strip().replace("_", " ")
        if bare and not re.fullmatch(r"</?[\w-]+>", bare):
            seen[bare] = seen.get(bare, 0) + 1
    for tag, n in seen.items():
        if n > 1 and len(tag) > 2:
            issues.append(Issue("info", "duplicate_tag", f"{tag} appears {n} times", field))
    return issues


def _context(text: str, start: int, end: int, pad: int = 20) -> str:
    return text[max(0, start - pad) : min(len(text), end + pad)].replace("\n", " ")


def _inside(ranges: list[tuple[int, int]], index: int) -> bool:
    return any(start <= index < end for start, end in ranges)


def _beside_latin(text: str, start: int, end: int) -> bool:
    # Full-width punctuation is normal in CJK sentences; only flag it next to Latin tags
    left = text[:start].rstrip(" \t")
    right = text[end:].lstrip(" \t")
    return any(c.isascii() and c.isalnum() for c in (left[-1:], right[:1]) if c)


def check_quality_overlap(user_prompt: str, model: Model, quality: str) -> list[Issue]:
    if quality == "none":
        return []
    auto = [t.strip() for t in presets.quality_tags(model, quality).split(",")]
    lowered = user_prompt.lower()
    dup = [w for w in auto if re.search(rf"(?<![\w]){re.escape(w)}(?![\w])", lowered)]
    if not dup:
        return []
    return [
        Issue(
            "info",
            "quality_duplicated",
            f"the prompt already contains {', '.join(dup)}, which quality tags add again; set --quality none to control them yourself",
            "prompt",
        )
    ]


def check_text_marker_prefix(user_prompt: str, quality: str, transparent: bool) -> list[Issue]:
    """Quality tags go before the first text:, so a prompt starting with text: becomes "...no texttext:"."""
    if quality == "none" and not transparent:
        return []
    match = P.TEXT_MARKER.search(P.split_mix_chunks(user_prompt)[0])
    if match is None or match.start() != 0 or match.group(0)[0].lower() != "t":
        return []
    return [
        Issue(
            "warning",
            "text_marker_prefix",
            'the prompt starts with text:, so quality tags are glued on as "...no texttext:"; put a tag first or set --quality none',
            "prompt",
            user_prompt[:30],
        )
    ]


def check_uc_preset_overlap(negative: str, model: Model, uc_preset: str, verbatim: bool) -> list[Issue]:
    if verbatim or uc_preset == "none":
        return []
    preset_text = presets.uc_preset_text(model, uc_preset)
    if not preset_text or preset_text not in negative:
        return []
    return [
        Issue(
            "info",
            "uc_preset_duplicated",
            f"the negative prompt already contains the {uc_preset} UC preset, which --uc-preset adds again; set --uc-preset none to control it yourself",
            "negative",
        )
    ]


def estimate_tokens(text: str) -> int:
    """Rough estimate for the Qwen tokenizer: about 1.3 tokens per English word and 1 per CJK character."""
    cjk = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", text))
    words = len(re.findall(r"[A-Za-z0-9]+", text))
    punct = len(re.findall(r"[,:{}\[\]()|]", text))
    return int(words * 1.3 + cjk + punct * 0.5)


@dataclass(frozen=True)
class TokenCount:
    prompt: int
    negative: int
    limit: int
    # False when the tokenizer file could not be loaded and the counts are estimates
    exact: bool

    def to_json(self) -> dict:
        return {"prompt": self.prompt, "negative": self.negative, "limit": self.limit, "exact": self.exact}


def count_tokens(
    model: Model, tok: Tokenizer | None, prompt: str, negative: str, characters: list[tuple[str, str]]
) -> TokenCount:
    """Counted as the official prompt boxes count: every mix segment separately, characters added to the base prompt;
    estimated without a tokenizer."""
    measure = (lambda text: len(tok.encode(text))) if tok is not None else estimate_tokens

    def field(text: str) -> int:
        return sum(measure(segment) for segment in P.split_mix_chunks(P.longest_alternatives(text)))

    return TokenCount(
        prompt=field(prompt) + sum(field(c) for c, _ in characters),
        negative=field(negative) + sum(field(n) for _, n in characters),
        limit=model.token_limit,
        exact=tok is not None,
    )


def check_tokens(count: TokenCount, model: Model) -> list[Issue]:
    about = "" if count.exact else "about "
    issues = []
    for field, n in (("prompt", count.prompt), ("negative", count.negative)):
        if n > count.limit:
            issues.append(
                Issue(
                    "error",
                    "token_limit",
                    f"{about}{n} tokens, over the {model.name} limit of {count.limit}; the rest is cut off",
                    field,
                )
            )
        elif n > count.limit * 0.9:
            issues.append(
                Issue(
                    "warning",
                    "token_near_limit",
                    f"{about}{n} tokens, close to the {model.name} limit of {count.limit}",
                    field,
                )
            )
    return issues


def check_text_length(prompt: str, model: Model, field: str = "prompt") -> list[Issue]:
    """The official documentation gives how many characters of text each model renders, spaces and newlines
    included; only the whitespace separating the text from text: is left out."""
    if model.text_limit is None:
        return []
    for segment in P.split_mix_chunks(prompt):
        match = P.TEXT_MARKER.search(segment)
        length = len(segment[match.end() :].lstrip()) if match else 0
        if length > model.text_limit:
            return [
                Issue(
                    "warning",
                    "text_too_long",
                    f"the text after text: is {length} characters; {model.name} renders about {model.text_limit} reliably",
                    field,
                )
            ]
    return []


def run(
    model: Model,
    prompt: str,
    negative: str = "",
    characters: list[tuple[str, str]] | None = None,
    quality: str = "standard",
    effective_prompt: str | None = None,
    uc_preset: str = "none",
    transparent: bool = False,
    negative_verbatim: bool = False,
    tokens: TokenCount | None = None,
) -> list[Issue]:
    issues = check_text(prompt, "prompt") + check_text(negative, "negative")
    for i, (cp, cn) in enumerate(characters or [], start=1):
        issues += check_text(cp, f"character{i}") + check_text(cn, f"character{i}_negative")
    issues += check_quality_overlap(prompt, model, quality)
    issues += check_text_marker_prefix(prompt, quality, transparent and model.transparent)
    issues += check_uc_preset_overlap(negative, model, uc_preset, negative_verbatim)
    issues += check_text_length(effective_prompt or prompt, model)
    for i, (cp, _) in enumerate(characters or [], start=1):
        issues += check_text_length(cp, model, f"character{i}")
    if tokens is not None:
        issues += check_tokens(tokens, model)
    return issues
