"""Token counts with the tokenizer files the official client downloads: Qwen3.5 BPE for V5, T5 Unigram for V4.5."""

import itertools
import re
import struct
import unicodedata
from collections.abc import Callable, Iterator
from typing import Any, Protocol

import regex

from nai_cli import decode, values
from nai_cli import prompt as P

FILES = {"qwen": "qwen35_tokenizer.def", "t5": "t5_tokenizer.def"}
# Part of the official download URL; a new version means new files
VERSION = 2

_JS_SPACE = re.compile("[" + re.escape(P.JS_SPACE) + "]+")
_T5_BRACKETS = re.compile(r"[\[\]{}]")
_T5_WEIGHTS = re.compile(r"-?[0-9]*\.?[0-9]*::")
_LONE_SURROGATE = re.compile(r"[\ud800-\udfff]")
# What a damaged or foreign file raises while parsing, besides ValueError
_BAD_FILE = (LookupError, TypeError, AttributeError, ValueError, regex.error)


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


def _utf16_units(text: str) -> str:
    """Astral characters become surrogate pairs, since JavaScript walks strings by UTF-16 code unit."""
    data = text.encode("utf-16-le", "surrogatepass")
    return "".join(map(chr, struct.unpack(f"<{len(data) // 2}H", data)))


class _Node:
    __slots__ = ("prev", "token_id", "total")

    def __init__(self, token_id: int, total: float, prev: "_Node | None") -> None:
        self.token_id = token_id
        self.total = total
        self.prev = prev


def _best_after(candidates: list[_Node], token_id: int, score: float) -> _Node:
    best, total = None, 0.0
    for node in candidates:
        value = node.total + score
        # The first of equal scores wins
        if best is None or value > total:
            best, total = node, value
    return _Node(token_id, total, best)


def _pre_tokenizer(config: dict[str, Any]) -> Callable[[list[str]], list[str]]:
    kind = config["type"]
    if kind == "Sequence":
        steps = [_pre_tokenizer(c) for c in config["pretokenizers"]]

        def sequence(words: list[str]) -> list[str]:
            for step in steps:
                words = step(words)
            return words

        return sequence
    if kind == "WhitespaceSplit":
        return lambda words: [part for word in words for part in _JS_SPACE.split(word)]
    if kind == "Metaspace":
        replacement = config["replacement"]
        add_prefix = config["add_prefix_space"]
        prefix = config.get("str_rep") or replacement
        if not isinstance(replacement, str) or not isinstance(prefix, str):
            raise ValueError("Metaspace replacement must be text")

        def metaspace(words: list[str]) -> list[str]:
            out = []
            for word in words:
                # JavaScript's replace with a string pattern changes only the first space
                word = word.replace(" ", prefix, 1)
                out.append(prefix + word if add_prefix and not word.startswith(replacement) else word)
            return out

        return metaspace
    raise ValueError(f"unsupported pre-tokenizer: {kind}")


class T5:
    """SentencePiece Unigram: the best-scoring split by Viterbi over every vocabulary piece."""

    def __init__(self, config: dict[str, Any]) -> None:
        model = config["model"]
        if model["type"] != "Unigram" or config["normalizer"]["type"] != "Precompiled":
            raise ValueError("unsupported T5 tokenizer")
        pieces = [piece for piece, _ in model["vocab"]]
        self.scores = [float(score) for _, score in model["vocab"]]
        self.unk_id: int = model["unk_id"]
        if not values.is_whole(self.unk_id) or not 0 <= self.unk_id < len(pieces):
            raise ValueError(f"unk_id {self.unk_id!r} is not in the vocabulary")
        # The official client starts the minimum at 1e6 and scores unknown pieces 10 below it
        self.unk_score = min([1e6, *self.scores]) - 10
        self.scores[self.unk_id] = self.unk_score
        # Later duplicates win, as in the official Map
        self.ids = {piece: i for i, piece in enumerate(pieces)}
        self.eos_id = self.ids["</s>"]
        self.trie: dict[str, Any] = {}
        for piece in pieces:
            node = self.trie
            for ch in piece:
                node = node.setdefault(ch, {})
            node[""] = True
        self.pre_tokenize = _pre_tokenizer(config["pre_tokenizer"])

    def encode(self, text: str) -> list[int]:
        if not text:
            return [self.eos_id]
        text = _T5_WEIGHTS.sub("", _T5_BRACKETS.sub("", text))
        ids: list[int] = []
        for word in self.pre_tokenize([text]):
            ids += self._viterbi(_utf16_units(word))
        ids.append(self.eos_id)
        return ids

    def _pieces(self, units: str, start: int) -> Iterator[tuple[int, int, float]]:
        # Pieces are keyed by code point but matched by code unit, so astral characters never match
        node = self.trie
        single = False
        for end in range(start, len(units)):
            node = node.get(units[end])
            if node is None:
                break
            if "" in node:
                token_id = self.ids[units[start : end + 1]]
                single = single or end == start
                yield end + 1 - start, token_id, self.scores[token_id]
        if not single:
            yield 1, self.unk_id, self.unk_score

    def _viterbi(self, units: str) -> list[int]:
        ending: list[list[_Node]] = [[] for _ in range(len(units) + 1)]
        ending[0].append(_Node(-1, 0.0, None))
        for start in range(len(units)):
            for length, token_id, score in self._pieces(units, start):
                ending[start + length].append(_best_after(ending[start], token_id, score))
        ids = []
        node = _best_after(ending[len(units)], -1, 0.0).prev
        while node is not None and node.prev is not None:
            ids.append(node.token_id)
            node = node.prev
        ids.reverse()
        return ids


def _byte_chars() -> list[str]:
    """GPT-2's printable stand-ins for the 256 byte values."""
    printable = [*range(0x21, 0x7F), *range(0xA1, 0xAD), *range(0xAE, 0x100)]
    chars = {b: chr(b) for b in printable}
    for b in range(256):
        if b not in chars:
            chars[b] = chr(256 + len(chars) - len(printable))
    return [chars[b] for b in range(256)]


class _SpecialNode:
    __slots__ = ("children", "value")

    def __init__(self) -> None:
        self.children: dict[str, _SpecialNode] = {}
        self.value: str | None = None


class Qwen:
    """Byte-level BPE after splitting off special tokens and splitting the rest with the official pattern."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.encoder: dict[str, int] = config["vocab"]
        # The official client switches to a raw byte mode for such vocabularies, which is not ported
        if any(token.startswith("0x") for token in self.encoder):
            raise ValueError("byte-token vocabularies are not supported")
        self.ranks = {(left, right): rank for rank, (left, right) in enumerate(config["merges"])}
        settings = config.get("config") or {}
        self.normalization: str | None = settings.get("normalization")
        self.max_chars: int | None = settings.get("maxEncodeChars")
        if self.normalization not in (None, "", "NFC", "NFD", "NFKC", "NFKD"):
            raise ValueError("unknown Qwen normalization")
        if self.max_chars is not None and (not values.is_whole(self.max_chars) or self.max_chars < 0):
            raise ValueError("maxEncodeChars must be a non-negative integer")
        self.ignore_merges = bool(settings.get("ignoreMerges"))
        self.pattern = regex.compile(settings["splitRegex"])
        self.specials = {t: self.encoder[t] for t in config["specialTokens"] if t in self.encoder}
        # Specials missing from the vocabulary are still split off, then encoded as text
        self.special_root = _SpecialNode()
        for token in config["specialTokens"]:
            node = self.special_root
            for ch in token:
                node = node.children.setdefault(ch, _SpecialNode())
            node.value = token
        self.byte_chars = _byte_chars()
        self.cache: dict[str, list[int]] = {}

    def encode(self, text: str) -> list[int]:
        if self.normalization:
            text = unicodedata.normalize(self.normalization, text)
        chars = list(text)
        if self.max_chars and len(chars) > self.max_chars:
            chunks = ["".join(chars[i : i + self.max_chars]) for i in range(0, len(chars), self.max_chars)]
        else:
            chunks = [text]
        ids: list[int] = []
        for chunk in chunks:
            for word in self._split(chunk):
                special = self.specials.get(word)
                if special is not None:
                    ids.append(special)
                else:
                    ids += self._bpe(self._to_unicode(word))
        return ids

    def _split(self, text: str) -> list[str]:
        """Longest special-token matches, backtracking after a partial match, scanned as the official client does."""
        words: list[str] = []
        pending = ""
        candidate = ""
        node = self.special_root
        matched: _SpecialNode | None = None

        def flush() -> None:
            nonlocal pending, candidate, node
            if pending:
                words.extend(m.group(0) for m in self.pattern.finditer(pending))
                pending = ""
            if candidate:
                words.append(candidate)
                candidate = ""
                node = self.special_root

        i = 0
        while i < len(text):
            ch = text[i]
            child = node.children.get(ch)
            if child is not None:
                node = child
                candidate += ch
                i += 1
            elif not candidate:
                pending += ch
                i += 1
            elif matched is not None and matched.value:
                rest = candidate[len(matched.value) :]
                candidate = matched.value
                i -= len(rest)
                matched = None
                flush()
            else:
                pending += candidate[0]
                i -= len(candidate) - 1
                candidate = ""
                node = self.special_root
            if node.value and candidate == node.value:
                matched = node
        if candidate:
            if matched is not None and matched.value:
                rest = candidate[len(matched.value) :]
                candidate = matched.value
                flush()
                pending = rest
            else:
                pending += candidate
                candidate = ""
        flush()
        return words

    def _to_unicode(self, word: str) -> str:
        # TextEncoder writes lone surrogates as U+FFFD
        return "".join(self.byte_chars[b] for b in _LONE_SURROGATE.sub("\ufffd", word).encode("utf-8"))

    def _bpe(self, word: str) -> list[int]:
        if self.ignore_merges and word in self.encoder:
            return [self.encoder[word]]
        cached = self.cache.get(word)
        if cached is not None:
            return cached
        parts = list(word)
        while len(parts) > 1:
            pair = min(itertools.pairwise(parts), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            parts = _merge(parts, *pair)
        ids = [self.encoder[p] for p in parts if p in self.encoder]
        self.cache[word] = ids
        return ids


def _merge(parts: list[str], left: str, right: str) -> list[str]:
    merged = []
    i = 0
    while i < len(parts):
        if parts[i] == left and i + 1 < len(parts) and parts[i + 1] == right:
            merged.append(left + right)
            i += 2
        else:
            merged.append(parts[i])
            i += 1
    return merged


def parse(kind: str, raw: bytes) -> Tokenizer:
    """Raises ValueError for anything that is not a usable tokenizer file."""
    try:
        config = values.parse_json(decode.inflate(raw, decode.RAW))
        return T5(config) if kind == "t5" else Qwen(config)
    except _BAD_FILE as e:
        raise ValueError(f"not a {kind} tokenizer file: {type(e).__name__}: {e}") from e
