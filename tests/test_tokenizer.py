import contextlib
import json
import zlib

import httpx
import pytest

from nai_cli import api, caches, fs, lint, models, tokenizer

SPACE = "\N{LOWER ONE EIGHTH BLOCK}"
# Read before the test fixture redirects the data folder and blocks downloads
REAL_FILES = {kind: caches.tokenizer_path(kind) for kind in tokenizer.FILES}
REAL_FETCH = api.fetch_tokenizer
# Ids the official client's tokenizer code returns for these prompts
GOLDEN: dict[str, tuple[list[int], list[int]]] = {
    "1girl, solo, blue hair, looking at viewer": (
        [209, 18722, 6, 6729, 6, 1692, 1268, 6, 479, 44, 17831, 1],
        [16, 27620, 11, 13138, 11, 6105, 6655, 11, 3221, 506, 24890],
    ),
    "masterpiece, best quality, {{blue eyes}}, [[smile]], 1.5::red dress::, -1::hat::": (
        [20975, 6, 200, 463, 6, 1692, 2053, 6, 3993, 6, 1131, 3270, 6, 3, 547, 1],
        [
            13236,
            21666,
            11,
            1786,
            4131,
            11,
            5685,
            11855,
            6213,
            36892,
            4177,
            3445,
            448,
            19867,
            220,
            16,
            13,
            20,
            476,
            1114,
            8250,
            476,
            11,
            471,
            16,
            476,
            8925,
            476,
        ],
    ),
    "  multiple   spaces\tand\nnewlines  ": (
        [3, 1317, 4856, 11, 126, 6972, 3, 1],
        [220, 5081, 256, 12258, 50711, 198, 902, 7718, 256],
    ),
    "日本語のプロンプト、青い髪": (
        [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
        [247359, 202988, 152474, 19073, 5205, 96556, 16098, 209645],
    ),
    "蓝色头发，看着观众": (
        [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
        [104145, 108891, 3709, 98844, 100687],
    ),
    "emoji 😀👍🏽 test": (
        [3, 15, 51, 21892, 3, 2, 2, 2, 2, 2, 2, 794, 1],
        [36280, 87209, 9008, 239, 235, 9008, 237, 121, 1228],
    ),
    "<|endoftext|> special <|im_start|>": (
        [3, 2, 9175, 989, 858, 6327, 9175, 3155, 534, 3, 2, 9175, 603, 834, 10208, 9175, 3155, 1],
        [248044, 3175, 220, 248045],
    ),
    "fur dataset, wolf, anthro": (
        [4223, 17953, 6, 3, 19747, 6, 46, 8514, 1],
        [80039, 10031, 11, 35336, 11, 21577, 298],
    ),
    "-2::monochrome::, 0.5::sketch::": (
        [7414, 10363, 526, 6, 13278, 1],
        [12, 17, 476, 1588, 88558, 476, 11, 220, 15, 13, 20, 476, 4729, 2900, 476],
    ),
    "AAAA bbbb CcCc": (
        [22656, 188, 3, 115, 115, 115, 115, 205, 75, 254, 75, 1],
        [24881, 292, 51365, 351, 66, 94381],
    ),
}


def deflate(config_json: dict) -> bytes:
    packer = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return packer.compress(json.dumps(config_json).encode()) + packer.flush()


def t5_config(vocab: list[list]) -> dict:
    return {
        "normalizer": {"type": "Precompiled", "precompiled_charsmap": ""},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "WhitespaceSplit"},
                {"type": "Metaspace", "replacement": SPACE, "add_prefix_space": True},
            ],
        },
        "model": {"type": "Unigram", "unk_id": 2, "vocab": vocab},
    }


T5_VOCAB = [
    ["<pad>", 0.0],
    ["</s>", 0.0],
    ["<unk>", 0.0],
    [SPACE, -3.0],
    [SPACE + "a", -1.0],
    ["b", -2.0],
    ["ab", -1.5],
    [SPACE + "ab", -2.8],
    ["x", -1.0],
    ["y", -1.0],
    ["xy", -2.0],
]

ACUTE_E = chr(0xC3) + chr(0xA9)
QWEN_VOCAB = {
    "a": 0,
    "b": 1,
    "c": 2,
    "\N{LATIN CAPITAL LETTER G WITH DOT ABOVE}": 3,
    "ab": 4,
    "abc": 5,
    "\N{LATIN CAPITAL LETTER G WITH DOT ABOVE}a": 6,
    "<|endoftext|>": 7,
    "<": 8,
    "|": 9,
    ">": 10,
    "e": 11,
    "n": 12,
    "d": 13,
    ACUTE_E: 14,
    "|m": 15,
}


def qwen_config() -> dict:
    return {
        "config": {"splitRegex": r" ?\p{L}+|\p{N}|\s+|[^\s\p{L}\p{N}]+", "normalization": "NFC"},
        "specialTokens": ["<|endoftext|>", "<|missing|>"],
        "vocab": QWEN_VOCAB,
        "merges": [
            ["a", "b"],
            ["ab", "c"],
            ["\N{LATIN CAPITAL LETTER G WITH DOT ABOVE}", "a"],
            [chr(0xC3), chr(0xA9)],
            ["|", "m"],
        ],
    }


@pytest.mark.parametrize(
    ("text", "ids"),
    [
        ("ab", [7, 1]),
        ("a b", [4, 3, 5, 1]),
        ("", [1]),
        ("az", [4, 2, 1]),
        # A piece wins over its parts when they score the same, as it starts earlier
        ("xy", [3, 10, 1]),
        ("{a}", [4, 1]),
        ("1.5::[[ab]]::", [7, 1]),
    ],
)
def test_t5_unigram(text, ids):
    assert tokenizer.parse("t5", deflate(t5_config(T5_VOCAB))).encode(text) == ids


def test_t5_astral_characters_are_unknown_per_code_unit():
    t5 = tokenizer.parse("t5", deflate(t5_config([*T5_VOCAB, ["\N{GRINNING FACE}", -1.0]])))
    assert t5.encode("a\N{GRINNING FACE}") == [4, 2, 2, 1]


@pytest.mark.parametrize(
    ("text", "ids"),
    [
        ("abc", [5]),
        (" a", [6]),
        ("a<|endoftext|>b", [0, 7, 1]),
        # A partial special token is text
        ("<|end", [8, 9, 11, 12, 13]),
        # Not in the vocabulary, but still split off whole, so |m merges across the pattern's boundary
        ("a<|missing|>", [0, 8, 15, 12, 9, 10]),
        ("e\N{COMBINING ACUTE ACCENT}", [14]),
        ("z", []),
    ],
)
def test_qwen_bpe(text, ids):
    assert tokenizer.parse("qwen", deflate(qwen_config())).encode(text) == ids


def test_qwen_rejects_byte_token_vocabularies():
    data = qwen_config()
    data["vocab"] = {**QWEN_VOCAB, "0x41": 99}
    with pytest.raises(ValueError):
        tokenizer.parse("qwen", deflate(data))


def test_load_downloads_once_and_caches(monkeypatch):
    raw = deflate(t5_config(T5_VOCAB))
    fetched = []

    def fetch(name, version):
        fetched.append((name, version))
        return raw

    monkeypatch.setattr(api, "fetch_tokenizer", fetch)
    assert caches.load_tokenizer("t5").encode("ab") == [7, 1]
    assert caches.tokenizer_path("t5").read_bytes() == raw
    caches.load_tokenizer.cache_clear()
    assert caches.load_tokenizer("t5") is not None
    assert fetched == [("t5_tokenizer.def", tokenizer.VERSION)]


@pytest.mark.parametrize(
    ("kind", "option", "value"),
    [
        ("qwen", "normalization", "unknown"),
        ("qwen", "maxEncodeChars", "4"),
        ("qwen", "maxEncodeChars", -1),
        ("t5", "replacement", 1),
        ("t5", "str_rep", 1),
    ],
)
def test_unusable_count_options_are_refused_before_caching(monkeypatch, kind, option, value):
    good = qwen_config() if kind == "qwen" else t5_config(T5_VOCAB)
    good_bytes = deflate(good)
    settings = good["config"] if kind == "qwen" else good["pre_tokenizer"]["pretokenizers"][1]
    settings[option] = value
    bad_bytes = deflate(good)
    path = caches.tokenizer_path(kind)
    monkeypatch.setattr(api, "fetch_tokenizer", lambda *_: bad_bytes)
    assert caches.load_tokenizer(kind) is None and not path.exists()

    caches.load_tokenizer.cache_clear()
    path.parent.mkdir(parents=True)
    path.write_bytes(bad_bytes)
    monkeypatch.setattr(api, "fetch_tokenizer", lambda *_: good_bytes)
    assert caches.load_tokenizer(kind).encode("ab") == tokenizer.parse(kind, good_bytes).encode("ab")
    assert path.read_bytes() == good_bytes


def test_load_rejects_bad_downloads(monkeypatch):
    monkeypatch.setattr(api, "fetch_tokenizer", lambda name, version: b"<html>")
    assert caches.load_tokenizer("qwen") is None
    assert not caches.tokenizer_path("qwen").exists()


@pytest.mark.parametrize("unk_id", [None, "2", 99, -1])
def test_damaged_files_are_refused(unk_id):
    cfg = t5_config(T5_VOCAB)
    cfg["model"]["unk_id"] = unk_id
    with pytest.raises(ValueError):
        tokenizer.parse("t5", deflate(cfg))
    cfg = t5_config(T5_VOCAB)
    cfg["model"]["vocab"] = [["a"]]
    with pytest.raises(ValueError):
        tokenizer.parse("t5", deflate(cfg))


def test_unwritable_cache_leaves_no_partial_file(monkeypatch):
    monkeypatch.setattr(api, "fetch_tokenizer", lambda name, version: deflate(t5_config(T5_VOCAB)))

    def fail(src, dst):
        raise PermissionError("locked")

    monkeypatch.setattr(fs.os, "replace", fail)
    assert caches.load_tokenizer("t5").encode("ab") == [7, 1]
    folder = caches.tokenizer_path("t5").parent
    assert list(folder.iterdir()) == []


def _serving(monkeypatch, content):
    seen = []

    @contextlib.contextmanager
    def stream(method, url, **kwargs):
        seen.append(url)
        yield httpx.Response(200, content=content, request=httpx.Request(method, url))

    monkeypatch.setattr(api.httpx, "stream", stream)
    return seen


def test_fetch_uses_official_url(monkeypatch):
    seen = _serving(monkeypatch, b"x")
    assert REAL_FETCH("t5_tokenizer.def", 2) == b"x"
    assert seen == ["https://novelai.net/tokenizer/compressed/t5_tokenizer.def?v=2&static=true"]


def test_a_tokenizer_larger_than_any_is_not_kept(monkeypatch):
    monkeypatch.setattr(api, "_MAX_TOKENIZER", 4)
    _serving(monkeypatch, b"x" * 5)
    with pytest.raises(ValueError):
        REAL_FETCH("t5_tokenizer.def", 2)


class WordTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


def test_count_like_official_prompt_boxes():
    v5 = models.resolve("v5")
    words = WordTokenizer()
    count = lint.count_tokens(v5, words, "a b|c", "n", [("one two three", "x y")])
    assert count == lint.TokenCount(prompt=2 + 1 + 3, negative=1 + 2, limit=1471, exact=True)
    # Each ||a|b|| group counts as its longest option, ties going to the later one
    assert lint.count_tokens(v5, words, "x ||a|b c|| y", "", []).prompt == 4
    assert lint.count_tokens(v5, words, "||a b|c d||", "", []).prompt == 2


def test_count_falls_back_to_estimate_without_a_tokenizer():
    count = lint.count_tokens(models.resolve("v4.5"), None, "1girl, solo", "", [])
    assert not count.exact and count.limit == 512 and count.prompt == lint.estimate_tokens("1girl, solo")


def test_token_issues_per_field():
    v5c = models.resolve("v5-curated")
    issues = lint.check_tokens(lint.TokenCount(prompt=704, negative=650, limit=703, exact=True), v5c)
    assert [(i.code, i.field) for i in issues] == [("token_limit", "prompt"), ("token_near_limit", "negative")]
    assert issues[0].message.startswith("704 tokens")
    estimated = lint.check_tokens(lint.TokenCount(prompt=800, negative=0, limit=703, exact=False), v5c)
    assert estimated[0].message.startswith("about 800 tokens")


@pytest.mark.parametrize("kind", ["t5", "qwen"])
def test_matches_official_tokenizer(kind):
    path = REAL_FILES[kind]
    if not path.exists():
        pytest.skip("the tokenizer file has not been downloaded on this machine")
    tok = tokenizer.parse(kind, path.read_bytes())
    for text, (t5_ids, qwen_ids) in GOLDEN.items():
        assert tok.encode(text) == (t5_ids if kind == "t5" else qwen_ids), text


def test_supplied_tokenizer_is_used_even_when_falsey():
    class EmptyCache(WordTokenizer):
        def __len__(self):
            return 0

    count = lint.count_tokens(models.resolve("v5"), EmptyCache(), "one two three four", "", [])
    assert count.exact and count.prompt == 4
