import pytest

from nai_cli import prompt as P
from nai_cli import tidy


def test_suffix_goes_into_first_mix_chunk_only():
    assert P.apply_suffix("cat|dog", "q") == "cat, q|dog"


def test_suffix_ignores_escaped_separator():
    assert P.split_mix_chunks("a ||x|y|| b|c") == ["a ||x|y|| b", "c"]


def test_suffix_stays_before_text_marker():
    # Appended directly, keeping the trailing comma
    assert P.apply_suffix("girl, sign, text: HELLO", "q") == "girl, sign,, q text: HELLO"
    # The official client glues the suffix to a leading text: marker
    assert P.apply_suffix("text: HELLO", "q") == "qtext: HELLO"
    assert P.apply_suffix("a, Text: x, text: y", "q") == "a,, q Text: x, Text: y"


def test_suffix_keeps_trailing_comma_like_official():
    assert P.append_suffix("wind,", "q") == "wind,, q"
    assert P.append_suffix("a\n", "q") == "a\n, q"
    assert P.append_suffix("  ", "q") == "  , q"
    assert P.append_suffix("", "q") == "q"


def test_strip_suffix_is_inverse():
    for text in ["wind,", "a, b", "girl, sign, text: HELLO", "a|b"]:
        ok, back = P.strip_suffix(P.apply_suffix(text, "very aesthetic"), "very aesthetic")
        assert ok and back == text


def test_mix_chunk_limit_merges_tail():
    chunks = P.split_mix_chunks("|".join("abcdefgh"))
    assert len(chunks) == 6 and chunks[-1] == "f|g|h"


def test_auto_text_collects_quotes_from_base_and_characters():
    chars = [P.AutoTextCharacter('girl, says "hi"', 0.2, 0.5), P.AutoTextCharacter("boy, 'yo'", 0.8, 0.5)]
    out = P.apply_auto_text('sign reading "OPEN",', chars, use_coords=False)
    assert out == 'sign reading "OPEN", teXt: OPEN\n\nhi\n\nyo'


def test_auto_text_reverses_order_for_cjk():
    out = P.apply_auto_text("「你好」「世界」", [], use_coords=False)
    assert out.endswith("teXt: 世界\n\n你好")


def test_auto_text_cjk_share_counts_utf16_like_official():
    # Each emoji is two UTF-16 units, so one kana in five units is not mostly CJK
    out = P.apply_auto_text('1girl, "あ", "😀😀"', [], use_coords=False)
    assert out.endswith("teXt: あ\n\n😀😀")


def test_auto_text_skipped_when_manual_text_marker():
    assert P.apply_auto_text('Text: "hi"', [], False) == 'Text: "hi"'


def test_apostrophe_is_not_a_quote():
    assert P.auto_text_block("girl's room, don't", [], False) is None


def test_auto_text_reading_order_with_coords():
    chars = [
        P.AutoTextCharacter('"B"', 0.8, 0.2),
        P.AutoTextCharacter('"A"', 0.2, 0.2),
        P.AutoTextCharacter('"C"', 0.5, 0.9),
    ]
    assert P.auto_text_block("x", chars, use_coords=True) == "teXt: A\n\nB\n\nC"


def test_strip_auto_text_roundtrip():
    chars = [P.AutoTextCharacter('says "hi"')]
    text = P.apply_auto_text('sign "OPEN"', chars, False)
    assert P.strip_auto_text(text, chars, False) == 'sign "OPEN"'


def test_negative_gets_preset_then_nsfw_and_keeps_body_verbatim():
    preset = "lowres, bad"
    assert P.compose_negative("a,b, ", preset, nsfw=False) == "lowres, bad, a,b, "
    assert P.compose_negative("", preset, nsfw=True) == "nsfw, lowres, bad"
    assert P.compose_negative("x|y", preset, nsfw=True) == "nsfw, lowres, bad, x|y"
    assert P.compose_negative("|y", preset, nsfw=False) == "lowres, bad|y"
    assert P.compose_negative("x", "", nsfw=False) == "x"


def test_strip_negative_is_inverse():
    preset = "lowres, bad"
    for body in ["a,b, ", "", "x|y", "|y"]:
        for nsfw in (True, False):
            composed = P.compose_negative(body, preset, nsfw)
            ok, back = P.strip_uc_preset(P.strip_nsfw_prefix(composed), preset)
            assert ok and back == body
    assert P.strip_uc_preset("x, lowres, bad", preset) == (False, "x, lowres, bad")
    assert P.strip_nsfw_prefix("nsfw") == ""


def test_weight_marks_and_numeric_close():
    kinds = lambda t: [(m.kind, m.number) for m in P.weight_marks(t)]  # noqa: E731
    assert kinds("1.2::year_2025::, x") == [("open", "1.2"), ("numeric_close", "2025")]
    assert kinds("1.2::year 2025::") == [("open", "1.2"), ("numeric_close", "2025")]
    assert kinds("1.2::x.::") == [("open", "1.2"), ("numeric_close", ".")]
    assert kinds("2::nsfw::, -1::bad::") == [("open", "2"), ("close", ""), ("open", "-1"), ("close", "")]
    # A number as its own tag switches weights rather than closing
    assert kinds("1.2::a, 0.8::b::") == [("open", "1.2"), ("switch", "0.8"), ("close", "")]
    assert kinds("1.5 ::tag::") == [("spaced", "1.5"), ("close", "")]
    assert kinds("tag::1.5, x") == [("number_after", "1.5")]
    assert kinds("score_9::x") == [("stray", "9")]
    assert kinds("1.2::a") == [("open", "1.2"), ("unclosed", "1.2")]


def test_space_numeric_closings_only_touches_closings():
    assert tidy.space_numeric_closings("1.2::year 2025::, 1.1::score_9::") == "1.2::year 2025 ::, 1.1::score_9 ::"
    for text in ["1.2::a, 0.8::b::", "2::x::", "1.2::year 2025 ::"]:
        assert tidy.space_numeric_closings(text) == text


def test_format_prompt_keeps_literals():
    assert tidy.format_prompt("1girl，  long_hair ,, smile,") == "1girl, long_hair, smile,"
    assert tidy.format_prompt("long_hair, ^_^, >_<", underscores=True) == "long hair, ^_^, >_<"
    text = '"Hello,  world" ,a\n  b_c , <style_x>, text: Hi ,  there | x_y'
    assert tidy.format_prompt(text, underscores=True) == '"Hello,  world", a\n  b c, <style_x>, text: Hi ,  there | x y'
    formatted = tidy.format_prompt("a ,b，c", underscores=True)
    assert tidy.format_prompt(formatted, underscores=True) == formatted


@pytest.mark.parametrize(
    "text, expected",
    [
        ("(long hair:1.5), smile", "1.5::long hair::, smile"),
        ("[ugly:0.8]", "0.8::ugly::"),
        ("a, (b:1.2), (c:1.2), d", "a, 1.2::b::, 1.2::c::, d"),
        ("(long hair), [x]", "(long hair), [x]"),
        ("\\(text\\) (cat:1.1)", "(text) 1.1::cat::"),
        ("girl2 (x:1.5)", "girl2 1.5::x::"),
        ("(a:1.0), b", "a, b"),
        ("(a:-1)", "-1::a::"),
        ("(x:nan)", "(x:nan)"),
        ("plain \\(x\\)", "plain (x)"),
        ("(a:1.25000)(b:1.25)", "1.25::ab::"),
        # Quoted text is copied into the image as typed
        ('"(sign:1.2)", (cat:1.1)', '"(sign:1.2)", 1.1::cat::'),
        ("text: (hello:1.5)", "text: (hello:1.5)"),
    ],
)
def test_sd_weights_convert(text, expected):
    assert tidy.convert_sd_weights(text) == expected


def test_fixed_tags_join_with_commas():
    assert P.with_fixed("cat", "artist:x", "best") == "artist:x, cat, best"
    assert P.with_fixed("", "artist:x", " ") == "artist:x"


@pytest.mark.parametrize(
    ("text", "stored", "drawn", "expected"),
    [
        ("||cat|dog||, smile", "||cat|dog||, smile", "dog, smile", "dog, smile"),
        ("||cat|dog||, frown", "||cat|dog||, smile", "cat, smile", "cat, frown"),
        # A group the image did not have stays to be drawn again
        ("||red|blue||, x", "||cat|dog||", "dog", "||red|blue||, x"),
        # Repeated groups take the image's draws in order
        ("||a|b||, ||a|b||", "||a|b||, ||a|b||", "b, a", "b, a"),
        # The stored prompt must explain what was sent, or nothing is filled
        ("||cat|dog||", "||cat|dog||", "bird", "||cat|dog||"),
        ("plain", "||cat|dog||", "cat", "plain"),
    ],
)
def test_fill_drawn_takes_the_options_the_image_drew(text, stored, drawn, expected):
    assert P.fill_drawn(text, stored, drawn) == expected
