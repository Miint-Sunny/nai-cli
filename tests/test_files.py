"""Files written: never over another file, numbered when a chosen name is taken."""

import json
import os
import sys
from pathlib import Path

import pytest
from helpers import FakeClient, fake_clients, run, set_filename
from PIL import Image
from samples import nai_image, png_bytes

from nai_cli import config, fs, naming


def test_numbered_keeps_double_suffixes(tmp_path):
    assert fs.numbered(tmp_path / "a.png", 2).name == "a-2.png"
    assert fs.numbered(tmp_path / "c.naiv4vibebundle.png", 1).name == "c-1.naiv4vibebundle.png"
    assert fs.numbered(tmp_path / "2026-11-18.naiv4vibebundle", 1).name == "2026-11-18-1.naiv4vibebundle"


def test_create_never_replaces_a_file(tmp_path):
    path = fs.create(tmp_path / "a.txt", "first")
    with pytest.raises(FileExistsError):
        fs.create(path, "second")
    assert path.read_text(encoding="utf-8") == "first"


def test_a_name_taken_meanwhile_gets_the_next_number(tmp_path, monkeypatch):
    target = tmp_path / "a.png"
    real = os.open

    def racing(path, flags, mode=0o777):
        # Another process takes the free name between the choice and the write
        if Path(path) == target and not target.exists():
            target.write_bytes(b"theirs")
        return real(path, flags, mode)

    monkeypatch.setattr(fs.os, "open", racing)
    written = fs.create_numbered(target, b"mine")
    assert written.name == "a-1.png" and target.read_bytes() == b"theirs" and written.read_bytes() == b"mine"


def test_failed_write_leaves_no_partial_file(tmp_path):
    class Unwritable(str):
        def encode(self, *args):
            raise OSError("disk full")

    with pytest.raises(OSError):
        fs.create(tmp_path / "a.txt", Unwritable("x"))
    assert not (tmp_path / "a.txt").exists()


def test_replace_swaps_the_whole_file(tmp_path):
    path = tmp_path / "f" / "c.toml"
    fs.create(path, "old")
    fs.replace(path, "new")
    assert path.read_text(encoding="utf-8") == "new" and [p.name for p in path.parent.iterdir()] == ["c.toml"]


def _png(path, color="red"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)
    return path


def test_strip_names_through_dots_and_numbers_same_stems(capsys, tmp_path):
    _png(tmp_path / "pic.png")
    # POSIX resolves .. through a folder that must exist
    (tmp_path / "sub").mkdir()
    code, out = run(capsys, "strip", str(tmp_path / "sub" / ".." / "pic.png"))
    assert code == 0 and Path(out["files"][0]["file"]).name == "pic_clean.png"
    # Two sources of the same name into one folder: the second copy is numbered, not written over the first
    a, b = _png(tmp_path / "x" / "pic.png", "red"), _png(tmp_path / "y" / "pic.png", "blue")
    code, out = run(capsys, "strip", str(a), str(b), "-o", str(tmp_path / "out"))
    names = [Path(f["file"]).name for f in out["files"]]
    assert code == 0 and names == ["pic.png", "pic-1.png"]
    assert Image.open(tmp_path / "out" / "pic-1.png").getpixel((0, 0)) == (0, 0, 255)


def test_a_name_taken_in_another_case_is_never_written_over(capsys, tmp_path):
    _png(tmp_path / "pic.png")
    code, out = run(capsys, "strip", str(tmp_path / "pic.png"))
    first = Path(out["files"][0]["file"])
    taken = first.with_name("PIC_CLEAN.png")
    first.rename(taken)
    taken.write_bytes(b"other")
    # Asked before the second copy is written, which would make the lower-case name exist anyway
    case_blind = first.exists()
    code, out = run(capsys, "strip", str(tmp_path / "pic.png"))
    name = Path(out["files"][0]["file"]).name
    # Numbered where case does not tell names apart, as on Windows and macOS; its own name elsewhere, as on Linux
    expected = "pic_clean-1.png" if case_blind else "pic_clean.png"
    assert code == 0 and name == expected and taken.read_bytes() == b"other"


def test_vibe_names_are_numbered_when_taken(capsys, with_token, monkeypatch, tmp_path):
    img = _png(tmp_path / "v.png")
    (tmp_path / "v.naiv4vibe").write_text("mine", encoding="utf-8")
    clients = fake_clients(monkeypatch)
    code, out = run(capsys, "vibe", str(img))
    assert code == 0 and Path(out["file"]).name == "v-1.naiv4vibe"
    assert (tmp_path / "v.naiv4vibe").read_text(encoding="utf-8") == "mine"
    code, out = run(capsys, "vibe", str(img), "--info", "0.5", "-o", str(tmp_path / "v.naiv4vibe"))
    assert code == 0 and Path(out["file"]).name == "v-2.naiv4vibe"
    assert (tmp_path / "v.naiv4vibe").read_text(encoding="utf-8") == "mine"
    assert sum(c.calls.count("encode_vibe") for c in clients) == 2


def test_bundle_into_the_same_image_twice_numbers_the_copy(capsys, with_token, monkeypatch, tmp_path):
    img = _png(tmp_path / "v.png")
    cover = _png(tmp_path / "cover.png", "blue")
    fake_clients(monkeypatch)
    first = run(capsys, "vibe", str(img), "--into", str(cover))[1]["file"]
    code, out = run(capsys, "vibe", str(img), "--into", str(cover))
    assert code == 0 and Path(out["file"]).name == "cover-1.naiv4vibebundle.png" and Path(first).exists()


def test_generated_image_numbered_when_taken(capsys, with_token, tmp_path):
    out_dir = tmp_path / "out"
    code, out = run(capsys, "gen", "--prompt", "x", "--seed", "4", "-o", str(out_dir))
    first = Path(out["images"][0])
    code, out = run(capsys, "gen", "--prompt", "x", "--seed", "4", "-o", str(out_dir))
    assert code == 0 and Path(out["images"][0]) != first and first.exists()
    before = first.read_bytes()
    code, out = run(capsys, "gen", "--prompt", "x", "-o", str(first))
    assert code == 0 and Path(out["images"][0]).name.startswith(first.stem + "-")
    assert first.read_bytes() == before


def test_only_a_template_with_type_names_tool_results(capsys, with_token, tmp_path):
    img = tmp_path / "src.png"
    nai_image(img, w=512, h=512, prompt="cat, smile, very aesthetic, masterpiece, no text")
    # Without {type} every result would take the name of its source, so the official client keeps source_tool
    for template in (None, "{prompt}"):
        if template:
            set_filename(template)
        for made in (tmp_path / "out").rglob("*.png"):
            made.unlink()
        code, out = run(capsys, "upscale", str(img), "--yes")
        assert code == 0 and Path(out["images"][0]).name == "src_upscale.png", template
        code, out = run(capsys, "tool", "lineart", str(img), "--yes")
        assert code == 0 and Path(out["images"][0]).name == "src_lineart.png", template
        code, out = run(capsys, "tool", "emotion", str(img), "--emotion", "happy", "--yes")
        assert code == 0 and Path(out["images"][0]).name == "src_emotion_happy.png", template
    path = config.default_path()
    kept = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("filename = ")]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    set_filename("{prompt} {type}")
    code, out = run(capsys, "upscale", str(img), "--yes")
    # The prompt the user typed, without the tags the official client added
    assert code == 0 and Path(out["images"][0]).name == "cat, smile upscale.png"


def test_config_init_force_replaces(capsys, tmp_path):
    path = tmp_path / "c.toml"
    code, out = run(capsys, "--config", str(path), "config", "init")
    assert code == 0 and out["created"] is True
    path.write_text("[output]\n", encoding="utf-8")
    code, out = run(capsys, "--config", str(path), "config", "init")
    assert code == 0 and out["created"] is False and path.read_text(encoding="utf-8") == "[output]\n"
    code, out = run(capsys, "--config", str(path), "config", "init", "--force")
    assert code == 0 and out["created"] is True and path.read_text(encoding="utf-8") != "[output]\n"
    code, out = run(capsys, "--config", str(path), "config", "--force")
    assert code == 1 and out["code"] == "bad_args"


def _output(text):
    path = config.default_path()
    s = path.read_text(encoding="utf-8").replace("[output]\n", f"[output]\n{text}\n", 1)
    path.write_text(s, encoding="utf-8")


def test_folder_rule_from_the_config(capsys, with_token, tmp_path):
    _output('folder = "{model}/{steps}"\nfilename = "{seed} {sampler}"')
    code, out = run(capsys, "gen", "--prompt", "x", "--seed", "5", "--steps", "20", "--sampler", "euler")
    path = Path(out["images"][0])
    assert code == 0 and path.parent.name == "20" and path.parent.parent.name == "nai-diffusion-5-full"
    assert path.name == "5 k_euler.png"
    code, out = run(capsys, "config")
    assert out["folder"] == "{model}/{steps}"


def test_empty_folder_rule_saves_in_the_output_folder(capsys, with_token, tmp_path):
    _output('folder = ""')
    code, out = run(capsys, "gen", "--prompt", "x")
    assert code == 0 and Path(out["images"][0]).parent == tmp_path / "out"
    # A folder given for the run never takes the rule
    path = config.default_path()
    path.write_text(path.read_text(encoding="utf-8").replace('folder = ""', 'folder = "{year}"'), encoding="utf-8")
    code, out = run(capsys, "gen", "--prompt", "x", "-o", str(tmp_path / "here"))
    assert code == 0, out
    assert Path(out["images"][0]).parent == tmp_path / "here"


def test_bad_folder_rule_refused_before_sending(capsys, with_token):
    _output('folder = "../elsewhere"')
    code, out = run(capsys, "gen", "--prompt", "x")
    assert code == 1 and out["code"] == "config_missing" and "[output] folder" in out["message"]


def test_folder_the_system_refuses_falls_back_to_the_output_folder(capsys, with_token, tmp_path, monkeypatch):
    _output('folder = "{model}"')
    real = fs.create

    def refuse_folders(path, data, private=False):
        if path.parent.name == "nai-diffusion-5-full":
            raise OSError("access denied")
        return real(path, data, private)

    monkeypatch.setattr(fs, "create", refuse_folders)
    code, out = run(capsys, "gen", "--prompt", "x", "--seed", "3")
    assert code == 0 and Path(out["images"][0]).parent == tmp_path / "out"
    assert any("default name" in w for w in out["warnings"])


def test_a_file_in_the_place_of_a_folder_is_not_a_name_taken(tmp_path):
    blocker = tmp_path / "out"
    blocker.write_text("x", encoding="utf-8")
    for write in (fs.create, fs.create_numbered, fs.replace):
        with pytest.raises(NotADirectoryError, match="out is a file"):
            write(blocker / "sub" / "a.png", b"data")
    assert blocker.read_text(encoding="utf-8") == "x" and not any(tmp_path.glob("**/*.png"))


def test_output_under_a_file_is_refused_before_paying(capsys, with_token, tmp_path):
    blocker = tmp_path / "taken"
    blocker.write_text("x", encoding="utf-8")
    code, out = run(capsys, "gen", "--prompt", "x", "-o", str(blocker / "a.png"))
    assert code == 1 and out["code"] == "bad_args" and "is a file" in out["message"]
    cfg = tmp_path / "cfg-file"
    cfg.write_text("x", encoding="utf-8")
    code, out = run(capsys, "--config", str(cfg / "config.toml"), "config", "init")
    assert code == 1 and out["code"] == "bad_args"


def test_tool_results_keep_their_labels_and_index(capsys, with_token, tmp_path):
    img = tmp_path / "src.png"
    nai_image(img, w=512, h=512, seed=777, prompt="cat")
    code, out = run(capsys, "tool", "bg-removal", str(img), "--yes")
    names = [Path(p).name for p in out["images"]]
    assert code == 0 and names == [
        "src_bg-removal_masked.png",
        "src_bg-removal_generated.png",
        "src_bg-removal_blend.png",
    ]
    set_filename("{prompt} {type} {index}")
    code, out = run(capsys, "tool", "bg-removal", str(img), "--yes")
    names = [Path(p).name for p in out["images"]]
    assert code == 0 and names == [
        "cat bg-removal 1_masked.png",
        "cat bg-removal 2_generated.png",
        "cat bg-removal 3_blend.png",
    ]


def test_copies_are_filed_by_the_settings_they_keep(capsys, tmp_path):
    path = config.default_path()
    path.write_text(path.read_text(encoding="utf-8").replace("[output]\n", '[output]\nfolder = "s{seed}"\n'), "utf-8")
    img = tmp_path / "pic.png"
    nai_image(img, w=64, h=64, seed=77)
    region = "rect:0.1,0.1,0.5,0.5"
    _, out = run(capsys, "redact", str(img), "--region", region, "--keep-metadata")
    kept = Path(out["files"][0]["file"])
    _, out = run(capsys, "strip", str(img))
    clean = Path(out["files"][0]["file"])
    assert (kept.parent.name, kept.name) == ("s77", "pic_redacted.png")
    # Nothing to file the stripped copy by, so the rule gives only its fixed text
    assert (clean.parent.name, clean.name) == ("s", "pic_clean.png") and img.read_bytes()


def test_a_home_folder_path_works_in_every_option(capsys, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    nai_image(home / "pic.png", w=64, h=64)
    (home / "prompt.txt").write_text("cat", encoding="utf-8")
    for argv in (
        ["meta", "~/pic.png"],
        ["strip", "~/pic.png", "-o", "~/copies"],
        ["redact", "~/pic.png", "--region", "rect:0.1,0.1,0.5,0.5"],
        ["gen", "--from-image", "~/pic.png", "--dry-run"],
        ["i2i", "--image", "~/pic.png", "--prompt", "@~/prompt.txt", "--dry-run"],
        ["upscale", "~/pic.png", "--dry-run"],
        ["vibe", "~/pic.png", "--dry-run"],
        ["gen", "--model", "v4.5", "--prompt", "x", "--vibe", "~/pic.png:0.5", "--dry-run"],
    ):
        code, out = run(capsys, *argv)
        assert code == 0, (argv, out)
    assert (home / "copies" / "pic.png").exists()


def test_names_fit_in_255_bytes_and_keep_their_label(capsys, tmp_path):
    assert naming.fit("猫" * 200, "_redacted").endswith("猫_redacted")
    assert len(naming.fit("猫" * 200, "_redacted").encode()) <= 255 - len("-9999.naiv4vibebundle.png")
    # A cut never splits a character
    assert naming.fit("a" + "猫" * 200).encode().decode() == naming.fit("a" + "猫" * 200)
    meta = naming.Meta(prompt="猫" * 170, seed=1)
    stem = naming.render(naming.parse("{prompt}"), meta)
    assert len((stem + "-9999.png").encode()) <= 255


@pytest.mark.parametrize("stem", ["a" * 235, "猫" * 76])
@pytest.mark.parametrize(
    "command", [["upscale"], ["tool", "lineart"], ["tool", "bg-removal"]], ids=["upscale", "lineart", "bg-removal"]
)
def test_a_long_source_name_never_loses_a_paid_result(capsys, with_token, tmp_path, stem, command):
    source = tmp_path / f"{stem}.png"
    Image.new("RGB", (64, 64)).save(source)
    code, out = run(capsys, *command, str(source), "--yes")
    assert code == 0, out
    for saved in out["images"]:
        name = Path(saved).name
        assert Path(saved).is_file() and len(name.encode()) <= 255 and command[-1] in name


@pytest.mark.parametrize(
    "argv",
    [
        ["gen", "--prompt", "x", "--n", "2"],
        ["inpaint", "--region", "rect:0,0,1,1", "--prompt", "x", "--keep-raw"],
        ["tool", "bg-removal"],
    ],
    ids=["two images", "keep-raw", "bg-removal"],
)
def test_a_long_name_given_with_o_is_cut_to_fit_and_keeps_its_labels(capsys, with_token, tmp_path, monkeypatch, argv):
    monkeypatch.setattr(FakeClient, "generate", lambda self, body: [png_bytes()] * body["parameters"]["n_samples"])
    source = tmp_path / "src.png"
    Image.new("RGB", (64, 64)).save(source)
    image = ["--image", str(source)] if argv[0] == "inpaint" else [str(source)] if argv[0] == "tool" else []
    exact = tmp_path / "o" / ("e" * 250 + ".png")
    code, out = run(capsys, *argv, *image, "-o", str(exact), "--yes")
    assert code == 0, out
    saved = out["images"] + out.get("raw_images", [])
    assert len(saved) >= 2 and len(set(saved)) == len(saved)
    assert all(Path(p).is_file() and len(Path(p).name.encode()) <= 255 for p in saved)


def test_the_fallback_name_holds_no_text_of_the_run(capsys, with_token, tmp_path, monkeypatch):
    real = fs.create

    def refuse_named(path, data, private=False):
        # A name the file system refuses, made of the prompt or of -o
        if "secretprompt" in path.name or "given" in path.name:
            raise OSError("invalid name")
        return real(path, data, private)

    monkeypatch.setattr(fs, "create", refuse_named)
    set_filename("{prompt}")
    for extra in ([], ["-o", str(tmp_path / "exact" / "given.png")]):
        code, out = run(capsys, "gen", "--prompt", "secretprompt", "--seed", "3", *extra)
        assert code == 0, out
        (saved,) = out["images"]
        assert "secretprompt" not in saved and "given" not in Path(saved).name
        assert Path(saved).parent == (tmp_path / "exact" if extra else tmp_path / "out")


@pytest.mark.parametrize("where", ["NUL", "CON", "COM1", "dots..."])
def test_a_folder_the_system_takes_for_another_is_refused_or_saved_in(capsys, with_token, tmp_path, monkeypatch, where):
    # Which of these Windows treats as devices depends on its version; none may take a paid image and lose it
    clients = fake_clients(monkeypatch)

    def saved_or_refused(out):
        if out["ok"]:
            return all(Path(p).is_file() and Path(p).read_bytes() for p in out["images"])
        return out["code"] == "bad_args" and not any("generate" in c.calls for c in clients)

    _, out = run(capsys, "gen", "--prompt", "x", "-o", str(tmp_path / where / "out.png"))
    assert saved_or_refused(out), out
    if sys.platform == "win32" and where == "NUL":
        assert out["code"] == "bad_args"
    path = config.default_path()
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("dir = ")]
    lines.insert(lines.index("[output]") + 1, f"dir = {json.dumps(str(tmp_path / where))}")
    path.write_text("\n".join(lines), "utf-8")
    _, out = run(capsys, "gen", "--prompt", "x")
    assert saved_or_refused(out), out


def test_a_folder_holding_the_name_is_a_name_taken(capsys, tmp_path):
    source = tmp_path / "pic.png"
    Image.new("RGB", (8, 8)).save(source)
    (tmp_path / "copies" / "pic.png").mkdir(parents=True)
    code, out = run(capsys, "strip", str(source), "-o", str(tmp_path / "copies"))
    assert code == 0 and Path(out["files"][0]["file"]).name == "pic-1.png", out
    with pytest.raises(FileExistsError):
        fs.create(tmp_path / "copies" / "pic.png", b"x")


def test_vibe_o_is_saved_as_every_other_o(capsys, with_token, tmp_path):
    img = tmp_path / "a.png"
    nai_image(img, w=512, h=512)
    target = tmp_path / "new" / "deeper" / "x.naiv4vibe"
    code, out = run(capsys, "vibe", str(img), "-o", str(target))
    assert code == 0 and out["file"] == str(target), out
    # Encoded now, so a second export needs nothing paid, and the name taken gets the next number
    code, out = run(capsys, "vibe", str(img), "-o", str(target))
    assert code == 0 and out["file"] == str(target.with_name("x-1.naiv4vibe")) and out["cost"]["spent"] == 0
    code, out = run(capsys, "vibe", str(img), "-o", str(tmp_path / "B.NAIV4VIBEBUNDLE"))
    assert code == 0 and out["vibes"] == 1 and json.loads(Path(out["file"]).read_text("utf-8"))["identifier"]
    blocking = tmp_path / "file"
    blocking.write_text("x", encoding="utf-8")
    for extra in (["--dry-run"], []):
        code, out = run(capsys, "vibe", str(img), "-o", str(blocking / "x.naiv4vibe"), *extra)
        assert (code, out["code"]) == (1, "bad_args"), (extra, out)


def test_vibe_o_too_long_or_refused_by_the_system_fails_before_paying(capsys, with_token, monkeypatch, tmp_path):
    clients = fake_clients(monkeypatch)
    source = tmp_path / "s.png"
    Image.new("RGB", (64, 64), "red").save(source)
    code, out = run(capsys, "vibe", str(source), "-o", str(tmp_path / ("e" * 250 + ".naiv4vibe")))
    assert code == 0 and out["file"].endswith(".naiv4vibe") and len(Path(out["file"]).name.encode()) <= 255, out
    if sys.platform == "win32":
        clients.clear()
        Image.new("RGB", (64, 64), "blue").save(source)
        for extra in (["--dry-run"], []):
            code, out = run(capsys, "vibe", str(source), "-o", str(tmp_path / "bad*name.naiv4vibe"), *extra)
            assert (code, out["code"]) == (1, "bad_args"), (extra, out)
        assert not any("encode_vibe" in c.calls for c in clients)


def test_a_dry_run_refuses_the_folders_saving_refuses_and_leaves_none(capsys, tmp_path):
    before = sorted(p.name for p in tmp_path.iterdir())
    new = tmp_path / "new" / "deeper"
    code, out = run(capsys, "gen", "--prompt", "x", "-o", str(new / "a.png"), "--dry-run")
    assert code == 0 and not (tmp_path / "new").exists(), out
    if sys.platform == "win32":
        for bad in ("bad*dir", "trailing "):
            code, out = run(capsys, "gen", "--prompt", "x", "-o", str(tmp_path / bad / "a.png"), "--dry-run")
            assert (code, out["code"]) == (1, "bad_args"), (bad, out)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


@pytest.mark.parametrize(
    "argv",
    [
        ["gen", "--prompt", "x", "-o", "bad\x00name.png", "--dry-run"],
        ["gen", "--prompt", "cat\x00dog", "--dry-run"],
        ["meta", "a\x00.png"],
        ["batch", "jobs\x00.jsonl", "--dry-run"],
    ],
)
def test_a_nul_in_any_argument_is_refused(capsys, argv):
    code, out = run(capsys, *argv)
    assert (code, out["code"]) == (1, "bad_args"), out


def test_a_path_too_long_to_look_up_names_nothing(capsys, tmp_path):
    long = tmp_path / ("x" * 300) / ("y" * 5000)
    assert not fs.exists(long) and not fs.is_file(long) and not fs.is_folder(long)
    code, out = run(capsys, "meta", str(long))
    assert code == 1 and out["code"] != "internal", out


@pytest.mark.parametrize(
    ("data", "code"),
    [
        ('\ufeffnovelai_token = ""\n'.encode(), 0),
        ('novelai_token = ""\n'.encode("utf-16"), 1),
        (b'novelai_token = "\xff"\n', 1),
        (b'novelai_token = ""\n[output]\ndir = "a\\u0000b"\n', 1),
    ],
)
def test_a_config_is_read_with_a_byte_order_mark_and_refused_when_not_utf8(capsys, tmp_path, data, code):
    path = tmp_path / "other.toml"
    path.write_bytes(data)
    got, out = run(capsys, "--config", str(path), "models")
    assert got == code and (code == 0 or out["code"] == "config_missing"), out


def test_a_dry_run_leaves_a_folder_another_process_made_meanwhile(capsys, monkeypatch, tmp_path):
    theirs = tmp_path / "shared"
    real_mkdir = os.mkdir

    def mkdir(path, *args, **kwargs):
        # Another nai makes the folder between this one's look and its own try
        if Path(path) == theirs and not theirs.exists():
            real_mkdir(path, *args, **kwargs)
            raise FileExistsError(path)
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "mkdir", mkdir)
    code, out = run(capsys, "gen", "--prompt", "x", "-o", str(theirs / "mine"), "--dry-run")
    assert code == 0, out
    assert theirs.is_dir() and not (theirs / "mine").exists()


def test_a_long_o_name_is_cut_to_fit_and_numbered_when_taken(capsys, with_token, tmp_path):
    exact = tmp_path / ("q" * 252 + ".png")
    code, out = run(capsys, "gen", "--prompt", "x", "-o", str(exact))
    first = Path(out["images"][0])
    assert code == 0 and first != exact and len(first.name.encode()) <= 255
    before = first.read_bytes()
    code, out = run(capsys, "gen", "--prompt", "x", "-o", str(exact))
    assert code == 0 and Path(out["images"][0]) == first.with_name(first.stem + "-1.png")
    assert first.read_bytes() == before


def test_trying_a_name_never_writes_it(tmp_path, monkeypatch):
    target = tmp_path / "out" / "would-export.naiv4vibe"
    written = []
    create = fs.create

    def record(path, data, *args, **kwargs):
        written.append(Path(path))
        return create(path, data, *args, **kwargs)

    monkeypatch.setattr(fs, "create", record)
    (tmp_path / "out").mkdir()
    target.write_bytes(b"theirs")
    fs.try_names(target.parent, target)
    assert written and target not in written and target.read_bytes() == b"theirs"
    assert list((tmp_path / "out").iterdir()) == [target]


@pytest.mark.parametrize("exact", [False, True])
def test_a_directory_occupying_a_vibe_filename_is_numbered(capsys, with_token, tmp_path, exact):
    img = _png(tmp_path / "v.png")
    target = tmp_path / "v.naiv4vibe"
    target.mkdir()
    existing = target / "keep.txt"
    existing.write_bytes(b"keep")
    args = ["-o", str(target)] if exact else []
    code, out = run(capsys, "vibe", str(img), *args)
    assert code == 0 and Path(out["file"]) == tmp_path / "v-1.naiv4vibe", out
    assert existing.read_bytes() == b"keep"


def test_unencodable_output_stem_never_loses_a_paid_image(capsys, with_token, tmp_path):
    for stem in ("a\udc80b", "a\udc80" + "x" * 250):
        code, out = run(capsys, "gen", "--prompt", "cat", "-o", str(tmp_path / (stem + ".png")))
        assert code == 0 and Path(out["images"][0]).read_bytes() and out["cost"]["spent"] == 30, out


@pytest.mark.parametrize(("failure", "expected"), [(ValueError, "bad_args"), (KeyboardInterrupt, "interrupted")])
def test_copy_set_rolls_back_only_its_own_files(capsys, tmp_path, monkeypatch, failure, expected):
    first, last = _png(tmp_path / "first.png"), _png(tmp_path / "last.png")
    folder = tmp_path / "copies"
    folder.mkdir()
    kept = folder / first.name
    kept.write_bytes(b"keep")
    create = fs.create_numbered

    def write(path, data):
        if path.name == last.name:
            raise failure("cannot finish the copy")
        return create(path, data)

    monkeypatch.setattr(fs, "create_numbered", write)
    code, out = run(capsys, "strip", str(first), str(last), "-o", str(folder))
    assert code != 0 and out["code"] == expected
    assert list(folder.iterdir()) == [kept] and kept.read_bytes() == b"keep"


def test_vibe_dry_run_names_the_next_free_export_without_creating_it(capsys, with_token, tmp_path):
    img = _png(tmp_path / "cat.png")
    target = tmp_path / "cover.naiv4vibebundle.png"
    target.write_bytes(b"original")
    fs.numbered(target, 1).mkdir()
    args = ["vibe", str(img), "--into", str(img), "-o", str(target)]
    code, dry = run(capsys, *args, "--dry-run")
    wanted = fs.numbered(target, 2)
    assert code == 0 and Path(dry["file"]) == wanted and not wanted.exists()
    code, actual = run(capsys, *args)
    assert code == 0 and actual["file"] == dry["file"] and target.read_bytes() == b"original"


def test_filename_template_names_images(capsys, with_token, tmp_path):
    set_filename("{prompt} s-{seed} {index}{type|pre:-}")
    code, out = run(capsys, "gen", "--prompt", "tabby cat", "--seed", "9")
    assert code == 0 and Path(out["images"][0]).name == "tabby cat s-9 1.png"
    code, out = run(capsys, "gen", "--prompt", "tabby cat", "--seed", "9")
    assert Path(out["images"][0]).name == "tabby cat s-9 1-1.png"
    src = tmp_path / "src.png"
    Image.new("RGB", (832, 1216)).save(src)
    code, out = run(capsys, "i2i", "--image", str(src), "--prompt", "tabby cat", "--seed", "3")
    assert code == 0 and Path(out["images"][0]).name == "tabby cat s-3 1-image2image.png"


def test_filename_template_refused_before_any_request(capsys, with_token, monkeypatch):
    set_filename("{prompt|first:0}")
    clients = fake_clients(monkeypatch)
    code, out = run(capsys, "gen", "--prompt", "x")
    assert code == 1 and out["code"] == "config_missing" and "needs a positive number" in out["message"]
    assert clients == []


def test_filename_template_falls_back_when_the_name_cannot_be_saved(capsys, with_token, monkeypatch, tmp_path):
    set_filename("{prompt}")
    real_create = fs.create

    def refuse_named(path, data, private=False):
        if path.name.startswith("tabby cat"):
            raise OSError("name refused")
        return real_create(path, data, private)

    monkeypatch.setattr(fs, "create", refuse_named)
    code, out = run(capsys, "gen", "--prompt", "tabby cat", "--seed", "4")
    assert code == 0 and Path(out["images"][0]).name.endswith("_4.png")
    assert any(w.startswith("cannot save as tabby cat.png: name refused; the default name") for w in out["warnings"])


def test_tool_results_named_by_the_source_settings(capsys, with_token, tmp_path):
    set_filename("{prompt|tags:1} {type} {resolution}")
    img = tmp_path / "pic.png"
    nai_image(img, w=512, h=512)
    code, out = run(capsys, "tool", "lineart", str(img))
    assert code == 0 and Path(out["images"][0]).name == "1girl lineart 1024x1024.png"
    code, out = run(capsys, "tool", "emotion", str(img), "--emotion", "happy")
    assert Path(out["images"][0]).name == "1girl emotion-happy 1024x1024.png"
    # Without settings to name it by, the default name is kept
    plain = tmp_path / "plain.png"
    Image.new("RGB", (512, 512)).save(plain)
    code, out = run(capsys, "tool", "lineart", str(plain))
    assert code == 0 and Path(out["images"][0]).name == "plain_lineart.png"


def test_output_names_a_file_or_a_folder(capsys, with_token, monkeypatch, tmp_path):
    monkeypatch.setattr(FakeClient, "generate", lambda self, body: [png_bytes()] * body["parameters"]["n_samples"])
    target = tmp_path / "shots" / "cat.png"
    code, out = run(capsys, "gen", "--prompt", "x", "--seed", "2", "--n", "2", "-o", str(target))
    assert code == 0 and [Path(p).name for p in out["images"]] == ["cat.png", "cat_1.png"]
    # A file given is never overwritten; the image gets the next number
    before = target.read_bytes()
    code, out = run(capsys, "gen", "--prompt", "x", "-o", str(target))
    assert code == 0 and Path(out["images"][0]).name == "cat-1.png" and target.read_bytes() == before
    code, out = run(capsys, "gen", "--prompt", "x", "-o", str(tmp_path / "cat.jpg"))
    assert code == 1 and "PNG" in out["message"]
    folder = tmp_path / "flat"
    code, out = run(capsys, "gen", "--prompt", "x", "--seed", "3", "-o", str(folder))
    assert code == 0 and Path(out["images"][0]).parent == folder


def test_output_for_tools_and_previews(capsys, with_token, tmp_path):
    img = tmp_path / "pic.png"
    nai_image(img, w=512, h=512)
    code, out = run(capsys, "upscale", str(img), "-o", str(tmp_path / "big.png"))
    assert code == 0 and Path(out["images"][0]).name == "big.png"
    mask = tmp_path / "m.png"
    Image.new("L", (512, 512), 255).save(mask)
    code, out = run(
        capsys,
        "inpaint",
        "--image",
        str(img),
        "--mask",
        str(mask),
        "--prompt",
        "x",
        "--preview",
        "-o",
        str(tmp_path / "p.png"),
    )
    assert code == 0 and Path(out["preview"]).name == "p_preview.png"
