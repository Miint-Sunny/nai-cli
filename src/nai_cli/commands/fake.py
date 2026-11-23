"""fake: copies of images with made-up settings, for sharing without telling what made them."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nai_cli import config, errors, fake, fs, meta
from nai_cli.commands import copies, texts
from nai_cli.commands.context import Context


@dataclass(frozen=True)
class FakeSettings:
    targets: copies.Targets
    plan: fake.Plan


def resolve(opts: dict[str, Any], ctx: Context) -> FakeSettings:
    sets = [fake.parse_set(item) for item in opts.get("set") or []]
    return FakeSettings(copies.resolve(opts, ctx), _plan(opts.get("preset"), sets, ctx))


def run(s: FakeSettings, ctx: Context) -> dict[str, Any]:
    warnings: list[str] = []

    def make(source: Path) -> bytes:
        return copies.with_settings(source, lambda size: s.plan.settings(s.plan.own(lambda: _own(source)), size))

    out = copies.write(s.targets, "fake", make, warnings)
    return {**out, "warnings": out.get("warnings", [])}


def _plan(spec: str | None, sets: list[tuple[str, Any]], ctx: Context) -> fake.Plan:
    """What the copies' settings are made of. --preset is a built-in or a saved preset, @FILE.json a template, or
    any other text, which goes into every field; the fields --set gives change it. With --set alone the settings
    are the image's own, changed where told to."""
    given = dict(sets)
    if spec is None:
        if not given:
            raise errors.NaiError(errors.BAD_ARGS, "give --preset SPEC or --set KEY=VALUE")
        return fake.Own(sets=given)
    if spec.startswith("@") and not spec.startswith("@@"):
        return fake.Preset(meta=fake.read_preset(texts.text_arg(spec)), sets=given)
    name = texts.text_arg(spec)
    saved = config.config_dir() / "fakes" / f"{name}.json"
    if fs.is_file(saved):
        # A preset of the user's is used in place of a built-in one of the same name
        return fake.Preset(meta=fake.read_preset(texts.text_arg(f"@{saved}")), sets=given)
    return fake.Text(text=fake.BUILTIN.get(name, name), sets=given)


def _own(source: Path) -> dict[str, str] | None:
    """The image's own settings, None where it keeps none."""
    try:
        return meta.read_raw(source)
    except errors.NaiError as e:
        if e.code != errors.NO_METADATA:
            raise
        return None
