"""The config file and the data directory: limits, defaults, accounts and local models."""

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nai_cli import api, errors, fs, values

TOKEN_ENV = "NOVELAI_TOKEN"
# The account of novelai_token, which the [accounts] tables cannot take the name of
DEFAULT_ACCOUNT = "default"
# The longest wait between batch jobs, in seconds
MAX_INTERVAL = 86400

TEMPLATE = """\
# Persistent API token from NovelAI account settings (starts with pst-). NOVELAI_TOKEN takes precedence when set
novelai_token = ""

# More accounts, chosen with --account NAME or nai account use NAME. A relay that speaks the NovelAI API takes
# its own token and address; image_url defaults to api_url. Email logins are added with nai account login.
# [accounts.second]
# token = "pst-..."
# [accounts.relay]
# token = "..."
# api_url = "https://relay.example.com"

[output]
# Where images are saved; empty means images/ under the data directory, or ~/Pictures/nai-cli on macOS
dir = ""
# Folders under dir in the same syntax as filename, such as "{year}/{month}" or "{model}"; empty saves in dir
folder = "{date}"
# File names in the official client's template syntax, such as "{prompt} s-{seed}"; empty keeps HHMMSS_seed.
# Tokens: prompt seed date time year month day model modelname resolution width height index id type characters
# steps cfg sampler
filename = ""

[cost]
# Requests estimated above this many Anlas return need_confirm and only run with --yes
max_per_request = 100

[batch]
# Seconds to wait between jobs that sent requests, or [min, max] for a random wait
interval = 0

[local]
# Models run on this computer need pip install "nai-cli[local]"; nothing is downloaded unless asked.
# The ONNX tagger nai interrogate uses, and its tag list; empty labels looks beside the model
tagger = ""
tagger_labels = ""
# wd14, cl, cl-v2, eva02 or pixai; empty guesses from the file names
tagger_kind = ""
# Where nai local download sam puts the model object: regions use; empty means models/sam under the data directory
sam_dir = ""

[defaults]
# Generation defaults for all models. Command-line options win; anything unset here uses the model's default
# model = "v5"
# V5 Full at "high" or "medium" effort when --effort is not given; the official client now defaults to medium
# effort = "high"
# size = "832x1216"
# steps = 28
# cfg = 4
# sampler = "k_euler_ancestral"
# cfg_rescale = 0
# noise_schedule = "karras"
# quality = "standard"
# uc_preset = "heavy"
# n = 1
# true follows the official client: prepend nsfw to the negative prompt unless the prompt mentions nsfw.
# false (default) never adds it, and removes nsfw from the negative prompt when the prompt mentions nsfw
official_nsfw = false
# Split ComfyUI multi-character prompts (COUPLE, AND with MASK or AREA, a character per | line) into characters.
# Off by default: a single | also divides official prompts
comfyui_characters = false
# Let V5 Full and V5 Curated take a noise schedule and Variety+, which the official client hides for V5
v5_hidden_options = false
# Transparent images (V5) use premultiplied alpha instead of straight alpha, the official client's other mode
premultiplied_alpha = false
# Tidy prompts before sending: full-width commas, extra whitespace, empty tags.
# Quoted text, text: segments, <markers> and kaomoji are left untouched
format_prompt = false
# Also replace underscores with spaces when tidying
format_underscores = false
# Insert a space before a closing :: that follows a number (1.2::year 2025:: -> 1.2::year 2025 ::);
# otherwise the number is read as a new weight
space_numeric_close = false
# Convert SD weights such as (tag:1.2) to 1.2::tag:: before sending
convert_sd_weights = false
# Fixed tags joined around every prompt typed for a run with ", "; prompts taken from an image are left as stored
prompt_prefix = ""
prompt_suffix = ""
negative_prefix = ""
negative_suffix = ""
"""

# One folder per day
DEFAULT_FOLDER = "{date}"

# Target type of each value; ranges are checked per model by the CLI
DEFAULT_KEYS: dict[str, type] = {
    "model": str,
    "effort": str,
    "size": str,
    "steps": int,
    "cfg": float,
    "sampler": str,
    "cfg_rescale": float,
    "noise_schedule": str,
    "quality": str,
    "uc_preset": str,
    "n": int,
    "official_nsfw": bool,
    "v5_hidden_options": bool,
    "comfyui_characters": bool,
    "premultiplied_alpha": bool,
    "format_prompt": bool,
    "format_underscores": bool,
    "space_numeric_close": bool,
    "convert_sd_weights": bool,
    "prompt_prefix": str,
    "prompt_suffix": str,
    "negative_prefix": str,
    "negative_suffix": str,
}


def config_dir() -> Path:
    if sys.platform == "win32":
        return _base("APPDATA", "AppData", "Roaming") / "nai-cli"
    return _base("XDG_CONFIG_HOME", ".config") / "nai-cli"


def data_dir() -> Path:
    if sys.platform == "win32":
        return _base("LOCALAPPDATA", "AppData", "Local") / "nai-cli"
    return _base("XDG_DATA_HOME", ".local", "share") / "nai-cli"


def default_output_dir() -> Path:
    """Where images are saved when [output] dir is empty. Finder hides the data directory, so on macOS they go
    to the Pictures folder, unless XDG_DATA_HOME chooses the data directory or it already holds images."""
    images = data_dir() / "images"
    if sys.platform != "darwin" or _variable("XDG_DATA_HOME") or images.is_dir():
        return images
    return _home("Pictures", "nai-cli", variable="[output] dir")


def _base(variable: str, *under_home: str) -> Path:
    """The folder a variable names, else the usual one under the home folder. An empty or relative value is
    ignored, as the XDG specification says, and the home folder is only looked up when needed."""
    return _variable(variable) or _home(*under_home, variable=variable)


def _variable(name: str) -> Path | None:
    value = os.environ.get(name, "")
    return Path(value) if value and Path(value).is_absolute() else None


def _home(*under_home: str, variable: str) -> Path:
    try:
        return Path.home().joinpath(*under_home)
    except RuntimeError as e:
        raise errors.NaiError(
            errors.CONFIG_MISSING, f"no home folder is known; set {variable} to the folder to use"
        ) from e


@dataclass(frozen=True)
class AccountConfig:
    token: str
    endpoint: api.Endpoint = api.OFFICIAL


def default_sam_dir() -> Path:
    return data_dir() / "models" / "sam"


@dataclass(frozen=True)
class LocalModels:
    tagger: Path | None = None
    tagger_labels: Path | None = None
    tagger_kind: str = ""
    sam_dir: Path = field(default_factory=default_sam_dir)


@dataclass(frozen=True)
class Config:
    path: Path
    novelai_token: str
    token_source: str | None
    output_dir: Path
    max_per_request: int
    # As written in the file; empty names files HHMMSS_seed
    filename: str = ""
    folder: str = DEFAULT_FOLDER
    defaults: dict[str, Any] = field(default_factory=dict)
    # Seconds between batch jobs that sent requests, drawn between the two
    interval: tuple[float, float] = (0.0, 0.0)
    accounts: dict[str, AccountConfig] = field(default_factory=dict)
    local: LocalModels = field(default_factory=LocalModels)

    def require_novelai(self) -> str:
        if not self.novelai_token:
            raise errors.NaiError(
                errors.CONFIG_MISSING, f"no NovelAI token: set {TOKEN_ENV} or novelai_token in {self.path}"
            )
        return self.novelai_token


def default_path() -> Path:
    return config_dir() / "config.toml"


def _table(raw: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise errors.NaiError(errors.CONFIG_MISSING, f"{path}: [{key}] must be a table")
    return value


def interval(pair: list[Any]) -> tuple[float, float]:
    """A wait as [seconds] twice or [min, max], each from 0 to a day."""
    if len(pair) != 2 or not all(_seconds(v) for v in pair) or pair[0] > pair[1]:
        raise ValueError(f"needs seconds up to {MAX_INTERVAL}, or [min, max] with min <= max: {pair!r}")
    return float(pair[0]), float(pair[1])


def _seconds(value: Any) -> bool:
    return values.is_number(value) and 0 <= value <= MAX_INTERVAL


def load(path: Path) -> Config:
    raw: dict = {}
    try:
        # Notepad saves UTF-8 with a byte order mark
        raw = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        pass
    except OSError as e:
        raise errors.NaiError(errors.CONFIG_MISSING, f"{path}: cannot read: {e}") from e
    except UnicodeDecodeError as e:
        raise errors.NaiError(errors.CONFIG_MISSING, f"{path}: not UTF-8 text, so it cannot be read: {e}") from e
    except tomllib.TOMLDecodeError as e:
        raise errors.NaiError(errors.CONFIG_MISSING, f"{path}: cannot parse: {e}") from e
    if _holds_nul(raw):
        raise errors.NaiError(errors.CONFIG_MISSING, f"{path}: a value holds a NUL character")
    env_token = os.environ.get(TOKEN_ENV, "").strip()
    file_token = raw.get("novelai_token", "")
    if not isinstance(file_token, str):
        raise errors.NaiError(errors.CONFIG_MISSING, f"{path}: novelai_token must be a string")
    file_token = file_token.strip()
    return Config(
        path=path,
        novelai_token=env_token or file_token,
        token_source="env" if env_token else ("config" if file_token else None),
        output_dir=_output_dir(raw, path),
        filename=_output_text(raw, path, "filename", ""),
        folder=_output_text(raw, path, "folder", DEFAULT_FOLDER),
        max_per_request=_limit(raw, path, "max_per_request", 100, " of Anlas"),
        defaults=_defaults(raw, path),
        interval=_interval(raw, path),
        accounts=_accounts(raw, path),
        local=_local(raw, path),
    )


def _output_dir(raw: dict[str, Any], path: Path) -> Path:
    value = _table(raw, "output", path).get("dir", "")
    if not isinstance(value, str):
        raise errors.NaiError(errors.CONFIG_MISSING, f"{path}: [output] dir must be a string: {value!r}")
    return values.path(value or default_output_dir())


def _local(raw: dict[str, Any], path: Path) -> LocalModels:
    table = _table(raw, "local", path)
    given = {}
    for key in ("tagger", "tagger_labels", "tagger_kind", "sam_dir"):
        value = table.get(key, "")
        if not isinstance(value, str):
            raise errors.NaiError(errors.CONFIG_MISSING, f"{path}: [local] {key} must be a string: {value!r}")
        given[key] = value.strip()

    def path_of(key: str) -> Path | None:
        return values.path(given[key]) if given[key] else None

    return LocalModels(
        tagger=path_of("tagger"),
        tagger_labels=path_of("tagger_labels"),
        tagger_kind=given["tagger_kind"],
        sam_dir=path_of("sam_dir") or default_sam_dir(),
    )


def _output_text(raw: dict[str, Any], path: Path, key: str, default: str) -> str:
    value = _table(raw, "output", path).get(key, default)
    if not isinstance(value, str):
        raise errors.NaiError(errors.CONFIG_MISSING, f"{path}: [output] {key} must be a string: {value!r}")
    return value


def _limit(raw: dict[str, Any], path: Path, key: str, default: int, unit: str) -> int:
    value = _table(raw, "cost", path).get(key, default)
    if not values.is_whole(value) or value < 0:
        raise errors.NaiError(errors.CONFIG_MISSING, f"{path}: [cost] {key} must be a whole number{unit}: {value!r}")
    return value


def _defaults(raw: dict[str, Any], path: Path) -> dict[str, Any]:
    raw = _table(raw, "defaults", path)
    unknown = set(raw) - set(DEFAULT_KEYS)
    if unknown:
        raise errors.NaiError(
            errors.CONFIG_MISSING,
            f"{path}: unknown keys in [defaults]: {', '.join(sorted(unknown))}",
            choices=list(DEFAULT_KEYS),
        )
    out: dict[str, Any] = {}
    for key, value in raw.items():
        kind = DEFAULT_KEYS[key]
        # A float would be cut silently to a whole number, and 1 must not pass as a switch
        right = {
            bool: isinstance(value, bool),
            str: isinstance(value, str),
            int: values.is_whole(value),
            float: values.is_number(value),
        }[kind]
        if not right:
            raise errors.NaiError(errors.CONFIG_MISSING, f"{path}: [defaults] {key} has the wrong type: {value!r}")
        out[key] = kind(value)
    return out


def _interval(raw: dict[str, Any], path: Path) -> tuple[float, float]:
    value = _table(raw, "batch", path).get("interval", 0)
    pair = value if isinstance(value, list) else [value, value]
    try:
        return interval(pair)
    except ValueError as e:
        raise errors.NaiError(errors.CONFIG_MISSING, f"{path}: [batch] interval: {e}") from None


def _accounts(raw: dict[str, Any], path: Path) -> dict[str, AccountConfig]:
    out = {}
    for name, table in _table(raw, "accounts", path).items():
        where = f"{path}: [accounts.{name}]"
        if name == DEFAULT_ACCOUNT:
            raise errors.NaiError(errors.CONFIG_MISSING, f"{where}: {name} is the account of novelai_token")
        if not isinstance(table, dict) or set(table) - {"token", "api_url", "image_url"}:
            raise errors.NaiError(errors.CONFIG_MISSING, f"{where} takes token, api_url and image_url")
        token, api_url, image_url = (table.get(k, "") for k in ("token", "api_url", "image_url"))
        if not all(isinstance(v, str) for v in (token, api_url, image_url)) or not token.strip():
            raise errors.NaiError(errors.CONFIG_MISSING, f"{where} needs a token, and its values must be strings")
        try:
            endpoint = api.endpoint(api_url, image_url) if api_url else api.OFFICIAL
        except ValueError as e:
            raise errors.NaiError(errors.CONFIG_MISSING, f"{where}: {e}") from None
        if image_url and not api_url:
            raise errors.NaiError(errors.CONFIG_MISSING, f"{where}: image_url needs api_url")
        out[name] = AccountConfig(token.strip(), endpoint)
    return out


def _holds_nul(value: Any) -> bool:
    if isinstance(value, str):
        return "\x00" in value
    if isinstance(value, dict):
        return any(_holds_nul(k) or _holds_nul(v) for k, v in value.items())
    return isinstance(value, list) and any(map(_holds_nul, value))


def init(path: Path, overwrite: bool) -> bool:
    """Write the template, readable only by the user since it holds the token; returns whether a file was
    written. An existing file is kept unless overwrite is given."""
    if overwrite:
        fs.replace(path, TEMPLATE, private=True)
        return True
    try:
        fs.create(path, TEMPLATE, private=True)
    except FileExistsError:
        return False
    return True
