"""Random prompts as the official client's dice button writes them for models with character prompts."""

import functools
import json
import math
import random
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from typing import Any

# Each random character's negative prompt and centre, as the official client fills them
CHARACTER_NEGATIVE = "lowres, aliasing, "
CHARACTER_CENTER = (0.0, 0.0)


class Recipe(Enum):
    # The flags an entry declares take effect, and a few faulty records are corrected
    CORRECTED = "corrected"
    # The official code as it runs: a pick never sets its entry's flags, so entries needing them never come up
    OFFICIAL = "official"


@dataclass(frozen=True)
class RandomPrompt:
    prompt: str
    characters: tuple[str, ...]


def generate(recipe: Recipe = Recipe.CORRECTED, rng: random.Random | None = None) -> RandomPrompt:
    engine = (_Official if recipe is Recipe.OFFICIAL else _Corrected)(rng or random.Random())
    prompt, *characters = engine.scene()
    return RandomPrompt(prompt, tuple(characters))


# An entry is [text, weight, flags it sets, flags it needs, flags that rule it out]
Entry = list[Any]

_ERRATA = (
    # No entry sets feel; every other sock needs feet
    ("legwear", ["socks", 12, [], ["feel"], []], ["socks", 12, [], ["feet"], []]),
    # A doubled weight pushes the eyes exclusion out of its field
    ("accessory", ["opaque glasses", 12, 12, [], [], ["eyes"]], ["opaque glasses", 12, [], [], ["eyes"]]),
    # Flags are a list; the bare string set nothing, so a second location could follow
    ("setting", ["ocean", 12, "location", [], ["location"]], ["ocean", 12, ["location"], [], ["location"]]),
    # Closed eyes, flagged as the other closed-eye entries are
    ("action", ["sleeping", 12, [], [], []], ["sleeping", 12, ["eyes"], [], ["eyes", "noeyes"]]),
    ("expression", ["^_^", 12, [], [], []], ["^_^", 12, ["eyes"], [], ["eyes", "noeyes"]]),
)
_GENDER_FLAGS = {"f": "female", "m": "male"}
_FOOTLESS_LEGWEAR = {"leggings", "leg warmers"}


@functools.cache
def word_lists() -> dict[str, list[Entry]]:
    """The official client's lists, verbatim."""
    with resources.files("nai_cli").joinpath("data/random_prompt.json").open(encoding="utf-8") as f:
        return json.load(f)


@functools.cache
def corrected_lists() -> dict[str, list[Entry]]:
    lists = {name: list(entries) for name, entries in word_lists().items()}
    for group, original, corrected in _ERRATA:
        entries = lists[group]
        entries[entries.index(original)] = corrected
    return lists


def _field(entry: Entry, index: int) -> list[str]:
    return entry[index] if len(entry) > index else []


class _Official:
    """The dice button's recipe, step for step, so each tag comes up as often as it does there."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.lists = self.load()

    def scene(self) -> list[str]:
        tags: list[str] = []
        state: set[str] = set()
        people = self.literal(((1, 70), (2, 20), (3, 7), (0, 5)))
        if people == 0:
            return [self.no_humans(state)]
        if self.chance(0.5):
            tags.append(self.pick("style", state))
        genders = [self.literal((("m", 30), ("f", 60), ("o", 0))) for _ in range(people)]
        # Prepended girls first, so boys and then others end up in front
        for gender, word in (("f", "girl"), ("m", "boy"), ("o", "other")):
            count = genders.count(gender)
            if count:
                tags.insert(0, f"{count}{word}" + ("s" if count > 1 else ""))
        if self.chance(0.8):
            background = self.pick("background", state)
            tags.append(background)
            if background == "scenery" and self.chance(0.5):
                tags += [self.pick("setting", state) for _ in range(self.below(3, 1))]
        if self.chance(0.3):
            tags.append(self.pick("angle", state))
        if self.chance(0.7) and (framing := self.pick("framing", state)):
            tags.append(framing)
        characters = [
            ", ".join([word, *self.character(state, gender, people)])
            for gender, word in (("f", "girl"), ("m", "boy"), ("o", "other"))
            for _ in range(genders.count(gender))
        ]
        if self.chance(0.2):
            count = self.below(4)
            if people == 2:
                count = self.below(3)
            tags += [self.pick("object", state) for _ in range(count)]
        if self.chance(0.25):
            tags += [self.pick("effect", state) for _ in range(self.below(3, 1))]
        if self.chance(0.2):
            tags.append(self.pick("year", state))
        if self.chance(0.1):
            tags.append(self.pick("focus", state))
        unique = dict.fromkeys(", ".join(tags).split(", "))
        return [", ".join(t for t in unique if t), *characters]

    def no_humans(self, state: set[str]) -> str:
        tags = ["no humans"]
        if self.chance(0.5):
            tags.append(self.pick("style", state))
        tags.append(self.pick("landscape", state))
        tags += [self.pick("setting", state) for _ in range(self.literal(((2, 15), (3, 50), (4, 15), (5, 5))))]
        objects = self.literal(((0, 15), (1, 10), (2, 20), (3, 20), (4, 20), (5, 15)))
        tags += [self.pick("object", state) for _ in range(objects)]
        return ", ".join(tags)

    def character(self, scene: set[str], gender: str, people: int) -> list[str]:
        tags: list[str] = []
        state = self.character_state(scene, gender)
        if self.chance(0.1):
            tags.append(self.pick("animal", state))
        self.after_animal(state)
        if self.chance(0.4):
            tags.append(self.pick("skin", state))
        if self.chance(0.05):
            tags.append(self.pick("eye_special", state))
        if not self.eyes_hidden(state):
            if self.chance(0.2):
                tags.append(self.pick("eye_shape", state))
            if self.chance(0.8) and "nocoloreyes" not in state:
                tags.append(self.pick("eye_color", state))
        for group, odds in (("hair_length", 0.8), ("hair_braid", 0.7), ("hair_color", 0.7)):
            if self.chance(odds):
                tags.append(self.pick(group, state))
        if self.chance(0.1):
            tags += [self.pick("hair_multicolor", state), self.pick("hair_color", state)]
        for group, odds in (("hair_style", 0.3), ("bangs", 0.4)):
            if self.chance(odds):
                tags.append(self.pick(group, state))
        if gender == "f" and self.chance(0.8):
            tags.append(self.pick("breasts", state))
        tags += [self.pick("body", state) for _ in range(self.count(people))]
        if self.chance(0.2):
            tags.append(self.pick("hat", state))
            if self.chance(0.2):
                tags.append(self.pick("hat_extra", state))
        elif self.chance(0.3):
            tags.append(self.pick("headwear", state))
        clothes = self.literal((("uniform", 25), ("swimsuit", 5), ("bodysuit", 5), ("normal clothes", 40)))
        covered = self.separates(tags, state, gender) if clothes == "normal clothes" else False
        if clothes != "normal clothes":
            tags.append(self.pick(clothes, state))
        self.after_clothes(state, covered)
        for group, odds in (("expression", 0.6), ("action", 0.4)):
            if self.chance(odds):
                tags.append(self.pick(group, state))
        tags += [self.pick("accessory", state) for _ in range(self.count(people))]
        return [t for t in tags if t]

    def separates(self, tags: list[str], state: set[str], gender: str) -> bool:
        """Clothes for a character in normal clothes; whether something covers the feet."""
        legwear = ""
        if gender == "f" and self.chance(0.5):
            legwear = self.pick("legwear", state)
            tags.append(legwear)
            if self.chance(0.2):
                tags.append(self.pick("legwear_extra", state))
        footwear = ""
        if gender == "f" and self.chance(0.2):
            tags.append(self.colored("dress", state))
        else:
            if self.chance(0.85):
                tags.append(self.colored("top", state))
            if "legs" in state:
                if self.chance(0.85):
                    tags.append(self.colored("bottom", state))
                if "feet" in state and self.chance(0.6):
                    footwear = self.colored("footwear", state)
                    tags.append(footwear)
        return bool(footwear) or bool(legwear and legwear not in _FOOTLESS_LEGWEAR)

    def colored(self, group: str, state: set[str]) -> str:
        use = self.chance(0.5)
        color = self.pick("color", state)
        value = self.pick(group, state)
        return f"{color} {value}" if value and use else value

    def count(self, people: int) -> int:
        if people == 1:
            return self.literal(((0, 10), (1, 30), (2, 15), (3, 5)))
        if people == 2:
            return self.literal(((0, 20), (1, 40), (2, 10)))
        return self.literal(((0, 30), (1, 30)))

    def pick(self, group: str, state: set[str]) -> str:
        eligible = [
            e
            for e in self.lists[group]
            if not any(f in state for f in _field(e, 4)) and all(f in state for f in _field(e, 3))
        ]
        ticket, reached = self.below(sum(e[1] for e in eligible), 1), 0
        for entry in eligible:
            reached += entry[1]
            if ticket <= reached:
                state.update(self.flags(entry))
                return entry[0]
        return ""

    def literal(self, choices: tuple[tuple[Any, int], ...]) -> Any:
        ticket, reached = self.below(sum(w for _, w in choices), 1), 0
        for value, weight in choices:
            reached += weight
            if ticket <= reached:
                return value
        return choices[0][0]

    def chance(self, odds: float) -> bool:
        return self.rng.random() < odds

    def below(self, high: int, low: int = 0) -> int:
        # The official draw, which never reaches high
        return math.floor(self.rng.random() * (high - low)) + low

    # The steps where the two recipes differ

    def load(self) -> dict[str, list[Entry]]:
        return word_lists()

    def flags(self, entry: Entry) -> list[str]:
        # The official code adds back what the entry needed, which is already set
        return _field(entry, 3)

    def character_state(self, scene: set[str], gender: str) -> set[str]:
        return set(scene)

    def after_animal(self, state: set[str]) -> None:
        pass

    def eyes_hidden(self, state: set[str]) -> bool:
        # Checks a tag, though state only holds flags
        return "no eyes" in state

    def after_clothes(self, state: set[str], covered: bool) -> None:
        pass


class _Corrected(_Official):
    def load(self) -> dict[str, list[Entry]]:
        return corrected_lists()

    def flags(self, entry: Entry) -> list[str]:
        return _field(entry, 2)

    def character_state(self, scene: set[str], gender: str) -> set[str]:
        # The lists need gender flags that nothing sets
        return scene | ({_GENDER_FLAGS[gender]} if gender in _GENDER_FLAGS else set())

    def after_animal(self, state: set[str]) -> None:
        # A legless body cancels the legs and feet that framing set for the whole scene
        if "nolegs" in state:
            state -= {"legs", "feet"}

    def eyes_hidden(self, state: set[str]) -> bool:
        return "noeyes" in state

    def after_clothes(self, state: set[str], covered: bool) -> None:
        # Toenail polish needs barefoot, which nothing sets
        if "feet" in state and not covered:
            state.add("barefoot")
