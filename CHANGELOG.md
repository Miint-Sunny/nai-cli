# Changelog

## Unreleased

- `nai meta` reads both copies of the settings NovelAI writes, the text and the stealth data in the alpha channel. `layers` says whether they agree, and when the text was changed after NovelAI signed it, the settings come from the stealth data. `--layer` reads one of them only.

## 0.1.0 - 2026-11-20

First release.
