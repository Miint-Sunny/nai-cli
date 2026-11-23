# Changelog

## Unreleased

- `nai meta` also reads settings hidden in the RGB channels (`stealth_rgbinfo`, `stealth_rgbcomp`).
- Copies made by `nai strip`, `nai redact` and `nai watermark` no longer keep stealth data hidden in the RGB channels. Only the bits it occupies change.
- `nai strip`, `nai redact` and `nai watermark` take folders, with `-r` for the folders inside them. With `-o`, a copy keeps its path inside the folder given.

## 0.1.0 - 2026-11-20

First release.
