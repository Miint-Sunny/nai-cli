# Changelog

## Unreleased

- `nai meta` also reads settings hidden in the RGB channels (`stealth_rgbinfo`, `stealth_rgbcomp`).
- Copies made by `nai strip`, `nai redact` and `nai watermark` no longer keep stealth data hidden in the RGB channels. Only the bits it occupies change.
- `nai strip`, `nai redact` and `nai watermark` take folders, with `-r` for the folders inside them. With `-o`, a copy keeps its path inside the folder given.
- `--rename` names copies by date and number, such as `20261123-0001.png`, instead of after their source. NovelAI's file names hold the start of the prompt and the seed, and copying an image named so warns when `--rename` is not given.
- `nai strip --keep-format` keeps each copy in its source's format. JPEG and NovelAI's lossy WebP lose their metadata without being encoded again, and with it what follows the image in the file; lossless WebP stays lossless. Every strip copy is now read back before it is written, and refused if any metadata is left.

## 0.1.0 - 2026-11-20

First release.
