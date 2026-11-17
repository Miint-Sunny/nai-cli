# Vibes and precise references

Vibes carry the look of an image into a generation; precise references carry a character or style.

## Which models

`--vibe` and `--vibes-from` need a V4.5 or V4 model, and `--ref` a V4.5 model (`--model v4.5`). On V5 they fail with `model_unsupported`; tell the user and let them pick the model.

## Using Vibes

- `--vibe` takes `.naiv4vibe` and `.naiv4vibebundle` files as well as images. An image is encoded first, for 2 Anlas, unless it was encoded with the same settings before.
- `--vibes-from IMAGE` reuses the Vibes kept in a generated image for the model that made the image, without encoding them again. A different target model cannot re-encode them: the settings hold encodings, not the source images.
- Generation costs an extra 5 Anlas per reference per output image, even with Opus. Two output images with one precise reference add 10 Anlas to the base price.

## Saving Vibes

`nai vibe SOURCE...` saves Vibes as a `.naiv4vibe` file, or a `.naiv4vibebundle` for several, for 2 Anlas per image encoded, or for nothing when the images were encoded with the same settings before. Check it with `--dry-run` first, as with generation. Reuse the saved file with `--vibe` to avoid paying again.
