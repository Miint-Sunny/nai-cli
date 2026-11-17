# Inpainting

`nai inpaint` repaints part of an image and composites the result back onto the original. The files in `images` are finished images; `--keep-raw` also saves what the server returned.

## Choose the area

Combine any of these:

- `--region SHAPE` in fractions of the canvas, from `0,0` at the top left to `1,1` at the bottom right: `rect:x1,y1,x2,y2`, `ellipse:x1,y1,x2,y2` (its bounding box), `poly:x,y,x,y,x,y,...` or `object:x,y`, the object under the point, found by the local SAM model once `nai local download sam` has fetched it. Estimate the coordinates by looking at the image. A leading `-` subtracts a shape, as in `--region -rect:...`, and regions apply in the order given.
- `--mask IMAGE` when the user provides a mask. White or opaque areas are repainted.
- `--expand L,T,R,B` to add canvas in pixels on each side, or one number for all sides. The new area is repainted automatically.
- `--grow PX` to widen the whole area.

## Check with a preview

1. Run with `--preview`. It sends nothing paid and saves an image whose path is in `preview`.
2. Look at it. Red marks the area the server will repaint, in 8-pixel cells. A green box marks the focus crop.
3. If the red area misses the target or covers too much, adjust the coordinates or `--grow` and preview again. Leave a small margin around the target.
4. Run the same command without `--preview`.

## Write the prompt

Describe the whole image, then what the area should become, as in `1girl, sitting, cafe, open hand, five fingers`. For an image made with NovelAI, `nai meta` gives the original prompt to start from.

## Focus and cost

When the area is small, nai-cli repaints a crop around it at about 1 MP and composites it back, which is free with Opus and keeps the full resolution. `inpaint.focus` in the output shows the crop and the size sent; `null` means the whole image is sent.

If `warnings` says the canvas is scaled down to fit the generation limit, the result will be smaller than the original. Make the area smaller so the crop fits in 1 MP, or ask the user whether to add `--focus`, which keeps the size but may cost Anlas.

## Characters

`--char-pos x,y` is relative to the whole image even when focusing; nai-cli maps it into the crop.
