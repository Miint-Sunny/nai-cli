"""gen, i2i, inpaint, enhance and variations: price the request, pay under the lock, save the images."""

import base64
from dataclasses import replace
from datetime import datetime
from typing import Any

from nai_cli import api, caches, cost, images, inpaint, naming, values, vibe
from nai_cli import request as R
from nai_cli.charge import Charge, authorize, cost_json, keep_usable
from nai_cli.commands import output
from nai_cli.commands.context import Context
from nai_cli.commands.generation import Enhanced, GenSettings, Repaint
from nai_cli.commands.spec import Prepared, attach_vibes, prepare

# The label the official client gives each kind of image, which {type} names; plain generation has none
_TYPE_LABELS = {"i2i": "image2image", "inpaint": "inpaint", "enhance": "enhanced", "variations": "variations"}


def run(s: GenSettings, ctx: Context) -> dict[str, Any]:
    prep = prepare(s)
    built = prep.built
    preview = _preview(s, prep)
    # A run that sends nothing paid is priced here; a real one against the balance under the lock
    dry_warnings: list[str] = []
    account = ctx.dry_account(dry_warnings) if s.dry_run or preview else None
    estimate = _price(s, prep, account, len(prep.pending_encodes))
    result = _result(s, prep, estimate, account)
    result["warnings"] += dry_warnings
    # A preview is refused on the same grounds as the request it previews
    cost.require_valid(estimate)
    if preview is not None and not s.dry_run:
        return _save_preview(s, ctx, preview, built, result)
    if s.dry_run:
        return _dry_run(s, ctx, prep, estimate, account, result)
    # Fails without a token before any folder is made
    ctx.client()
    output.ready(s.output)
    with ctx.charged(result["warnings"], command=s.action, model=built.body["model"], seed=built.seed) as charge:
        estimate, built = _pay_vibes(s, ctx, prep, charge, result)
        files, raw_files = _generate(s, ctx, prep, built, charge, result["warnings"])
    result["images"] = files
    if raw_files:
        result["raw_images"] = raw_files
    result["cost"] = cost_json(estimate, None, charge)
    if charge.client.retries:
        result["retries"] = charge.client.retries
    return result


def _dry_run(
    s: GenSettings,
    ctx: Context,
    prep: Prepared,
    estimate: cost.Estimate,
    account: api.Account | None,
    result: dict[str, Any],
) -> dict[str, Any]:
    """The request priced and checked as the real run checks it before sending, with the refusal it would meet."""
    output.check(s.output)
    result["dry_run"] = True
    if _preview(s, prep) is not None:
        # A preview sends nothing paid, so no limit refuses it and a batch adds nothing for it
        result["preview"] = None
        return result
    refused = ctx.refusal(estimate.anlas, account, s.yes, result["cost"])
    if refused:
        result["refused"] = refused
    return result


def _preview(s: GenSettings, prep: Prepared) -> inpaint.Job | None:
    """The repaint job to show instead of sending, when --preview asks for it."""
    return prep.job if isinstance(s.canvas, Repaint) and s.canvas.area.preview else None


def _price(s: GenSettings, prep: Prepared, account: api.Account | None, to_encode: int) -> cost.Estimate:
    built, spec = prep.built, prep.spec
    sent = built.body["parameters"]
    billed_width, billed_height = (
        s.canvas.plan.billed_size if isinstance(s.canvas, Enhanced) else (sent["width"], sent["height"])
    )
    return cost.generation(
        built.model,
        billed_width,
        billed_height,
        sent["steps"],
        n_samples=spec.n_samples,
        strength=_cost_strength(spec),
        opus=account.opus if account else True,
        quota_exhausted=account.quota_negative if account else False,
        precise_references=len(spec.references),
        vibes=len(spec.vibes) if R.vibes_used(spec) else 0,
        vibes_to_encode=to_encode,
    )


def _cost_strength(spec: R.GenSpec) -> float:
    # img2img is discounted by strength and inpainting by inpaint strength
    if spec.action == "img2img":
        return spec.strength
    if spec.action == "infill":
        return spec.inpaint_strength
    return 1.0


def _result(s: GenSettings, prep: Prepared, estimate: cost.Estimate, account: api.Account | None) -> dict[str, Any]:
    built, spec = prep.built, prep.spec
    result: dict[str, Any] = {
        "ok": True,
        "model": built.body["model"],
        "action": s.action,
        "seed": built.seed,
        "size": f"{spec.width}x{spec.height}",
        "prompt": built.prompt,
        "negative": built.negative,
        "characters": [{"prompt": c.prompt, "negative": c.negative} for c in built.characters],
        "cost": cost_json(estimate, account),
        "lint": [i.to_json() for i in prep.issues],
        "tokens": prep.tokens.to_json(),
        "warnings": prep.warnings,
    }
    if prep.job is not None:
        result["inpaint"] = prep.job.to_json()
    if s.show_body or s.dry_run:
        result["body"] = redact(built.body)
    if isinstance(s.canvas, Enhanced):
        result["enhance"] = s.canvas.plan.to_json()
    return result


def redact(body: dict[str, Any]) -> dict[str, Any]:
    """The body as sent, with images replaced by their length."""
    body = values.as_js(body)
    params = dict(body["parameters"])
    for key in ("image", "mask"):
        if key in params:
            params[key] = f"<{len(params[key])} base64 characters>"
    for key in ("reference_image_multiple", "director_reference_images"):
        if key in params:
            params[key] = [f"<{len(v)} base64 characters>" for v in params[key]]
    return {**body, "parameters": params}


def _save_preview(
    s: GenSettings, ctx: Context, job: inpaint.Job, built: R.Built, result: dict[str, Any]
) -> dict[str, Any]:
    when = ctx.now()
    picture = images.to_png(job.preview())
    image = output.named(s.output, naming_meta(s, built), picture, when, seed=built.seed, index=0)
    result["preview"] = output.save_unpaid(
        picture, s.output.destination.image_path(image.meta, built.seed, suffix="_preview", name=image.name)
    )
    return result


def _pay_vibes(
    s: GenSettings, ctx: Context, prep: Prepared, charge: Charge, result: dict[str, Any]
) -> tuple[cost.Estimate, R.Built]:
    """Price against the balance read under the lock, then encode the Vibes still missing."""
    spec, built = prep.spec, prep.built
    # Another process may have encoded the same Vibes while this one waited
    pending = attach_vibes(spec, prep.vibe_sources)
    estimate = _price(s, prep, charge.before, len(pending))
    authorize(estimate.anlas, spec.n_samples, charge, ctx.config, s.yes, cost_json(estimate, charge.before))
    fresh = {}
    for encode in pending:
        source = encode.source
        encoding = charge.pay(
            charge.client.encode_vibe,
            base64.b64decode(source.image_b64 or ""),
            spec.model.id,
            source.information_extracted,
            source.mask,
        )
        fresh.update(dict.fromkeys(encode.positions, encoding))
        remember(source, spec.model.id, encoding, result["warnings"])
    if prep.vibe_sources:
        attach_vibes(spec, prep.vibe_sources, fresh)
        built = R.build(replace(spec, seed=built.seed))
        if "body" in result:
            result["body"] = redact(built.body)
    return estimate, built


def remember(source: vibe.VibeSource, model: str, encoding: str, warnings: list[str]) -> None:
    # The encoding is used from memory, so a cache that cannot be written only means paying again next time
    try:
        caches.remember_vibe(source, model, encoding)
    except OSError as e:
        warnings.append(f"the Vibe encoding could not be cached, so encoding this image again costs Anlas: {e}")


def _generate(
    s: GenSettings,
    ctx: Context,
    prep: Prepared,
    built: R.Built,
    charge: Charge,
    warnings: list[str],
) -> tuple[list[str], list[str]]:
    def save(pictures: list[bytes | None]) -> tuple[list[str], list[str]]:
        return _save_images(s, ctx, charge, pictures, built, prep.job, warnings)

    try:
        pictures = charge.pay(charge.client.generate, built.body)
    except api.DamagedImages as e:
        keep_usable(e, save, warnings)
    return save(pictures)


def _save_images(
    s: GenSettings,
    ctx: Context,
    charge: Charge,
    pictures: list[bytes | None],
    built: R.Built,
    job: inpaint.Job | None,
    warnings: list[str],
) -> tuple[list[str], list[str]]:
    keep_raw = isinstance(s.canvas, Repaint) and s.canvas.area.keep_raw
    named = naming_meta(s, built)
    files, raw_files = [], []
    for i, data in enumerate(pictures):
        # Damaged in the reply; the others keep their own numbers
        if data is None:
            continue
        result = data
        if job is not None:
            try:
                result = job.paste(data)
            except Exception as e:
                # Anlas is already spent, so keep the server image even if compositing fails
                warnings.append(
                    f"compositing failed, saved the image as returned by the server: {type(e).__name__}: {e}"
                )
        when: datetime = ctx.now()
        # The server gives each image of a batch the next seed; sizes are the server's, as the official names use
        image = output.named(s.output, named, data, when, seed=built.seed + i, index=i)
        files.append(output.save_image(charge.keep, result, s.output, image, warnings, built.seed, i))
        if job is not None and keep_raw:
            raw_files.append(output.save_image(charge.keep, data, s.output, image, warnings, built.seed, i, "_raw"))
    return files, raw_files


def naming_meta(s: GenSettings, built: R.Built) -> naming.Meta:
    """{prompt} is the prompt as written, without the fixed tags or tidying; the settings are those sent."""
    params = built.body["parameters"]
    return naming.Meta(
        prompt=s.prompt.value,
        model=s.model.id,
        steps=params.get("steps"),
        cfg=params.get("scale"),
        sampler=params.get("sampler", ""),
        type=_TYPE_LABELS.get(s.command, ""),
        characters=tuple(naming.Character(c.prompt.value) for c in s.characters),
    )
