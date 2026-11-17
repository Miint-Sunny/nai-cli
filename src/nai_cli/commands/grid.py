"""grid: run a command once per cell of settings varied along two axes, then save a labelled sheet of the images."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nai_cli import errors, fs, grid, images, naming, sheet
from nai_cli.commands import batch
from nai_cli.commands.context import Context
from nai_cli.commands.texts import text_arg


@dataclass(frozen=True)
class GridSettings:
    grid: grid.Grid
    options: batch.RunOptions
    sheet: bool


def resolve(opts: dict[str, Any], ctx: Context) -> GridSettings:
    try:
        x = grid.parse_axis([_entry(e) for e in opts.get("x") or []])
        y = grid.parse_axis([_entry(e) for e in opts.get("y") or []])
        built = grid.build(list(opts.get("job") or []), x, y, ctx.parse_job)
        options = batch.run_options(opts, ctx)
    except errors.NaiError as e:
        raise batch.as_summary(e) from None
    return GridSettings(built, options, opts.get("sheet") is not False)


def _entry(text: str) -> tuple[str, str]:
    name, sep, values = text.partition("=")
    if not sep:
        raise errors.NaiError(errors.BAD_ARGS, f"an axis is PARAM=VALUES: {text}")
    # @FILE gives the values from a file, one prompt per line
    return name, text_arg(values)


def run(s: GridSettings, ctx: Context) -> dict[str, Any]:
    g = s.grid
    if not s.options.dry_run:
        _try_cells(g, ctx)
    ran = batch.run_jobs(tuple(c.job for c in g.cells), set(), s.options, ctx)
    layout: dict[str, Any] = {
        "columns": g.columns,
        "rows": g.rows,
        "x": g.x.title if g.x else None,
        "y": g.y.title if g.y else None,
    }
    if s.sheet and not s.options.dry_run:
        try:
            layout["sheet"] = _save_sheet(g, ran.results)
        except KeyboardInterrupt:
            ran.totals.code = 130
            layout["sheet"] = None
        except (OSError, ValueError, errors.NaiError) as e:
            # The cells are paid for and saved; only the sheet is missing
            layout["sheet"] = None
            layout["sheet_error"] = str(e)
    return batch.summary(ran, s.options.dry_run, grid=layout)


def _try_cells(g: grid.Grid, ctx: Context) -> None:
    """Every cell as a dry run before the first is sent, so a value the command refuses, such as steps past the
    limit, fails the grid before any cell is paid for."""
    for cell in g.cells:
        result, code = ctx.run_job(cell.job.argv, ctx.job(cell.job.argv, dry_run=True))
        if code == 130 or ctx.shared.interrupted:
            raise batch.as_summary(errors.NaiError(errors.INTERRUPTED, "interrupted"))
        if code:
            message = f"cell {cell.job.id} cannot run, so no cell was sent: {result.get('message')}"
            raise batch.as_summary(errors.NaiError(result.get("code") or errors.BAD_ARGS, message))


def _save_sheet(g: grid.Grid, results: dict[str, dict[str, Any]]) -> str | None:
    """NAME_grid.png beside the first image; nothing when no cell made one."""
    files = {c.job.id: _first_image(results.get(c.job.id)) for c in g.cells}
    first = next((f for f in files.values() if f is not None), None)
    if first is None:
        return None
    rows: list[list[Any]] = [[None] * g.columns for _ in range(g.rows)]
    for c in g.cells:
        path = files[c.job.id]
        rows[c.row][c.column] = images.open_image(path) if path is not None else None
    picture = sheet.render(
        sheet.Sheet(
            rows,
            list(g.x.labels) if g.x else [],
            list(g.y.labels) if g.y else [],
            g.x.title if g.x else "",
            g.y.title if g.y else "",
        )
    )
    return str(fs.create_numbered(naming.copy_path(first, "grid"), images.to_png(picture)))


def _first_image(result: dict[str, Any] | None) -> Path | None:
    found = (result or {}).get("images") or []
    return Path(found[0]) if found and result and result.get("ok") else None
