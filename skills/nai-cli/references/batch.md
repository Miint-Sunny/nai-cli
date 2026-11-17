# Batches and grids

`nai batch` runs many commands from one file, and `nai grid` runs one command for each combination of settings.

## Batch

Write one argument list per line to a JSONL file, such as `["gen", "--prompt", "cat"]`, or an object `{"id": "cat-1", "args": [...]}` to name the job. Run `nai batch FILE --dry-run` first to see the total `estimate`. Each job is priced on its own, so a job that uses an image an earlier one makes fails in the dry run; run those in a second batch.

Every job prints one line with `"type": "job"` as soon as it ends, and the last line, with `"type": "summary"`, totals them. Add `--yes` inside the jobs the user agreed to pay for, since each job keeps its own limit. `--stop-on-error` stops at the first failure, and `--interval S` spaces the requests.

Append the output of each run to one file. After a failure, `nai batch FILE --skip-done OUTPUT` skips only the jobs that succeeded. A job that failed with `maybe_processed` may already have been charged: show the user its `cost` and ask before resuming.

## Grid

To compare settings, `nai grid --x steps=20,28 --y cfg=4,6 gen --prompt ...` runs one image per cell and saves a labelled sheet. Check the total with `--dry-run` first. The grid picks one random seed for all cells unless the base command supplies `--seed`, an axis varies it (for example `--x seed=1,2`), or `--from-image` is asked to import it with `--import seed` or `--import all`.
