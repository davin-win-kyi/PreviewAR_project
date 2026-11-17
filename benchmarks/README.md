# benchmarks

Stage 3 output: `scale_eval.json`, written by `benchmarking_rrmse.py` from `mesh_outputs/meshes.json`. The checked-in file is identical to `paper_benchmarks/run#3/scale_eval.json`. See the [main README](../README.md) for the full pipeline.

## How the evaluation works

For each record in `meshes.json`:

1. Read the listed dimensions from `standardized_dimensions`, in inches.
2. Measure the mesh: Blender runs `utils/blender_bbox.py` on the GLB and reports the bounding box of its largest mesh object, converted from meters to inches.
3. Sort both sets of dimensions largest-first, so the comparison doesn't depend on axis order.
4. Divide each set by its largest value to get proportions, e.g. (84, 35, 33.5) becomes (1, 0.42, 0.40).
5. Compute the RMSE between the two sets of proportions. This is the relative RMSE (RRMSE); multiplied by 100, it's a percentage.

The meshes aren't generated at real-world scale, so RRMSE, which only compares proportions, is the headline metric. Absolute errors in inches are recorded as well.

Running it requires `blender` on your `PATH`. The script has no command-line options; edit the defaults of `evaluate()` to change paths.

## `scale_eval.json`

| Field | Meaning |
|---|---|
| `summary.mean_relative_rmse_percent` | Headline metric: mean RRMSE over evaluated records, in percent |
| `summary.mean_relative_rmse_normalized` | The same, as a fraction |
| `summary.rmse_absolute_in` | RMSE in inches between the sorted absolute dimensions, over all records |
| `summary.n_ok`, `summary.n_fail` | Records evaluated, and records skipped because of a missing GLB, unreadable dimensions, or a Blender error |
| `items.<name>` | Per product: measured and listed dimensions (`glb_absolute`, `actual_absolute`), their proportions (`glb_relative`, `actual_relative`), and `metrics` |

`items` is keyed by product name, so a product with several records in `meshes.json` appears once in `items`, but `summary` counts it once per record. See [mesh_outputs/](../mesh_outputs/) for why products can have several records.
