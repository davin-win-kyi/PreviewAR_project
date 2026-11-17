# paper_benchmarks

Results from three complete runs of the pipeline on the same 50 products: five each of beds, cabinets, chairs, couches, desks, lamps, shelves, stools, tables, and wardrobes. See the [main README](../README.md) for the full pipeline.

| Path | Contents |
|---|---|
| `run#N/preprocessing_runtime.json` | Stage 1 runtimes, copied from `outputs_preprocessing/` |
| `run#N/runtimes.json` | Stage 2 runtimes, copied from `mesh_outputs/` |
| `run#N/scale_eval.json` | Stage 3 results, copied from `benchmarks/` |
| `benchmark_summary.json` | Per-run and overall averages, written by `finalize_benchmarks.py` |

`finalize_benchmarks.py` averages `runtime_sec` over the records with `status: "ok"` in each runtime file, takes `mean_relative_rmse_normalized` and `mean_relative_rmse_percent` from each `scale_eval.json`, and averages each value across runs. To add a run, create `run#4/` with the same three files and run `python finalize_benchmarks.py` from the repository root.

The GLBs from these runs are in [backblaze_glbs/](../backblaze_glbs/), and run 3's complete outputs are in [cache/](../cache/).

## Results

From `benchmark_summary.json`:

| Run | Stage 1 (s per product) | Stage 2 (s per product) | Mean RRMSE | Records evaluated |
|---|---|---|---|---|
| run#1 | 42.7 | 67.8 | 4.90% | 51 |
| run#2 | 47.6 | 69.8 | 5.13% | 56 |
| run#3 | 48.6 | 62.7 | 4.61% | 53 |
| **Mean** | **46.3** | **66.8** | **4.88%** | |

The mean RRMSE averages over every record in each run's `meshes.json`. Products whose mesh was regenerated during a run have more than one record pointing to the same GLB, which is why each run has more than 50 records evaluated; see [mesh_outputs/](../mesh_outputs/).
