# cache

A snapshot of the stage 1 and stage 2 outputs for all 50 products in benchmark run 3 (see [paper_benchmarks/](../paper_benchmarks/)), laid out like the top-level output folders.

```
cache/
├── outputs_preprocessing/   impainted_images/ (50 images), impainted_images.json, preprocessing_runtime.json
└── mesh_outputs/            glbs/ (50 GLBs), meshes.json, runtimes.json
```

The GLBs and `meshes.json` are identical to `backblaze_glbs/Run#3/`. The runtime files differ slightly from the copies in `paper_benchmarks/run#3/`: two products have different stage 1 times, and `runtimes.json` has one extra stage 2 record at the end.

No code reads this folder directly. To reuse these results instead of recomputing them, copy `cache/outputs_preprocessing/` and `cache/mesh_outputs/` over the top-level folders of the same names: stages 1 and 2 skip products whose outputs already exist. Paths inside the JSON files are absolute paths from the original machine.
