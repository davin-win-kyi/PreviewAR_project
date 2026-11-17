# backblaze_glbs

The meshes produced in the three benchmark runs in [paper_benchmarks/](../paper_benchmarks/), one folder per run. Each run has the same 50 file names, one per product.

```
backblaze_glbs/
├── Run#1/
│   ├── glbs/<object_type>__<name>.glb   50 meshes
│   └── meshes.json                      that run's mesh_outputs/meshes.json
├── Run#2/
└── Run#3/
```

The `path` fields in each `meshes.json` are absolute paths from the machine that produced them, so match records to files by file name. Some records are duplicates; see [mesh_outputs/](../mesh_outputs/). Run 3's GLBs and `meshes.json` are identical to those in [cache/](../cache/).

The folder is named after Backblaze B2 cloud storage, but no script in this repository uploads to or reads from it. `backblaze_glbs.json`, in the repository root, is a `meshes.json`-style list of 40 of these GLBs, referenced by file name.
