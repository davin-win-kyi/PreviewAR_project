# mesh_outputs

Stage 2 output, written by `mesh_generation_main.py`; pass `--out_dir` to write elsewhere. Stage 3 (`benchmarking_rrmse.py`) reads `meshes.json`. See the [main README](../README.md) for the full pipeline.

The checked-in contents come from test runs: the records in the JSON files don't correspond to the three GLBs in `glbs/`.

| Path | Contents |
|---|---|
| `glbs/<object_type>__<name>.glb` | The textured mesh for each product, e.g. `lamp__lamp#1.glb` |
| `meshes.json` | One record per generation attempt (below) |
| `runtimes.json` | Seconds per attempt, from uploading the image to the GLB being saved |

## `meshes.json`

```json
{
  "meshes": [
    {
      "object_type": "lamp",
      "name": "lamp#1",
      "standardized_dimensions": { "l": 14, "w": 7, "h": 16 },
      "path": "/absolute/path/to/mesh_outputs/glbs/lamp__lamp#1.glb",
      "status": "ok"
    }
  ]
}
```

## `runtimes.json`

```json
{
  "runtimes": [
    { "description": "lamp", "runtime_sec": 55.6, "status": "ok" }
  ]
}
```

Failed attempts have `"status": "fail"`, and in `runtimes.json` an `error` message.

## Re-running

- A product is skipped if its GLB already exists. Delete the GLB to generate it again.
- Both JSON files are only ever appended to. Regenerating a product therefore leaves its earlier record in place, with both records pointing to the same GLB. Failed attempts stay as well, and once a retry succeeds, their `path` exists too.
- `benchmarking_rrmse.py` evaluates every record regardless of `status`, so remove stale records (or keep only the last record per `name`) before running stage 3. Otherwise products with several records are counted several times in the averages.
