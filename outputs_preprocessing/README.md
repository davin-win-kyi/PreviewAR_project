# outputs_preprocessing

Stage 1 output, written by `preprocess_main.py`; pass `--output-dir` to write elsewhere. Stage 2 reads `impainted_images.json`. See the [main README](../README.md) for the full pipeline.

The checked-in contents come from a single test product (`bed#1`).

| Path | Contents |
|---|---|
| `impainted_images/<name>.png` | The clean product image from Gemini, one per product |
| `impainted_images.json` | One record per input product (below) |
| `preprocessing_runtime.json` | Per-product wall-clock time for photo ranking, prompts, segmentation, and inpainting, plus the average over successful products |

## `impainted_images.json`

```json
{
  "impainted_images": [
    {
      "url": "https://www.example.com/products/123",
      "object_type": "couch",
      "site": "amazon",
      "name": "couch#1",
      "standardized_dimensions": { "l": 84.0, "w": 35.0, "h": 33.5 },
      "impainted_image_path": "/absolute/path/to/outputs_preprocessing/impainted_images/couch#1.png",
      "status": "ok"
    }
  ]
}
```

`status` is `"ok"` or `"fail"`. Failed products have an empty `impainted_image_path`.

## `preprocessing_runtime.json`

```json
{
  "runtimes": [
    { "url": "https://www.example.com/products/123", "object_type": "couch", "runtime_sec": 42.7, "status": "ok" }
  ],
  "average_runtime_sec": 42.7
}
```

Failed products also have an `error` message.

## Re-running

- Both JSON files are rewritten on every run and list only the products in the current input.
- A product is skipped if its record in `impainted_images.json` points to an image that exists (its old records are kept unchanged), or if `impainted_images/<name>.png` exists. Delete the image to process the product again.
- Paths in the JSON files are absolute.
