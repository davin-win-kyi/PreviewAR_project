# Impaint_Processing

The last step of stage 1. Gemini's image model turns the composite from [Image_Processing/](../Image_Processing/) into a clean picture of the product alone on a white background, filling in the parts that were covered. See the [main README](../README.md) for the full pipeline.

## Usage

```python
from Impaint_Processing.nano_banana import run_full_inpaint_with_manual_mask

result = run_full_inpaint_with_manual_mask(
    "sam_outputs/targets/target_000/final_composite.png",
    target_class="couch.",               # the positive prompt
    classes_in_crop="pillow. blanket.",  # the negative prompt (may be empty)
)
result["inpainted_image_path"]  # "uploads/inpainted_result.png"
```

`run_full_inpaint_with_manual_mask` calls `get_inpainted_image_path` in `inpaint_with_gemini.py`, which:

1. builds the prompt with `build_inpaint_prompt`: keep exactly one complete product; rebuild masked or hidden regions to match its material, color, and shape; center it on a plain white background with no shadows or other objects; and remove the objects listed in `classes_in_crop`;
2. sends the prompt and the image to `gemini-3-pro-image-preview` (`recontext_masked_area`);
3. saves the returned image to `uploads/inpainted_result.png`, replacing the previous result.

If Gemini returns no image, nothing is saved, but the function still returns that path. `preprocess_main.py` copies the newest file in `uploads/` to `outputs_preprocessing/impainted_images/<name>.png`, so run it with `--clear-uploads-between-items`. That way a missing result fails the product instead of reusing the previous product's image.

## Files

| File | Contents |
|---|---|
| `nano_banana.py` | `run_full_inpaint_with_manual_mask()`, the entry point. Its `conf`, `iou`, `use_gpt`, and `adjustment_factor` parameters are left over from an earlier mask generator and are ignored. |
| `inpaint_with_gemini.py` | Prompt building and the Gemini call. `inpaint_image_with_mask()` and `replace_black_with_white()` come from an earlier approach and aren't used by the pipeline. |
| `nano_banana_2.py` | A standalone Gemini ranker that picks the best product photo from a list of image URLs (`rank_images_from_urls`). The pipeline doesn't use it; it ranks local files with `WebScrape_Processing/rank_images_from_local_paths.py` instead. Exits on import if `GEMINI_API_KEY` is unset. |

## Configuration

- Needs `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in the environment or in `.env`.
- Importing `inpaint_with_gemini.py` creates `uploads/` in the working directory.
