# utils

Blender scripts and small helpers used by other parts of the repository.

| File | Run with | What it does |
|---|---|---|
| `blender_bbox.py` | `blender -b -P utils/blender_bbox.py -- mesh.glb` | Imports a GLB into an empty scene, picks the mesh object with the largest bounding-box volume, and prints one line of JSON. `glb_absolute` holds its world-space dimensions in inches, largest first; `debug` holds the per-axis size, the bounding box in meters, and the volume. Used by `benchmarking_rrmse.py`. |
| `view_glbs.py` | `blender -P utils/view_glbs.py -- outputs/relevant_glbs.json` | Lays out GLBs in a grid, four per row, each labeled with its name and score, for visual inspection. Reads the output of `rag_products.py`: a list of items (or an object with a `glbs` list), each with `path_to_glb` and optionally `clean_name` and `score`. |
| `mask_utils.py` | Python import | `combine_images_with_mask()` turns an image on a white background, plus a mask, into a black-and-white inpainting guide. Only used by `inpaint_image_with_mask()` in `Impaint_Processing/`, which the pipeline doesn't call. |
| `text_utils.py` | Python import | `format_object_list(["A", "B", "C"])` returns `"A, B, or C"`. Imported by `Impaint_Processing/inpaint_with_gemini.py` but not called. |

The Blender scripts need Blender's bundled Python (`bpy`) and can't be run with plain `python`.
