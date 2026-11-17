# Mesh_Processing

Runs Hunyuan3D-2 in-process to turn an image into a textured GLB. The batch pipeline doesn't use this module: stage 2 (`mesh_generation_main.py`) drives Hunyuan3D-2's Gradio app instead. It's used by the older `main.py` and for timing generation directly. See the [main README](../README.md) for the full pipeline.

## Usage

```python
from Mesh_Processing.get_mesh import get_pipelines, generate_glb_from_image

shape_pipe, paint_pipe = get_pipelines()  # load once

glb_path, shape_sec, paint_sec, export_sec = generate_glb_from_image(
    image_path="outputs_preprocessing/impainted_images/couch#1.png",
    out_glb="outputs/couch#1.glb",
    shape_pipeline=shape_pipe,
    paint_pipeline=paint_pipe,
)
```

- `get_pipelines()` loads `Hunyuan3DDiTFlowMatchingPipeline` (shape) and `Hunyuan3DPaintPipeline` (texture) from `tencent/Hunyuan3D-2`, with attention slicing and xFormers memory-efficient attention enabled on the paint pipeline.
- `generate_glb_from_image()` generates the shape, paints it, exports the GLB, and returns the path plus the seconds spent on each of the three steps. It synchronizes CUDA around each step so the timings are accurate.

`main.py` still calls an older two-argument version of `generate_glb_from_image()` and needs updating to this signature.

Running `python -m Mesh_Processing.get_mesh` from the repository root generates a mesh from a demo image and prints the timings. Its paths are absolute, so edit them in the `__main__` block first.

## Requirements

- `hy3dgen`, installed from `hunyuan/` together with its texture extensions (see the [main README's setup section](../README.md#setup)).
- xFormers, for `enable_xformers_memory_efficient_attention()`.
- A CUDA GPU.

`outputs/` holds two sample GLBs from standalone runs.
