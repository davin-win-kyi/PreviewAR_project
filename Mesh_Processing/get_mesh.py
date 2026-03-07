from __future__ import annotations

from pathlib import Path
import time
import torch

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline


def get_pipelines():
    """
    Load default Hunyuan3D-2 shape and paint pipelines.
    This load time is NOT included in generation benchmark.
    """
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2"
    )

    paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        "tencent/Hunyuan3D-2"
    )

    paint_pipeline.enable_attention_slicing()
    # and/or (if installed)
    paint_pipeline.enable_xformers_memory_efficient_attention()

    return shape_pipeline, paint_pipeline


def generate_glb_from_image(
    image_path: str,
    out_glb: str,
    shape_pipeline,
    paint_pipeline,
):
    """
    Run shape + texture generation using preloaded pipelines.
    """

    # ---- Shape ----
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_shape_start = time.time()

    with torch.inference_mode():
        mesh = shape_pipeline(image=image_path)[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    shape_time = time.time() - t_shape_start

    # ---- Paint ----
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_paint_start = time.time()

    with torch.inference_mode():
        mesh = paint_pipeline(mesh, image=image_path)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    paint_time = time.time() - t_paint_start

    # ---- Export ----
    t_export_start = time.time()

    out_glb = Path(out_glb)
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_glb))

    export_time = time.time() - t_export_start

    return str(out_glb), shape_time, paint_time, export_time


if __name__ == "__main__":
    # 1) Load pipelines once (NOT benchmarked)
    shape_pipe, paint_pipe = get_pipelines()

    # 2) Benchmark only mesh + texture generation
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_total_start = time.time()

    glb_path, shape_sec, paint_sec, export_sec = generate_glb_from_image(
        image_path="/home/davin123/PreviewAR_project/hunyuan/assets/demo.png",
        out_glb="outputs/updated_mesh.glb",
        shape_pipeline=shape_pipe,
        paint_pipeline=paint_pipe,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_sec = time.time() - t_total_start

    print(f"[shape]  {shape_sec:.2f}s")
    print(f"[paint]  {paint_sec:.2f}s")
    print(f"[export] {export_sec:.2f}s")
    print(f"[total]  {total_sec:.2f}s")
    print("Saved mesh:", glb_path)