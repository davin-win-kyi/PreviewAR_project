from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from pathlib import Path


def generate_glb_from_image(
    image_path: str,
    out_glb: str = "output.glb",
):
    image_path = str(image_path)

    # 1) Shape generation (image -> mesh)
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2"
    )

    shape_outputs = shape_pipeline(image=image_path)
    if shape_outputs is None or len(shape_outputs) == 0:
        raise RuntimeError("Shape pipeline returned no meshes")

    mesh = shape_outputs[0]

    # 2) Texture / painting pass
    paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        "tencent/Hunyuan3D-2"
    )

    mesh = paint_pipeline(mesh, image=image_path)

    # 3) Save GLB
    out_glb = Path(out_glb)
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_glb))

    return str(out_glb)


if __name__ == "__main__":
    glb_path = generate_glb_from_image(
        image_path="/home/davin123/PreviewAR_project/uploads/inpainted_result.png",
        out_glb="outputs/mesh.glb",
    )
    print("Saved mesh:", glb_path)

# /home/davin123/PreviewAR_project/uploads/inpainted_result.png