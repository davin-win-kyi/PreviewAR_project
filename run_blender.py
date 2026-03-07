#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path


def glb_to_blend(glb_path: Path, blend_path: Path, blender_bin: str = "/home/davin123/Downloads/blender-5.0.1-linux-x64/blender") -> None:
    glb_path = glb_path.resolve()
    blend_path = blend_path.resolve()

    if not glb_path.exists():
        raise FileNotFoundError(f"GLB not found: {glb_path}")

    blend_path.parent.mkdir(parents=True, exist_ok=True)

    # Blender runs this expression *inside Blender* (headless), imports glb, then saves blend.
    expr = (
        "import bpy;"
        "bpy.ops.wm.read_factory_settings(use_empty=True);"  # start clean
        f"bpy.ops.import_scene.gltf(filepath=r'{str(glb_path)}');"
        f"bpy.ops.wm.save_as_mainfile(filepath=r'{str(blend_path)}');"
    )

    subprocess.run(
        [
            blender_bin,
            "--background",
            "--factory-startup",
            "--python-expr",
            expr,
        ],
        check=True,
    )

    if not blend_path.exists():
        raise RuntimeError(f"Conversion finished but .blend was not created: {blend_path}")


def open_blend_in_blender(blend_path: Path, blender_bin: str = "blender") -> None:
    blend_path = blend_path.resolve()
    if not blend_path.exists():
        raise FileNotFoundError(f"BLEND not found: {blend_path}")

    # Launch Blender UI and open the .blend
    subprocess.run([blender_bin, str(blend_path)], check=True)


def convert_then_open(glb: str, blender_bin: str = "/home/davin123/Downloads/blender-5.0.1-linux-x64/blender") -> None:
    glb_path = Path(glb)
    # Save .blend next to glb (same name)
    blend_path = glb_path.with_suffix(".blend")

    glb_to_blend(glb_path, blend_path, blender_bin=blender_bin)
    open_blend_in_blender(blend_path, blender_bin=blender_bin)


if __name__ == "__main__":
    # Change this to your glb path
    convert_then_open("/home/davin123/PreviewAR_project/outputs/mesh.glb")


