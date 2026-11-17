#!/usr/bin/env python3
"""
Get_Impainting_Images.py

Entrypoint:
- Uses prebuilt Grounded-SAM2 models if provided
- Otherwise can still build models locally as a fallback
- Runs positive/negative + writes prompt folders
- Generates per-target inpainting composites + masks + json
- Returns the path to the master json index
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from Image_Processing.Grounded_SAM2_Config import GroundedSAM2Config, build_models
from Image_Processing.Grounded_SAM2_Run import run_pos_neg_and_write, normalize_prompt, cleanup_cuda
from Image_Processing.Generate_Impainting_Images import generate_impainting_images

import gc
import torch


def main(
    *,
    image_path: str,
    positive_prompt: str,
    negative_prompt: str,
    out_root: str = "sam_outputs",
    grounding_model_name: str = "IDEA-Research/grounding-dino-tiny",
    sam2_checkpoint: str = "/home/davin123/PreviewAR_project/grounded_sam2/checkpoints/sam2.1_hiera_large.pt",
    sam2_model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    box_threshold: float = 0.4,
    text_threshold: float = 0.3,
    crop_pad: int = 0,
    force_cpu: bool = False,
    dump_json: bool = True,
    # Prebuilt reusable models
    device: Optional[str] = None,
    sam2_predictor=None,
    processor=None,
    grounding_model=None,
) -> str:
    """
    Runs the full Grounded-SAM2 -> inpainting pipeline.

    If device/sam2_predictor/processor/grounding_model are provided,
    they are reused instead of being rebuilt.

    Returns:
        Path to the master inpainting index JSON.
    """

    built_locally = False

    if any(x is None for x in (device, sam2_predictor, processor, grounding_model)):
        cfg = GroundedSAM2Config(
            grounding_model=grounding_model_name,
            sam2_checkpoint=sam2_checkpoint,
            sam2_model_config=sam2_model_config,
            force_cpu=force_cpu,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            crop_pad=crop_pad,
        )
        device, sam2_predictor, processor, grounding_model = build_models(cfg)
        built_locally = True

    out = run_pos_neg_and_write(
        image_path=image_path,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        out_root=out_root,
        processor=processor,
        grounding_model=grounding_model,
        sam2_predictor=sam2_predictor,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        dump_json=dump_json,
    )

    index_json_path = generate_impainting_images(
        image_path=image_path,
        img_bgr=out["img_bgr"],
        positive_prompt=normalize_prompt(positive_prompt),
        negative_prompt=normalize_prompt(negative_prompt),
        pos_boxes=out["pos_boxes"],
        pos_masks=out["pos_masks"],
        neg_boxes=out["neg_boxes"],
        out_root=out_root,
        crop_pad=crop_pad,
        dump_json=dump_json,
    )

    print(f"Master inpainting index json: {index_json_path}")

    # Per-item cleanup
    del out
    cleanup_cuda()

    # Only destroy models if this function built them locally
    if built_locally:
        del sam2_predictor
        del processor
        del grounding_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    return index_json_path


if __name__ == "__main__":
    index_json = main(
        image_path="/home/davin123/PreviewAR_project/outputs/best_image.png",
        positive_prompt="couch.",
        negative_prompt="pillow. blanket. books. lamp.",
        out_root="sam_outputs",
        crop_pad=16,
        force_cpu=False,
    )

    print("Returned index JSON:", index_json)