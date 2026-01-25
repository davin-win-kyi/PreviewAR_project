#!/usr/bin/env python3
"""
Grounded_SAM2_Config.py

- Centralizes configuration
- Builds and returns models (SAM2 predictor + GroundingDINO processor/model)
- Applies the same autocast/TF32 settings as your original script
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from grounded_sam2.sam2.build_sam import build_sam2
from grounded_sam2.sam2.sam2_image_predictor import SAM2ImagePredictor



@dataclass
class GroundedSAM2Config:
    grounding_model: str = "IDEA-Research/grounding-dino-tiny"
    sam2_checkpoint: str = "/home/davin123/PreviewAR_project/grounded_sam2/checkpoints/sam2.1_hiera_large.pt"
    sam2_model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    force_cpu: bool = False

    # thresholds used in inference
    box_threshold: float = 0.4
    text_threshold: float = 0.3

    # crop settings used later
    crop_pad: int = 0


def resolve_device(force_cpu: bool = False) -> str:
    return "cuda" if (torch.cuda.is_available() and not force_cpu) else "cpu"


def configure_torch_for_inference(device: str) -> None:
    """
    Mimics the behavior in your original code:
      torch.autocast(...).__enter__()
      enable TF32 on Ampere+ GPUs
    """
    # NOTE: This "enters" autocast globally like your script.
    # If you prefer scoped autocast, do it with `with torch.autocast(...):` around inference calls.
    torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        if props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


def build_models(cfg: GroundedSAM2Config):
    """
    Returns:
      device (str),
      sam2_predictor (SAM2ImagePredictor),
      processor (AutoProcessor),
      grounding_model (AutoModelForZeroShotObjectDetection)
    """
    device = resolve_device(cfg.force_cpu)
    configure_torch_for_inference(device)

    sam2_model = build_sam2(cfg.sam2_model_config, cfg.sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    processor = AutoProcessor.from_pretrained(cfg.grounding_model)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.grounding_model).to(device)

    return device, sam2_predictor, processor, grounding_model
