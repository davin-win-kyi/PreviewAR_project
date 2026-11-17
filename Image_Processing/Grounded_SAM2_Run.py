#!/usr/bin/env python3
"""
Grounded_SAM2_Run.py

- Contains: run_grounded_sam2, write_prompt_folder, and utilities they need
- Provides a single function to:
    * run positive prompt + negative prompt
    * write the prompt folders
    * return (pos_boxes, pos_masks, neg_masks, image metadata)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pycocotools.mask as mask_util
import supervision as sv
import torch
from PIL import Image
from supervision.draw.color import ColorPalette
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor



from grounded_sam2.sam2.sam2_image_predictor import SAM2ImagePredictor


import gc

import gc
import torch

# def cleanup_cuda(models_to_delete: list):
#     """
#     Forcefully removes models from VRAM and clears cache.
#     """
#     for model in models_to_delete:
#         if model is not None:
#             # Move to CPU first to ensure it's off the GPU
#             model.cpu()
#             del model
    
#     # Trigger Python's garbage collector
#     gc.collect()
    
#     # Clear the PyTorch CUDA cache
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()

import gc
import torch

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()



def ensure_3d_masks(masks: np.ndarray) -> np.ndarray:
    """SAM2 can return (N,1,H,W) or (N,H,W). Return (N,H,W)."""
    if masks is None:
        return masks
    if masks.ndim == 4:
        return masks.squeeze(1)
    return masks


def normalize_prompt(text: str) -> str:
    t = text.strip().lower()
    if not t.endswith("."):
        t += "."
    return t


def single_mask_to_rle(mask_bool_hw: np.ndarray) -> dict:
    """mask_bool_hw: HxW bool -> COCO RLE dict."""
    rle = mask_util.encode(np.array(mask_bool_hw[:, :, None], order="F", dtype=np.uint8))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def run_grounded_sam2(
    *,
    text_prompt: str,
    image_pil: Image.Image,
    processor: AutoProcessor,
    grounding_model: AutoModelForZeroShotObjectDetection,
    sam2_predictor: SAM2ImagePredictor,
    device: str,
    box_threshold: float,
    text_threshold: float,
):
    text = normalize_prompt(text_prompt)

    image_np = np.array(image_pil.convert("RGB"))
    sam2_predictor.set_image(image_np)
    del image_np

    inputs = processor(images=image_pil, text=text, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image_pil.size[::-1]],
    )

    del outputs
    del inputs
    cleanup_cuda()

    boxes_t = results[0].get("boxes", None)
    scores_t = results[0].get("scores", None)
    labels = results[0].get("labels", [])

    if boxes_t is None or len(labels) == 0:
        h, w = image_pil.size[1], image_pil.size[0]
        return [], [], [], np.zeros((0, h, w), dtype=bool)

    input_boxes = boxes_t.detach().cpu().numpy()
    confidences = scores_t.detach().cpu().numpy().tolist()

    del boxes_t
    del scores_t
    cleanup_cuda()

    masks, sam_scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    masks = ensure_3d_masks(masks)
    masks_bool = masks.astype(bool)

    del masks
    del sam_scores
    del logits
    cleanup_cuda()

    return input_boxes.tolist(), confidences, labels, masks_bool


def write_prompt_folder(
    *,
    folder: Path,
    prompt_name: str,
    text_prompt: str,
    image_path: str,
    image_pil: Image.Image,
    boxes: list,
    scores: list,
    labels: list,
    masks_bool: np.ndarray,
    dump_json: bool,
):
    folder.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.imread(image_path)

    if len(boxes) > 0:
        boxes_np = np.array(boxes, dtype=float)
        class_ids = np.arange(len(labels))
        dets = sv.Detections(xyxy=boxes_np, mask=masks_bool, class_id=class_ids)

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()

        labels_for_draw = [f"{cls} {conf:.2f}" for cls, conf in zip(labels, scores)]

        bbox_frame = box_annotator.annotate(scene=img_bgr.copy(), detections=dets)
        bbox_frame = label_annotator.annotate(scene=bbox_frame, detections=dets, labels=labels_for_draw)
        cv2.imwrite(str(folder / "bbox.jpg"), bbox_frame)

        overlay = mask_annotator.annotate(scene=bbox_frame.copy(), detections=dets)
        cv2.imwrite(str(folder / "segmented_overlay.jpg"), overlay)
    else:
        cv2.imwrite(str(folder / "bbox.jpg"), img_bgr)
        cv2.imwrite(str(folder / "segmented_overlay.jpg"), img_bgr)

    if dump_json:
        h, w = image_pil.size[1], image_pil.size[0]
        payload = {
            "prompt_name": prompt_name,
            "text_prompt": normalize_prompt(text_prompt),
            "image_path": image_path,
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
            "annotations": [],
        }
        if len(boxes) > 0:
            mask_rles = [single_mask_to_rle(m) for m in masks_bool]
            payload["annotations"] = [
                {
                    "class_name": cls,
                    "bbox": box,
                    "segmentation": rle,
                    "score": float(score),
                }
                for cls, box, rle, score in zip(labels, boxes, mask_rles, scores)
            ]
        (folder / "results.json").write_text(json.dumps(payload, indent=4))


def run_pos_neg_and_write(
    *,
    image_path: str,
    positive_prompt: str,
    negative_prompt: str,
    out_root: str,
    processor: AutoProcessor,
    grounding_model: AutoModelForZeroShotObjectDetection,
    sam2_predictor: SAM2ImagePredictor,
    device: str,
    box_threshold: float,
    text_threshold: float,
    dump_json: bool = True,
):
    out_root_p = Path(out_root)
    pos_dir = out_root_p / "positive_prompt"
    neg_dir = out_root_p / "negative_prompt"

    image_pil = Image.open(image_path).convert("RGB")
    img_bgr = cv2.imread(image_path)

    pos_boxes, pos_scores, pos_labels, pos_masks = run_grounded_sam2(
        text_prompt=positive_prompt,
        image_pil=image_pil,
        processor=processor,
        grounding_model=grounding_model,
        sam2_predictor=sam2_predictor,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    write_prompt_folder(
        folder=pos_dir,
        prompt_name="positive",
        text_prompt=positive_prompt,
        image_path=image_path,
        image_pil=image_pil,
        boxes=pos_boxes,
        scores=pos_scores,
        labels=pos_labels,
        masks_bool=pos_masks,
        dump_json=dump_json,
    )

    neg_boxes, neg_scores, neg_labels, neg_masks = run_grounded_sam2(
        text_prompt=negative_prompt,
        image_pil=image_pil,
        processor=processor,
        grounding_model=grounding_model,
        sam2_predictor=sam2_predictor,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    write_prompt_folder(
        folder=neg_dir,
        prompt_name="negative",
        text_prompt=negative_prompt,
        image_path=image_path,
        image_pil=image_pil,
        boxes=neg_boxes,
        scores=neg_scores,
        labels=neg_labels,
        masks_bool=neg_masks,
        dump_json=dump_json,
    )

    return {
        "img_bgr": img_bgr,
        "pos_boxes": pos_boxes,
        "pos_masks": pos_masks,
        "pos_labels": pos_labels,
        "pos_scores": pos_scores,
        "neg_boxes": neg_boxes,
        "neg_masks": neg_masks,
        "neg_labels": neg_labels,
        "neg_scores": neg_scores,
        "pos_dir": str(pos_dir),
        "neg_dir": str(neg_dir),
    }