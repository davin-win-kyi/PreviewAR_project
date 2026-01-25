#!/usr/bin/env python3
"""
Generate_Impainting_Images.py

- Generates per-target crops based on POSITIVE boxes
- Uses NEGATIVE union mask to subtract negatives
- Saves crop, masks, overlay, composite, and per-target json
- Returns the path to a "master json" index of all targets (plus per-target json paths)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
import pycocotools.mask as mask_util
import supervision as sv
from supervision.draw.color import ColorPalette


def clamp_xyxy(xyxy, w, h):
    x1, y1, x2, y2 = xyxy
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w - 1, x2)))
    y2 = int(max(0, min(h - 1, y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 == x1:
        x2 = min(w - 1, x1 + 1)
    if y2 == y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def union_masks(masks_bool_nhw: np.ndarray, h: int, w: int) -> np.ndarray:
    if masks_bool_nhw is None or masks_bool_nhw.size == 0:
        return np.zeros((h, w), dtype=bool)
    return np.any(masks_bool_nhw, axis=0)


def single_mask_to_rle(mask_bool_hw: np.ndarray) -> dict:
    rle = mask_util.encode(np.array(mask_bool_hw[:, :, None], order="F", dtype=np.uint8))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def save_binary_mask_png(path: Path, mask_bool_hw: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = (mask_bool_hw.astype(np.uint8) * 255)
    cv2.imwrite(str(path), img)


def overlay_single_mask(image_bgr: np.ndarray, mask_bool_hw: np.ndarray, label: str) -> np.ndarray:
    h, w = mask_bool_hw.shape
    ys, xs = np.where(mask_bool_hw)
    if len(xs) == 0:
        xyxy = np.array([[0, 0, 1, 1]], dtype=float)
        mask_nhw = np.zeros((1, h, w), dtype=bool)
    else:
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        xyxy = np.array([[x1, y1, x2, y2]], dtype=float)
        mask_nhw = mask_bool_hw[None, :, :]

    dets = sv.Detections(xyxy=xyxy, mask=mask_nhw, class_id=np.array([0]))

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()

    frame = box_annotator.annotate(scene=image_bgr.copy(), detections=dets)
    frame = label_annotator.annotate(scene=frame, detections=dets, labels=[label])
    frame = mask_annotator.annotate(scene=frame, detections=dets)
    return frame


def composite_pos_neg(crop_bgr: np.ndarray, pos_mask_crop: np.ndarray, neg_mask_crop: np.ndarray) -> np.ndarray:
    pos = pos_mask_crop.astype(bool)
    neg = neg_mask_crop.astype(bool)
    pos_only = np.logical_and(pos, np.logical_not(neg))

    out = np.zeros_like(crop_bgr)          # black everywhere
    out[neg] = (255, 255, 255)             # white where negative
    out[pos_only] = crop_bgr[pos_only]     # original pixels where positive-only
    return out


def generate_impainting_images(
    *,
    image_path: str,
    img_bgr: np.ndarray,
    positive_prompt: str,
    negative_prompt: str,
    pos_boxes: List[List[float]],
    pos_masks: np.ndarray,     # (N,H,W) bool
    neg_masks: np.ndarray,     # (M,H,W) bool
    out_root: str,
    crop_pad: int = 0,
    dump_json: bool = True,
) -> str:
    """
    Creates:
      out_root/targets/target_###/{crop.jpg, positive_mask.png, negative_mask.png, final_mask.png,
                                  final_composite.png, final_overlay.jpg, target.json}
    Also writes:
      out_root/targets/impainting_index.json  (master index)

    Returns:
      path to the master json
    """
    H, W = img_bgr.shape[:2]
    out_root_p = Path(out_root)
    targets_dir = out_root_p / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    neg_union = union_masks(neg_masks, H, W)
    n_targets = len(pos_boxes)

    index_payload: Dict[str, Any] = {
        "image_path": image_path,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "targets_dir": str(targets_dir),
        "num_targets": int(n_targets),
        "targets": [],
    }

    for i in range(n_targets):
        x1, y1, x2, y2 = [int(v) for v in pos_boxes[i]]
        pad = max(0, int(crop_pad))
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(W - 1, x2 + pad)
        y2p = min(H - 1, y2 + pad)
        x1p, y1p, x2p, y2p = clamp_xyxy((x1p, y1p, x2p, y2p), W, H)

        crop_dir = targets_dir / f"target_{i:03d}"
        crop_dir.mkdir(parents=True, exist_ok=True)

        crop_bgr = img_bgr[y1p:y2p, x1p:x2p].copy()
        cv2.imwrite(str(crop_dir / "crop.jpg"), crop_bgr)

        pos_mask_i_full = pos_masks[i] if (pos_masks is not None and pos_masks.shape[0] > i) else np.zeros((H, W), dtype=bool)
        pos_mask_crop = pos_mask_i_full[y1p:y2p, x1p:x2p]
        neg_mask_crop = neg_union[y1p:y2p, x1p:x2p]

        final_mask_crop = np.logical_and(pos_mask_crop, np.logical_not(neg_mask_crop))

        save_binary_mask_png(crop_dir / "positive_mask.png", pos_mask_crop)
        save_binary_mask_png(crop_dir / "negative_mask.png", neg_mask_crop)
        save_binary_mask_png(crop_dir / "final_mask.png", final_mask_crop)

        final_composite = composite_pos_neg(crop_bgr, pos_mask_crop, neg_mask_crop)
        cv2.imwrite(str(crop_dir / "final_composite.png"), final_composite)

        overlay = overlay_single_mask(crop_bgr, final_mask_crop, label=f"target_{i}")
        cv2.imwrite(str(crop_dir / "final_overlay.jpg"), overlay)

        target_json_path = crop_dir / "target.json"
        if dump_json:
            payload = {
                "target_index": i,
                "image_path": image_path,
                "crop": {
                    "xyxy": [x1p, y1p, x2p, y2p],
                    "width": int(crop_bgr.shape[1]),
                    "height": int(crop_bgr.shape[0]),
                },
                "positive_prompt": positive_prompt.strip(),
                "negative_prompt": negative_prompt.strip(),
                "masks": {
                    "positive_rle": single_mask_to_rle(pos_mask_crop),
                    "negative_rle": single_mask_to_rle(neg_mask_crop),
                    "final_rle": single_mask_to_rle(final_mask_crop),
                },
                "stats": {
                    "positive_pixels": int(pos_mask_crop.sum()),
                    "negative_pixels": int(neg_mask_crop.sum()),
                    "final_pixels": int(final_mask_crop.sum()),
                },
                "rendering": {
                    "final_composite.png": {
                        "positive_only": "original RGB pixels",
                        "negative": "white (255,255,255)",
                        "background": "black (0,0,0)",
                        "priority": "negative overrides positive",
                        "rule": "final_mask = positive AND NOT negative",
                    }
                },
                "artifacts": {
                    "crop.jpg": "crop image",
                    "positive_mask.png": "binary mask (0/255)",
                    "negative_mask.png": "binary mask (0/255)",
                    "final_mask.png": "binary mask (0/255)",
                    "final_composite.png": "inpainting composite image",
                    "final_overlay.jpg": "visual overlay",
                },
            }
            target_json_path.write_text(json.dumps(payload, indent=4))

        index_payload["targets"].append(
            {
                "target_index": i,
                "folder": str(crop_dir),
                "target_json": str(target_json_path),
                "crop_xyxy": [x1p, y1p, x2p, y2p],
                "final_composite": str(crop_dir / "final_composite.png"),
                "final_mask": str(crop_dir / "final_mask.png"),
                "crop": str(crop_dir / "crop.jpg"),
            }
        )

    index_path = targets_dir / "impainting_index.json"
    index_path.write_text(json.dumps(index_payload, indent=4))
    return str(index_path)
