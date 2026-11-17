#!/usr/bin/env python3
"""
Generate_Impainting_Images.py

- Generates per-target crops based on POSITIVE boxes
- Builds an inpaint mask from NEGATIVE BOXES (not negative masks)
- Optionally constrains the inpaint mask to a "couch region" derived from the positive mask (dilate)
- Saves crop, masks, overlay, composite, and per-target json
- Writes a master index json

Key idea:
  inpaint_mask = union(negative boxes) within the positive crop (bbox)
  optional: inpaint_mask &= dilated(pos_mask_crop)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
import pycocotools.mask as mask_util
import supervision as sv


import gc
import torch

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()



def clamp_xyxy_exclusive(xyxy, w, h):
    """
    Clamp to [0,w]x[0,h] with x2/y2 treated as EXCLUSIVE endpoints.
    Ensures x2 > x1 and y2 > y1.
    """
    x1, y1, x2, y2 = xyxy
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(1, min(w, x2)))   # allow x2==w
    y2 = int(max(1, min(h, y2)))   # allow y2==h

    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)

    return x1, y1, x2, y2


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


def build_box_mask(
    boxes_xyxy: List[List[float]],
    h: int,
    w: int,
    *,
    shrink_px: int = 0,
    dilate_px: int = 0,
    feather_px: int = 0,
) -> np.ndarray:
    """
    Create a mask from boxes in CROP-LOCAL coords. Returns bool mask (h,w).
    - shrink_px reduces box size (helps reduce rectangle bias)
    - dilate_px expands box mask (optional)
    - feather_px blurs edges then thresholds (optional "rounding")
    """
    m = np.zeros((h, w), dtype=np.uint8)

    for b in boxes_xyxy:
        x1, y1, x2, y2 = [int(v) for v in b]
        # clamp to crop-local exclusive bounds
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(1, min(w, x2))
        y2 = max(1, min(h, y2))

        if x2 <= x1 + 1 or y2 <= y1 + 1:
            continue

        if shrink_px > 0:
            x1 = min(x2 - 1, x1 + shrink_px)
            y1 = min(y2 - 1, y1 + shrink_px)
            x2 = max(x1 + 1, x2 - shrink_px)
            y2 = max(y1 + 1, y2 - shrink_px)

        m[y1:y2, x1:x2] = 255

    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        m = cv2.dilate(m, k)

    if feather_px > 0:
        # blur then threshold back to binary (gives softer/rounded edges)
        k = 2 * feather_px + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
        m = (m > 127).astype(np.uint8) * 255

    return (m > 0)


def intersect_boxes_with_crop(
    neg_boxes_full: List[List[float]],
    crop_xyxy: List[int],
) -> List[List[int]]:
    """
    Convert full-image neg boxes into crop-local boxes (exclusive end).
    """
    x1p, y1p, x2p, y2p = crop_xyxy
    out: List[List[int]] = []
    for bx1, by1, bx2, by2 in neg_boxes_full:
        bx1, by1, bx2, by2 = int(bx1), int(by1), int(bx2), int(by2)

        ix1 = max(bx1, x1p)
        iy1 = max(by1, y1p)
        ix2 = min(bx2, x2p)
        iy2 = min(by2, y2p)

        if ix2 > ix1 and iy2 > iy1:
            out.append([ix1 - x1p, iy1 - y1p, ix2 - x1p, iy2 - y1p])
    return out


def composite_preview_whiten_inpaint(crop_bgr: np.ndarray, inpaint_mask: np.ndarray) -> np.ndarray:
    """
    Human-friendly preview: keep original crop, whiten only inpaint area.
    """
    out = crop_bgr.copy()
    out[inpaint_mask.astype(bool)] = (255, 255, 255)
    return out


def composite_final_black_white_rgb(
    crop_bgr: np.ndarray,
    pos_mask_crop: np.ndarray,
    inpaint_mask: np.ndarray,
) -> np.ndarray:
    """
    Final composite:
    - WHITE where inpaint_mask is True (negative boxes in positive bbox / region)
    - ORIGINAL RGB where pos_mask_crop is True AND not inpaint
    - BLACK everywhere else
    """
    pos = pos_mask_crop.astype(bool)
    hole = inpaint_mask.astype(bool)

    out = np.zeros_like(crop_bgr)               # black everywhere
    out[pos & ~hole] = crop_bgr[pos & ~hole]    # keep positive-only original pixels
    out[hole] = (255, 255, 255)                 # white hole region
    return out



def generate_impainting_images(
    *,
    image_path: str,
    img_bgr: np.ndarray,
    positive_prompt: str,
    negative_prompt: str,
    pos_boxes: List[List[float]],
    pos_masks: np.ndarray,                # (N,H,W) bool-ish
    neg_boxes: List[List[float]],         # (K,4) full-image coords xyxy
    out_root: str,
    crop_pad: int = 0,
    # tuning knobs for the negative box mask
    neg_shrink_px: int = 12,
    neg_dilate_px: int = 1,
    neg_feather_px: int = 0,
    # optionally constrain negatives to be "on the couch"
    constrain_to_pos_region: bool = True,
    pos_region_dilate_px: int = 20,
    dump_json: bool = True,
) -> str:
    """
    Writes per-target folders:
      target_###/{crop.jpg, positive_mask.png, inpaint_mask.png, final_mask.png,
                 preview_whited.png, final_overlay.jpg, target.json}

    Returns path to master index JSON.
    """
    H, W = img_bgr.shape[:2]
    out_root_p = Path(out_root)
    targets_dir = out_root_p / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    n_targets = len(pos_boxes)

    index_payload: Dict[str, Any] = {
        "image_path": image_path,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "targets_dir": str(targets_dir),
        "num_targets": int(n_targets),
        "targets": [],
        "neg_mask_mode": "negative_boxes",
        "neg_box_mask_params": {
            "shrink_px": int(neg_shrink_px),
            "dilate_px": int(neg_dilate_px),
            "feather_px": int(neg_feather_px),
            "constrain_to_pos_region": bool(constrain_to_pos_region),
            "pos_region_dilate_px": int(pos_region_dilate_px),
        }
    }

    for i in range(n_targets):
        x1, y1, x2, y2 = [int(v) for v in pos_boxes[i]]

        pad = max(0, int(crop_pad))
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(W, x2 + pad)  # EXCLUSIVE
        y2p = min(H, y2 + pad)  # EXCLUSIVE
        x1p, y1p, x2p, y2p = clamp_xyxy_exclusive((x1p, y1p, x2p, y2p), W, H)

        crop_dir = targets_dir / f"target_{i:03d}"
        crop_dir.mkdir(parents=True, exist_ok=True)

        crop_bgr = img_bgr[y1p:y2p, x1p:x2p].copy()
        cv2.imwrite(str(crop_dir / "crop.jpg"), crop_bgr)

        # Positive mask crop
        pos_mask_i_full = (
            pos_masks[i] if (pos_masks is not None and pos_masks.shape[0] > i)
            else np.zeros((H, W), dtype=bool)
        )
        pos_mask_crop = (pos_mask_i_full[y1p:y2p, x1p:x2p] > 0)

        # Negative boxes -> crop-local boxes -> box mask
        crop_xyxy = [x1p, y1p, x2p, y2p]
        neg_boxes_in_crop = intersect_boxes_with_crop(neg_boxes, crop_xyxy)

        crop_h, crop_w = crop_bgr.shape[:2]
        inpaint_mask = build_box_mask(
            neg_boxes_in_crop,
            crop_h,
            crop_w,
            shrink_px=neg_shrink_px,
            dilate_px=neg_dilate_px,
            feather_px=neg_feather_px,
        )

        final_composite = composite_final_black_white_rgb(crop_bgr, pos_mask_crop, inpaint_mask)
        cv2.imwrite(str(crop_dir / "final_composite.png"), final_composite)


        # Optionally constrain inpaint to a "couch region" proxy (dilated positive mask)
        pos_region = pos_mask_crop
        if constrain_to_pos_region and pos_mask_crop.any():
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * pos_region_dilate_px + 1, 2 * pos_region_dilate_px + 1),
            )
            pos_region = (cv2.dilate((pos_mask_crop.astype(np.uint8) * 255), k) > 0)
            inpaint_mask = np.logical_and(inpaint_mask, pos_region)

        # Final keep-mask: couch pixels minus inpaint region
        final_mask_crop = np.logical_and(pos_mask_crop, np.logical_not(inpaint_mask))

        # Save masks
        save_binary_mask_png(crop_dir / "positive_mask.png", pos_mask_crop)
        save_binary_mask_png(crop_dir / "inpaint_mask.png", inpaint_mask)
        save_binary_mask_png(crop_dir / "final_mask.png", final_mask_crop)

        # Previews
        preview_whited = composite_preview_whiten_inpaint(crop_bgr, inpaint_mask)
        cv2.imwrite(str(crop_dir / "preview_whited.png"), preview_whited)

        overlay = overlay_single_mask(crop_bgr, final_mask_crop, label=f"target_{i}")
        cv2.imwrite(str(crop_dir / "final_overlay.jpg"), overlay)

        target_json_path = crop_dir / "target.json"
        if dump_json:
            payload = {
                "target_index": i,
                "image_path": image_path,
                "crop": {
                    "xyxy": [x1p, y1p, x2p, y2p],
                    "width": int(crop_w),
                    "height": int(crop_h),
                },
                "positive_prompt": positive_prompt.strip(),
                "negative_prompt": negative_prompt.strip(),
                "neg_boxes_in_crop": neg_boxes_in_crop,
                "masks": {
                    "positive_rle": single_mask_to_rle(pos_mask_crop),
                    "inpaint_rle": single_mask_to_rle(inpaint_mask),
                    "final_rle": single_mask_to_rle(final_mask_crop),
                },
                "stats": {
                    "positive_pixels": int(pos_mask_crop.sum()),
                    "inpaint_pixels": int(inpaint_mask.sum()),
                    "final_pixels": int(final_mask_crop.sum()),
                },
                "artifacts": {
                    "crop.jpg": "crop image",
                    "positive_mask.png": "binary mask (0/255)",
                    "inpaint_mask.png": "binary inpaint mask (0/255) derived from negative boxes",
                    "final_mask.png": "binary keep mask = positive minus inpaint",
                    "preview_whited.png": "preview with inpaint region whitened",
                    "final_overlay.jpg": "visual overlay of final mask",
                },
            }
            target_json_path.write_text(json.dumps(payload, indent=4))

        index_payload["targets"].append(
            {
                "target_index": i,
                "folder": str(crop_dir),
                "target_json": str(target_json_path),
                "crop_xyxy": [x1p, y1p, x2p, y2p],
                "crop": str(crop_dir / "crop.jpg"),
                "inpaint_mask": str(crop_dir / "inpaint_mask.png"),
                "final_mask": str(crop_dir / "final_mask.png"),
                "final_composite": str(crop_dir / "final_composite.png"),
                "preview_whited": str(crop_dir / "preview_whited.png"),
            }
        )

    index_path = targets_dir / "impainting_index.json"
    index_path.write_text(json.dumps(index_payload, indent=4))
    # del out
    # cleanup_cuda()
    return str(index_path)

