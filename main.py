#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util
import supervision as sv
import torch
from PIL import Image
from supervision.draw.color import ColorPalette
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Your repo utility (keep if you have it)
from utils.supervision_utils import CUSTOM_COLOR_MAP


def ensure_3d_masks(masks: np.ndarray) -> np.ndarray:
    """SAM2 can return (N,1,H,W) or (N,H,W). Return (N,H,W)."""
    if masks is None:
        return masks
    if masks.ndim == 4:
        return masks.squeeze(1)
    return masks


def single_mask_to_rle(mask_bool_hw: np.ndarray) -> dict:
    """mask_bool_hw: HxW bool -> COCO RLE dict."""
    rle = mask_util.encode(np.array(mask_bool_hw[:, :, None], order="F", dtype=np.uint8))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def union_masks(masks_bool_nhw: np.ndarray, h: int, w: int) -> np.ndarray:
    """Union (OR) of (N,H,W) bool masks -> (H,W) bool."""
    if masks_bool_nhw is None or masks_bool_nhw.size == 0:
        return np.zeros((h, w), dtype=bool)
    return np.any(masks_bool_nhw, axis=0)


def clamp_xyxy(xyxy, w, h):
    x1, y1, x2, y2 = xyxy
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w - 1, x2)))
    y2 = int(max(0, min(h - 1, y2)))
    # ensure proper order
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    # make non-empty crop
    if x2 == x1:
        x2 = min(w - 1, x1 + 1)
    if y2 == y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def overlay_single_mask(image_bgr: np.ndarray, mask_bool_hw: np.ndarray, label: str) -> np.ndarray:
    """Overlay a single mask using supervision."""
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

    box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))

    frame = box_annotator.annotate(scene=image_bgr.copy(), detections=dets)
    frame = label_annotator.annotate(scene=frame, detections=dets, labels=[label])
    frame = mask_annotator.annotate(scene=frame, detections=dets)
    return frame


def save_binary_mask_png(path: Path, mask_bool_hw: np.ndarray):
    """Save bool mask as 0/255 PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = (mask_bool_hw.astype(np.uint8) * 255)
    cv2.imwrite(str(path), img)


def normalize_prompt(text: str) -> str:
    t = text.strip().lower()
    if not t.endswith("."):
        t += "."
    return t


def composite_pos_neg(crop_bgr: np.ndarray, pos_mask_crop: np.ndarray, neg_mask_crop: np.ndarray) -> np.ndarray:
    """
    Output image where:
      - pos-only pixels: original crop pixels
      - negative pixels: white
      - elsewhere: black
    Priority: negative overrides positive if overlap exists.
    """
    pos = pos_mask_crop.astype(bool)
    neg = neg_mask_crop.astype(bool)
    pos_only = np.logical_and(pos, np.logical_not(neg))

    out = np.zeros_like(crop_bgr)          # black everywhere
    out[neg] = (255, 255, 255)             # white where negative
    out[pos_only] = crop_bgr[pos_only]     # original pixels where positive-only
    return out


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
    """Return boxes, scores, labels, masks_bool (N,H,W)."""
    text = normalize_prompt(text_prompt)

    sam2_predictor.set_image(np.array(image_pil.convert("RGB")))

    inputs = processor(images=image_pil, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image_pil.size[::-1]],  # (H,W)
    )

    boxes_t = results[0].get("boxes", None)
    scores_t = results[0].get("scores", None)
    labels = results[0].get("labels", [])

    if boxes_t is None or len(labels) == 0:
        h, w = image_pil.size[1], image_pil.size[0]
        return [], [], [], np.zeros((0, h, w), dtype=bool)

    input_boxes = boxes_t.detach().cpu().numpy()  # (N,4)
    confidences = scores_t.detach().cpu().numpy().tolist()

    masks, sam_scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    masks = ensure_3d_masks(masks)
    masks_bool = masks.astype(bool)

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

        box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--img-path", default="notebooks/images/couch.jpeg")
    parser.add_argument("--positive-prompt", default="couch.")
    parser.add_argument("--negative-prompt", default="pillow.")
    parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--out-root", default="sam_outputs")
    parser.add_argument("--no-dump-json", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--box-threshold", type=float, default=0.4)
    parser.add_argument("--text-threshold", type=float, default=0.3)
    parser.add_argument("--crop-pad", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    dump_json = not args.no_dump_json

    # Autocast settings
    torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
    if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Build models once
    sam2_model = build_sam2(args.sam2_model_config, args.sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    processor = AutoProcessor.from_pretrained(args.grounding_model)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.grounding_model).to(device)

    # Load image
    img_path = args.img_path
    image_pil = Image.open(img_path)
    img_bgr = cv2.imread(img_path)
    H, W = img_bgr.shape[:2]

    out_root = Path(args.out_root)
    pos_dir = out_root / "positive_prompt"
    neg_dir = out_root / "negative_prompt"
    targets_dir = out_root / "targets"

    # Run full-image positive/negative
    pos_boxes, pos_scores, pos_labels, pos_masks = run_grounded_sam2(
        text_prompt=args.positive_prompt,
        image_pil=image_pil,
        processor=processor,
        grounding_model=grounding_model,
        sam2_predictor=sam2_predictor,
        device=device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    neg_boxes, neg_scores, neg_labels, neg_masks = run_grounded_sam2(
        text_prompt=args.negative_prompt,
        image_pil=image_pil,
        processor=processor,
        grounding_model=grounding_model,
        sam2_predictor=sam2_predictor,
        device=device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    # Save prompt folders
    write_prompt_folder(
        folder=pos_dir,
        prompt_name="positive",
        text_prompt=args.positive_prompt,
        image_path=img_path,
        image_pil=image_pil,
        boxes=pos_boxes,
        scores=pos_scores,
        labels=pos_labels,
        masks_bool=pos_masks,
        dump_json=dump_json,
    )

    write_prompt_folder(
        folder=neg_dir,
        prompt_name="negative",
        text_prompt=args.negative_prompt,
        image_path=img_path,
        image_pil=image_pil,
        boxes=neg_boxes,
        scores=neg_scores,
        labels=neg_labels,
        masks_bool=neg_masks,
        dump_json=dump_json,
    )

    # negative union for cropping
    neg_union = union_masks(neg_masks, H, W)

    # Per-target crops are based on POSITIVE detections (targets)
    n_targets = len(pos_boxes)
    targets_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_targets):
        x1, y1, x2, y2 = [int(v) for v in pos_boxes[i]]
        pad = max(0, int(args.crop_pad))
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(W - 1, x2 + pad)
        y2p = min(H - 1, y2 + pad)
        x1p, y1p, x2p, y2p = clamp_xyxy((x1p, y1p, x2p, y2p), W, H)

        crop_dir = targets_dir / f"target_{i:03d}"
        crop_dir.mkdir(parents=True, exist_ok=True)

        crop_bgr = img_bgr[y1p:y2p, x1p:x2p].copy()
        cv2.imwrite(str(crop_dir / "crop.jpg"), crop_bgr)

        pos_mask_i_full = pos_masks[i] if pos_masks.shape[0] > i else np.zeros((H, W), dtype=bool)
        pos_mask_crop = pos_mask_i_full[y1p:y2p, x1p:x2p]
        neg_mask_crop = neg_union[y1p:y2p, x1p:x2p]

        # Final boolean mask = positive minus negative
        final_mask_crop = np.logical_and(pos_mask_crop, np.logical_not(neg_mask_crop))

        # Save binary masks
        save_binary_mask_png(crop_dir / "positive_mask.png", pos_mask_crop)
        save_binary_mask_png(crop_dir / "negative_mask.png", neg_mask_crop)
        save_binary_mask_png(crop_dir / "final_mask.png", final_mask_crop)

        # Save the composite image you requested:
        # - pos-only => original pixels
        # - negative => white
        # - else => black
        final_composite = composite_pos_neg(crop_bgr, pos_mask_crop, neg_mask_crop)
        cv2.imwrite(str(crop_dir / "final_composite.png"), final_composite)

        # (Optional) also save a labeled overlay (supervision-style)
        overlay = overlay_single_mask(crop_bgr, final_mask_crop, label=f"target_{i}")
        cv2.imwrite(str(crop_dir / "final_overlay.jpg"), overlay)

        # Per-target json
        if dump_json:
            payload = {
                "target_index": i,
                "image_path": img_path,
                "crop": {
                    "xyxy": [x1p, y1p, x2p, y2p],
                    "width": int(crop_bgr.shape[1]),
                    "height": int(crop_bgr.shape[0]),
                },
                "positive_prompt": normalize_prompt(args.positive_prompt),
                "negative_prompt": normalize_prompt(args.negative_prompt),
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
            }
            (crop_dir / "target.json").write_text(json.dumps(payload, indent=4))

    print(f"Saved full-image positive outputs to: {pos_dir}")
    print(f"Saved full-image negative outputs to: {neg_dir}")
    print(f"Saved per-target crops + masks + json + composite to: {targets_dir}")
    print(f"Targets indexed 0..{max(0, n_targets-1)}")


if __name__ == "__main__":
    main()