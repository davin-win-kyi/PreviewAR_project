#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from WebScrape_Processing.rank_images_from_local_paths import rank_images_from_local_paths
from WebScrape_Processing.prompt import infer_positive_and_negative_prompt

from Image_Processing.Get_Impainting_Images import main as impaint_main
from Image_Processing.Grounded_SAM2_Config import GroundedSAM2Config, build_models
from Impaint_Processing.nano_banana import run_full_inpaint_with_manual_mask

import gc
import torch


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def safe_json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def clear_directory(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return

    for child in path.iterdir():
        try:
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        except Exception as e:
            print(f"[WARN] Could not remove {child}: {e}")


def newest_file_in_dir(path: Path) -> Optional[Path]:
    if not path.exists():
        return None

    files = [p for p in path.rglob("*") if p.is_file()]
    if not files:
        return None

    return max(files, key=lambda p: p.stat().st_mtime)


def sanitize_filename(text: str, max_len: int = 120) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_", "#") else "_" for c in text)
    cleaned = cleaned.strip("_")
    return cleaned[:max_len] or "item"


def sanitize_object_type(text: str, max_len: int = 80) -> str:
    cleaned = "".join(c.lower() if c.isalnum() or c in ("-", "_") else "_" for c in text)
    cleaned = cleaned.strip("_")
    return cleaned[:max_len] or "object"


def load_products(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input JSON: {path.resolve()}")

    data = safe_json_load(path)

    # Support either:
    # 1) top-level list
    # 2) {"products": [...]}
    if isinstance(data, list):
        return data

    if isinstance(data, dict) and isinstance(data.get("products"), list):
        return data["products"]

    raise ValueError(
        "Expected either a top-level JSON list or an object with a 'products' list"
    )


def resolve_local_image_path(raw_path: str, project_root: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def build_name_mapping(products: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Creates a deterministic mapping:
      url -> <object_type>#n
    based on product order in the input JSON.
    Keeps numbering separate for each object_type.
    """
    counts: Dict[str, int] = defaultdict(int)
    mapping: Dict[str, str] = {}

    for i, product in enumerate(products):
        url = str(product.get("url", "")).strip()
        object_type = str(product.get("object_type", "object")).strip() or "object"
        base = sanitize_object_type(object_type)

        counts[base] += 1
        generated_name = f"{base}#{counts[base]}"

        key = url if url else f"__index__:{i}"
        mapping[key] = generated_name

    return mapping


def resolve_existing_path(raw_path: str, project_root: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


# ------------------------------------------------------------
# Existing JSON loading / lookup
# ------------------------------------------------------------

def load_existing_impainted_records(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load existing impainted_images.json and index records by:
      1) name
      2) url
    """
    if not path.exists():
        return {}

    try:
        data = safe_json_load(path)
    except Exception as e:
        print(f"[WARN] Could not read existing impainted json {path}: {e}")
        return {}

    records = data.get("impainted_images", [])
    if not isinstance(records, list):
        return {}

    indexed: Dict[str, Dict[str, Any]] = {}

    for rec in records:
        if not isinstance(rec, dict):
            continue

        name = str(rec.get("name", "")).strip()
        url = str(rec.get("url", "")).strip()

        if name:
            indexed[f"name::{name}"] = rec
        if url:
            indexed[f"url::{url}"] = rec

    return indexed


def load_existing_runtime_records(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load existing preprocessing_runtime.json and index runtime records by:
      1) url
      2) object_type fallback
    """
    if not path.exists():
        return {}

    try:
        data = safe_json_load(path)
    except Exception as e:
        print(f"[WARN] Could not read existing runtime json {path}: {e}")
        return {}

    records = data.get("runtimes", [])
    if not isinstance(records, list):
        return {}

    indexed: Dict[str, Dict[str, Any]] = {}

    for rec in records:
        if not isinstance(rec, dict):
            continue

        url = str(rec.get("url", "")).strip()
        object_type = str(rec.get("object_type", "")).strip()

        if url:
            indexed[f"url::{url}"] = rec
        elif object_type:
            indexed[f"object_type::{object_type}"] = rec

    return indexed


def get_existing_impainted_record(
    *,
    existing_impainted_records: Dict[str, Dict[str, Any]],
    generated_name: str,
    url: str,
) -> Optional[Dict[str, Any]]:
    rec = (
        existing_impainted_records.get(f"name::{generated_name}")
        or existing_impainted_records.get(f"url::{url}")
    )
    return deepcopy(rec) if rec is not None else None


def get_existing_runtime_record(
    *,
    existing_runtime_records: Dict[str, Dict[str, Any]],
    url: str,
    object_type: str,
) -> Optional[Dict[str, Any]]:
    rec = (
        existing_runtime_records.get(f"url::{url}")
        or existing_runtime_records.get(f"object_type::{object_type}")
    )
    return deepcopy(rec) if rec is not None else None


# ------------------------------------------------------------
# Core per-product processing
# ------------------------------------------------------------

def process_one_product(
    product: Dict[str, Any],
    *,
    project_root: Path,
    output_dir: Path,
    impainted_dir: Path,
    uploads_dir: Path,
    clear_uploads_between_items: bool,
    device: str,
    sam2_predictor,
    processor,
    grounding_model,
    box_threshold: float,
    text_threshold: float,
    crop_pad: int,
    generated_name: str,
    existing_impainted_records: Dict[str, Dict[str, Any]],
    existing_runtime_records: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], bool]:
    """
    Returns:
      (impainted_record, runtime_record_or_none, reused_existing_json)

    Behavior:
      1) If item already exists in impainted_images.json and its image path exists,
         reuse the old impainted JSON object exactly as-is.
      2) If a matching runtime object exists in preprocessing_runtime.json,
         reuse that old runtime JSON object exactly as-is too.
      3) Otherwise run the normal pipeline and create fresh records.
    """

    url = str(product.get("url", "")).strip()
    object_type = str(product.get("object_type", "product"))
    site = product.get("site", "")
    standardized_dimensions = product.get("standardized_dimensions", None)
    images = product.get("images", [])

    impainted_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 0) Reuse existing JSON objects exactly as-is if possible
    # ------------------------------------------------------------
    existing_impainted = get_existing_impainted_record(
        existing_impainted_records=existing_impainted_records,
        generated_name=generated_name,
        url=url,
    )

    if existing_impainted:
        existing_path_str = str(existing_impainted.get("impainted_image_path", "")).strip()
        if existing_path_str:
            existing_path = resolve_existing_path(existing_path_str, project_root)
            if existing_path.exists():
                existing_runtime = get_existing_runtime_record(
                    existing_runtime_records=existing_runtime_records,
                    url=url,
                    object_type=object_type,
                )
                print(f"  [skip] Reusing existing JSON objects: {generated_name}")
                return existing_impainted, existing_runtime, True

    # ------------------------------------------------------------
    # Fallback: if there is no prior JSON record, but a matching image
    # exists on disk, reuse image and create fresh JSON objects.
    # ------------------------------------------------------------
    existing_candidates = []
    base_name = sanitize_filename(generated_name)

    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = impainted_dir / f"{base_name}{ext}"
        if candidate.exists():
            existing_candidates.append(candidate)

    if existing_candidates:
        existing_path = existing_candidates[0]
        print(f"  [skip] Reusing existing impainted image on disk: {existing_path}")
        return {
            "url": url,
            "object_type": object_type,
            "site": site,
            "name": generated_name,
            "standardized_dimensions": standardized_dimensions,
            "impainted_image_path": str(existing_path),
            "status": "ok",
        }, None, False

    if not isinstance(images, list) or not images:
        raise RuntimeError("No images found for this product.")

    local_image_paths: List[str] = []
    image_url_lookup: Dict[str, str] = {}

    for img in images:
        if not isinstance(img, dict):
            continue

        raw_local_path = (img.get("local_path") or "").strip()
        image_url = (img.get("url") or "").strip()

        if not raw_local_path:
            continue

        resolved = resolve_local_image_path(raw_local_path, project_root)
        if resolved.exists():
            resolved_str = str(resolved)
            local_image_paths.append(resolved_str)
            image_url_lookup[resolved_str] = image_url
        else:
            print(f"  [missing] raw local_path: {raw_local_path}")
            print(f"           resolved path: {resolved}")

    if not local_image_paths:
        raise RuntimeError("No valid local image paths found in images[].local_path")

    # 1) Rerank local images
    ranking = rank_images_from_local_paths(
        image_paths=local_image_paths,
        target_object=object_type,
    )

    best_image_path = (ranking.get("best_image_path") or "").strip()
    if not best_image_path:
        raise RuntimeError("rank_images_from_local_paths returned empty best_image_path")

    local_best = Path(best_image_path)

    if not local_best.exists():
        raise RuntimeError(f"Best image path does not exist: {local_best}")

    # 2) Prompt inference
    classes = product.get("product_names") or []
    if not isinstance(classes, list) or not classes:
        classes = [object_type]

    target_info = infer_positive_and_negative_prompt(
        image_path=str(local_best),
        class_name=classes[0] if classes else object_type,
    )

    positive_prompt = target_info.get("positive_prompt", "")
    negative_prompt = target_info.get("negative_prompt", "")

    if not positive_prompt:
        raise RuntimeError("infer_positive_and_negative_prompt returned empty positive_prompt")

    # 3) Run inpainting stage 1 using prebuilt Grounded-SAM2 models
    if clear_uploads_between_items:
        clear_directory(uploads_dir)

    impaint_index_json = impaint_main(
        image_path=str(local_best),
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        device=device,
        sam2_predictor=sam2_predictor,
        processor=processor,
        grounding_model=grounding_model,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        crop_pad=crop_pad,
    )

    idx_path = Path(impaint_index_json)
    if not idx_path.exists():
        raise RuntimeError(f"impaint_main returned missing index json: {impaint_index_json}")

    index_data = safe_json_load(idx_path)

    try:
        final_composite_path = index_data["targets"][0]["final_composite"]
    except Exception:
        raise RuntimeError(
            "Could not find index_data['targets'][0]['final_composite'] from impaint index json"
        )

    del index_data
    cleanup_cuda()

    # 4) Run inpainting stage 2
    _ = run_full_inpaint_with_manual_mask(
        final_composite_path,
        target_class=positive_prompt,
        classes_in_crop=negative_prompt,
    )
    cleanup_cuda()

    # 5) Pick newest output from uploads/
    newest = newest_file_in_dir(uploads_dir)
    if newest is None:
        raise RuntimeError("No inpainted outputs found in uploads/ after run_full_inpaint_with_manual_mask")

    # 6) Copy to outputs_preprocessing/impainted_images/
    ext = newest.suffix if newest.suffix else ".png"
    out_name = f"{base_name}{ext}"
    out_path = impainted_dir / out_name

    shutil.copy2(newest, out_path)

    new_impainted_record = {
        "url": url,
        "object_type": object_type,
        "site": site,
        "name": generated_name,
        "standardized_dimensions": standardized_dimensions,
        "impainted_image_path": str(out_path),
        "status": "ok",
    }

    return new_impainted_record, None, False


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-json",
        default="webscrape_output/all_products.json",
        help="JSON file containing scraped product objects",
    )
    ap.add_argument(
        "--output-dir",
        default="outputs_preprocessing",
        help="Where to write impainted_images/ and json outputs",
    )
    ap.add_argument(
        "--uploads-dir",
        default="uploads",
        help="Folder where your inpaint step writes outputs",
    )
    ap.add_argument(
        "--clear-uploads-between-items",
        action="store_true",
        help="Delete uploads/* before each item to avoid mixing outputs",
    )

    # Grounded-SAM2 config args
    ap.add_argument(
        "--grounding-model",
        default="IDEA-Research/grounding-dino-tiny",
        help="HuggingFace grounding model name",
    )
    ap.add_argument(
        "--sam2-checkpoint",
        default="/home/davin123/PreviewAR_project/grounded_sam2/checkpoints/sam2.1_hiera_large.pt",
        help="Path to SAM2 checkpoint",
    )
    ap.add_argument(
        "--sam2-model-config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 model config path",
    )
    ap.add_argument(
        "--box-threshold",
        type=float,
        default=0.4,
        help="Grounding box threshold",
    )
    ap.add_argument(
        "--text-threshold",
        type=float,
        default=0.3,
        help="Grounding text threshold",
    )
    ap.add_argument(
        "--crop-pad",
        type=int,
        default=0,
        help="Padding around crop boxes",
    )
    ap.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force Grounded-SAM2 to run on CPU",
    )

    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent
    input_json = (project_root / args.input_json).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    uploads_dir = (project_root / args.uploads_dir).resolve()

    impainted_dir = output_dir / "impainted_images"
    impainted_json_path = output_dir / "impainted_images.json"
    runtime_json_path = output_dir / "preprocessing_runtime.json"

    products = load_products(input_json)
    if not products:
        raise SystemExit(f"No products found in: {input_json}")

    name_mapping = build_name_mapping(products)

    output_dir.mkdir(parents=True, exist_ok=True)
    impainted_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    existing_impainted_records = load_existing_impainted_records(impainted_json_path)
    existing_runtime_records = load_existing_runtime_records(runtime_json_path)

    cfg = GroundedSAM2Config(
        grounding_model=args.grounding_model,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_model_config=args.sam2_model_config,
        force_cpu=bool(args.force_cpu),
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        crop_pad=args.crop_pad,
    )

    device, sam2_predictor, processor, grounding_model = build_models(cfg)

    impainted_records: List[Dict[str, Any]] = []
    runtimes: List[Dict[str, Any]] = []

    try:
        for i, product in enumerate(products, start=1):
            url = str(product.get("url", "")).strip()
            object_type = str(product.get("object_type", "product"))
            generated_name = name_mapping.get(url, name_mapping.get(f"__index__:{i-1}", f"object#{i}"))

            print(f"\n[{i}/{len(products)}] Processing: {generated_name}")

            t0 = time.perf_counter()
            try:
                rec, existing_runtime_rec, reused_existing_json = process_one_product(
                    product,
                    project_root=project_root,
                    output_dir=output_dir,
                    impainted_dir=impainted_dir,
                    uploads_dir=uploads_dir,
                    clear_uploads_between_items=bool(args.clear_uploads_between_items),
                    device=device,
                    sam2_predictor=sam2_predictor,
                    processor=processor,
                    grounding_model=grounding_model,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    crop_pad=args.crop_pad,
                    generated_name=generated_name,
                    existing_impainted_records=existing_impainted_records,
                    existing_runtime_records=existing_runtime_records,
                )

                runtime_sec = round(time.perf_counter() - t0, 4)

                impainted_records.append(rec)

                if reused_existing_json and existing_runtime_rec is not None:
                    runtimes.append(existing_runtime_rec)
                else:
                    runtimes.append({
                        "url": rec.get("url", ""),
                        "object_type": rec.get("object_type", ""),
                        "runtime_sec": runtime_sec,
                        "status": "ok",
                    })

                print(f"[OK] {generated_name}")
                print(f"     impainted_image_path: {rec.get('impainted_image_path', '')}")

            except Exception as e:
                runtime_sec = round(time.perf_counter() - t0, 4)

                existing_impainted = get_existing_impainted_record(
                    existing_impainted_records=existing_impainted_records,
                    generated_name=generated_name,
                    url=url,
                )
                existing_runtime = get_existing_runtime_record(
                    existing_runtime_records=existing_runtime_records,
                    url=url,
                    object_type=object_type,
                )

                if existing_impainted:
                    existing_path_str = str(existing_impainted.get("impainted_image_path", "")).strip()
                    if existing_path_str:
                        existing_path = resolve_existing_path(existing_path_str, project_root)
                        if existing_path.exists():
                            impainted_records.append(existing_impainted)
                        else:
                            impainted_records.append({
                                "url": product.get("url", ""),
                                "object_type": product.get("object_type", "product"),
                                "site": product.get("site", ""),
                                "name": generated_name,
                                "standardized_dimensions": product.get("standardized_dimensions", None),
                                "impainted_image_path": "",
                                "status": "fail",
                            })
                    else:
                        impainted_records.append({
                            "url": product.get("url", ""),
                            "object_type": product.get("object_type", "product"),
                            "site": product.get("site", ""),
                            "name": generated_name,
                            "standardized_dimensions": product.get("standardized_dimensions", None),
                            "impainted_image_path": "",
                            "status": "fail",
                        })
                else:
                    impainted_records.append({
                        "url": product.get("url", ""),
                        "object_type": product.get("object_type", "product"),
                        "site": product.get("site", ""),
                        "name": generated_name,
                        "standardized_dimensions": product.get("standardized_dimensions", None),
                        "impainted_image_path": "",
                        "status": "fail",
                    })

                if existing_runtime is not None:
                    runtimes.append(existing_runtime)
                else:
                    runtimes.append({
                        "url": product.get("url", ""),
                        "object_type": product.get("object_type", "product"),
                        "runtime_sec": runtime_sec,
                        "status": "fail",
                        "error": str(e),
                    })

                print(f"[FAIL] {generated_name}: {e}")
            finally:
                cleanup_cuda()

    finally:
        del sam2_predictor
        del processor
        del grounding_model
        cleanup_cuda()

    avg_runtime = None
    ok_vals = [
        r["runtime_sec"]
        for r in runtimes
        if isinstance(r, dict) and r.get("status") == "ok" and isinstance(r.get("runtime_sec"), (int, float))
    ]
    if ok_vals:
        avg_runtime = round(sum(ok_vals) / len(ok_vals), 4)

    impainted_json_path.write_text(
        json.dumps({"impainted_images": impainted_records}, indent=2),
        encoding="utf-8",
    )

    runtime_json_path.write_text(
        json.dumps(
            {
                "runtimes": runtimes,
                "average_runtime_sec": avg_runtime,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f" - Wrote: {impainted_json_path}")
    print(f" - Wrote: {runtime_json_path}")


if __name__ == "__main__":
    main()