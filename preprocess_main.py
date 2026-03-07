#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from WebScrape_Processing.rank_images_from_local_paths import rank_images_from_local_paths
from WebScrape_Processing.prompt import infer_positive_and_negative_prompt

# Replace these imports with your actual module paths
from Image_Processing.Get_Impainting_Images import main as impaint_main
from Impaint_Processing.nano_banana import run_full_inpaint_with_manual_mask

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

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
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)
    cleaned = cleaned.strip("_")
    return cleaned[:max_len] or "item"


def load_products(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input JSON: {path.resolve()}")

    data = safe_json_load(path)
    if not isinstance(data, list):
        raise ValueError("Expected a top-level JSON list in webscrapping_output/all_products.json")
    return data


def resolve_local_image_path(raw_path: str, project_root: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


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
) -> Dict[str, Any]:
    """
    Process one product from all_products.json:
      1) gather local images
      2) rerank local images
      3) infer positive/negative prompt from best image
      4) run two-stage inpainting
      5) copy newest uploads output into impainted_images/
      6) return output record
    """

    url = product.get("url", "")
    object_type = product.get("object_type", "product")
    site = product.get("site", "")
    scraped_at = product.get("scraped_at", "")
    scrape_runtime_seconds = product.get("runtime_seconds", None)
    name = product.get("name", "")
    description = product.get("description", "")
    dims = product.get("dimensions", {})
    standardized_dimensions = product.get("standardized_dimensions", None)
    images = product.get("images", [])

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

    best_image_url = image_url_lookup.get(best_image_path, "")
    local_best = Path(best_image_path)

    if not local_best.exists():
        raise RuntimeError(f"Best image path does not exist: {local_best}")

    # 2) Prompt inference
    # You can pass object_type directly, or if you want old multi-class behavior:
    classes = product.get("product_names") or []
    if not isinstance(classes, list) or not classes:
        classes = [object_type]

    # Your current prompt function takes image_path + class_name
    # so we use object_type (or classes[0] if preferred)
    target_info = infer_positive_and_negative_prompt(
        image_path=str(local_best),
        class_name=classes[0] if classes else object_type,
    )

    positive_prompt = target_info.get("positive_prompt", "")
    negative_prompt = target_info.get("negative_prompt", "")

    if not positive_prompt:
        raise RuntimeError("infer_positive_and_negative_prompt returned empty positive_prompt")

    # 3) Run inpainting stage 1
    if clear_uploads_between_items:
        clear_directory(uploads_dir)

    impaint_index_json = impaint_main(
        image_path=str(local_best),
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
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

    # 4) Run inpainting stage 2
    _ = run_full_inpaint_with_manual_mask(
        final_composite_path,
        target_class=positive_prompt,
        classes_in_crop=negative_prompt,
    )

    # 5) Pick newest output from uploads/
    newest = newest_file_in_dir(uploads_dir)
    if newest is None:
        raise RuntimeError("No inpainted outputs found in uploads/ after run_full_inpaint_with_manual_mask")

    # 6) Copy to outputs_preprocessing/impainted_images/
    impainted_dir.mkdir(parents=True, exist_ok=True)

    folder_tag = sanitize_filename(name or object_type or url)
    ext = newest.suffix if newest.suffix else ".png"
    out_name = f"{sanitize_filename(object_type)}__{folder_tag}{ext}"
    out_path = impainted_dir / out_name

    shutil.copy2(newest, out_path)

    return {
        "url": url,
        "object_type": object_type,
        "site": site,
        "scraped_at": scraped_at,
        "scrape_runtime_seconds": scrape_runtime_seconds,
        "name": name,
        "description": description,
        "dimensions": dims if isinstance(dims, dict) else None,
        "standardized_dimensions": standardized_dimensions,
        "images": images,

        "best_image_path": str(local_best),
        "best_image_url": best_image_url,
        "ranking_reasoning": ranking.get("reasoning", ""),
        "ranking_scores": ranking.get("scores", {}),

        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,

        "impainted_image_path": str(out_path),
    }


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

    output_dir.mkdir(parents=True, exist_ok=True)
    impainted_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    impainted_records: List[Dict[str, Any]] = []
    runtimes: List[Dict[str, Any]] = []

    for i, product in enumerate(products, start=1):
        tag = sanitize_filename(product.get("name") or product.get("object_type") or f"item_{i}")
        print(f"\n[{i}/{len(products)}] Processing: {tag}")

        t0 = time.perf_counter()
        try:
            rec = process_one_product(
                product,
                project_root=project_root,
                output_dir=output_dir,
                impainted_dir=impainted_dir,
                uploads_dir=uploads_dir,
                clear_uploads_between_items=bool(args.clear_uploads_between_items),
            )

            runtime_sec = round(time.perf_counter() - t0, 4)
            rec["preprocessing_runtime_sec"] = runtime_sec
            rec["status"] = "ok"

            impainted_records.append(rec)
            runtimes.append({
                "url": rec.get("url", ""),
                "object_type": rec.get("object_type", ""),
                "runtime_sec": runtime_sec,
                "status": "ok",
            })

            print(f"[OK] {tag}")
            print(f"     best_image_path: {rec.get('best_image_path', '')}")
            print(f"     impainted_image_path: {rec.get('impainted_image_path', '')}")

        except Exception as e:
            runtime_sec = round(time.perf_counter() - t0, 4)

            impainted_records.append({
                "url": product.get("url", ""),
                "object_type": product.get("object_type", "product"),
                "name": product.get("name", ""),
                "description": product.get("description", ""),
                "dimensions": product.get("dimensions", {}),
                "images": product.get("images", []),
                "best_image_path": "",
                "best_image_url": "",
                "positive_prompt": "",
                "negative_prompt": "",
                "impainted_image_path": "",
                "preprocessing_runtime_sec": runtime_sec,
                "status": "fail",
                "error": str(e),
            })

            runtimes.append({
                "url": product.get("url", ""),
                "object_type": product.get("object_type", "product"),
                "runtime_sec": runtime_sec,
                "status": "fail",
                "error": str(e),
            })

            print(f"[FAIL] {product.get('object_type', 'product')}: {e}")

    avg_runtime = None
    ok_vals = [r["runtime_sec"] for r in runtimes if r.get("status") == "ok"]
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