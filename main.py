#!/usr/bin/env python3
"""
main.py

Minimal entrypoint:
- Calls WebScrape_Processing.webscrape_processing.fetch_url_info_and_best_image
- Prints the returned payload

Usage:
  python main.py "https://www.amazon.com/..."
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse

from pathlib import Path
import os
import re
import requests


# --- Ensure repo root is on PYTHONPATH ---
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --- Import the scraper wrapper ---
from WebScrape_Processing.webscrape_processing import fetch_url_info_and_best_image
from WebScrape_Processing.prompt import infer_positive_and_negative_prompt
from Image_Processing.Get_Impainting_Images import main as impaint_main
from Impaint_Processing.nano_banana import run_full_inpaint_with_manual_mask 
from Mesh_Processing.get_mesh import generate_glb_from_image

def download_image_to_local(image_url: str, out_dir: str = "outputs/images") -> str:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # Try to keep a reasonable filename
    name = image_url.split("?")[0].rstrip("/").split("/")[-1] or "image"
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)

    # Ensure it has an extension (some URLs may not)
    if "." not in name:
        name += ".jpg"

    local_path = out_dir_p / name

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }
    r = requests.get(image_url, headers=headers, timeout=30)
    r.raise_for_status()

    local_path.write_bytes(r.content)
    return str(local_path)

def main(url) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-images", type=int, default=30)
    parser.add_argument("--print-scrape", action="store_true")
    parser.add_argument("--model", default="gpt-5")

    args = parser.parse_args()

    # Webscrape Data
    result = fetch_url_info_and_best_image(
        url=url,
        output_dir=args.output_dir,
        max_images=args.max_images,
        print_scrape=args.print_scrape,
        model=args.model,
    )

    # Pretty-print the result
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))


    # positive objects + negative objects
    best_image_url = result['best_image_selection']['best_image']['image_url']
    classes = result['best_image_selection']['product_names'] 
    # best_image_url = "https://m.media-amazon.com/images/I/71v0slLGRlL._AC_SL1500_.jpg"
    # classes = ["couch", "sofa", "furniture"]

    # 1) Ask GPT for positive/negative prompts using the URL
    target_info = infer_positive_and_negative_prompt(best_image_url, classes)

    # 2) Download image locally (because cv2.imread / PIL.Image.open need a file path)
    best_image_local_path = download_image_to_local(best_image_url, out_dir="outputs/images")
    print(f"Saved image to: {best_image_local_path}")

    # 3) Run inpainting pipeline using the LOCAL PATH
    impaint_index_json = impaint_main(
        image_path=best_image_local_path,
        positive_prompt=target_info["positive_prompt"],
        negative_prompt=target_info["negative_prompt"],
    )
    print(f"Inpainting index JSON: {impaint_index_json}")

    with open(impaint_index_json, "r") as f:
        index_data = json.load(f)

    final_composite_path = index_data["targets"][0]["final_composite"]

    # Impaint the images
    impaint_result = run_full_inpaint_with_manual_mask(final_composite_path)

    print("Impaint result: ", impaint_result)

    # pass the best image into the call to Hunyuan3D
    mesh_path = generate_glb_from_image(
        image_path=impaint_result,
        out_glb="outputs/mesh.glb",
    )

    print("Generated mesh at: ", mesh_path)



if __name__ == "__main__":
    url = "https://www.amazon.com/Gericco-3-Seater-Cushion-Upholstered-Apartment/dp/B0D2D4CJ4V/ref=sxbs_pa_sp_search_thematic_btf_sspa?content-id=amzn1.sym.cbc375c4-5382-432d-9a3c-e8a9cffc06cd%3Aamzn1.sym.cbc375c4-5382-432d-9a3c-e8a9cffc06cd&crid=18NUP8JHGYLVE&cv_ct_cx=couch&keywords=couch&pd_rd_i=B0D2D4CJ4V&pd_rd_r=0fa592d2-f4b0-4567-9534-bd78faaae599&pd_rd_w=2ri4B&pd_rd_wg=m8OV2&pf_rd_p=cbc375c4-5382-432d-9a3c-e8a9cffc06cd&pf_rd_r=0KN28VC13TSQBH978W2C&qid=1769328704&sbo=RZvfv%2F%2FHxDF%2BO5021pAnSA%3D%3D&sprefix=couch%2Caps%2C239&sr=1-4-7cf413b8-4ee0-4602-a464-4db99a71081c-spons&aref=nYqX97ABAn&sp_csd=d2lkZ2V0TmFtZT1zcF9zZWFyY2hfdGhlbWF0aWNfYnRm&th=1"
    main(url)