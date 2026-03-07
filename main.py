#!/usr/bin/env python3
"""
main.py

Pipeline:
1) Scrape product page + pick best image + extract dimensions
2) (Optional) Create high-level description via GPT using scraped text/bullets/title
3) Inpaint + generate mesh (.glb)
4) Write minimal metadata schema to outputs/metadata/dimensions.json (upsert by clean_name)

Per-object JSON schema:
{
  "<clean_name>": {
    "url": str,
    "product_name": str,
    "dimensions": {"length": float|None, "width": float|None, "height": float|None} | None,
    "units": "inches",
    "high_level_description": str|None,
    "path_to_glb": str|None
  }
}
"""

from __future__ import annotations

import sys
import os
import re
import argparse
import json
import time
import torch
import requests
from pathlib import Path
from PIL import Image

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

from dotenv import load_dotenv
load_dotenv()


# ----------------------------
# Helpers
# ----------------------------
def sanitize_filename(name: str, max_length: int = 60) -> str:
    if not name:
        return "unknown_product"
    s = name.lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_-]", "", s)
    s = re.sub(r"_+", "_", s)
    return s[:max_length].strip("_")


def download_image_to_local(image_url: str, out_dir: str = "outputs/images") -> str:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    name = image_url.split("?")[0].rstrip("/").split("/")[-1] or "image"
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
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


def upsert_dimensions_json(dimensions_json_path: Path, clean_name: str, payload: dict) -> None:
    """
    Upserts payload into a single dimensions.json file keyed by clean_name.
    Safe against missing file / empty file / partial writes.
    """
    dimensions_json_path.parent.mkdir(parents=True, exist_ok=True)

    data = {}
    if dimensions_json_path.exists():
        try:
            data = json.loads(dimensions_json_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}

    data[clean_name] = payload

    tmp_path = dimensions_json_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(dimensions_json_path)


def show_image(image_path: str):
    try:
        if os.environ.get("DISPLAY"):
            Image.open(image_path).show()
    except Exception:
        pass


def extract_page_text_for_summary(result: dict, max_chars: int = 5000) -> str:
    """
    Best-effort extraction of product text from scraper result.
    Adjust keys as you learn your scraper's output structure.
    """
    parts = []

    title = result.get("product_title") or result.get("title") or result.get("page_title")
    if title:
        parts.append(f"TITLE: {title}")

    bullets = result.get("bullets") or result.get("feature_bullets") or result.get("highlights") or []
    if isinstance(bullets, str):
        bullets = [bullets]
    bullets = [b for b in bullets if isinstance(b, str) and b.strip()]
    if bullets:
        parts.append("BULLETS:\n" + "\n".join(f"- {b}" for b in bullets[:20]))

    desc = result.get("description") or result.get("product_description") or result.get("overview") or ""
    if isinstance(desc, str) and desc.strip():
        parts.append("DESCRIPTION:\n" + desc.strip())

    text = "\n\n".join(parts).strip()
    return text[:max_chars]


def gpt_high_level_description(page_text: str, model: str = "gpt-5") -> str | None:
    """
    Returns a 1-2 sentence description or None if unavailable.
    Requires OPENAI_API_KEY in env.
    """
    if not page_text.strip():
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    system = "Write a concise 1-2 sentence high-level product description. Be factual and avoid fluff."
    user = f"Create a 1-2 sentence high-level description of the product from this page text:\n\n{page_text}"

    try:
        r = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=60,
        )
        r.raise_for_status()
        out = r.json()

        # Defensive parsing (Responses API)
        # Typical: out["output"][0]["content"][0]["text"]
        for item in out.get("output", []):
            for c in item.get("content", []):
                t = c.get("text")
                if isinstance(t, str) and t.strip():
                    return t.strip()

        return None
    except Exception:
        return None


def main(url: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-images", type=int, default=30)
    parser.add_argument("--print-scrape", action="store_true")
    parser.add_argument("--model", default="gpt-5")

    try:
        args = parser.parse_args()
    except Exception:
        args = argparse.Namespace(output_dir="outputs", max_images=30, print_scrape=False, model="gpt-5")

    print("\n--- Starting Processing for URL ---")

    # 1) Webscrape data
    result = fetch_url_info_and_best_image(
        url=url,
        output_dir=args.output_dir,
        max_images=args.max_images,
        print_scrape=args.print_scrape,
        model=args.model,
    )

    # get the high_level_description:
    high_level_description = result["best_image_selection"].get("high_level_description")
    print("High Level Description: ", high_level_description)

    # get the attributes:
    attributes = result["best_image_selection"].get("attributes")
    print("Attributes: ", attributes)

    # 2) Determine product name
    raw_name = result.get("product_title") or result.get("title")
    if not raw_name:
        try:
            if "/dp/" in url:
                raw_name = url.split("/dp/")[0].split("/")[-1]
            else:
                raw_name = url.split("?")[0].rstrip("/").split("/")[-1]
        except Exception:
            raw_name = None
    if not raw_name or len(raw_name) < 2:
        raw_name = f"item_{int(time.time())}"

    clean_name = sanitize_filename(raw_name)
    print(f"Product Name Identified: {raw_name}")
    print(f"Filename Safe Name: {clean_name}")

    # 3) Extract dimensions
    dims = (result.get("best_image_selection") or {}).get("dimensions")
    # Optional normalization:
    # dims = dims or {"length": None, "width": None, "height": None}

    # 5) Get best image info
    best_image_url = result["best_image_selection"]["best_image"]["image_url"]
    classes = result["best_image_selection"]["product_names"]

    # 6) Ask GPT for positive/negative prompts
    target_info = infer_positive_and_negative_prompt(best_image_url, classes)

    # 7) Download image locally
    best_image_local_path = download_image_to_local(best_image_url, out_dir=f"{args.output_dir}/images")
    print(f"Saved local image: {best_image_local_path}")

    # 8) Run inpainting pipeline
    impaint_index_json = impaint_main(
        image_path=best_image_local_path,
        positive_prompt=target_info["positive_prompt"],
        negative_prompt=target_info["negative_prompt"],
    )

    with open(impaint_index_json, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    final_composite_path = index_data["targets"][0]["final_composite"]

    # 9) Final inpaint
    _ = run_full_inpaint_with_manual_mask(
        final_composite_path,
        target_class=target_info["positive_prompt"],
        classes_in_crop=target_info["negative_prompt"],
    )

    # Find inpaint output (your current temp behavior)
    folder_path = "uploads"
    files = os.listdir(folder_path) if os.path.isdir(folder_path) else []
    if not files:
        raise RuntimeError("No inpainted outputs found in 'uploads' folder.")
    inpainted_result_path = os.path.join(folder_path, files[0])

    # 10) Generate mesh (.glb)
    mesh_out_dir = Path(args.output_dir) / "meshes"
    mesh_out_dir.mkdir(parents=True, exist_ok=True)
    unique_glb_path = mesh_out_dir / f"{clean_name}.glb"

    print(f"Generating mesh: {unique_glb_path} ...")
    mesh_path = generate_glb_from_image(
        image_path=inpainted_result_path,
        out_glb=str(unique_glb_path),
    )
    print(f"SUCCESS: Generated mesh at: {mesh_path}")

    # 11) Save minimal metadata schema (upsert)
    final_payload = {
        "url": url,
        "product_name": raw_name,
        "dimensions": dims,
        "units": "inches",
        "high_level_description": high_level_description,
        "attributes": attributes,
        "path_to_glb": str(mesh_path),
    }

    dimensions_json_path = Path(args.output_dir) / "metadata" / "dimensions.json"
    upsert_dimensions_json(dimensions_json_path, clean_name, final_payload)


def clear_directory(folder_path: str):
    folder = Path(folder_path)
    if not folder.exists():
        return
    for item in folder.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            import shutil
            shutil.rmtree(item)


if __name__ == "__main__":
    urls = [
        # Amazon Lamps
        # "https://www.amazon.com/LEPOWER-Century-Standing-Studying-Bedroom/dp/B08HQ3XFG1?crid=27SZAX9I5KDFH&dib=eyJ2IjoiMSJ9.ZTER_haswx3H0Ci0sjTTOf2Bq6-SCiO4KXvWxQreDLWiFuyOB7g5659mv3RvbNVC92tFYj7UMCNIury-E3JECMvPjvg80IwDKdK0ziT_SvZtiFfHA6dSC4bFpA12gxPJL29bD--lxDzTvOpz3lLpvsBM1ywgau5n3HZmeq_IzpMtqod-jF7AnT0tNtel-557lnE_7-TYsW1Z6YcUMsfqkh3bMnRAyJswlUlf6VcN9-m0tOyQfyyzKot57_TRodkcgDB6woYqFLJgM9xapAFgfwVvJda2SB4-TaLD2UzW7gQ.jJR-yhuqFjkemwL0UJTit1i2yrOYffq9R3bR-jDKvu4&dib_tag=se&keywords=lamp&qid=1770685015&sprefix=lamp%2Caps%2C244&sr=8-56&xpid=SY5PnJ-2ssz5A&th=1",
        # "https://www.amazon.com/white-crown-Dimmable-Charging-Sensitive/dp/B0868HY623?crid=2692LTC728TVB&dib=eyJ2IjoiMSJ9.n0I8MkX6heBfH9NbZSquXwuo-xHt6hKc00rzM3vyXSJvLgQcpqJz2vYIz-WpmxDzuil4fh6akzvnQC672aApHvgi_HoGMdRz4tb7DrEGTtW6DEVDytojSooTmGj-Xx4sl5U41Kmvnxt6cccVmiXIizQfqRAQQKufcexrWhgdnukSVLAlYOcphlrvoDkQkTbYJaS-PPzScFXLyd2K-UOKw_iYWYdmodO2rw07yZHmLn3g7OpyBIZlklpHgXa-EWMSsr36WVpF3e_iYZKJcT1O_pEB1ECwhzxdQO2qzOfpfTo.AEOwZ57-tPEPBroyn2_ATErJqSQ6Rd6IOjoJdRgvsdo&dib_tag=se&keywords=lamp&qid=1770684986&sprefix=lamp%2Caps%2C196&sr=8-55&th=1",
        # "https://www.amazon.com/Small-Bedroom-Lamps-Color-Temperatures/dp/B0BHVP949X?crid=DYK3VOB10S8S&dib=eyJ2IjoiMSJ9.n0I8MkX6heBfH9NbZSquXwuo-xHt6hKc00rzM3vyXSJvLgQcpqJz2vYIz-WpmxDzuil4fh6akzvnQC672aApHvgi_HoGMdRz4tb7DrEGTtW6DEVDytojSooTmGj-Xx4sl5U41Kmvnxt6cccVmiXIizQfqRAQQKufcexrWhgdnukSVLAlYOcphlrvoDkQkTbYJaS-PPzScFXLyd2K-UOKw_iYWYdmodO2rw07yZHmLn3g7OpyBIZlklpHgXa-EWMSsr36WVpF3e_iYZKJcT1O_pEB1ECwhzxdQO2qzOfpfTo.AEOwZ57-tPEPBroyn2_ATErJqSQ6Rd6IOjoJdRgvsdo&dib_tag=se&keywords=lamp&qid=1770684968&sprefix=lamp%2Caps%2C200&sr=8-26&th=1",
        # "https://www.amazon.com/SUNMORY-Temperatures-Standing-Adjustable-Lampshade/dp/B0CDW797MM?crid=1ZY90QOHAIN32&dib=eyJ2IjoiMSJ9.6SPBRyAjzx798VzNLkKD3PhUEJAn8WEcYS6ZcBsQnuNVfQGHg-wLzzztfYE55yFDeM1z9UIG8LKFG2ysqOb3e8Tc7EIYgVBttS7--w11T9q_q9QYSZABnyLDvf_HO8LC8SqXPYHe2EOY6eQS-8CY8Jya0LNvGt6WJoHlhLqxKi1K9PldjYfMX8_DVdl8Ktptd62cke_17CO5Z7Y0CQLqNG8jB1G7D-uq3oov2Jdf2kG-1zkCHVTaKOk-vt8qhVEZB9wx8hHb0QgrDFuaa9kDCw1PxEog80Og85hSyc5lpIo.G_FaNbEfl9IFJuCP3EiMycN1Zddv_Yh9urSIcDW8cwg&dib_tag=se&keywords=lamp&qid=1770684940&sprefix=lamp%2Caps%2C375&sr=8-18&th=1",
        # "https://www.amazon.com/Simple-Standing-Hanging-Bedroom-Without/dp/B07Z5NQPDS?crid=2D4XCKGIW1CH6&dib=eyJ2IjoiMSJ9.LBoiu2UKUwJVugUsAtsQlVG__9i3NKv3d-JqLlPbobkUo2ZCqwQxOMvCiGcTp3f_PDk5zdBNeIpyTSw-HprmVdZsnnq3h_rgeuDVLF4jwcRxS0AIWdpJRCC1AAQp5GXuHkK1fW7IvGDdt6A1iMTLeGmux6ohKCNe_qgAbKr97isnV1B86rz3zEI0WxvEphALRPZgNBPGhUGA_QRdGVa3J8y_jPm_U2jbsH_uztm2jIaemg0K5lqNj71sJQsAVk53xyeqC6QOHjpuUkJuq78IDM-O61iX_k9zM4ZDxdQP_0o.CphiZRmfjr54Y_76fyNmXjEe8cTYZTz7hM9WNgzQQYs&dib_tag=se&keywords=lamp&qid=1770684928&sprefix=lamp%2Caps%2C228&sr=8-10&th=1",
        # # Amazon Desks
        "https://www.amazon.com/Claiks-Electric-Standing-Adjustable-Height/dp/B0BZ7GXM4M?crid=27K5J79LYFPUA&dib=eyJ2IjoiMSJ9.dzyq_aW_fMv9_myfbaA1o6LsMGBG1BIO9a4eiM475d9_lZLsDPPeu0KUp6EMPBDxefynNRZ0cz2SP0LR2V6oHLIke3wgyCXIQu6IjUvjcBgkgN23nTzQKUOeK2kKb9XKceM8p1j_dagDXCqYbTADk05VxeqJ9E9kremF1bxv5DO2BYxt3cbgZcjHVWPxFfEq3jbjqU6pgw1-1tA8GnTwrQWFY2eL8QxdWlmiW78gCvPN53kFIVtFpGqUcEn5Qgj2ZBYV6kTCo0QtPZDdIjG3NyCd_tCi5snaRsZKlcA-W-4.sMe0ILpX1h4_1N1cHss_mNOZ2hkxwiAygT4e4Uwxj6g&dib_tag=se&keywords=desk&qid=1770685299&sprefix=des%2Caps%2C208&sr=8-9&th=1",
        "https://www.amazon.com/HUANUO-Electric-Adjustable-Workstation-LightWalnut/dp/B0F9X3FDYY?crid=27K5J79LYFPUA&dib=eyJ2IjoiMSJ9.dzyq_aW_fMv9_myfbaA1o6LsMGBG1BIO9a4eiM475d9_lZLsDPPeu0KUp6EMPBDxefynNRZ0cz2SP0LR2V6oHLIke3wgyCXIQu6IjUvjcBgkgN23nTzQKUOeK2kKb9XKceM8p1j_dagDXCqYbTADk05VxeqJ9E9kremF1bxv5DO2BYxt3cbgZcjHVWPxFfEq3jbjqU6pgw1-1tA8GnTwrQWFY2eL8QxdWlmiW78gCvPN53kFIVtFpGqUcEn5Qgj2ZBYV6kTCo0QtPZDdIjG3NyCd_tCi5snaRsZKlcA-W-4.sMe0ILpX1h4_1N1cHss_mNOZ2hkxwiAygT4e4Uwxj6g&dib_tag=se&keywords=desk&qid=1770685299&sprefix=des%2Caps%2C208&sr=8-10",
        # "https://www.amazon.com/Huuger-Outlets-Computer-Drawers-Shelves/dp/B0D9Q542SF?crid=27K5J79LYFPUA&dib=eyJ2IjoiMSJ9.dzyq_aW_fMv9_myfbaA1o6LsMGBG1BIO9a4eiM475d9_lZLsDPPeu0KUp6EMPBDxefynNRZ0cz2SP0LR2V6oHLIke3wgyCXIQu6IjUvjcBgkgN23nTzQKUOeK2kKb9XKceM8p1j_dagDXCqYbTADk05VxeqJ9E9kremF1bxv5DO2BYxt3cbgZcjHVWPxFfEq3jbjqU6pgw1-1tA8GnTwrQWFY2eL8QxdWlmiW78gCvPN53kFIVtFpGqUcEn5Qgj2ZBYV6kTCo0QtPZDdIjG3NyCd_tCi5snaRsZKlcA-W-4.sMe0ILpX1h4_1N1cHss_mNOZ2hkxwiAygT4e4Uwxj6g&dib_tag=se&keywords=desk&qid=1770685299&sprefix=des%2Caps%2C208&sr=8-17",
        # "https://www.amazon.com/Bestier-Executive-Computer-Industrial-Pedestals/dp/B0DLSSTZX2?crid=27K5J79LYFPUA&dib=eyJ2IjoiMSJ9.rXZgYzD7rPjRQugLbcXP_znmA2oej3rKcHzf1abj43iFrq_Dg3BGXhzffwBWlcDeeDuQq-ePE7zlJjv3JmUGp9O_w3NyHNLHarbO1fTcBJhHxVoUgl1kkOvmwLKU47cc-_J6agrjRDS5DtXUj4Mlhk_ERRAINQWyrymsjL5CmsF13wpA7Q0nHqeXxBOgw7ygRweUwSKc-SRPyBSYY05pLMNhSMJTtw0uocGpS9Xm-JdMM-PgC7ubuo_1LB8cGvc2Fo99Of6AOLHoRlKJVQhVync1E5q0rZohs3LXi147WdM.LA3vTyvv7m8BFSGXVDMWMhrME2RlTNa3MVjIAf_V1dw&dib_tag=se&keywords=desk&qid=1770685318&sprefix=des%252Caps%252C208&sr=8-54&xpid=glqC3aiWNpb2G",
        # "https://www.amazon.com/Tribesigns-Executive-Workstations-Conference-Furniture/dp/B0DSL3ZV6P?crid=27K5J79LYFPUA&dib=eyJ2IjoiMSJ9.op2QPD6_ll8Uqf5KoaqNx3XY4nBPB8ZgyoQqANV8Dy8U3qQzgszoO7PGfAzpolR0rwHaNHzBKEEnjKsHuuZMr8TjfI86yAGtDKxiooKHPQZIcmLNrfZ4OmGNhcgAxF6kR2NPDr1LS9P6WK39kPZLiX-ssWEBhPNuAxdv7ChIcILsW22G9y3d-wPPD3pBsCnT58Vm2QxXmSPv8kD3mni986Pz8oAr8jpZdRdtINI1uq11vJ2J3yHb4hJsk3GN8cnvf6tfBBVhjNyVSrluuFtgly2Lu7XsK7IERF18lILTWdM.uKSC_fjq_hFLiLVVW4UyYUpXSYn91akTcvoH8lepq28&dib_tag=se&keywords=desk&qid=1770685339&sprefix=des%2Caps%2C208&sr=8-104&xpid=glqC3aiWNpb2G&th=1",
    ]

    for i, url in enumerate(urls, 1):
        print(f"\n{'='*80}")
        print(f"Processing URL {i}/{len(urls)}")
        print(f"{'='*80}\n")

        try:
            main(url)
        except Exception as e:
            print(f"!!! Error processing URL {i}: {e}")

        print("Cleaning GPU memory between items...")
        torch.cuda.empty_cache()
        # clear_directory("uploads")




