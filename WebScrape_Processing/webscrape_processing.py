#!/usr/bin/env python3
"""
Wrapper:
- Extracts URL info (company_name, product_name) via extract_url_info.extract_with_gpt5(url)
- Selects/downloads best image via best_image_selector.get_best_image_url(url, ...)
- Returns one unified payload for downstream use

Assumptions:
- extract_url_info.py provides: extract_with_gpt5(url) -> dict
- best_image_selector provides: get_best_image_url(url, **kwargs) -> dict

Portability:
- Imports are resolved relative to this file (no hard-coded machine paths)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

load_dotenv()  



# --- Stable local imports (repo-relative) ---
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent  # change to THIS_FILE.parents[1] if needed
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local modules
from extract_url_info import extract_with_gpt5

# If your selector file is named select_best_product_image.py, import from that:
from best_image_selector import get_best_image_url


def fetch_url_info_and_best_image(
    url: str,
    *,
    # passthrough params to get_best_image_url
    max_images: int = 30,
    print_scrape: bool = False,
    output_dir: Union[str, Path] = "outputs",
    best_image_filename: str = "best_image.png",
    model: str = "gpt-5",
    save_result_json: bool = True,
    result_json_filename: str = "selection_result.json",
) -> Dict[str, Any]:
    """
    High-level wrapper that returns:
      {
        "url": ...,
        "company_name": ...,
        "seed_product_names": [...],      # from extract_url_info
        "best_image_selection": {...},    # from best_image_selector (includes dimensions, best_image, etc.)
      }

    Notes:
    - We call extract_url_info first to validate the URL and get company/product names.
    - We call best_image_selector to scrape + rank + optionally download/save image.
    """

    # 1) Get URL info (company/product names)
    info = extract_with_gpt5(url)
    if not isinstance(info, dict):
        raise RuntimeError("extract_with_gpt5(url) did not return a dict.")

    company_name = info.get("company_name") or ""
    product_name = info.get("product_name") or []

    # normalize product_name to list[str]
    if isinstance(product_name, str):
        seed_product_names: List[str] = [product_name]
    elif isinstance(product_name, list):
        seed_product_names = [str(x) for x in product_name]
    else:
        seed_product_names = [str(product_name)] if product_name is not None else []

    # 2) Select best image (this internally scrapes and may call GPT for dims/aliases/ranking)
    best_image_selection = get_best_image_url(
        url,
        max_images=max_images,
        print_scrape=print_scrape,
        output_dir=output_dir,
        best_image_filename=best_image_filename,
        model=model,
        save_result_json=save_result_json,
        result_json_filename=result_json_filename,
    )

    # 3) Return unified payload
    return {
        "url": url,
        "company_name": company_name,
        "seed_product_names": seed_product_names,
        "url_info": info,  # keep the full dict in case it has extra fields
        "best_image_selection": best_image_selection,
    }


if __name__ == "__main__":
    # Example usage
    test_url = "https://www.amazon.com/Gericco-3-Seater-Cushion-Upholstered-Apartment/dp/B0D2D4CJ4V/ref=sxbs_pa_sp_search_thematic_btf_sspa?content-id=amzn1.sym.cbc375c4-5382-432d-9a3c-e8a9cffc06cd%3Aamzn1.sym.cbc375c4-5382-432d-9a3c-e8a9cffc06cd&crid=18NUP8JHGYLVE&cv_ct_cx=couch&keywords=couch&pd_rd_i=B0D2D4CJ4V&pd_rd_r=0fa592d2-f4b0-4567-9534-bd78faaae599&pd_rd_w=2ri4B&pd_rd_wg=m8OV2&pf_rd_p=cbc375c4-5382-432d-9a3c-e8a9cffc06cd&pf_rd_r=0KN28VC13TSQBH978W2C&qid=1769328704&sbo=RZvfv%2F%2FHxDF%2BO5021pAnSA%3D%3D&sprefix=couch%2Caps%2C239&sr=1-4-7cf413b8-4ee0-4602-a464-4db99a71081c-spons&aref=nYqX97ABAn&sp_csd=d2lkZ2V0TmFtZT1zcF9zZWFyY2hfdGhlbWF0aWNfYnRm&th=1"
    out = fetch_url_info_and_best_image(
        test_url,
        output_dir="outputs",
        max_images=30,
    )
    print(out["best_image_selection"].get("best_image", {}))
