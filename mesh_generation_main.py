#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import ElementClickInterceptedException, StaleElementReferenceException


# ============================================================
# File validity / stability helpers
# ============================================================

def wait_for_file_stable(
    path: Path,
    timeout_sec: float = 180.0,
    poll_interval_sec: float = 0.25,
    stable_for_sec: float = 2.0,
) -> None:
    t0 = time.time()
    last_size = -1
    stable_start: Optional[float] = None

    while (time.time() - t0) < timeout_sec:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            size = -1

        if size > 0 and size == last_size:
            if stable_start is None:
                stable_start = time.time()
            if (time.time() - stable_start) >= stable_for_sec:
                return
        else:
            stable_start = None
            last_size = size

        time.sleep(poll_interval_sec)

    raise RuntimeError(f"File never became stable within {timeout_sec:.1f}s: {path}")


def assert_is_glb(path: Path) -> None:
    with path.open("rb") as f:
        magic = f.read(4)
    if magic != b"glTF":
        raise RuntimeError(f"Not a valid GLB (magic={magic}) at: {path}")


def atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass
    shutil.copy2(src, tmp)
    tmp.replace(dst)


# ============================================================
# Cache helpers
# ============================================================

def _list_subdirs(cache_dir: Path) -> Dict[Path, float]:
    out: Dict[Path, float] = {}
    if not cache_dir.exists():
        return out
    for p in cache_dir.iterdir():
        if p.is_dir():
            try:
                out[p] = p.stat().st_mtime
            except FileNotFoundError:
                pass
    return out


def _find_textured_mesh_glb(folder: Path) -> Optional[Path]:
    target = "textured_mesh.glb"
    try:
        for p in folder.rglob(target):
            if p.is_file() and p.name == target:
                return p
    except FileNotFoundError:
        return None
    return None


def wait_for_fresh_textured_mesh_after_click(
    cache_dir: Path,
    click_time: float,
    timeout_sec: float = 300.0,
    poll_interval_sec: float = 0.25,
) -> Tuple[Path, Path]:
    cache_dir = cache_dir.resolve()
    t0 = time.time()

    while (time.time() - t0) < timeout_sec:
        folders = _list_subdirs(cache_dir)
        if folders:
            newest_folders = sorted(folders.keys(), key=lambda p: folders[p], reverse=True)
            for folder in newest_folders[:30]:
                mesh = _find_textured_mesh_glb(folder)
                if mesh is None:
                    continue
                try:
                    mtime = mesh.stat().st_mtime
                except FileNotFoundError:
                    continue
                if mtime >= click_time:
                    return folder, mesh

        time.sleep(poll_interval_sec)

    raise RuntimeError(
        f"Timed out after {timeout_sec:.1f}s waiting for fresh textured_mesh.glb after click."
    )


# ============================================================
# Selenium helpers
# ============================================================

def build_driver(headless: bool = False) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1400,900")
    return webdriver.Chrome(options=opts)


def _js_click(driver: webdriver.Chrome, el) -> None:
    driver.execute_script("arguments[0].scrollIntoView({block:'center', inline:'center'});", el)
    driver.execute_script("arguments[0].click();", el)


def click_gen_textured_shape(driver: webdriver.Chrome) -> None:
    wait = WebDriverWait(driver, 30)
    btn = wait.until(
        EC.presence_of_element_located((By.XPATH, "//button[normalize-space(.)='Gen Textured Shape']"))
    )
    for _ in range(10):
        try:
            if btn.is_displayed() and btn.is_enabled():
                try:
                    btn.click()
                    return
                except ElementClickInterceptedException:
                    _js_click(driver, btn)
                    return
        except StaleElementReferenceException:
            btn = wait.until(
                EC.presence_of_element_located((By.XPATH, "//button[normalize-space(.)='Gen Textured Shape']"))
            )
        time.sleep(0.2)
    raise RuntimeError("Failed to click 'Gen Textured Shape' button.")


def set_target_face_number(driver: webdriver.Chrome, faces: int) -> None:
    export_tab = None
    for xp in [
        "//button[normalize-space(.)='Export']",
        "//div[@role='tab' and normalize-space(.)='Export']",
        "//*[self::div or self::span][normalize-space(.)='Export']",
    ]:
        els = driver.find_elements(By.XPATH, xp)
        export_tab = next((e for e in els if e.is_displayed()), None)
        if export_tab:
            break

    if export_tab:
        try:
            export_tab.click()
        except Exception:
            _js_click(driver, export_tab)
        time.sleep(0.2)

    label = None
    for xp in [
        "//*[contains(normalize-space(.), 'Target Face Number')]",
        "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'target face number')]",
    ]:
        els = driver.find_elements(By.XPATH, xp)
        label = next((e for e in els if e.is_displayed()), None)
        if label:
            break

    face_input = None
    if label:
        for container_xp in [
            "./ancestor::*[self::div or self::fieldset][1]",
            "./ancestor::*[self::div or self::fieldset][2]",
            "./ancestor::*[self::div or self::fieldset][3]",
        ]:
            try:
                container = label.find_element(By.XPATH, container_xp)
                nums = [
                    i for i in container.find_elements(By.CSS_SELECTOR, "input[type='number']")
                    if i.is_displayed() and i.is_enabled()
                ]
                if nums:
                    face_input = nums[0]
                    break
            except Exception:
                continue

    if face_input is None:
        nums = [
            e for e in driver.find_elements(By.CSS_SELECTOR, "input[type='number']")
            if e.is_displayed() and e.is_enabled()
        ]
        if nums:
            face_input = nums[-1]

    if face_input is None:
        raise RuntimeError("Could not find Target Face Number input.")

    driver.execute_script(
        """
        const el = arguments[0];
        const v = arguments[1];
        el.focus();
        el.value = v;
        el.dispatchEvent(new Event('input', {bubbles:true}));
        el.dispatchEvent(new Event('change', {bubbles:true}));
        """,
        face_input,
        str(int(faces)),
    )
    time.sleep(0.2)


def selenium_upload_set_faces_generate_get_mesh(
    driver: webdriver.Chrome,
    image_path: Path,
    cache_dir: Path,
    timeout_sec: float,
    target_faces: int,
) -> Path:
    wait = WebDriverWait(driver, 60)
    image_path = image_path.resolve()
    cache_dir = cache_dir.resolve()

    file_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']")))
    file_input.send_keys(str(image_path))
    time.sleep(1.0)

    set_target_face_number(driver, faces=target_faces)

    click_gen_textured_shape(driver)
    click_time = time.time()

    _, mesh_path = wait_for_fresh_textured_mesh_after_click(
        cache_dir=cache_dir,
        click_time=click_time,
        timeout_sec=timeout_sec,
    )
    return mesh_path


# ============================================================
# Metadata helpers
# ============================================================

def infer_object_type_from_filename(p: Path) -> str:
    stem = p.stem
    if "__" in stem:
        left = stem.split("__", 1)[0].strip()
        if left:
            return left
    return "item"


def load_impainted_index(index_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    records = data.get("impainted_images")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Expected key 'impainted_images' containing a non-empty list.")
    return records


def normalize_dimensions(d: Any) -> Dict[str, Any]:
    if isinstance(d, dict):
        return d
    return {}


def resolve_input_path(raw_path: str, project_root: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p.resolve()
    return (project_root / p).resolve()


# ============================================================
# Main
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impainted_index", default="outputs_preprocessing/impainted_images.json")
    ap.add_argument("--out_dir", default="mesh_outputs", help="Where to write glbs + meshes.json + runtimes.json")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if GLB already exists")

    ap.add_argument("--gradio_url", default="http://127.0.0.1:8080/")
    ap.add_argument("--gradio_cache_dir", default="hunyuan/gradio_cache")
    ap.add_argument("--cache_timeout_sec", type=float, default=300.0)

    ap.add_argument("--target_faces", type=int, default=1_000_000, help="Target Face Number to set in UI")
    ap.add_argument("--headless", type=int, default=0, help="1=headless, 0=show browser window")

    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent

    index_path = resolve_input_path(args.impainted_index, project_root)
    if not index_path.exists():
        print(f"[ERROR] Missing index: {index_path}")
        return 1

    try:
        records = load_impainted_index(index_path)
    except Exception as e:
        print(f"[ERROR] Could not parse index: {e}")
        return 1

    out_dir = resolve_input_path(args.out_dir, project_root)
    glb_dir = (out_dir / "glbs").resolve()
    glb_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = resolve_input_path(args.gradio_cache_dir, project_root)

    meshes_out: List[Dict[str, Any]] = []
    runtimes_out: List[Dict[str, Any]] = []

    driver = build_driver(headless=bool(args.headless))
    try:
        driver.get(args.gradio_url)
        wait = WebDriverWait(driver, 60)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.gradio-container")))
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']")))
        print("[selenium] Gradio loaded and ready.")
        print(f"[paths] project_root={project_root}")
        print(f"[paths] cache_dir={cache_dir}")
        print(f"[paths] glb_dir={glb_dir}")

        total = len(records)
        for i, rec in enumerate(records, start=1):
            img_raw = rec.get("impainted_image_path")
            if not isinstance(img_raw, str) or not img_raw.strip():
                print(f"[{i}/{total}] [SKIP] missing impainted_image_path")
                runtimes_out.append({
                    "description": rec.get("object_type", "item"),
                    "runtime_sec": 0.0,
                    "status": "skip_missing_path",
                })
                continue

            img_path = resolve_input_path(img_raw, project_root)
            if not img_path.exists():
                print(f"[{i}/{total}] [SKIP] image missing: {img_path}")
                runtimes_out.append({
                    "description": rec.get("object_type", "item"),
                    "runtime_sec": 0.0,
                    "status": "skip_missing_file",
                })
                continue

            object_type = rec.get("object_type")
            if not isinstance(object_type, str) or not object_type.strip():
                object_type = infer_object_type_from_filename(img_path)
            object_type = object_type.strip()

            dims = normalize_dimensions(rec.get("dimensions"))
            text_desc = rec.get("description")
            if not isinstance(text_desc, str):
                text_desc = ""

            product_name = rec.get("name", "")
            source_url = rec.get("url", "")
            best_image_path = rec.get("best_image_path", "")
            best_image_url = rec.get("best_image_url", "")
            positive_prompt = rec.get("positive_prompt", "")
            negative_prompt = rec.get("negative_prompt", "")
            preprocessing_runtime_sec = rec.get("preprocessing_runtime_sec", None)
            ranking_reasoning = rec.get("ranking_reasoning", "")
            ranking_scores = rec.get("ranking_scores", {})
            source_images = rec.get("images", [])
            scrape_runtime_seconds = rec.get("scrape_runtime_seconds", None)
            site = rec.get("site", "")
            scraped_at = rec.get("scraped_at", "")

            out_glb = glb_dir / f"{object_type}__{img_path.stem}.glb"

            if args.skip_existing and out_glb.exists():
                print(f"[{i}/{total}] [SKIP] exists: {out_glb.name}")
                meshes_out.append({
                    "description": object_type,
                    "text_description": text_desc,
                    "name": product_name,
                    "dimensions": dims,
                    "path": str(out_glb),
                    "source_image": str(img_path),
                    "source_url": source_url,
                    "best_image_path": best_image_path,
                    "best_image_url": best_image_url,
                    "positive_prompt": positive_prompt,
                    "negative_prompt": negative_prompt,
                    "site": site,
                    "scraped_at": scraped_at,
                    "scrape_runtime_seconds": scrape_runtime_seconds,
                    "preprocessing_runtime_sec": preprocessing_runtime_sec,
                    "ranking_reasoning": ranking_reasoning,
                    "ranking_scores": ranking_scores,
                    "images": source_images,
                    "status": "skip",
                })
                runtimes_out.append({
                    "description": object_type,
                    "runtime_sec": 0.0,
                    "status": "skip",
                })
                continue

            print(f"[{i}/{total}] Gradio(Selenium) generating: {object_type} :: {img_path.name}")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()

            try:
                mesh_path = Path(
                    selenium_upload_set_faces_generate_get_mesh(
                        driver=driver,
                        image_path=img_path,
                        cache_dir=cache_dir,
                        timeout_sec=float(args.cache_timeout_sec),
                        target_faces=int(args.target_faces),
                    )
                ).resolve()

                wait_for_file_stable(mesh_path, timeout_sec=240.0, stable_for_sec=2.0)
                assert_is_glb(mesh_path)
                atomic_copy(mesh_path, out_glb)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_sec = time.time() - t0

                meshes_out.append({
                    "description": object_type,
                    "text_description": text_desc,
                    "name": product_name,
                    "dimensions": dims,
                    "path": str(out_glb),
                    "source_image": str(img_path),
                    "source_url": source_url,
                    "best_image_path": best_image_path,
                    "best_image_url": best_image_url,
                    "positive_prompt": positive_prompt,
                    "negative_prompt": negative_prompt,
                    "site": site,
                    "scraped_at": scraped_at,
                    "scrape_runtime_seconds": scrape_runtime_seconds,
                    "preprocessing_runtime_sec": preprocessing_runtime_sec,
                    "ranking_reasoning": ranking_reasoning,
                    "ranking_scores": ranking_scores,
                    "images": source_images,
                    "mesh_generation_runtime_sec": float(total_sec),
                    "status": "ok",
                })
                runtimes_out.append({
                    "description": object_type,
                    "runtime_sec": float(total_sec),
                    "status": "ok",
                })

                print(f"  [OK] total={total_sec:.2f}s")
                print(f"       source={mesh_path} ({mesh_path.stat().st_size} bytes)")
                print(f"       saved ={out_glb} ({out_glb.stat().st_size} bytes)")

            except Exception as e:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_sec = time.time() - t0
                print(f"  [FAIL] {e}")
                runtimes_out.append({
                    "description": object_type,
                    "runtime_sec": float(total_sec),
                    "status": "fail",
                    "error": str(e),
                })
                meshes_out.append({
                    "description": object_type,
                    "text_description": text_desc,
                    "name": product_name,
                    "dimensions": dims,
                    "path": str(out_glb),
                    "source_image": str(img_path),
                    "source_url": source_url,
                    "best_image_path": best_image_path,
                    "best_image_url": best_image_url,
                    "positive_prompt": positive_prompt,
                    "negative_prompt": negative_prompt,
                    "site": site,
                    "scraped_at": scraped_at,
                    "scrape_runtime_seconds": scrape_runtime_seconds,
                    "preprocessing_runtime_sec": preprocessing_runtime_sec,
                    "ranking_reasoning": ranking_reasoning,
                    "ranking_scores": ranking_scores,
                    "images": source_images,
                    "mesh_generation_runtime_sec": float(total_sec),
                    "status": "fail",
                    "error": str(e),
                })

    finally:
        driver.quit()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meshes.json").write_text(
        json.dumps({"meshes": meshes_out}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "runtimes.json").write_text(
        json.dumps({"runtimes": runtimes_out}, indent=2),
        encoding="utf-8",
    )

    print(f"\nWrote: {(out_dir / 'meshes.json').resolve()}")
    print(f"Wrote: {(out_dir / 'runtimes.json').resolve()}")
    print(f"GLBs:  {glb_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())