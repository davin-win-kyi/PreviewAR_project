#!/usr/bin/env python3
# eval_glb_scaling.py

from __future__ import annotations
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

INCHES_PER_M = 39.37007874015748
INCHES_PER_CM = 0.3937007874015748
INCHES_PER_MM = 0.03937007874015748


# ----------------------------
# IO helpers
# ----------------------------
def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))

def to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str):
            s = x.strip()
            if s == "":
                return None
            s = s.replace(",", "")
            return float(s)
        return float(x)
    except Exception:
        return None


# ----------------------------
# Unit + GT dimension parsing
# ----------------------------
def normalize_unit(u: str) -> str:
    u = (u or "").strip().lower().replace(".", "")
    aliases = {
        "in": "in", "inch": "in", "inches": "in",
        "cm": "cm", "centimeter": "cm", "centimeters": "cm",
        "mm": "mm", "millimeter": "mm", "millimeters": "mm",
        "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    }
    return aliases.get(u, u)

def inches_per_unit(u: str) -> Optional[float]:
    u = normalize_unit(u)
    if u == "in":
        return 1.0
    if u == "cm":
        return INCHES_PER_CM
    if u == "mm":
        return INCHES_PER_MM
    if u == "m":
        return INCHES_PER_M
    return None

def extract_three_numbers_from_string(s: str) -> Optional[Tuple[float, float, float]]:
    if not isinstance(s, str):
        return None
    s2 = s.lower().replace("×", "x")
    nums = re.findall(r"[-+]?\d*\.?\d+", s2)
    if len(nums) < 3:
        return None
    return (float(nums[0]), float(nums[1]), float(nums[2]))

def get_first_present(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def parse_dims_from_dimensions_obj(dim_obj: Any) -> Optional[Tuple[float, float, float]]:
    if dim_obj is None:
        return None

    if isinstance(dim_obj, str):
        return extract_three_numbers_from_string(dim_obj)

    if isinstance(dim_obj, (list, tuple)) and len(dim_obj) >= 3:
        a, b, c = to_float(dim_obj[0]), to_float(dim_obj[1]), to_float(dim_obj[2])
        if a is not None and b is not None and c is not None:
            return (a, b, c)

    if not isinstance(dim_obj, dict):
        return None

    L = to_float(get_first_present(dim_obj, ["length", "len", "l", "a"]))
    W = to_float(get_first_present(dim_obj, ["width", "wid", "w", "b"]))
    H = to_float(get_first_present(dim_obj, ["height", "ht", "h", "depth", "thickness", "c"]))

    if L is None or W is None or H is None:
        x = to_float(get_first_present(dim_obj, ["x", "X"]))
        y = to_float(get_first_present(dim_obj, ["y", "Y"]))
        z = to_float(get_first_present(dim_obj, ["z", "Z"]))
        if x is not None and y is not None and z is not None:
            return (x, y, z)

    if L is not None and W is not None and H is not None:
        return (L, W, H)

    for v in dim_obj.values():
        if isinstance(v, str):
            triple = extract_three_numbers_from_string(v)
            if triple is not None:
                return triple

    return None

def parse_dims_from_text_description(text: str) -> Optional[Tuple[float, float, float]]:
    """
    Try to extract named dimensions from product prose, e.g.
    'measures 4.5” tall, 12” deep and 22” wide'
    Returns (length, width, height) in inches.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    s = text.lower()
    s = s.replace("″", '"').replace("”", '"').replace("“", '"').replace("’", "'")

    number = r"(\d+(?:\.\d+)?)"
    unit = r'(?:\s*(?:inches|inch|in|"))?'

    patterns = {
        "height": [
            rf"{number}{unit}\s*(?:tall|high|height)",
            rf"(?:height)\s*(?:is|:)?\s*{number}{unit}",
        ],
        "width": [
            rf"{number}{unit}\s*(?:wide|width)",
            rf"(?:width)\s*(?:is|:)?\s*{number}{unit}",
        ],
        "depth": [
            rf"{number}{unit}\s*(?:deep|depth)",
            rf"(?:depth)\s*(?:is|:)?\s*{number}{unit}",
        ],
        "length": [
            rf"{number}{unit}\s*(?:long|length)",
            rf"(?:length)\s*(?:is|:)?\s*{number}{unit}",
        ],
    }

    found: Dict[str, float] = {}

    for key, pats in patterns.items():
        for pat in pats:
            m = re.search(pat, s)
            if m:
                found[key] = float(m.group(1))
                break

    if "length" in found and "width" in found and "height" in found:
        return (found["length"], found["width"], found["height"])

    if "depth" in found and "width" in found and "height" in found:
        return (found["depth"], found["width"], found["height"])

    triple = extract_three_numbers_from_string(s)
    if triple is not None:
        return triple

    return None

def parse_gt_dims_in_inches(item: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Read ground-truth dimensions from item["standardized_dimensions"].

    Expected examples:
      {"l": 12.0, "w": 30.0, "h": 4.5}
      {"length": 12.0, "width": 30.0, "height": 4.5}

    Assumes these values are already in inches.
    """
    dim_obj = item.get("standardized_dimensions")
    if dim_obj is None:
        raise ValueError("Missing standardized_dimensions in mesh item.")

    triple = parse_dims_from_dimensions_obj(dim_obj)
    if triple is None:
        raise ValueError("Could not parse 3 dimensions from standardized_dimensions.")

    return triple


# ----------------------------
# Metric helpers
# ----------------------------
def mse(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2) / 3.0

def rmse(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt(mse(a, b))

def sorted_dims(d: Tuple[float, float, float]) -> Tuple[float, float, float]:
    a = sorted(d, reverse=True)
    return (a[0], a[1], a[2])

def normalize_dims_by_max(d: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Convert absolute dimensions into relative proportions by dividing by the largest dimension.
    Example:
      (20, 10, 10) -> (1.0, 0.5, 0.5)
      (2, 1, 1)    -> (1.0, 0.5, 0.5)
    """
    m = max(d)
    if m <= 0:
        raise ValueError(f"Cannot normalize non-positive dimensions: {d}")
    return (d[0] / m, d[1] / m, d[2] / m)


# ----------------------------
# Blender bbox helpers
# ----------------------------
def call_blender_bbox(blender_bin: str, bbox_script: Path, glb_path: Path) -> Dict[str, Any]:
    cmd = [blender_bin, "-b", "-P", str(bbox_script), "--", str(glb_path)]
    p = subprocess.run(cmd, capture_output=True, text=True)

    if p.returncode != 0:
        raise RuntimeError(
            f"Blender failed for {glb_path}:\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )

    lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") and ln.endswith("}"):
            try:
                return json.loads(ln)
            except Exception:
                continue

    raise RuntimeError(
        f"No JSON output from Blender for {glb_path}.\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
    )

def glb_dims_in_inches_from_blender_output(bbox: Dict[str, Any]) -> Tuple[float, float, float]:
    ga = bbox.get("glb_absolute")
    if ga is None:
        raise KeyError(f"Blender JSON missing 'glb_absolute'. Keys: {list(bbox.keys())}")

    if isinstance(ga, dict):
        return (float(ga["a"]), float(ga["b"]), float(ga["c"]))

    if isinstance(ga, (list, tuple)) and len(ga) >= 3:
        return (float(ga[0]), float(ga[1]), float(ga[2]))

    raise TypeError(f"Unsupported type for glb_absolute: {type(ga)}")


# ----------------------------
# Main evaluation
# ----------------------------
def evaluate(
    metadata_path: Path = Path("mesh_outputs/meshes.json"),
    index_out_path: Path = Path("benchmarks/scale_eval.json"),
    blender_bin: str = "blender",
    bbox_script: Path = Path("utils/blender_bbox.py"),
    use_sorted_matching: bool = True,
    resolve_relative_glb_paths: bool = True,
    debug_print_parsed_dims: bool = True,
) -> Dict[str, Any]:

    raw = load_json(metadata_path)

    if isinstance(raw, dict) and "meshes" in raw:
        items = raw["meshes"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("Expected metadata to be either a list or a dict containing a 'meshes' list.")

    print("\n===== GLBs referenced in metadata =====")
    for i, item in enumerate(items):
        key = item.get("name") or item.get("description") or f"item_{i:03d}"
        glb_raw = item.get("path") or item.get("path_to_glb") or item.get("glb") or item.get("glb_path")
        if not glb_raw:
            print(f"{key}: MISSING path")
            continue

        p = Path(glb_raw)
        print(f"{key}: {p}  |  exists={p.exists()}")
    print("========================================\n")

    results: Dict[str, Any] = {}
    mse_abs_list: List[float] = []
    mse_rel_ratio_list: List[float] = []
    rel_rmse_ratio_list: List[float] = []
    count_ok = 0
    count_fail = 0

    for i, item in enumerate(items):
        key = item.get("name") or item.get("description") or Path(item.get("path", f"item_{i:03d}")).stem

        try:
            gt_tuple_raw = parse_gt_dims_in_inches(item)
        except Exception as e:
            results[key] = {"error": f"GT parse error: {e}"}
            count_fail += 1
            continue

        glb_raw = item.get("path") or item.get("path_to_glb") or item.get("glb") or item.get("glb_path")
        if not glb_raw:
            results[key] = {"error": "Missing path/path_to_glb/glb/glb_path."}
            count_fail += 1
            continue

        glb_path = Path(glb_raw)
        if resolve_relative_glb_paths and not glb_path.is_absolute():
            glb_path = glb_path.resolve()

        if not glb_path.exists():
            results[key] = {"error": f"GLB not found: {glb_path}", "path_to_glb": str(glb_path)}
            count_fail += 1
            continue

        try:
            bbox = call_blender_bbox(blender_bin, bbox_script, glb_path)
            glb_tuple_raw = glb_dims_in_inches_from_blender_output(bbox)
        except Exception as e:
            results[key] = {"error": f"Blender bbox error: {e}", "path_to_glb": str(glb_path)}
            count_fail += 1
            continue

        gt_tuple = gt_tuple_raw
        glb_tuple = glb_tuple_raw

        if use_sorted_matching:
            gt_m = sorted_dims(gt_tuple)
            glb_m = sorted_dims(glb_tuple)
        else:
            gt_m = gt_tuple
            glb_m = glb_tuple

        # Absolute-dimension metric
        mse_abs = mse(glb_m, gt_m)

        # Pure relative/proportion metric
        gt_rel = normalize_dims_by_max(gt_m)
        glb_rel = normalize_dims_by_max(glb_m)

        mse_rel_ratio = mse(glb_rel, gt_rel)
        rmse_rel_ratio = math.sqrt(mse_rel_ratio)

        # Since both vectors are already normalized to relative proportions,
        # this is the relative-scale RRMSE
        rel_rmse_norm = rmse_rel_ratio
        rel_rmse_percent = rel_rmse_norm * 100.0

        if debug_print_parsed_dims:
            print(
                f"[{key}] "
                f"GT_abs(in)={gt_tuple_raw} "
                f"GLB_abs(in)={glb_tuple_raw} "
                f"GT_rel={gt_rel} "
                f"GLB_rel={glb_rel} "
                f"RRMSE%={rel_rmse_percent:.4f}"
            )

        entry = {
            "name": item.get("name"),
            "description": item.get("description"),
            "url": item.get("source_url") or item.get("url"),
            "path_to_glb": str(glb_path),

            "glb_absolute": {"a": glb_m[0], "b": glb_m[1], "c": glb_m[2]},
            "glb_relative": {"a": glb_rel[0], "b": glb_rel[1], "c": glb_rel[2]},

            "actual_absolute": {"a": gt_m[0], "b": gt_m[1], "c": gt_m[2]},
            "actual_relative": {"a": gt_rel[0], "b": gt_rel[1], "c": gt_rel[2]},

            "metrics": {
                "mse_absolute_in2": mse_abs,
                "rmse_absolute_in": math.sqrt(mse_abs),

                "mse_relative_ratio": mse_rel_ratio,
                "rmse_relative_ratio": rmse_rel_ratio,

                "relative_rmse_normalized": rel_rmse_norm,
                "relative_rmse_percent": rel_rmse_percent,
            },

            "matching": "sorted_desc" if use_sorted_matching else "xyz_direct",

            "debug": {
                "raw_gt_dims_in": {"a": gt_tuple_raw[0], "b": gt_tuple_raw[1], "c": gt_tuple_raw[2]},
                "raw_glb_dims_in": {"a": glb_tuple_raw[0], "b": glb_tuple_raw[1], "c": glb_tuple_raw[2]},
                "blender_json_keys": list(bbox.keys()),
                "gt_parse_source": "dimensions_or_text_description",
            },
        }

        results[key] = entry
        mse_abs_list.append(mse_abs)
        mse_rel_ratio_list.append(mse_rel_ratio)
        rel_rmse_ratio_list.append(rel_rmse_norm)
        count_ok += 1

    summary = {
        "n_ok": count_ok,
        "n_fail": count_fail,

        "mean_mse_absolute_in2": (sum(mse_abs_list) / len(mse_abs_list)) if mse_abs_list else None,
        "rmse_absolute_in": math.sqrt(sum(mse_abs_list) / len(mse_abs_list)) if mse_abs_list else None,

        "mean_mse_relative_ratio": (sum(mse_rel_ratio_list) / len(mse_rel_ratio_list)) if mse_rel_ratio_list else None,
        "rmse_relative_ratio": math.sqrt(sum(mse_rel_ratio_list) / len(mse_rel_ratio_list)) if mse_rel_ratio_list else None,

        "mean_relative_rmse_normalized": (sum(rel_rmse_ratio_list) / len(rel_rmse_ratio_list)) if rel_rmse_ratio_list else None,
        "mean_relative_rmse_percent": (100.0 * sum(rel_rmse_ratio_list) / len(rel_rmse_ratio_list)) if rel_rmse_ratio_list else None,
    }

    out = {"summary": summary, "items": results}
    index_out_path.parent.mkdir(parents=True, exist_ok=True)
    index_out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote report: {index_out_path}")
    return out


if __name__ == "__main__":
    evaluate()