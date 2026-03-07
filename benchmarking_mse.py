#!/usr/bin/env python3
# eval_glb_scaling.py

from __future__ import annotations
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

# These are still used for GT unit conversion, but NOT for parsing Blender extents anymore.
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

def parse_gt_dims_in_inches(item: Dict[str, Any]) -> Tuple[float, float, float]:
    units_raw = item.get("units") or item.get("unit") or item.get("dimension_units") or ""
    units = normalize_unit(str(units_raw))

    dim_obj = item.get("dimensions") or item.get("dimension") or item.get("measurements") or item.get("measurement") or {}
    if isinstance(dim_obj, dict):
        inner_units = dim_obj.get("units") or dim_obj.get("unit")
        if inner_units:
            units = normalize_unit(str(inner_units))

    mult = inches_per_unit(units)
    if mult is None:
        raise ValueError(f"Unsupported units '{units_raw}' (normalized='{units}'). Expected in/cm/mm/m.")

    triple = parse_dims_from_dimensions_obj(dim_obj)
    if triple is None:
        L = to_float(get_first_present(item, ["length", "len", "l"]))
        W = to_float(get_first_present(item, ["width", "wid", "w"]))
        H = to_float(get_first_present(item, ["height", "ht", "h", "depth", "thickness"]))
        if L is not None and W is not None and H is not None:
            triple = (L, W, H)

    if triple is None:
        raise ValueError("Could not parse 3 ground-truth dimensions from metadata.")

    return (triple[0] * mult, triple[1] * mult, triple[2] * mult)

# ----------------------------
# Scaling metrics helpers
# ----------------------------
def best_fit_uniform_scale(glb_dims: Tuple[float, float, float],
                           gt_dims: Tuple[float, float, float]) -> float:
    d0, d1, d2 = glb_dims
    g0, g1, g2 = gt_dims
    num = g0*d0 + g1*d1 + g2*d2
    den = d0*d0 + d1*d1 + d2*d2
    return (num / den) if den > 0 else 0.0

def mse(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) / 3.0

def sorted_dims(d: Tuple[float, float, float]) -> Tuple[float, float, float]:
    a = sorted(d, reverse=True)
    return (a[0], a[1], a[2])

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

    # Expect one JSON line. Take the last JSON-ish line.
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
    """
    Your current tools/blender_bbox.py returns:
      - glb_absolute as dict {"a","b","c"} OR list/tuple [..,..,..]
    We accept both.
    """
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
    metadata_path: Path = Path("outputs/metadata/dimensions.json"),
    index_out_path: Path = Path("outputs/metadata/scale_eval.json"),
    blender_bin: str = "blender",
    bbox_script: Path = Path("utils/blender_bbox.py"),  # IMPORTANT: match your actual file
    use_sorted_matching: bool = True,
    resolve_relative_glb_paths: bool = True,
    debug_print_parsed_dims: bool = True,
) -> Dict[str, Any]:

    data: Dict[str, Any] = load_json(metadata_path)

    # Print paths & existence
    print("\n===== GLBs referenced in metadata =====")
    for key, item in data.items():
        glb_raw = item.get("path_to_glb") or item.get("glb") or item.get("glb_path")
        if not glb_raw:
            print(f"{key}: MISSING path_to_glb")
            continue

        p = Path(glb_raw)
        if resolve_relative_glb_paths and not p.is_absolute():
            p = p
        print(f"{key}: {p}  |  exists={p.exists()}")
    print("========================================\n")

    results: Dict[str, Any] = {}
    mse_abs_list: List[float] = []
    mse_rel_list: List[float] = []
    rel_rmse_norm_list: List[float] = []
    count_ok = 0
    count_fail = 0

    for key, item in data.items():
        # 1) Parse GT dims
        try:
            gt_tuple_raw = parse_gt_dims_in_inches(item)
        except Exception as e:
            results[key] = {"error": f"GT parse error: {e}"}
            count_fail += 1
            continue

        # 2) Resolve GLB path
        glb_raw = item.get("path_to_glb") or item.get("glb") or item.get("glb_path")
        if not glb_raw:
            results[key] = {"error": "Missing path_to_glb/glb/glb_path."}
            count_fail += 1
            continue

        glb_path = Path(glb_raw)
        if resolve_relative_glb_paths and not glb_path.is_absolute():
            glb_path = glb_path

        if not glb_path.exists():
            results[key] = {"error": f"GLB not found: {glb_path}", "path_to_glb": str(glb_path)}
            count_fail += 1
            continue

        # 3) Run Blender bbox + parse glb dims from output
        try:
            bbox = call_blender_bbox(blender_bin, bbox_script, glb_path)
            glb_tuple_raw = glb_dims_in_inches_from_blender_output(bbox)
        except Exception as e:
            results[key] = {"error": f"Blender bbox error: {e}", "path_to_glb": str(glb_path)}
            count_fail += 1
            continue

        # 4) Match dims (sorted or direct)
        gt_tuple = gt_tuple_raw
        glb_tuple = glb_tuple_raw

        if use_sorted_matching:
            gt_m = sorted_dims(gt_tuple)
            glb_m = sorted_dims(glb_tuple)
        else:
            gt_m = gt_tuple
            glb_m = glb_tuple

        # 5) Metrics
        mse_abs = mse(glb_m, gt_m)
        s_star = best_fit_uniform_scale(glb_m, gt_m)
        glb_rel = (s_star * glb_m[0], s_star * glb_m[1], s_star * glb_m[2])
        mse_rel = mse(glb_rel, gt_m)

        rmse_rel = math.sqrt(mse_rel)
        mean_gt = (gt_m[0] + gt_m[1] + gt_m[2]) / 3.0
        rel_rmse_norm = (rmse_rel / mean_gt) if mean_gt > 0 else None

        if debug_print_parsed_dims:
            print(f"[{key}] GT(in)={gt_tuple_raw} GLB(in)={glb_tuple_raw} s*={s_star:.4f} relRMSE%={(rel_rmse_norm*100 if rel_rmse_norm is not None else None)}")

        entry = {
            "url": item.get("url"),
            "path_to_glb": str(glb_path),

            # Required map structure
            "glb_absolute": {"a": glb_m[0], "b": glb_m[1], "c": glb_m[2]},
            "glb_relative": {"a": glb_rel[0], "b": glb_rel[1], "c": glb_rel[2], "scale": s_star},
            "actual_absolute": {"a": gt_m[0], "b": gt_m[1], "c": gt_m[2]},
            "actual_relative": {"a": gt_m[0], "b": gt_m[1], "c": gt_m[2]},

            "metrics": {
                "mse_absolute_in2": mse_abs,
                "mse_relative_in2": mse_rel,
                "rmse_relative_in": rmse_rel,
                "relative_rmse_normalized": rel_rmse_norm,
                "relative_rmse_percent": (rel_rmse_norm * 100.0) if rel_rmse_norm is not None else None,
            },

            "matching": "sorted_desc" if use_sorted_matching else "xyz_direct",

            # Optional debug info
            "debug": {
                "raw_gt_dims_in": {"a": gt_tuple_raw[0], "b": gt_tuple_raw[1], "c": gt_tuple_raw[2]},
                "raw_glb_dims_in": {"a": glb_tuple_raw[0], "b": glb_tuple_raw[1], "c": glb_tuple_raw[2]},
                "blender_json_keys": list(bbox.keys()),
            },
        }

        results[key] = entry
        mse_abs_list.append(mse_abs)
        mse_rel_list.append(mse_rel)
        if rel_rmse_norm is not None:
            rel_rmse_norm_list.append(rel_rmse_norm)
        count_ok += 1

    summary = {
        "n_ok": count_ok,
        "n_fail": count_fail,

        "mean_mse_absolute_in2": (sum(mse_abs_list)/len(mse_abs_list)) if mse_abs_list else None,
        "rmse_absolute_in": math.sqrt(sum(mse_abs_list)/len(mse_abs_list)) if mse_abs_list else None,

        "mean_mse_relative_in2": (sum(mse_rel_list)/len(mse_rel_list)) if mse_rel_list else None,
        "rmse_relative_in": math.sqrt(sum(mse_rel_list)/len(mse_rel_list)) if mse_rel_list else None,

        "mean_relative_rmse_normalized": (sum(rel_rmse_norm_list)/len(rel_rmse_norm_list)) if rel_rmse_norm_list else None,
        "mean_relative_rmse_percent": (100.0 * sum(rel_rmse_norm_list)/len(rel_rmse_norm_list)) if rel_rmse_norm_list else None,
    }

    out = {"summary": summary, "items": results}
    index_out_path.parent.mkdir(parents=True, exist_ok=True)
    index_out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote report: {index_out_path}")
    return out

if __name__ == "__main__":
    evaluate()