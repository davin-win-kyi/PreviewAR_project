#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_meshes(raw: Any) -> List[Dict]:
    if isinstance(raw, dict) and "meshes" in raw:
        return raw["meshes"]
    if isinstance(raw, list):
        return raw
    raise ValueError("Expected JSON to be list or contain 'meshes'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-json",
        default="mesh_outputs/meshes_minimal.json",
        help="Path to meshes_minimal.json"
    )
    ap.add_argument(
        "--sort",
        choices=["count", "name"],
        default="count",
        help="Sort output by count or name"
    )
    args = ap.parse_args()

    data = load_json(Path(args.input_json))
    meshes = extract_meshes(data)

    counter = Counter()

    for item in meshes:
        obj_type = item.get("object_type", "unknown")
        counter[obj_type] += 1

    # Sorting
    if args.sort == "count":
        items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    else:
        items = sorted(counter.items(), key=lambda x: x[0])

    total = sum(counter.values())

    print("\n===== Object Type Counts =====")
    for obj_type, count in items:
        percent = (count / total) * 100 if total > 0 else 0
        print(f"{obj_type:20s} : {count:5d}  ({percent:6.2f}%)")

    print("================================")
    print(f"Total objects: {total}\n")


if __name__ == "__main__":
    main()