#!/usr/bin/env python3
# count_descriptions_and_sites.py

from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MESHES_JSON = Path("mesh_outputs/meshes.json")  # change if needed


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_value(v: Any, default: str = "UNKNOWN") -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def main() -> None:
    data = load_json(MESHES_JSON)

    if isinstance(data, dict) and "meshes" in data:
        meshes = data["meshes"]
    elif isinstance(data, list):
        meshes = data
    else:
        raise ValueError("meshes.json must be a list or a dict with a 'meshes' key")

    # Total count for each description
    description_counts = Counter()

    # Nested counts: description -> site -> count
    description_site_counts = defaultdict(Counter)

    for item in meshes:
        if not isinstance(item, dict):
            continue

        desc = normalize_value(item.get("description"))
        site = normalize_value(item.get("site"))

        description_counts[desc] += 1
        description_site_counts[desc][site] += 1

    print("Counts by description:")
    for desc, count in sorted(description_counts.items()):
        print(f"  {desc}: {count}")

    print("\nCounts by description and website:")
    for desc in sorted(description_site_counts.keys()):
        total = description_counts[desc]
        print(f"\n{desc}: {total}")
        for site, count in sorted(description_site_counts[desc].items()):
            print(f"  - {site}: {count}")


if __name__ == "__main__":
    main()