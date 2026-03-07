#!/usr/bin/env python3
"""
rag_products.py

Hybrid semantic + hard max-dimension filtering.
Supports:
- build index
- query via CLI
- hardcoded test mode (max-only constraints)

Edit TEST_PROMPT and MAX_DIM_CONSTRAINTS to test quickly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL_DEFAULT = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")


# =========================================================
# EMBEDDINGS
# =========================================================
def embed_text(text: str, model: str = EMBED_MODEL_DEFAULT) -> List[float]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    r = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": model, "input": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# =========================================================
# MAX-ONLY CONSTRAINTS
# =========================================================
@dataclass
class MaxConstraint:
    max_val: Optional[float] = None

    def matches(self, x: Optional[float]) -> bool:
        # If no max specified, always pass
        if self.max_val is None:
            return True

        # If dimension unknown, reject
        if x is None:
            return False

        return x <= self.max_val


def dims_match(item_dims: Optional[Dict[str, Any]],
               constraints: Dict[str, MaxConstraint]) -> bool:
    if not constraints:
        return True

    if not isinstance(item_dims, dict):
        return False

    for dim, constraint in constraints.items():
        value = item_dims.get(dim)
        if isinstance(value, (int, float)):
            value = float(value)
        else:
            try:
                value = float(value)
            except Exception:
                value = None

        if not constraint.matches(value):
            return False

    return True


# =========================================================
# TEXT FOR EMBEDDINGS
# =========================================================
def build_embedding_text(clean_name: str, item: Dict[str, Any]) -> str:
    pname = item.get("product_name") or clean_name
    desc = item.get("high_level_description") or ""
    dims = item.get("dimensions")

    parts = [
        f"name: {pname}",
        f"description: {desc}",
    ]

    if isinstance(dims, dict):
        parts.append(
            f"dimensions_in: length={dims.get('length')} "
            f"width={dims.get('width')} height={dims.get('height')}"
        )

    return "\n".join(parts)


# =========================================================
# INDEX BUILD
# =========================================================
def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_atomic(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def build_or_update_index(data_path: Path, index_path: Path):
    data = load_json(data_path)
    index = {}

    if index_path.exists():
        try:
            index = load_json(index_path)
        except Exception:
            index = {}

    updated = 0

    for clean_name, item in data.items():
        text = build_embedding_text(clean_name, item)

        needs_update = (
            clean_name not in index
            or index[clean_name].get("text") != text
        )

        if needs_update:
            vec = embed_text(text)
            index[clean_name] = {
                "text": text,
                "embedding": vec,
                "meta": item,
            }
            updated += 1

    save_json_atomic(index_path, index)
    print(f"Index updated. {updated} items embedded.")


# =========================================================
# QUERY
# =========================================================
def query_index(index_path: Path,
                prompt: str,
                top_k: int,
                similarity_threshold: float,
                constraints: Dict[str, MaxConstraint]):

    index = load_json(index_path)

    qvec = embed_text(prompt)

    scored: List[Tuple[str, float, Dict[str, Any]]] = []

    for clean_name, entry in index.items():
        score = cosine_similarity(qvec, entry["embedding"])
        scored.append((clean_name, score, entry["meta"]))

    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for clean_name, score, meta in scored:
        if score < similarity_threshold:
            continue

        if not dims_match(meta.get("dimensions"), constraints):
            continue

        results.append({
            "clean_name": clean_name,
            "score": score,
            "product_name": meta.get("product_name"),
            "dimensions": meta.get("dimensions"),
            "path_to_glb": meta.get("path_to_glb"),
        })

        if len(results) >= top_k:
            break

    return results


def run_query(
    index_path: Path,
    *,
    prompt: str,
    top_k: int = 5,
    similarity_threshold: float = 0.30,
    max_dims: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Run a query against the index with optional max-only dimension constraints.
    max_dims example: {"height": 60.0, "width": 20.0, "length": 20.0}
    """
    constraints: Dict[str, MaxConstraint] = {}
    if max_dims:
        constraints = {k: MaxConstraint(max_val=v) for k, v in max_dims.items()}

    print("\n======== QUERY ========")
    print("Prompt:", prompt)
    print("Top K:", top_k)
    print("Similarity threshold:", similarity_threshold)
    print("Max constraints:", max_dims or {})
    print("=======================\n")

    results = query_index(
        index_path=index_path,
        prompt=prompt,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        constraints=constraints,
    )

    print(json.dumps(results, indent=2))
    return results


def extract_glb_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert query results into a JSON-friendly output that focuses on GLB paths.
    Filters out entries with missing/nonexistent GLB files.
    """
    glbs = []
    missing = []

    for r in results:
        p = r.get("path_to_glb")
        if not isinstance(p, str) or not p.strip():
            missing.append({**r, "reason": "missing path_to_glb"})
            continue

        path = Path(p)
        if not path.exists():
            missing.append({**r, "reason": "glb file not found", "path_to_glb": p})
            continue

        glbs.append({
            "clean_name": r.get("clean_name"),
            "score": r.get("score"),
            "path_to_glb": str(path),
            "dimensions": r.get("dimensions"),
            "product_name": r.get("product_name"),
        })

    return {
        "n_returned": len(results),
        "n_glbs_found": len(glbs),
        "glbs": glbs,
        "missing": missing,
    }


def main(
    *,
    data_path: Path = Path("outputs/metadata/dimensions.json"),
    index_path: Path = Path("index.json"),
    prompt: str = "lamp",
    top_k: int = 5,
    similarity_threshold: float = 0.35,
    max_dims: Optional[Dict[str, float]] = None,
    out_json: Path = Path("outputs/relevant_glbs.json"),
) -> Dict[str, Any]:
    print("\n=== RAG PRODUCTS: AUTO MODE ===")

    if not data_path.exists():
        raise FileNotFoundError(
            f"\nData file not found:\n  {data_path}\n"
            "Run your metadata pipeline first."
        )

    # Build/update index
    build_or_update_index(data_path, index_path)

    # Run query
    results = run_query(
        index_path,
        prompt=prompt,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        max_dims=max_dims,
    )

    # NEW: focus output on GLBs
    glb_report = {
        "prompt": prompt,
        "top_k": top_k,
        "similarity_threshold": similarity_threshold,
        "max_dims": max_dims or {},
        "index_path": str(index_path),
        "data_path": str(data_path),
        **extract_glb_results(results),
    }

    # Save + print
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(glb_report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== RELEVANT GLBs ===")
    print(json.dumps(glb_report, indent=2, ensure_ascii=False))
    print(f"\nWrote: {out_json}")

    print("Relevant content: ", glb_report)

    return glb_report


import subprocess
if __name__ == "__main__":
    glb_report = main(
        prompt="lamp",
        top_k=10,
        similarity_threshold=0.35,
        max_dims={"height": 30.0, "width": 15.0, "length": 15.0},
    )

    if glb_report.get("n_glbs_found", 0) > 0:
        subprocess.run([
            "blender",
            "--python",
            "utils/view_glbs.py",
            "--",
            "outputs/relevant_glbs.json",
        ], check=True)
    else:
        print("No GLBs found. Not launching Blender.")