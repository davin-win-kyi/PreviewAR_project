import json
from pathlib import Path
from statistics import mean

BASE_DIR = Path("paper_benchmarks")
OUTPUT_JSON = BASE_DIR / "benchmark_summary.json"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def average_runtime_from_runtime_file(data: dict) -> float | None:
    """
    Expects a structure like:
    {
      "runtimes": [
        {"runtime_sec": 12.3, "status": "ok"},
        ...
      ]
    }

    Averages runtime_sec over entries with status == "ok".
    Falls back to all entries with runtime_sec if status is missing.
    """
    items = data.get("runtimes", [])
    vals = []

    for item in items:
        if not isinstance(item, dict):
            continue
        if "runtime_sec" not in item:
            continue

        status = item.get("status")
        if status is None or status == "ok":
            try:
                vals.append(float(item["runtime_sec"]))
            except (TypeError, ValueError):
                pass

    return mean(vals) if vals else None


def extract_rrmse_from_scale_eval(data: dict) -> tuple[float | None, float | None]:
    """
    Prefers summary.mean_relative_rmse_normalized.
    Also extracts summary.mean_relative_rmse_percent if present.
    """
    summary = data.get("summary", {})
    normalized = summary.get("mean_relative_rmse_normalized")
    percent = summary.get("mean_relative_rmse_percent")

    try:
        normalized = float(normalized) if normalized is not None else None
    except (TypeError, ValueError):
        normalized = None

    try:
        percent = float(percent) if percent is not None else None
    except (TypeError, ValueError):
        percent = None

    return normalized, percent


def main():
    run_dirs = sorted(
        [
            p for p in BASE_DIR.iterdir()
            if p.is_dir() and p.name.startswith("run#")
        ],
        key=lambda p: p.name
    )

    if not run_dirs:
        raise FileNotFoundError(f"No run folders found in {BASE_DIR}")

    per_run = []
    preprocessing_means = []
    runtimes_means = []
    rrmse_normalized_means = []
    rrmse_percent_means = []

    for run_dir in run_dirs:
        preprocessing_path = run_dir / "preprocessing_runtime.json"
        runtimes_path = run_dir / "runtimes.json"
        scale_eval_path = run_dir / "scale_eval.json"

        if not preprocessing_path.exists():
            print(f"[WARN] Missing: {preprocessing_path}")
            continue
        if not runtimes_path.exists():
            print(f"[WARN] Missing: {runtimes_path}")
            continue
        if not scale_eval_path.exists():
            print(f"[WARN] Missing: {scale_eval_path}")
            continue

        preprocessing_data = load_json(preprocessing_path)
        runtimes_data = load_json(runtimes_path)
        scale_eval_data = load_json(scale_eval_path)

        preprocessing_avg = average_runtime_from_runtime_file(preprocessing_data)
        runtimes_avg = average_runtime_from_runtime_file(runtimes_data)
        rrmse_normalized, rrmse_percent = extract_rrmse_from_scale_eval(scale_eval_data)

        if preprocessing_avg is not None:
            preprocessing_means.append(preprocessing_avg)
        if runtimes_avg is not None:
            runtimes_means.append(runtimes_avg)
        if rrmse_normalized is not None:
            rrmse_normalized_means.append(rrmse_normalized)
        if rrmse_percent is not None:
            rrmse_percent_means.append(rrmse_percent)

        per_run.append({
            "run_folder": run_dir.name,
            "preprocessing_runtime_avg_sec": preprocessing_avg,
            "runtimes_avg_sec": runtimes_avg,
            "scale_eval_mean_rrmse_normalized": rrmse_normalized,
            "scale_eval_mean_rrmse_percent": rrmse_percent,
        })

    output = {
        "base_dir": str(BASE_DIR),
        "num_runs_found": len(run_dirs),
        "num_runs_used": len(per_run),
        "per_run": per_run,
        "overall_averages": {
            "preprocessing_runtime_avg_sec_across_runs": (
                mean(preprocessing_means) if preprocessing_means else None
            ),
            "runtimes_avg_sec_across_runs": (
                mean(runtimes_means) if runtimes_means else None
            ),
            "scale_eval_mean_rrmse_normalized_across_runs": (
                mean(rrmse_normalized_means) if rrmse_normalized_means else None
            ),
            "scale_eval_mean_rrmse_percent_across_runs": (
                mean(rrmse_percent_means) if rrmse_percent_means else None
            ),
        }
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[OK] Wrote summary to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()