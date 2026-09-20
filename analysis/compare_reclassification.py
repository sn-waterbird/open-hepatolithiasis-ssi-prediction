"""Compare final and full models using paired held-out probabilities.

Compute continuous NRI and IDI from original-sample estimates, with paired
bootstrap percentile intervals and normal-approximation two-sided p values.
Require matching IDs and outcomes in both predictions and the fixed split."""

import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np


def estimates(y, old, new):
    y = np.asarray(y)
    old = np.asarray(old, dtype=float)
    new = np.asarray(new, dtype=float)
    if y.ndim != 1 or old.shape != y.shape or new.shape != y.shape:
        raise ValueError("Expected equal-length one-dimensional inputs")
    if not np.isin(y, [0, 1]).all() or set(y.tolist()) != {0, 1}:
        raise ValueError("Both binary outcome groups are required")
    if (
        not np.isfinite(old).all()
        or not np.isfinite(new).all()
        or ((old < 0) | (old > 1) | (new < 0) | (new > 1)).any()
    ):
        raise ValueError("Probabilities must be finite and in [0,1]")
    d = new - old
    event = y == 1
    return np.array(
        [
            np.sign(d)[event].mean() - np.sign(d)[~event].mean(),
            d[event].mean() - d[~event].mean(),
        ]
    )


def compare(y, old, new, n_bootstrap=1000, seed=42):
    y = np.asarray(y)
    old = np.asarray(old, dtype=float)
    new = np.asarray(new, dtype=float)
    point = estimates(y, old, new)
    if n_bootstrap < 2:
        raise ValueError("At least two bootstrap draws are required")
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_bootstrap):
        ix = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) == 2:
            boot.append(estimates(y[ix], old[ix], new[ix]))
    if len(boot) < 2:
        raise ValueError("Insufficient valid bootstrap draws")
    boot = np.asarray(boot)
    result = {
        "comparison": "Final retuned Top-4 uncalibrated vs full-64 uncalibrated CatBoost+S0",
        "direction": "positive favors reduced model",
        "n": len(y),
        "events": int(y.sum()),
        "bootstrap_requested": n_bootstrap,
        "bootstrap_n": len(boot),
        "bootstrap_seed": seed,
        "numpy_version": np.__version__,
        "p_method": "two-sided normal approximation, original-sample estimate / bootstrap SE",
    }
    for j, key in enumerate(["NRI", "IDI"]):
        se = float(boot[:, j].std(ddof=1))
        z = float(point[j] / se) if se > 0 else None
        result[key] = {
            "estimate": float(point[j]),
            "SE": se,
            "CI": np.percentile(boot[:, j], [2.5, 97.5]).tolist(),
            "Z": z,
            "p": math.erfc(abs(z) / math.sqrt(2)) if z is not None else None,
        }
    d = new - old
    event = y == 1
    result["counts"] = {
        name: {
            "n": int(mask.sum()),
            "up": int((d[mask] > 0).sum()),
            "down": int((d[mask] < 0).sum()),
            "ties": int((d[mask] == 0).sum()),
            "mean_delta": float(d[mask].mean()),
        }
        for name, mask in [("SSI", event), ("nonSSI", ~event)]
    }
    return result


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pair(final, full, split):
    val = [r for r in split if r["split"] == "VAL"]

    def unique(rows):
        keys = [r["ID"] for r in rows]
        if any((not k for k in keys)) or len(keys) != len(set(keys)):
            raise ValueError("Missing or duplicate IDs")
        return {r["ID"]: r for r in rows}

    fm, sm, rm = (unique(full), unique(val), unique(final))
    if set(fm) != set(sm) or set(sm) != set(rm):
        raise ValueError("VAL membership mismatch")
    rows = []
    for r in final:
        key = r["ID"]
        y = int(r["y_true"])
        if y != int(fm[key]["y_true"]) or y != int(sm[key]["Infection"]):
            raise ValueError("Outcome mismatch across paired inputs")
        rows.append(
            {
                "ID": key,
                "y_true": y,
                "p_full": float(fm[key]["pred_prob"]),
                "p_reduced": float(r["prob_uncalib"]),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Restricted analysis root",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/outputs/reclassification"),
        help="Controlled output directory",
    )
    args = parser.parse_args()
    rows = pair(
        read(args.root / "outputs/final_model/Final_Predictions_CatBoost+S0_Top4.csv"),
        read(args.root / "outputs/full_models/CatBoost+S0/indep_predictions.csv"),
        read(args.root / "data/split_folds.csv"),
    )
    if len(rows) != 70 or sum((r["y_true"] for r in rows)) != 11:
        raise ValueError("Study reproduction expects 70 VAL patients and 11 SSI events")
    result = compare(
        [r["y_true"] for r in rows],
        [r["p_full"] for r in rows],
        [r["p_reduced"] for r in rows],
    )
    result["parameters"] = {
        "reduced": {"depth": 3, "learning_rate": 0.03, "iterations": 300},
        "full": {"depth": 3, "learning_rate": 0.1, "iterations": 300},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "final_model_NRI_IDI_results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (args.out / "final_model_reclassification_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (args.out / "final_model_NRI_IDI_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(
            ["Metric", "Estimate", "SE", "Z", "p_value", "95%_CI_lower", "95%_CI_upper"]
        )
        for key in ["NRI", "IDI"]:
            r = result[key]
            w.writerow([key, r["estimate"], r["SE"], r["Z"], r["p"], *r["CI"]])
    print(
        json.dumps(
            {key: result[key] for key in ["n", "events", "NRI", "IDI"]}, indent=2
        )
    )
    print(
        "Patient-level paired outputs were written to the controlled output directory; do not publish them."
    )


if __name__ == "__main__":
    main()
