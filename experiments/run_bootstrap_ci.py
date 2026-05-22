"""
Experiment P1: Bootstrap CI for Wget (and optionally Trace) vs MAGIC.

Uses raw predictions from labelfree_experiments to bootstrap.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_curve, roc_auc_score


def metrics_from_scores(y, scores):
    if len(np.unique(y)) < 2:
        return None
    auc = roc_auc_score(y, scores)
    prec, rec, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best = int(np.argmax(f1))
    th = thresholds[min(best, len(thresholds) - 1)]
    pred = (scores >= th).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return {
        "auc": auc,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="labelfree_experiments")
    parser.add_argument("--out-dir", default="labelfree_experiments")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    rng = np.random.default_rng(args.bootstrap_seed)

    # MAGIC cited values
    cited = {
        "wget": {"magic_f1": 0.9388, "magic_fpr": 0.0400},
        "trace": {"magic_f1": 0.9910, "magic_fpr": 0.0014},
    }

    rows = []
    for ds in ["wget", "trace"]:
        for seed in seeds:
            # Try to load predictions
            pred_file = data_dir / f"{ds}_seed{seed}_predictions.npz"
            if not pred_file.exists():
                print(f"  [skip] {pred_file} not found")
                continue

            data = np.load(pred_file)
            # Use full test set scores for bootstrap
            if "scores_full" in data:
                scores = data["scores_full"]
                y = data["y_test_full"]
            else:
                scores = data["scores"]
                y = data["y_test"]

            print(f"  {ds} seed={seed}: {len(y)} samples ({y.sum():.0f} attack)")

            # Non-paired bootstrap
            for b in range(args.bootstrap_iters):
                idx = rng.integers(0, len(y), len(y))
                m = metrics_from_scores(y[idx], scores[idx])
                if m is None:
                    continue
                rows.append({
                    "dataset": ds,
                    "seed": seed,
                    "bootstrap": b,
                    "f1_minus_magic": m["f1"] - cited[ds]["magic_f1"],
                    "fpr_minus_magic": m["fpr"] - cited[ds]["magic_fpr"],
                    **m,
                })

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "bootstrap_ci_raw.csv", index=False)

    # Summary: per dataset × seed → CI
    summary_rows = []
    for (ds, seed), group in raw.groupby(["dataset", "seed"]):
        for col in ["f1", "fpr", "auc", "f1_minus_magic", "fpr_minus_magic"]:
            summary_rows.append({
                "dataset": ds,
                "seed": seed,
                "metric": col,
                "mean": group[col].mean(),
                "ci_low": group[col].quantile(0.025),
                "ci_high": group[col].quantile(0.975),
            })

    # Also compute aggregate across seeds
    for ds in ["wget", "trace"]:
        ds_group = raw[raw["dataset"] == ds]
        if ds_group.empty:
            continue
        for col in ["f1", "fpr", "auc", "f1_minus_magic", "fpr_minus_magic"]:
            summary_rows.append({
                "dataset": ds,
                "seed": "all",
                "metric": col,
                "mean": ds_group[col].mean(),
                "ci_low": ds_group[col].quantile(0.025),
                "ci_high": ds_group[col].quantile(0.975),
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "bootstrap_ci_summary.csv", index=False)

    # Print key results
    print("\n" + "=" * 60)
    print("Bootstrap CI Summary")
    print("=" * 60)
    agg = summary[summary["seed"] == "all"]
    for _, r in agg.iterrows():
        print(f"  {r['dataset']:12s} {r['metric']:20s}: {r['mean']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")

    # Key question: does CI for f1_minus_magic include 0?
    print("\n--- Significance Test: Does NIMBLE F1 - MAGIC F1 CI include 0? ---")
    for ds in ["wget", "trace"]:
        row = agg[(agg["dataset"] == ds) & (agg["metric"] == "f1_minus_magic")]
        if row.empty:
            continue
        r = row.iloc[0]
        includes_zero = r["ci_low"] <= 0 <= r["ci_high"]
        print(f"  {ds}: [{r['ci_low']:.4f}, {r['ci_high']:.4f}] → {'INCLUDES 0 (not significant)' if includes_zero else 'EXCLUDES 0 (significant)'}")


if __name__ == "__main__":
    main()
