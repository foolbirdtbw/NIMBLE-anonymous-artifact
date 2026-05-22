"""
Trace BQTS precision sweep.

This script reuses cached Trace embeddings and evaluates benign-quantile
threshold selection (BQTS) under stricter benign alert budgets. It does not
use attack labels to choose thresholds; labels are used only for reporting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(np.int8)
    tp = int(((y_true == 1) & (pred == 1)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    tn = int(((y_true == 0) & (pred == 0)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0
    return {
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def label_tuned_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(np.argmax(f1))
    threshold = thresholds[min(best, len(thresholds) - 1)]
    out = metrics_at_threshold(y_true, scores, float(threshold))
    out["threshold_method"] = "label-tuned upper bound"
    out["quantile"] = np.nan
    return out


def parse_csv_numbers(text: str, cast=float):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(cast(item))
    return values


def load_embeddings(cache_dir: Path, seed: int):
    path = cache_dir / f"trace_checkpoint-trace-seed{seed}_original_embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    return data["x_train"], data["x_test"], data["y_test"]


def evaluate_setting(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    n_estimators: int,
    max_samples,
    quantiles: list[float],
    val_size: int,
    fit_size: int | None,
) -> list[dict]:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    rng = np.random.default_rng(seed)
    n_val = min(val_size, x_train_s.shape[0] // 3)
    val_idx = rng.choice(x_train_s.shape[0], n_val, replace=False)
    train_mask = np.ones(x_train_s.shape[0], dtype=bool)
    train_mask[val_idx] = False
    fit_pool = np.flatnonzero(train_mask)
    if fit_size is not None and fit_size < fit_pool.size:
        fit_idx = rng.choice(fit_pool, fit_size, replace=False)
    else:
        fit_idx = fit_pool

    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",
        max_samples=max_samples,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(x_train_s[fit_idx])

    val_scores = -clf.decision_function(x_train_s[val_idx])
    test_scores = -clf.decision_function(x_test_s)

    rows = []
    lt = label_tuned_metrics(y_test, test_scores)
    rows.append(lt)
    for q in quantiles:
        threshold = float(np.quantile(val_scores, q))
        row = metrics_at_threshold(y_test, test_scores, threshold)
        row["threshold_method"] = f"benign-q{q:g}"
        row["quantile"] = q
        rows.append(row)
    for row in rows:
        row.update(
            {
                "seed": seed,
                "n_estimators": n_estimators,
                "max_samples": str(max_samples),
                "val_size": n_val,
                "fit_size": len(fit_idx),
            }
        )
    return rows


def summarize(raw: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    metrics = ["auc", "precision", "recall", "f1", "fpr", "tp", "fp", "tn", "fn"]
    group_cols = ["threshold_method", "n_estimators", "max_samples"]
    summary = raw.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).rstrip("_") for col in summary.columns]
    summary = summary.sort_values(
        ["precision_mean", "f1_mean", "recall_mean"], ascending=[False, False, False]
    )
    summary.to_csv(out_dir / "trace_bqts_precision_summary.csv", index=False)

    table_rows = []
    for _, r in summary.iterrows():
        table_rows.append(
            {
                "Threshold": r["threshold_method"],
                "iForest": f"{int(r['n_estimators'])}, max_samples={r['max_samples']}",
                "AUC": f"{r['auc_mean']*100:.2f} $\\pm$ {r['auc_std']*100:.2f}",
                "Precision": f"{r['precision_mean']*100:.2f} $\\pm$ {r['precision_std']*100:.2f}",
                "Recall": f"{r['recall_mean']*100:.2f} $\\pm$ {r['recall_std']*100:.2f}",
                "F1": f"{r['f1_mean']*100:.2f} $\\pm$ {r['f1_std']*100:.2f}",
                "FPR": f"{r['fpr_mean']*100:.2f} $\\pm$ {r['fpr_std']*100:.2f}",
            }
        )
    pd.DataFrame(table_rows).to_latex(
        out_dir / "trace_bqts_precision_table.tex", index=False, escape=False
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        default="reviewer_required_experiments_trace_multiseed_opt/embedding_cache",
    )
    parser.add_argument("--out-dir", default="trace_bqts_precision_sweep")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--n-estimators", default="200,500")
    parser.add_argument("--max-samples", default="1.0,0.5,auto")
    parser.add_argument("--quantiles", default="0.99,0.995,0.9975,0.999")
    parser.add_argument("--val-size", type=int, default=100000)
    parser.add_argument(
        "--fit-size",
        type=int,
        default=500000,
        help="Subsample benign training embeddings for iForest; <=0 uses all non-validation embeddings.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_csv_numbers(args.seeds, int)
    n_estimators_values = parse_csv_numbers(args.n_estimators, int)
    max_samples_values = []
    for item in args.max_samples.split(","):
        item = item.strip()
        max_samples_values.append("auto" if item == "auto" else float(item))
    quantiles = parse_csv_numbers(args.quantiles, float)
    fit_size = None if args.fit_size <= 0 else args.fit_size

    raw_path = out_dir / "trace_bqts_precision_raw.csv"
    rows = []
    done = set()
    if raw_path.exists():
        old = pd.read_csv(raw_path)
        rows = old.to_dict("records")
        done = {
            (int(r["seed"]), int(r["n_estimators"]), str(r["max_samples"]))
            for r in rows
        }
        print(f"Resuming from {raw_path} with {len(rows)} rows")

    for seed in seeds:
        print(f"\nLoading Trace embeddings for seed={seed}")
        x_train, x_test, y_test = load_embeddings(cache_dir, seed)
        for n_estimators in n_estimators_values:
            for max_samples in max_samples_values:
                key = (seed, n_estimators, str(max_samples))
                if key in done:
                    print(f"[skip] seed={seed} n={n_estimators} max_samples={max_samples}")
                    continue
                print(f"[run] seed={seed} n={n_estimators} max_samples={max_samples}")
                setting_rows = evaluate_setting(
                    x_train,
                    x_test,
                    y_test,
                    seed,
                    n_estimators,
                    max_samples,
                    quantiles,
                    args.val_size,
                    fit_size,
                )
                rows.extend(setting_rows)
                pd.DataFrame(rows).to_csv(raw_path, index=False)
                print(f"  saved {len(rows)} rows")

    raw = pd.DataFrame(rows)
    raw.to_csv(raw_path, index=False)
    summary = summarize(raw, out_dir)
    print("\nTop settings by precision:")
    cols = [
        "threshold_method",
        "n_estimators",
        "max_samples",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "fpr_mean",
        "fpr_std",
    ]
    print(summary[cols].head(20).to_string(index=False))
    print(f"\nDone: {out_dir}")


if __name__ == "__main__":
    main()
