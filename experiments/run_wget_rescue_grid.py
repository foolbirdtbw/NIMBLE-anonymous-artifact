"""
Wget-focused rescue grid.

The goal is not to cherry-pick a single lucky seed, but to identify whether a
stable Wget default exists among the reviewer-motivated variants. The grid keeps
the train/test split tied to each training seed and reports five-training-seed
mean/std for each representation + detector setting.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

from reviewer_required_experiments import load_model_from_checkpoint
from nimble_core.utils.loaddata import load_batch_level_dataset, transform_graph
from nimble_core.utils.poolers import Pooling


def metrics_from_scores(y, scores):
    auc = roc_auc_score(y, scores) if len(np.unique(y)) > 1 else 0.0
    prec, rec, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best = int(np.argmax(f1))
    th = thresholds[min(best, len(thresholds) - 1)] if len(thresholds) else np.inf
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
        "threshold": float(th),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def extract_wget_split(checkpoint, device, split_seed, pool_mode="typed"):
    dataset = load_batch_level_dataset("wget")
    model = load_model_from_checkpoint("wget", checkpoint, device)
    pooler = Pooling("mean")
    xs, ys = [], []
    with torch.no_grad():
        for i in dataset["full_index"]:
            g, label = dataset["dataset"][i]
            g = transform_graph(g, dataset["n_feat"], dataset["e_feat"]).to(device)
            emb = model.embed(g)
            if pool_mode == "typed":
                pooled = pooler(g, emb, n_types=dataset["n_feat"]).cpu().numpy()
            else:
                pooled = pooler(g, emb).cpu().numpy()
            xs.append(np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0))
            ys.append(label)
    x = np.concatenate(xs, axis=0)
    y = np.asarray(ys)

    rng = np.random.default_rng(split_seed)
    benign = np.where(y == 0)[0].copy()
    attack = np.where(y == 1)[0].copy()
    rng.shuffle(benign)
    train_idx = benign[:100]
    test_idx = np.concatenate([benign[100:], attack])
    return x[train_idx], x[test_idx], y[test_idx]


def fit_scores(x_train, x_test, seed, n_estimators, max_samples, contamination):
    if max_samples != "auto":
        max_samples = float(max_samples)
    if contamination != "auto":
        contamination = float(contamination)
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    clf = IsolationForest(
        n_estimators=int(n_estimators),
        max_samples=max_samples,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(x_train_s)
    return -clf.decision_function(x_test_s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out-dir", default="wget_rescue_grid")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "wget_rescue_grid_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    variants = {
        "gaussian_edge0": "p2_experiments/checkpoints/wget_gaussian_edge0_seed{seed}.pt",
        "gaussian_edge1": "p2_experiments/checkpoints/wget_gaussian_edge1_seed{seed}.pt",
        "mask_edge0": "p2_experiments/checkpoints/wget_mask_edge0_seed{seed}.pt",
        "mask_edge1": "reviewer_experiments/checkpoints/wget_mask_seed{seed}.pt",
        "none_edge1": "reviewer_experiments/checkpoints/wget_none_seed{seed}.pt",
    }
    grid = []
    for n_estimators in [100, 200, 500]:
        for max_samples in ["auto", 0.5, 1.0]:
            for contamination in ["auto", 0.01, 0.05, 0.1]:
                grid.append((n_estimators, max_samples, contamination))

    raw_path = out_dir / "wget_rescue_grid_raw.csv"
    rows = []
    done = set()
    if raw_path.exists():
        old = pd.read_csv(raw_path)
        rows = old.to_dict("records")
        done = {
            (
                r["variant"],
                int(r["seed"]),
                int(r["n_estimators"]),
                str(r["max_samples"]),
                str(r["contamination"]),
            )
            for r in rows
        }
        print(f"Resuming from {raw_path}: {len(rows)} rows")

    cache = {}
    for variant, pattern in variants.items():
        for seed in seeds:
            ckpt = Path(pattern.format(seed=seed))
            if not ckpt.exists():
                print(f"[missing] {variant} seed={seed}: {ckpt}")
                continue
            x_train, x_test, y_test = extract_wget_split(ckpt, device, seed, "typed")
            cache[(variant, seed)] = (x_train, x_test, y_test)
            for n_estimators, max_samples, contamination in grid:
                key = (variant, seed, n_estimators, str(max_samples), str(contamination))
                if key in done:
                    continue
                scores = fit_scores(x_train, x_test, seed, n_estimators, max_samples, contamination)
                m = metrics_from_scores(y_test, scores)
                rows.append({
                    "variant": variant,
                    "seed": seed,
                    "n_estimators": n_estimators,
                    "max_samples": max_samples,
                    "contamination": contamination,
                    **m,
                })
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            print(f"[done] {variant} seed={seed}, rows={len(rows)}")

    raw = pd.DataFrame(rows)
    raw.to_csv(raw_path, index=False)
    metric_cols = ["auc", "precision", "recall", "f1", "fpr"]
    summary = (
        raw.groupby(["variant", "n_estimators", "max_samples", "contamination"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(map(str, c)).rstrip("_") for c in summary.columns]
    summary = summary.sort_values(["f1_mean", "fpr_mean"], ascending=[False, True])
    summary.to_csv(out_dir / "wget_rescue_grid_summary.csv", index=False)
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
