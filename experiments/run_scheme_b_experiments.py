"""
Scheme-B rescue experiments.

This script evaluates a revised default candidate motivated by the reviewer
ablations:
  - StreamSpot: Gaussian corruption, no edge reconstruction, global pooling.
  - Wget: mask-token corruption, no edge reconstruction, type-aware pooling.
  - DARPA E3 Trace: Gaussian corruption, no edge reconstruction, iForest with
    max_samples=1.0.

It reuses checkpoints produced by run_p2_experiments.py where available and
trains only missing checkpoints.
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

from run_p2_experiments import (
    eval_batch,
    train_batch_ablation,
    train_trace_ablation,
)
from nimble_core.utils.loaddata import load_entity_level_dataset


def metrics_from_scores(y, scores):
    auc = roc_auc_score(y, scores) if len(np.unique(y)) > 1 else 0.0
    prec, rec, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best = int(np.argmax(f1))
    th = thresholds[min(best, len(thresholds) - 1)]
    pred = (scores >= th).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return dict(
        auc=auc,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        fpr=fp / (fp + tn) if fp + tn else 0.0,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def eval_trace_with_iforest(model, metadata, device, seed, n_estimators=200, max_samples=1.0):
    model.eval()
    malicious, _ = metadata["malicious"]
    n_train, n_test = metadata["n_train"], metadata["n_test"]

    with torch.no_grad():
        x_train = np.concatenate([
            np.nan_to_num(
                model.embed(load_entity_level_dataset("trace", "train", i).to(device)).cpu().numpy()
            )
            for i in range(n_train)
        ], axis=0)

        skip_benign = 0
        x_test_parts = []
        for i in range(n_test):
            g = load_entity_level_dataset("trace", "test", i).to(device)
            if i != n_test - 1:
                skip_benign += g.number_of_nodes()
            x_test_parts.append(np.nan_to_num(model.embed(g).cpu().numpy()))
            del g
        x_test_all = np.concatenate(x_test_parts, axis=0)

    y_all = np.zeros(x_test_all.shape[0], dtype=np.int8)
    y_all[malicious] = 1
    test_idx = np.array([i for i in range(x_test_all.shape[0]) if i >= skip_benign or y_all[i] == 1])
    x_test = x_test_all[test_idx]
    y_test = y_all[test_idx]

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    rng = np.random.default_rng(seed)
    n_fit = min(50000, x_train_s.shape[0])
    fit_idx = rng.choice(x_train_s.shape[0], n_fit, replace=False)

    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",
        max_samples=max_samples,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(x_train_s[fit_idx])
    scores = -clf.decision_function(x_test_s)
    return metrics_from_scores(y_test, scores)


def summarize(raw, out_dir):
    df = pd.DataFrame(raw)
    df.to_csv(out_dir / "scheme_b_raw.csv", index=False)
    metric_cols = ["auc", "precision", "recall", "f1", "fpr"]
    summary = (
        df.groupby(["dataset", "corruption", "edge_recon", "pooling", "iforest"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    summary.to_csv(out_dir / "scheme_b_summary.csv", index=False)

    rows = []
    for _, r in summary.iterrows():
        rows.append({
            "Dataset": r["dataset"],
            "Default candidate": f"{r['corruption']}, edge={r['edge_recon']}, {r['pooling']}, {r['iforest']}",
            "AUC": f"{r['auc_mean']*100:.2f} $\\pm$ {r['auc_std']*100:.2f}",
            "Precision": f"{r['precision_mean']*100:.2f} $\\pm$ {r['precision_std']*100:.2f}",
            "Recall": f"{r['recall_mean']*100:.2f} $\\pm$ {r['recall_std']*100:.2f}",
            "F1": f"{r['f1_mean']*100:.2f} $\\pm$ {r['f1_std']*100:.2f}",
            "FPR": f"{r['fpr_mean']*100:.2f} $\\pm$ {r['fpr_std']*100:.2f}",
        })
    latex_df = pd.DataFrame(rows)
    with open(out_dir / "scheme_b_table.tex", "w", encoding="utf-8") as f:
        f.write(latex_df.to_latex(index=False, escape=False))
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["streamspot", "wget", "trace"])
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out-dir", default="scheme_b_experiments")
    parser.add_argument("--checkpoint-dir", default="p2_experiments")
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scheme_b_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)
    print(f"Device: {device}")

    raw_path = out_dir / "scheme_b_raw.csv"
    rows = []
    done = set()
    if raw_path.exists():
        old = pd.read_csv(raw_path)
        rows = old.to_dict("records")
        done = {(r["dataset"], int(r["seed"])) for r in rows}
        print(f"Resuming from {raw_path}: {len(rows)} rows")

    configs = {
        "streamspot": dict(corruption="gaussian", edge_recon=False, pooling="global", iforest="100-auto-auto"),
        "wget": dict(corruption="mask", edge_recon=False, pooling="typed", iforest="100-auto-auto"),
        "trace": dict(corruption="gaussian", edge_recon=False, pooling="node-level", iforest="200-auto-1.0"),
    }

    for ds in args.datasets:
        cfg = configs[ds]
        for seed in seeds:
            if (ds, seed) in done:
                print(f"[skip] {ds} seed={seed}")
                continue
            print(f"\n[scheme-b] {ds} seed={seed} cfg={cfg}")
            if ds == "trace":
                model, metadata = train_trace_ablation(
                    seed,
                    cfg["corruption"],
                    cfg["edge_recon"],
                    device,
                    checkpoint_dir,
                    force=args.force_train,
                )
                metrics = eval_trace_with_iforest(model, metadata, device, seed)
            else:
                model, data = train_batch_ablation(
                    ds,
                    seed,
                    cfg["corruption"],
                    cfg["edge_recon"],
                    device,
                    checkpoint_dir,
                    force=args.force_train,
                )
                metrics = eval_batch(model, ds, data, device, seed)

            row = dict(
                dataset=ds,
                seed=seed,
                corruption=cfg["corruption"],
                edge_recon=cfg["edge_recon"],
                pooling=cfg["pooling"],
                iforest=cfg["iforest"],
                **metrics,
            )
            rows.append(row)
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summarize(rows, out_dir)
    print(f"\nAll done: {out_dir}")


if __name__ == "__main__":
    main()
