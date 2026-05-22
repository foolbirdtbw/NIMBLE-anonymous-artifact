"""
Wget training-length sweep for Scheme B.

This script retrains a small set of Wget representation variants for longer
budgets and reports five-training-seed means under the same label-tuned
iForest evaluation used by the batch-level experiments.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from run_p2_experiments import eval_batch, train_batch_ablation


def summarize(rows, out_dir):
    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "wget_epoch_sweep_raw.csv", index=False)
    metric_cols = ["auc", "precision", "recall", "f1", "fpr"]
    summary = (
        raw.groupby(["variant", "corruption", "edge_recon", "max_epoch"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(map(str, c)).rstrip("_") for c in summary.columns]
    summary = summary.sort_values(["f1_mean", "fpr_mean"], ascending=[False, True])
    summary.to_csv(out_dir / "wget_epoch_sweep_summary.csv", index=False)
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--epochs", default="10,20")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out-dir", default="wget_epoch_sweep")
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "wget_epoch_sweep_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    seeds = [int(s) for s in args.seeds.split(",")]
    epochs = [int(e) for e in args.epochs.split(",")]
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")

    variants = [
        ("gaussian_edge0", "gaussian", False),
        ("mask_edge1", "mask", True),
        ("mask_edge0", "mask", False),
        ("none_edge1", "none", True),
    ]

    raw_path = out_dir / "wget_epoch_sweep_raw.csv"
    rows = []
    done = set()
    if raw_path.exists():
        old = pd.read_csv(raw_path)
        rows = old.to_dict("records")
        done = {(r["variant"], int(r["max_epoch"]), int(r["seed"])) for r in rows}
        print(f"Resuming from {raw_path}: {len(rows)} rows")

    for max_epoch in epochs:
        epoch_dir = out_dir / f"epoch_{max_epoch}"
        for variant, corruption, edge_recon in variants:
            for seed in seeds:
                key = (variant, max_epoch, seed)
                if key in done:
                    print(f"[skip] {key}")
                    continue
                print(f"\n[wget-epoch] variant={variant} epoch={max_epoch} seed={seed}")
                model, data = train_batch_ablation(
                    "wget",
                    seed,
                    corruption,
                    edge_recon,
                    device,
                    epoch_dir,
                    max_epoch=max_epoch,
                    force=args.force_train,
                )
                metrics = eval_batch(model, "wget", data, device, seed)
                rows.append({
                    "dataset": "wget",
                    "variant": variant,
                    "corruption": corruption,
                    "edge_recon": edge_recon,
                    "max_epoch": max_epoch,
                    "seed": seed,
                    **metrics,
                })
                pd.DataFrame(rows).to_csv(raw_path, index=False)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summarize(rows, out_dir)


if __name__ == "__main__":
    main()
