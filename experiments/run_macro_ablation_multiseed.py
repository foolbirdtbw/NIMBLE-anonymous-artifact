"""
Five-seed macro-architecture ablation for NIMBLE.

This script upgrades the original single-run Table 7 style ablation into a
repeated-run experiment over the four macro variants:

  - full: GraphSAGE + denoising
  - no_denoising: GraphSAGE only
  - no_graphsage: GCN-style encoder + denoising
  - baseline: GCN-style encoder only

To isolate GraphSAGE and denoising, the optional edge-reconstruction head is
disabled for all variants by default.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from dgl.dataloading import GraphDataLoader
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from nimble_core.model.autoencoder_graphsage_denosing_ablation import build_model
from nimble_core.model.train import batch_level_train
from nimble_core.utils.loaddata import (
    load_batch_level_dataset,
    load_entity_level_dataset,
    load_metadata,
    transform_graph,
)
from nimble_core.utils.poolers import Pooling
from nimble_core.utils.utils import create_optimizer, set_random_seed


VARIANTS = {
    "full": {
        "label": "Full (GraphSAGE + Denoising)",
        "use_graphsage": True,
        "use_denoising": True,
    },
    "no_denoising": {
        "label": "w/o Denoising (GraphSAGE only)",
        "use_graphsage": True,
        "use_denoising": False,
    },
    "no_graphsage": {
        "label": "w/o GraphSAGE (GCN + Denoising)",
        "use_graphsage": False,
        "use_denoising": True,
    },
    "baseline": {
        "label": "w/o GraphSAGE & Denoising (GCN only)",
        "use_graphsage": False,
        "use_denoising": False,
    },
}


def make_args(dataset: str, n_dim: int, e_dim: int, variant: str, use_edge_recon: bool) -> SimpleNamespace:
    cfg = VARIANTS[variant]
    args = SimpleNamespace()
    args.dataset = dataset
    args.device = 0
    args.lr = 0.001
    args.weight_decay = 5e-4
    args.negative_slope = 0.2
    args.mask_rate = 0.5
    args.alpha_l = 3
    args.optimizer = "adam"
    args.loss_fn = "sce"
    args.pooling = "mean"
    args.eval_method = "iforest"
    args.n_dim = n_dim
    args.e_dim = e_dim
    args.use_graphsage = cfg["use_graphsage"]
    args.use_denoising = cfg["use_denoising"]
    args.use_edge_recon = use_edge_recon
    args.graphsage_rep_mode = "concat"
    args.aggregator = "mean"
    args.noise_std = 0.1
    args.lambda_weight = 0.5
    args.bounded_noise = False
    args.renorm_noise = False
    if dataset in ("streamspot", "wget"):
        args.num_hidden = 256
        args.num_layers = 4
        args.max_epoch = 5
    else:
        args.num_hidden = 64
        args.num_layers = 3
        args.max_epoch = 50
    return args


def metrics_from_scores(y_true: np.ndarray, scores: np.ndarray) -> dict:
    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_values = 2 * precision * recall / (precision + recall + 1e-9)
    best = int(np.argmax(f1_values))
    threshold = thresholds[min(best, len(thresholds) - 1)] if len(thresholds) else np.inf
    pred = (scores >= threshold).astype(np.int8)
    tp = int(((y_true == 1) & (pred == 1)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    tn = int(((y_true == 0) & (pred == 0)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    return {
        "auc": float(auc),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def fit_iforest_scores(
    x_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    n_estimators: int,
    max_samples,
    max_train_samples: int = 50000,
) -> np.ndarray:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    if x_train_s.shape[0] > max_train_samples:
        rng = np.random.default_rng(seed)
        fit_idx = rng.choice(x_train_s.shape[0], max_train_samples, replace=False)
        x_fit = x_train_s[fit_idx]
    else:
        x_fit = x_train_s
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",
        max_samples=max_samples,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(x_fit)
    return -clf.decision_function(x_test_s)


def train_batch(dataset: str, variant: str, seed: int, device, out_dir: Path, use_edge_recon: bool, force=False):
    tag = f"{dataset}_{variant}_edge{int(use_edge_recon)}_seed{seed}"
    ckpt = out_dir / "checkpoints" / f"{tag}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    data = load_batch_level_dataset(dataset)
    args = make_args(dataset, data["n_feat"], data["e_feat"], variant, use_edge_recon)
    if ckpt.exists() and not force:
        model = build_model(args)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device).eval()
        return model, data

    set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_model(args).to(device)
    optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    train_idx = list(data["train_index"])
    random.shuffle(train_idx)
    batch_size = 12 if dataset == "streamspot" else 1
    loader = GraphDataLoader(
        train_idx,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(torch.arange(len(train_idx))),
    )
    model = batch_level_train(model, data["dataset"], loader, optimizer, args.max_epoch, device, args.n_dim, args.e_dim)
    torch.save(model.state_dict(), ckpt)
    model.eval()
    return model, data


def eval_batch(model, dataset: str, data: dict, seed: int, device) -> dict:
    model.eval()
    pooler = Pooling("mean")
    pool_mode = "typed" if dataset == "wget" else "global"
    x_list, y_list = [], []
    with torch.no_grad():
        for i in data["full_index"]:
            g, label = data["dataset"][i]
            g = transform_graph(g, data["n_feat"], data["e_feat"]).to(device)
            emb = model.embed(g)
            if pool_mode == "typed":
                pooled = pooler(g, emb, n_types=data["n_feat"]).cpu().numpy()
            else:
                pooled = pooler(g, emb).cpu().numpy()
            x_list.append(np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0))
            y_list.append(label)
            del g
    x = np.concatenate(x_list, axis=0)
    y = np.asarray(y_list)
    train_count = 400 if dataset == "streamspot" else 100
    rng = np.random.default_rng(seed)
    benign_idx = np.where(y == 0)[0].copy()
    attack_idx = np.where(y == 1)[0].copy()
    rng.shuffle(benign_idx)
    train_idx = benign_idx[:train_count]
    test_idx = np.concatenate([benign_idx[train_count:], attack_idx])
    scores = fit_iforest_scores(
        x[train_idx],
        x[test_idx],
        seed=seed,
        n_estimators=100,
        max_samples="auto",
    )
    return metrics_from_scores(y[test_idx], scores)


def train_trace(variant: str, seed: int, device, out_dir: Path, use_edge_recon: bool, force=False):
    tag = f"trace_{variant}_edge{int(use_edge_recon)}_seed{seed}"
    ckpt = out_dir / "checkpoints" / f"{tag}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata("trace")
    args = make_args("trace", metadata["node_feature_dim"], metadata["edge_feature_dim"], variant, use_edge_recon)
    if ckpt.exists() and not force:
        model = build_model(args)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device).eval()
        return model, metadata

    set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_model(args).to(device)
    optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    for epoch in tqdm(range(args.max_epoch), desc=tag):
        epoch_loss = 0.0
        for i in range(metadata["n_train"]):
            g = load_entity_level_dataset("trace", "train", i).to(device)
            loss = model(g) / metadata["n_train"]
            optimizer.zero_grad()
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            del g, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if epoch % 10 == 0:
            tqdm.write(f"{tag} epoch={epoch} loss={epoch_loss:.4f}")
    torch.save(model.state_dict(), ckpt)
    model.eval()
    return model, metadata


def eval_trace(model, metadata: dict, seed: int, device) -> dict:
    model.eval()
    malicious, _ = metadata["malicious"]
    n_train, n_test = metadata["n_train"], metadata["n_test"]
    x_train_parts = []
    x_test_parts = []
    skip_benign = 0
    with torch.no_grad():
        for i in tqdm(range(n_train), desc="trace train embeddings"):
            g = load_entity_level_dataset("trace", "train", i).to(device)
            x_train_parts.append(np.nan_to_num(model.embed(g).cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0))
            del g
        for i in tqdm(range(n_test), desc="trace test embeddings"):
            g = load_entity_level_dataset("trace", "test", i).to(device)
            if i != n_test - 1:
                skip_benign += g.number_of_nodes()
            x_test_parts.append(np.nan_to_num(model.embed(g).cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0))
            del g
    x_train = np.concatenate(x_train_parts, axis=0)
    x_test_all = np.concatenate(x_test_parts, axis=0)
    y_all = np.zeros(x_test_all.shape[0], dtype=np.int8)
    y_all[malicious] = 1
    test_idx = np.asarray([i for i in range(x_test_all.shape[0]) if i >= skip_benign or y_all[i] == 1], dtype=np.int64)
    scores = fit_iforest_scores(
        x_train,
        x_test_all[test_idx],
        seed=seed,
        n_estimators=200,
        max_samples=1.0,
    )
    return metrics_from_scores(y_all[test_idx], scores)


def summarize(rows: list[dict], out_dir: Path):
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "macro_ablation_raw.csv", index=False)
    metric_cols = ["auc", "precision", "recall", "f1", "fpr"]
    summary = (
        df.groupby(["dataset", "variant", "variant_label", "edge_recon"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(col).rstrip("_") for col in summary.columns]
    summary.to_csv(out_dir / "macro_ablation_summary.csv", index=False)

    table_rows = []
    dataset_order = {"streamspot": 0, "wget": 1, "trace": 2}
    variant_order = {v: i for i, v in enumerate(VARIANTS)}
    for _, r in summary.sort_values(
        by=["dataset", "variant"],
        key=lambda s: s.map(dataset_order).fillna(s.map(variant_order)).fillna(99),
    ).iterrows():
        table_rows.append({
            "Dataset": r["dataset"],
            "Variant": r["variant_label"],
            "AUC": f"{r['auc_mean'] * 100:.2f} $\\pm$ {r['auc_std'] * 100:.2f}",
            "Precision": f"{r['precision_mean'] * 100:.2f} $\\pm$ {r['precision_std'] * 100:.2f}",
            "Recall": f"{r['recall_mean'] * 100:.2f} $\\pm$ {r['recall_std'] * 100:.2f}",
            "F1": f"{r['f1_mean'] * 100:.2f} $\\pm$ {r['f1_std'] * 100:.2f}",
            "FPR": f"{r['fpr_mean'] * 100:.2f} $\\pm$ {r['fpr_std'] * 100:.2f}",
        })
    table_df = pd.DataFrame(table_rows)
    with open(out_dir / "macro_ablation_table.tex", "w", encoding="utf-8") as f:
        f.write(table_df.to_latex(index=False, escape=False))
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["streamspot", "wget", "trace"])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()), choices=list(VARIANTS.keys()))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out-dir", default="macro_ablation_5seed")
    parser.add_argument("--edge-recon", action="store_true", help="Enable optional edge reconstruction for all variants.")
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "macro_ablation_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Datasets: {args.datasets}")
    print(f"Variants: {args.variants}")
    print(f"Seeds: {seeds}")
    print(f"Edge reconstruction: {args.edge_recon}")

    raw_path = out_dir / "macro_ablation_raw.csv"
    rows = []
    done = set()
    if raw_path.exists():
        old = pd.read_csv(raw_path)
        rows = old.to_dict("records")
        done = {(r["dataset"], r["variant"], int(r["seed"]), bool(r["edge_recon"])) for r in rows}
        print(f"Resuming from {raw_path}: {len(rows)} rows")

    for dataset in args.datasets:
        for variant in args.variants:
            for seed in seeds:
                key = (dataset, variant, seed, bool(args.edge_recon))
                if key in done:
                    print(f"[skip] {dataset} {variant} seed={seed}")
                    continue
                print(f"\n[macro] dataset={dataset} variant={variant} seed={seed}")
                if dataset == "trace":
                    model, meta = train_trace(variant, seed, device, out_dir, args.edge_recon, force=args.force_train)
                    metrics = eval_trace(model, meta, seed, device)
                else:
                    model, data = train_batch(dataset, variant, seed, device, out_dir, args.edge_recon, force=args.force_train)
                    metrics = eval_batch(model, dataset, data, seed, device)
                row = {
                    "dataset": dataset,
                    "variant": variant,
                    "variant_label": VARIANTS[variant]["label"],
                    "seed": seed,
                    "edge_recon": bool(args.edge_recon),
                    **metrics,
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(raw_path, index=False)
                summarize(rows, out_dir)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    summarize(rows, out_dir)
    print(f"\nDone: {out_dir}")


if __name__ == "__main__":
    main()
