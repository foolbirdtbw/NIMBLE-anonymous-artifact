"""
Experiment P0: Label-free threshold protocol + P1: Wget multi-seed.

For each dataset × seed:
  1. Train encoder (or load checkpoint)
  2. Extract embeddings
  3. Fit iForest on benign train embeddings
  4. Compute scores on test set
  5. Report metrics under:
     a) Label-tuned threshold (PR-curve optimal)
     b) Benign-quantile thresholds (95%, 97.5%, 99% of benign validation scores)
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import dgl
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from nimble_core.model.autoencoder_graphsage_denosing import build_model
from nimble_core.model.train import batch_level_train
from nimble_core.utils.loaddata import (
    load_batch_level_dataset,
    load_entity_level_dataset,
    load_metadata,
    transform_graph,
)
from nimble_core.utils.poolers import Pooling
from nimble_core.utils.utils import create_optimizer, set_random_seed

import warnings
warnings.filterwarnings("ignore")


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def make_args(dataset, n_dim, e_dim, **kw):
    a = SimpleNamespace()
    a.dataset = dataset
    a.device = kw.pop("device", 0)
    a.lr = kw.pop("lr", 0.001)
    a.weight_decay = kw.pop("weight_decay", 5e-4)
    a.negative_slope = kw.pop("negative_slope", 0.2)
    a.mask_rate = kw.pop("mask_rate", 0.5)
    a.alpha_l = kw.pop("alpha_l", 3)
    a.optimizer = kw.pop("optimizer", "adam")
    a.loss_fn = kw.pop("loss_fn", "sce")
    a.pooling = kw.pop("pooling", "mean")
    a.eval_method = kw.pop("eval_method", "iforest")
    a.n_dim = n_dim
    a.e_dim = e_dim
    if dataset in ("streamspot", "wget"):
        a.num_hidden = kw.pop("num_hidden", 256)
        a.num_layers = kw.pop("num_layers", 4)
        a.max_epoch = kw.pop("max_epoch", 5)
    else:
        a.num_hidden = kw.pop("num_hidden", 64)
        a.num_layers = kw.pop("num_layers", 3)
        a.max_epoch = kw.pop("max_epoch", 50)
    a.lambda_weight = kw.pop("lambda_weight", 0.5)
    a.noise_std = kw.pop("noise_std", 0.1)
    a.bounded_noise = kw.pop("bounded_noise", False)
    a.renorm_noise = kw.pop("renorm_noise", False)
    a.aggregator = kw.pop("aggregator", "mean")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


# ── metrics helpers ──────────────────────────────────────────────

def metrics_at_threshold(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tp = int(((y_true == 1) & (pred == 1)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    tn = int(((y_true == 0) & (pred == 0)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0
    return dict(auc=auc, precision=prec, recall=rec, f1=f1, fpr=fpr,
                threshold=float(threshold), tp=tp, fp=fp, tn=tn, fn=fn)


def label_tuned_metrics(y_true, scores):
    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, scores)
    f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
    best = int(np.argmax(f1_arr))
    th = thresholds[min(best, len(thresholds) - 1)]
    return metrics_at_threshold(y_true, scores, th)


# ── batch-level (StreamSpot / Wget) ─────────────────────────────

def train_batch_model(dataset_name, seed, data, device, out_dir, force=False, **kw):
    ckpt = out_dir / "checkpoints" / f"checkpoint-{dataset_name}-seed{seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    args = make_args(dataset_name, data["n_feat"], data["e_feat"], device=device.index if hasattr(device, 'index') and device.index is not None else 0, **kw)

    if ckpt.exists() and not force:
        model = build_model(args).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        return model

    set_random_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_model(args).to(device)
    optimizer = create_optimizer("adam", model, args.lr, args.weight_decay)
    batch_size = 12 if dataset_name == "streamspot" else 1

    train_idx = list(data["train_index"])
    random.shuffle(train_idx)
    from torch.utils.data.sampler import SubsetRandomSampler
    from dgl.dataloading import GraphDataLoader
    loader = GraphDataLoader(train_idx, batch_size=batch_size,
                             sampler=SubsetRandomSampler(torch.arange(len(train_idx))))

    model = batch_level_train(model, data["dataset"], loader, optimizer,
                              args.max_epoch, device, data["n_feat"], data["e_feat"])
    torch.save(model.state_dict(), ckpt)
    model.eval()
    return model


def extract_batch_embeddings(model, dataset_name, data, pool_mode, device):
    model.eval()
    pooler = Pooling("mean")
    x_list, y_list = [], []
    with torch.no_grad():
        for i in data["full_index"]:
            g = transform_graph(data["dataset"][i][0], data["n_feat"], data["e_feat"]).to(device)
            emb = model.embed(g)
            if pool_mode == "typed":
                pooled = pooler(g, emb, n_types=data["n_feat"]).cpu().numpy()
            else:
                pooled = pooler(g, emb).cpu().numpy()
            x_list.append(np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0))
            y_list.append(data["dataset"][i][1])
    return np.concatenate(x_list, axis=0), np.asarray(y_list)


def run_batch_label_free(dataset_name, x, y, seed, n_estimators=100,
                         quantiles=(0.95, 0.975, 0.99)):
    """
    Split benign into train/validation, fit iForest on train,
    compute benign-quantile thresholds from validation, evaluate on test.
    """
    train_count = 400 if dataset_name == "streamspot" else 100
    rng = np.random.default_rng(seed)
    benign_idx = np.where(y == 0)[0].copy()
    attack_idx = np.where(y == 1)[0].copy()
    rng.shuffle(benign_idx)

    benign_train = benign_idx[:train_count]
    benign_remaining = benign_idx[train_count:]

    # Split remaining benign into validation + test
    n_val = max(1, len(benign_remaining) // 2)
    benign_val = benign_remaining[:n_val]
    benign_test = benign_remaining[n_val:]

    x_train = x[benign_train]
    x_val = x[benign_val]
    x_test_benign = x[benign_test]
    x_test_attack = x[attack_idx]

    # Full test set for evaluation
    x_test = np.concatenate([x_test_benign, x_test_attack], axis=0)
    y_test = np.concatenate([np.zeros(len(benign_test)), np.ones(len(attack_idx))], axis=0)

    # Also full test including validation benign for label-tuned comparison
    x_test_full = np.concatenate([x[benign_remaining], x[attack_idx]], axis=0)
    y_test_full = np.concatenate([np.zeros(len(benign_remaining)), np.ones(len(attack_idx))], axis=0)

    # Fit scaler + iForest on train
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)
    x_test_full_s = scaler.transform(x_test_full)

    clf = IsolationForest(n_estimators=n_estimators, contamination="auto",
                          random_state=seed, n_jobs=-1)
    clf.fit(x_train_s)

    # Scores
    val_scores = -clf.decision_function(x_val_s)
    test_scores = -clf.decision_function(x_test_s)
    test_scores_full = -clf.decision_function(x_test_full_s)

    rows = []

    # 1) Label-tuned threshold (on full test)
    lt = label_tuned_metrics(y_test_full, test_scores_full)
    lt["threshold_method"] = "label-tuned"
    lt["quantile"] = None
    rows.append(lt)

    # 2) Benign-quantile thresholds
    for q in quantiles:
        th = np.quantile(val_scores, q)
        m = metrics_at_threshold(y_test, test_scores, th)
        m["threshold_method"] = f"benign-q{q}"
        m["quantile"] = q
        rows.append(m)

    return rows, {"val_scores": val_scores, "test_scores": test_scores,
                  "y_test": y_test, "test_scores_full": test_scores_full,
                  "y_test_full": y_test_full}


# ── entity-level (DARPA E3 Trace) ───────────────────────────────

def train_trace_model(seed, device, out_dir, force=False, **kw):
    metadata = load_metadata("trace")
    ckpt = out_dir / "checkpoints" / f"checkpoint-trace-seed{seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    args = make_args("trace", metadata["node_feature_dim"], metadata["edge_feature_dim"],
                     device=device.index if hasattr(device, 'index') and device.index is not None else 0, **kw)

    if ckpt.exists() and not force:
        model = build_model(args).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        return model, metadata

    set_random_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_model(args).to(device)
    optimizer = create_optimizer("adam", model, args.lr, args.weight_decay)

    for epoch in tqdm(range(args.max_epoch), desc=f"trace seed {seed}"):
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

    torch.save(model.state_dict(), ckpt)
    model.eval()
    return model, metadata


def extract_trace_embeddings(model, metadata, device):
    model.eval()
    malicious, _ = metadata["malicious"]
    n_train, n_test = metadata["n_train"], metadata["n_test"]

    x_train_parts = []
    with torch.no_grad():
        for i in tqdm(range(n_train), desc="trace train emb"):
            g = load_entity_level_dataset("trace", "train", i).to(device)
            x_train_parts.append(np.nan_to_num(model.embed(g).cpu().numpy()))
            del g
        x_train = np.concatenate(x_train_parts, axis=0)

        skip_benign = 0
        x_test_parts = []
        for i in tqdm(range(n_test), desc="trace test emb"):
            g = load_entity_level_dataset("trace", "test", i).to(device)
            if i != n_test - 1:
                skip_benign += g.number_of_nodes()
            x_test_parts.append(np.nan_to_num(model.embed(g).cpu().numpy()))
            del g
        x_test_all = np.concatenate(x_test_parts, axis=0)

    y_all = np.zeros(x_test_all.shape[0], dtype=np.int8)
    y_all[malicious] = 1
    test_idx = np.array([i for i in range(x_test_all.shape[0])
                         if i >= skip_benign or y_all[i] == 1])
    return x_train, x_test_all[test_idx], y_all[test_idx]


def run_trace_label_free(x_train, x_test, y_test, seed,
                         n_estimators=200, max_samples=1.0,
                         quantiles=(0.95, 0.975, 0.99)):
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    # Use a subset of training for validation scores
    rng = np.random.default_rng(seed)
    n_val = min(50000, x_train_s.shape[0])
    val_idx = rng.choice(x_train_s.shape[0], n_val, replace=False)
    x_val_s = x_train_s[val_idx]

    # Fit on remaining
    train_mask = np.ones(x_train_s.shape[0], dtype=bool)
    train_mask[val_idx] = False
    x_fit = x_train_s[train_mask] if train_mask.sum() > 100 else x_train_s

    clf = IsolationForest(n_estimators=n_estimators, contamination="auto",
                          max_samples=max_samples, random_state=seed, n_jobs=-1)
    clf.fit(x_fit)

    val_scores = -clf.decision_function(x_val_s)
    test_scores = -clf.decision_function(x_test_s)

    rows = []
    # Label-tuned
    lt = label_tuned_metrics(y_test, test_scores)
    lt["threshold_method"] = "label-tuned"
    lt["quantile"] = None
    rows.append(lt)

    # Benign-quantile
    for q in quantiles:
        th = np.quantile(val_scores, q)
        m = metrics_at_threshold(y_test, test_scores, th)
        m["threshold_method"] = f"benign-q{q}"
        m["quantile"] = q
        rows.append(m)

    return rows


# ── main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["streamspot", "wget", "trace"])
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out-dir", default="labelfree_experiments")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--skip-train", action="store_true",
                        help="Use existing checkpoints only")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_rows = []

    # ── StreamSpot & Wget ────────────────────────────────────────
    for ds in [d for d in args.datasets if d in ("streamspot", "wget")]:
        print(f"\n{'='*60}\n  Dataset: {ds}\n{'='*60}")
        data = load_batch_level_dataset(ds)
        pool_mode = "typed" if ds == "wget" else "global"

        for seed in seeds:
            print(f"\n--- {ds} seed={seed} ---")
            if args.skip_train:
                # Try to load existing checkpoint
                ckpt = out_dir / "checkpoints" / f"checkpoint-{ds}-seed{seed}.pt"
                if not ckpt.exists():
                    ckpt = Path(f"checkpoints/checkpoint-{ds}.pt")
                a = make_args(ds, data["n_feat"], data["e_feat"])
                model = build_model(a).to(device)
                model.load_state_dict(torch.load(ckpt, map_location=device))
                model.eval()
            else:
                model = train_batch_model(ds, seed, data, device, out_dir,
                                          force=args.force_train)

            x, y = extract_batch_embeddings(model, ds, data, pool_mode, device)
            rows, raw_data = run_batch_label_free(ds, x, y, seed)

            # Save raw predictions
            np.savez_compressed(
                out_dir / f"{ds}_seed{seed}_predictions.npz",
                y_test=raw_data["y_test"],
                scores=raw_data["test_scores"],
                y_test_full=raw_data["y_test_full"],
                scores_full=raw_data["test_scores_full"],
            )

            for r in rows:
                r["dataset"] = ds
                r["seed"] = seed
                r["pooling"] = pool_mode
                all_rows.append(r)

            # Incremental save
            pd.DataFrame(all_rows).to_csv(out_dir / "labelfree_raw.csv", index=False)
            print(f"  Saved {len(all_rows)} rows so far")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── DARPA E3 Trace ───────────────────────────────────────────
    if "trace" in args.datasets:
        print(f"\n{'='*60}\n  Dataset: trace\n{'='*60}")
        for seed in seeds:
            print(f"\n--- trace seed={seed} ---")
            if args.skip_train:
                metadata = load_metadata("trace")
                ckpt = out_dir / "checkpoints" / f"checkpoint-trace-seed{seed}.pt"
                if not ckpt.exists():
                    ckpt = Path("checkpoints/checkpoint-trace.pt")
                a = make_args("trace", metadata["node_feature_dim"],
                              metadata["edge_feature_dim"])
                model = build_model(a).to(device)
                model.load_state_dict(torch.load(ckpt, map_location=device))
                model.eval()
            else:
                model, metadata = train_trace_model(seed, device, out_dir,
                                                    force=args.force_train)

            x_train, x_test, y_test = extract_trace_embeddings(model, metadata, device)
            rows = run_trace_label_free(x_train, x_test, y_test, seed)

            for r in rows:
                r["dataset"] = "trace"
                r["seed"] = seed
                r["pooling"] = "node-level"
                all_rows.append(r)

            pd.DataFrame(all_rows).to_csv(out_dir / "labelfree_raw.csv", index=False)
            print(f"  Saved {len(all_rows)} rows so far")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "labelfree_raw.csv", index=False)

    # Summary: mean ± std per dataset × threshold_method
    metrics_cols = ["auc", "precision", "recall", "f1", "fpr"]
    summary = (
        df.groupby(["dataset", "threshold_method", "pooling"])[metrics_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    summary.to_csv(out_dir / "labelfree_summary.csv", index=False)

    # LaTeX table
    latex_rows = []
    for _, r in summary.iterrows():
        latex_rows.append({
            "Dataset": r["dataset"],
            "Threshold": r["threshold_method"],
            "AUC": f"{r['auc_mean']*100:.2f} $\\pm$ {r['auc_std']*100:.2f}",
            "F1": f"{r['f1_mean']*100:.2f} $\\pm$ {r['f1_std']*100:.2f}",
            "Precision": f"{r['precision_mean']*100:.2f} $\\pm$ {r['precision_std']*100:.2f}",
            "Recall": f"{r['recall_mean']*100:.2f} $\\pm$ {r['recall_std']*100:.2f}",
            "FPR": f"{r['fpr_mean']*100:.2f} $\\pm$ {r['fpr_std']*100:.2f}",
        })
    latex_df = pd.DataFrame(latex_rows)
    with open(out_dir / "labelfree_table.tex", "w", encoding="utf-8") as f:
        f.write(latex_df.to_latex(index=False, escape=False))

    print(f"\n{'='*60}")
    print(f"All done! Results in {out_dir}/")
    print(f"  labelfree_raw.csv        ({len(df)} rows)")
    print(f"  labelfree_summary.csv")
    print(f"  labelfree_table.tex")
    print(summary.to_string())


if __name__ == "__main__":
    main()
