"""
Matched-compute comparison between a MAGIC-style masked graph autoencoder and
NIMBLE.

The goal is not to reproduce MAGIC's published numbers with the authors'
original pipeline. Instead, this script answers the reviewer's component-level
question under a controlled in-repository protocol:

  * same processed datasets
  * same training seeds
  * same hidden size, layer count, epochs, optimizer, and benign train split
  * same pooling and downstream detector when isolating representation quality

Rows named "magic_style" use the original masked graph autoencoder in
nimble_core/model/autoencoder.py. Rows named "nimble" use
nimble_core/model/autoencoder_graphsage_denosing.py.
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from nimble_core.model.autoencoder import build_model as build_magic_style
from nimble_core.model.autoencoder_graphsage_denosing import build_model as build_nimble
from nimble_core.model.train import batch_level_train
from nimble_core.utils.loaddata import (
    load_batch_level_dataset,
    load_entity_level_dataset,
    load_metadata,
    transform_graph,
)
from nimble_core.utils.poolers import Pooling
from nimble_core.utils.utils import create_optimizer, set_random_seed


PIPELINES = {
    "magic_style": {
        "label": "MAGIC-style masked GAE",
        "builder": build_magic_style,
        "corruption": "mask token",
        "encoder": "GCNII",
        "auxiliary_objective": "attribute + edge reconstruction",
        "default_detector": "knn",
    },
    "nimble": {
        "label": "NIMBLE",
        "builder": build_nimble,
        "corruption": "Gaussian feature corruption",
        "encoder": "GraphSAGE",
        "auxiliary_objective": "attribute denoising + weighted edge reconstruction",
        "default_detector": "iforest",
    },
}


def make_args(dataset: str, n_dim: int, e_dim: int, pipeline: str, epochs: int) -> SimpleNamespace:
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
    args.noise_std = 0.1
    args.lambda_weight = 0.5
    args.bounded_noise = False
    args.renorm_noise = False
    args.aggregator = "mean"
    if dataset in ("streamspot", "wget"):
        args.num_hidden = 256
        args.num_layers = 4
    else:
        args.num_hidden = 64
        args.num_layers = 3
    args.max_epoch = epochs
    args.pipeline = pipeline
    return args


def set_seed(seed: int) -> None:
    set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def fit_scores(
    x_train: np.ndarray,
    x_test: np.ndarray,
    detector: str,
    seed: int,
    iforest_estimators: int,
    max_samples,
    max_train_samples: int,
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

    if detector == "iforest":
        clf = IsolationForest(
            n_estimators=iforest_estimators,
            contamination="auto",
            max_samples=max_samples,
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(x_fit)
        return -clf.decision_function(x_test_s)

    if detector == "knn":
        n_neighbors = min(10, max(2, int(x_fit.shape[0] * 0.02)))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
        nbrs.fit(x_fit)
        train_dist, _ = nbrs.kneighbors(x_fit, n_neighbors=n_neighbors)
        mean_distance = train_dist.mean() * n_neighbors / max(n_neighbors - 1, 1)
        test_dist, _ = nbrs.kneighbors(x_test_s, n_neighbors=n_neighbors)
        return test_dist.mean(axis=1) / (mean_distance + 1e-12)

    raise ValueError(f"Unknown detector: {detector}")


def train_batch_model(dataset_name: str, pipeline: str, seed: int, epochs: int, device, out_dir: Path, force=False):
    data = load_batch_level_dataset(dataset_name)
    args = make_args(dataset_name, data["n_feat"], data["e_feat"], pipeline, epochs)
    ckpt = out_dir / "checkpoints" / f"{dataset_name}_{pipeline}_seed{seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    model = PIPELINES[pipeline]["builder"](args)
    if ckpt.exists() and not force:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        return model.to(device).eval(), data

    set_seed(seed)
    model = model.to(device)
    optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    train_idx = list(data["train_index"])
    random.shuffle(train_idx)
    batch_size = 12 if dataset_name == "streamspot" else 1
    loader = GraphDataLoader(
        train_idx,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(torch.arange(len(train_idx))),
    )
    model = batch_level_train(model, data["dataset"], loader, optimizer, epochs, device, args.n_dim, args.e_dim)
    torch.save(model.state_dict(), ckpt)
    return model.eval(), data


def batch_embeddings(model, dataset_name: str, data: dict, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    pooler = Pooling("mean")
    pool_mode = "typed" if dataset_name == "wget" else "global"
    x_list, y_list = [], []
    with torch.no_grad():
        for i in data["full_index"]:
            g0, label = data["dataset"][i]
            g = transform_graph(g0, data["n_feat"], data["e_feat"]).to(device)
            emb = model.embed(g)
            if pool_mode == "typed":
                pooled = pooler(g, emb, n_types=data["n_feat"]).cpu().numpy()
            else:
                pooled = pooler(g, emb).cpu().numpy()
            x_list.append(np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0))
            y_list.append(label)
            del g
    return np.concatenate(x_list, axis=0), np.asarray(y_list)


def evaluate_batch_embeddings(x: np.ndarray, y: np.ndarray, dataset: str, seed: int, detector: str) -> dict:
    train_count = 400 if dataset == "streamspot" else 100
    rng = np.random.default_rng(seed)
    benign_idx = np.where(y == 0)[0].copy()
    attack_idx = np.where(y == 1)[0].copy()
    rng.shuffle(benign_idx)
    train_idx = benign_idx[:train_count]
    test_idx = np.concatenate([benign_idx[train_count:], attack_idx])
    scores = fit_scores(
        x[train_idx],
        x[test_idx],
        detector=detector,
        seed=seed,
        iforest_estimators=100,
        max_samples="auto",
        max_train_samples=50000,
    )
    return metrics_from_scores(y[test_idx], scores)


def train_trace_model(pipeline: str, seed: int, epochs: int, device, out_dir: Path, force=False):
    metadata = load_metadata("trace")
    args = make_args("trace", metadata["node_feature_dim"], metadata["edge_feature_dim"], pipeline, epochs)
    ckpt = out_dir / "checkpoints" / f"trace_{pipeline}_seed{seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    model = PIPELINES[pipeline]["builder"](args)
    if ckpt.exists() and not force:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        return model.to(device).eval(), metadata

    set_seed(seed)
    model = model.to(device)
    optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    for epoch in tqdm(range(epochs), desc=f"trace {pipeline} seed={seed}"):
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
            tqdm.write(f"trace {pipeline} seed={seed} epoch={epoch} loss={epoch_loss:.4f}")
    torch.save(model.state_dict(), ckpt)
    return model.eval(), metadata


def trace_embeddings(model, metadata: dict, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    malicious, _ = metadata["malicious"]
    x_train_parts, x_test_parts = [], []
    skip_benign = 0
    with torch.no_grad():
        for i in tqdm(range(metadata["n_train"]), desc="trace train embed"):
            g = load_entity_level_dataset("trace", "train", i).to(device)
            x_train_parts.append(np.nan_to_num(model.embed(g).cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0))
            del g
        for i in tqdm(range(metadata["n_test"]), desc="trace test embed"):
            g = load_entity_level_dataset("trace", "test", i).to(device)
            if i != metadata["n_test"] - 1:
                skip_benign += g.number_of_nodes()
            x_test_parts.append(np.nan_to_num(model.embed(g).cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0))
            del g
    x_train = np.concatenate(x_train_parts, axis=0)
    x_test = np.concatenate(x_test_parts, axis=0)
    y_test_all = np.zeros(x_test.shape[0], dtype=np.int8)
    y_test_all[malicious] = 1
    eval_idx = np.asarray([i for i in range(x_test.shape[0]) if i >= skip_benign or y_test_all[i] == 1], dtype=np.int64)
    return x_train, x_test[eval_idx], y_test_all[eval_idx]


def evaluate_trace_embeddings(x_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, seed: int, detector: str):
    scores = fit_scores(
        x_train,
        x_test,
        detector=detector,
        seed=seed,
        iforest_estimators=200,
        max_samples=1.0 if detector == "iforest" else "auto",
        max_train_samples=500000,
    )
    return metrics_from_scores(y_test, scores)


def summarize(rows: list[dict], out_dir: Path) -> None:
    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "magic_matched_compute_raw.csv", index=False)
    metric_cols = ["auc", "precision", "recall", "f1", "fpr"]
    group_cols = [
        "dataset",
        "pipeline",
        "pipeline_label",
        "corruption",
        "encoder",
        "auxiliary_objective",
        "pooling",
        "detector",
        "epochs",
        "num_hidden",
        "num_layers",
    ]
    summary = raw.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    summary.to_csv(out_dir / "magic_matched_compute_summary.csv", index=False)

    table = summary.copy()
    for col in metric_cols:
        table[f"{col}_text"] = table.apply(
            lambda r, c=col: f"{r[f'{c}_mean'] * 100:.2f} $\\pm$ {r[f'{c}_std'] * 100:.2f}",
            axis=1,
        )
    latex = table[
        [
            "dataset",
            "pipeline_label",
            "corruption",
            "encoder",
            "auxiliary_objective",
            "pooling",
            "detector",
            "auc_text",
            "precision_text",
            "recall_text",
            "f1_text",
            "fpr_text",
        ]
    ].rename(
        columns={
            "dataset": "Dataset",
            "pipeline_label": "Representation",
            "corruption": "Corruption",
            "encoder": "Encoder",
            "auxiliary_objective": "Auxiliary objective",
            "pooling": "Pooling",
            "detector": "Detector",
            "auc_text": "AUC",
            "precision_text": "Precision",
            "recall_text": "Recall",
            "f1_text": "F1",
            "fpr_text": "FPR",
        }
    )
    with open(out_dir / "magic_matched_compute_table.tex", "w", encoding="utf-8") as f:
        f.write(latex.to_latex(index=False, escape=False))
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["streamspot", "wget", "trace"])
    parser.add_argument("--pipelines", nargs="+", default=["magic_style", "nimble"], choices=list(PIPELINES))
    parser.add_argument("--detectors", nargs="+", default=["iforest", "knn"], choices=["iforest", "knn"])
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out-dir", default="magic_matched_compute")
    parser.add_argument("--batch-epochs", type=int, default=5)
    parser.add_argument("--trace-epochs", type=int, default=50)
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "magic_matched_compute_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    raw_path = out_dir / "magic_matched_compute_raw.csv"
    rows: list[dict] = []
    done = set()
    if raw_path.exists():
        old = pd.read_csv(raw_path)
        rows = old.to_dict("records")
        done = {(r["dataset"], r["pipeline"], int(r["seed"]), r["detector"]) for r in rows}
        print(f"Resuming {len(rows)} rows from {raw_path}")

    for dataset in args.datasets:
        for pipeline in args.pipelines:
            for seed in seeds:
                pending_detectors = [d for d in args.detectors if (dataset, pipeline, seed, d) not in done]
                if not pending_detectors:
                    continue
                print(f"\n[matched] dataset={dataset} pipeline={pipeline} seed={seed}")
                epochs = args.trace_epochs if dataset == "trace" else args.batch_epochs
                if dataset == "trace":
                    model, metadata = train_trace_model(pipeline, seed, epochs, device, out_dir, force=args.force_train)
                    x_train, x_test, y_test = trace_embeddings(model, metadata, device)
                    pooling = "node-level"
                else:
                    model, data = train_batch_model(dataset, pipeline, seed, epochs, device, out_dir, force=args.force_train)
                    x_all, y_all = batch_embeddings(model, dataset, data, device)
                    pooling = "type-aware" if dataset == "wget" else "global"

                for detector in pending_detectors:
                    if dataset == "trace":
                        metrics = evaluate_trace_embeddings(x_train, x_test, y_test, seed, detector)
                    else:
                        metrics = evaluate_batch_embeddings(x_all, y_all, dataset, seed, detector)
                    row = {
                        "dataset": dataset,
                        "pipeline": pipeline,
                        "pipeline_label": PIPELINES[pipeline]["label"],
                        "corruption": PIPELINES[pipeline]["corruption"],
                        "encoder": PIPELINES[pipeline]["encoder"],
                        "auxiliary_objective": PIPELINES[pipeline]["auxiliary_objective"],
                        "pooling": pooling,
                        "detector": detector,
                        "seed": seed,
                        "epochs": epochs,
                        "num_hidden": 64 if dataset == "trace" else 256,
                        "num_layers": 3 if dataset == "trace" else 4,
                        **metrics,
                    }
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(raw_path, index=False)
                    summarize(rows, out_dir)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    summarize(rows, out_dir)


if __name__ == "__main__":
    main()
