import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import dgl
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, roc_auc_score
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


TRACE_DEFAULTS = {
    "dataset": "trace",
    "num_hidden": 64,
    "num_layers": 3,
    "max_epoch": 50,
    "mask_rate": 0.5,
    "alpha_l": 3,
    "lambda_weight": 0.5,
    "noise_std": 0.1,
    "bounded_noise": False,
    "renorm_noise": False,
    "aggregator": "mean",
}


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_args(dataset, n_dim=None, e_dim=None, **overrides):
    args = SimpleNamespace()
    args.dataset = dataset
    args.device = overrides.pop("device", 0)
    args.lr = overrides.pop("lr", 0.001)
    args.weight_decay = overrides.pop("weight_decay", 5e-4)
    args.negative_slope = overrides.pop("negative_slope", 0.2)
    args.mask_rate = overrides.pop("mask_rate", 0.5)
    args.alpha_l = overrides.pop("alpha_l", 3)
    args.optimizer = overrides.pop("optimizer", "adam")
    args.loss_fn = overrides.pop("loss_fn", "sce")
    args.pooling = overrides.pop("pooling", "mean")
    args.eval_method = overrides.pop("eval_method", "iforest")
    if dataset in ("streamspot", "wget"):
        args.num_hidden = overrides.pop("num_hidden", 256)
        args.num_layers = overrides.pop("num_layers", 4)
        args.max_epoch = overrides.pop("max_epoch", 5 if dataset == "wget" else 5)
    else:
        for k, v in TRACE_DEFAULTS.items():
            setattr(args, k, overrides.pop(k, v))
    for k, v in overrides.items():
        setattr(args, k, v)
    if n_dim is not None:
        args.n_dim = n_dim
    if e_dim is not None:
        args.e_dim = e_dim
    return args


def load_model_from_checkpoint(dataset, checkpoint, device):
    if dataset in ("streamspot", "wget"):
        ds = load_batch_level_dataset(dataset)
        args = make_args(dataset, ds["n_feat"], ds["e_feat"])
    else:
        meta = load_metadata(dataset)
        args = make_args(dataset, meta["node_feature_dim"], meta["edge_feature_dim"])
    model = build_model(args)
    state = torch.load(checkpoint, map_location=device)
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()
    return model


def metrics_from_scores(y, scores):
    auc = roc_auc_score(y, scores)
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best = int(np.argmax(f1))
    if len(thresholds) == 0:
        threshold = np.inf
    else:
        threshold = thresholds[min(best, len(thresholds) - 1)]
    pred = (scores >= threshold).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
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


def fit_iforest_scores(x_train, x_test, seed, n_estimators=100, contamination="auto", max_samples="auto", max_train_samples=50000):
    if isinstance(contamination, str) and contamination != "auto":
        contamination = float(contamination)
    if isinstance(max_samples, str) and max_samples != "auto":
        max_samples = float(max_samples)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    if x_train.shape[0] > max_train_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x_train.shape[0], max_train_samples, replace=False)
        x_fit = x_train[idx]
    else:
        x_fit = x_train
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(x_fit)
    return -clf.decision_function(x_test)


def extract_wget_embeddings(checkpoint, device, pool_mode="typed"):
    dataset = load_batch_level_dataset("wget")
    model = load_model_from_checkpoint("wget", checkpoint, device)
    pooler = Pooling("mean")
    x_list, y_list = [], []
    with torch.no_grad():
        for i in dataset["full_index"]:
            g, label = dataset["dataset"][i]
            g = transform_graph(g, dataset["n_feat"], dataset["e_feat"]).to(device)
            emb = model.embed(g)
            if pool_mode == "typed":
                pooled = pooler(g, emb, n_types=dataset["n_feat"]).cpu().numpy()
            else:
                pooled = pooler(g, emb).cpu().numpy()
            x_list.append(np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0))
            y_list.append(label)
    x = np.concatenate(x_list, axis=0)
    y = np.asarray(y_list)
    rng = np.random.default_rng(0)
    benign = np.where(y == 0)[0].copy()
    attack = np.where(y == 1)[0].copy()
    rng.shuffle(benign)
    train_count = 100
    train_idx = benign[:train_count]
    test_idx = np.concatenate([benign[train_count:], attack])
    return x[train_idx], x[test_idx], y[test_idx]


def dagify_graph(g):
    src, dst = g.edges()
    low = torch.minimum(src, dst)
    high = torch.maximum(src, dst)
    keep = low != high
    low = low[keep]
    high = high[keep]
    edge_attrs = g.edata["attr"][keep] if "attr" in g.edata else None
    type_attrs = g.edata["type"][keep] if "type" in g.edata else None
    new_g = dgl.graph((low, high), num_nodes=g.num_nodes(), device=g.device)
    for key, value in g.ndata.items():
        new_g.ndata[key] = value
    if edge_attrs is not None:
        new_g.edata["attr"] = edge_attrs
    if type_attrs is not None:
        new_g.edata["type"] = type_attrs
    return new_g


def extract_trace_embeddings(checkpoint, device, graph_mode="original", cache_dir=None):
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"trace_{Path(checkpoint).stem}_{graph_mode}_embeddings.npz"
        if cache_path.exists():
            data = np.load(cache_path)
            return data["x_train"], data["x_test"], data["y_test"]

    metadata = load_metadata("trace")
    model = load_model_from_checkpoint("trace", checkpoint, device)
    malicious, _ = metadata["malicious"]
    n_train, n_test = metadata["n_train"], metadata["n_test"]

    x_train_parts = []
    with torch.no_grad():
        for i in tqdm(range(n_train), desc=f"trace train embeddings ({graph_mode})"):
            g = load_entity_level_dataset("trace", "train", i).to(device)
            if graph_mode == "dag":
                g = dagify_graph(g)
            x_train_parts.append(np.nan_to_num(model.embed(g).cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0))
            del g
        x_train = np.concatenate(x_train_parts, axis=0)

        skip_benign = 0
        x_test_parts = []
        for i in tqdm(range(n_test), desc=f"trace test embeddings ({graph_mode})"):
            g = load_entity_level_dataset("trace", "test", i).to(device)
            if i != n_test - 1:
                skip_benign += g.number_of_nodes()
            if graph_mode == "dag":
                g = dagify_graph(g)
            x_test_parts.append(np.nan_to_num(model.embed(g).cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0))
            del g
        x_test_all = np.concatenate(x_test_parts, axis=0)

    y_all = np.zeros(x_test_all.shape[0], dtype=np.int8)
    y_all[malicious] = 1
    test_idx = np.asarray([i for i in range(x_test_all.shape[0]) if i >= skip_benign or y_all[i] == 1], dtype=np.int64)
    x_test = x_test_all[test_idx]
    y_test = y_all[test_idx]
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, x_train=x_train, x_test=x_test, y_test=y_test)
    return x_train, x_test, y_test


def run_trace_multiseed(args):
    out_dir = ensure_dir(args.out_dir)
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(args.device if args.device >= 0 and torch.cuda.is_available() else "cpu")
    metadata = load_metadata("trace")
    rows = []

    for seed in seeds:
        ckpt = out_dir / "checkpoints" / f"checkpoint-trace-seed{seed}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        if not ckpt.exists() or args.force_train:
            set_random_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            train_args = make_args(
                "trace",
                metadata["node_feature_dim"],
                metadata["edge_feature_dim"],
                max_epoch=args.trace_epochs,
                lambda_weight=args.lambda_weight,
                device=args.device,
            )
            model = build_model(train_args).to(device)
            optimizer = create_optimizer(train_args.optimizer, model, train_args.lr, train_args.weight_decay)
            start = time.time()
            for epoch in tqdm(range(args.trace_epochs), desc=f"trace seed {seed} train"):
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
            train_time = time.time() - start
            torch.save(model.state_dict(), ckpt)
        else:
            train_time = np.nan

        x_train, x_test, y_test = extract_trace_embeddings(ckpt, device, "original", out_dir / "embedding_cache")
        scores = fit_iforest_scores(
            x_train,
            x_test,
            seed,
            n_estimators=args.iforest_estimators,
            contamination=args.contamination,
            max_samples=args.max_samples,
            max_train_samples=args.max_train_samples,
        )
        m = metrics_from_scores(y_test, scores)
        np.savez_compressed(out_dir / f"trace_seed{seed}_scores.npz", y=y_test, nimble_scores=scores)
        rows.append({"dataset": "trace", "seed": seed, "checkpoint": str(ckpt), "train_time_s": train_time, **m})
        pd.DataFrame(rows).to_csv(out_dir / "trace_multiseed_raw.csv", index=False)

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "trace_multiseed_raw.csv", index=False)
    summary = raw[["auc", "precision", "recall", "f1", "fpr"]].agg(["mean", "std"]).T.reset_index()
    summary.columns = ["metric", "mean", "std"]
    summary.to_csv(out_dir / "trace_multiseed_summary.csv", index=False)
    return raw, summary


def run_iforest_sensitivity(args):
    out_dir = ensure_dir(args.out_dir)
    device = torch.device(args.device if args.device >= 0 and torch.cuda.is_available() else "cpu")
    rows = []
    configs = []
    for n_est in [50, 100, 200]:
        for max_samples in ["auto", 0.5, 1.0]:
            for contamination in ["auto", 0.01, 0.05, 0.1]:
                configs.append((n_est, max_samples, contamination))

    datasets = []
    if args.wget_checkpoint:
        datasets.append(("wget", lambda: extract_wget_embeddings(args.wget_checkpoint, device, "typed")))
    if args.trace_checkpoint:
        datasets.append(("trace", lambda: extract_trace_embeddings(args.trace_checkpoint, device, "original", out_dir / "embedding_cache")))

    for dataset_name, loader in datasets:
        x_train, x_test, y_test = loader()
        for seed in [int(s) for s in args.detector_seeds.split(",")]:
            for n_est, max_samples, contamination in configs:
                scores = fit_iforest_scores(
                    x_train,
                    x_test,
                    seed,
                    n_estimators=n_est,
                    contamination=contamination,
                    max_samples=max_samples,
                    max_train_samples=args.max_train_samples,
                )
                m = metrics_from_scores(y_test, scores)
                rows.append({
                    "dataset": dataset_name,
                    "seed": seed,
                    "n_estimators": n_est,
                    "max_samples": max_samples,
                    "contamination": contamination,
                    **m,
                })
                pd.DataFrame(rows).to_csv(out_dir / "iforest_sensitivity_raw.csv", index=False)
    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "iforest_sensitivity_raw.csv", index=False)
    summary = raw.groupby(["dataset", "n_estimators", "max_samples", "contamination"])[["auc", "precision", "recall", "f1", "fpr"]].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(map(str, c)).rstrip("_") for c in summary.columns]
    summary.to_csv(out_dir / "iforest_sensitivity_summary.csv", index=False)
    return raw, summary


def run_cycle_dag(args):
    out_dir = ensure_dir(args.out_dir)
    device = torch.device(args.device if args.device >= 0 and torch.cuda.is_available() else "cpu")
    x_train_o, x_test_o, y_test = extract_trace_embeddings(args.trace_checkpoint, device, "original", out_dir / "embedding_cache")
    x_train_d, x_test_d, _ = extract_trace_embeddings(args.trace_checkpoint, device, "dag", out_dir / "embedding_cache")
    rows = []
    for mode, x_train, x_test in [("cyclic-original", x_train_o, x_test_o), ("first-seen-DAG", x_train_d, x_test_d)]:
        for seed in [int(s) for s in args.detector_seeds.split(",")]:
            scores = fit_iforest_scores(
                x_train,
                x_test,
                seed,
                n_estimators=args.iforest_estimators,
                contamination=args.contamination,
                max_samples=args.max_samples,
                max_train_samples=args.max_train_samples,
            )
            m = metrics_from_scores(y_test, scores)
            rows.append({"graph_construction": mode, "seed": seed, **m})
    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "cycle_dag_raw.csv", index=False)
    summary = raw.groupby("graph_construction")[["auc", "precision", "recall", "f1", "fpr"]].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(map(str, c)).rstrip("_") for c in summary.columns]
    cos = np.sum(x_test_o * x_test_d, axis=1) / (
        np.linalg.norm(x_test_o, axis=1) * np.linalg.norm(x_test_d, axis=1) + 1e-12
    )
    sim = {
        "mean_cosine": float(np.mean(cos)),
        "std_cosine": float(np.std(cos)),
        "median_cosine": float(np.median(cos)),
        "p05_cosine": float(np.quantile(cos, 0.05)),
        "p95_cosine": float(np.quantile(cos, 0.95)),
    }
    summary.to_csv(out_dir / "cycle_dag_summary.csv", index=False)
    with open(out_dir / "cycle_dag_embedding_similarity.json", "w", encoding="utf-8") as f:
        json.dump(sim, f, indent=2)
    return raw, summary, sim


def run_bootstrap_against_cited(args):
    out_dir = ensure_dir(args.out_dir)
    rows = []
    rng = np.random.default_rng(args.bootstrap_seed)
    cited = {
        "wget": {"magic_f1": 0.9388, "magic_fpr": 0.0400},
        "trace": {"magic_f1": 0.9910, "magic_fpr": 0.0014},
    }
    score_files = list(Path(args.out_dir).glob("trace_seed*_scores.npz"))
    for f in score_files:
        data = np.load(f)
        y = data["y"]
        scores = data["nimble_scores"]
        seed = int(f.stem.replace("trace_seed", "").replace("_scores", ""))
        for b in range(args.bootstrap_iters):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) < 2:
                continue
            m = metrics_from_scores(y[idx], scores[idx])
            rows.append({
                "dataset": "trace",
                "seed": seed,
                "bootstrap": b,
                "f1_minus_magic_cited": m["f1"] - cited["trace"]["magic_f1"],
                "fpr_minus_magic_cited": m["fpr"] - cited["trace"]["magic_fpr"],
                **m,
            })
    if args.wget_checkpoint:
        device = torch.device(args.device if args.device >= 0 and torch.cuda.is_available() else "cpu")
        x_train, x_test, y = extract_wget_embeddings(args.wget_checkpoint, device, "typed")
        for seed in [int(s) for s in args.detector_seeds.split(",")]:
            scores = fit_iforest_scores(x_train, x_test, seed, args.iforest_estimators, args.contamination, args.max_samples, args.max_train_samples)
            for b in range(args.bootstrap_iters):
                idx = rng.integers(0, len(y), len(y))
                if len(np.unique(y[idx])) < 2:
                    continue
                m = metrics_from_scores(y[idx], scores[idx])
                rows.append({
                    "dataset": "wget",
                    "seed": seed,
                    "bootstrap": b,
                    "f1_minus_magic_cited": m["f1"] - cited["wget"]["magic_f1"],
                    "fpr_minus_magic_cited": m["fpr"] - cited["wget"]["magic_fpr"],
                    **m,
                })
    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "bootstrap_vs_cited_magic_raw.csv", index=False)
    if raw.empty:
        return raw, pd.DataFrame()
    summary_rows = []
    for (dataset, seed), group in raw.groupby(["dataset", "seed"]):
        for col in ["f1", "fpr", "f1_minus_magic_cited", "fpr_minus_magic_cited"]:
            summary_rows.append({
                "dataset": dataset,
                "seed": seed,
                "metric": col,
                "mean": group[col].mean(),
                "ci_low": group[col].quantile(0.025),
                "ci_high": group[col].quantile(0.975),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "bootstrap_vs_cited_magic_summary.csv", index=False)
    return raw, summary


def write_tables(out_dir):
    out_dir = Path(out_dir)
    tex_parts = []
    for name in [
        "trace_multiseed_summary.csv",
        "iforest_sensitivity_summary.csv",
        "cycle_dag_summary.csv",
        "bootstrap_vs_cited_magic_summary.csv",
    ]:
        path = out_dir / name
        if path.exists():
            df = pd.read_csv(path)
            tex_parts.append(f"% {name}\n")
            tex_parts.append(df.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.4f}"))
            tex_parts.append("\n\n")
    (out_dir / "reviewer_required_tables.tex").write_text("".join(tex_parts), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reviewer_required_experiments")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--detector-seeds", default="0,1,2,3,4")
    parser.add_argument("--trace-epochs", type=int, default=50)
    parser.add_argument("--lambda-weight", type=float, default=0.5)
    parser.add_argument("--iforest-estimators", type=int, default=100)
    parser.add_argument("--contamination", default="auto")
    parser.add_argument("--max-samples", default="auto")
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--trace-checkpoint", default="checkpoints/checkpoint-trace.pt")
    parser.add_argument("--wget-checkpoint", default="checkpoints/checkpoint-wget.pt")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    parser.add_argument("--skip-trace-train", action="store_true")
    parser.add_argument("--skip-iforest", action="store_true")
    parser.add_argument("--skip-cycle-dag", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    if not args.skip_trace_train:
        run_trace_multiseed(args)
    if not args.skip_iforest:
        run_iforest_sensitivity(args)
    if not args.skip_cycle_dag:
        run_cycle_dag(args)
    if not args.skip_bootstrap:
        run_bootstrap_against_cited(args)
    write_tables(args.out_dir)


if __name__ == "__main__":
    main()
