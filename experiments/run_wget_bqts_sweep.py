"""
Wget BQTS parameter sweep.

This script evaluates strict benign-quantile threshold selection on Wget while
reusing existing Wget checkpoints. For each training seed, benign batches are
split into detector-fit, threshold-validation, and held-out benign test pools;
attack batches are used only after the threshold is fixed, for reporting.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from experiments.reviewer_required_experiments import load_model_from_checkpoint
from nimble_core.utils.loaddata import load_batch_level_dataset, transform_graph
from nimble_core.utils.poolers import Pooling


def parse_list(text, cast=str):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def normalize_value(x):
    if isinstance(x, str):
        if x.lower() == "auto":
            return "auto"
        if x.lower() in ("true", "false"):
            return x.lower() == "true"
    try:
        return float(x)
    except Exception:
        return x


def metric_at_threshold(y, scores, threshold):
    pred = (scores >= threshold).astype(np.int8)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    auc = roc_auc_score(y, scores) if len(np.unique(y)) > 1 else 0.0
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


def fit_preprocess(x_fit, x_val, x_test, scaler_name, pca_dim, seed):
    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "robust":
        scaler = RobustScaler()
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
    elif scaler_name == "none":
        scaler = None
    else:
        raise ValueError(f"Unknown scaler: {scaler_name}")

    if scaler is not None:
        x_fit = scaler.fit_transform(x_fit)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

    if pca_dim is not None and pca_dim > 0 and pca_dim < x_fit.shape[1]:
        dim = min(int(pca_dim), x_fit.shape[0], x_fit.shape[1])
        pca = PCA(n_components=dim, whiten=True, random_state=seed)
        x_fit = pca.fit_transform(x_fit)
        x_val = pca.transform(x_val)
        x_test = pca.transform(x_test)

    return x_fit, x_val, x_test


def score_iforest(x_fit, x_val, x_test, seed, n_estimators, max_samples, max_features, bootstrap):
    clf = IsolationForest(
        n_estimators=int(n_estimators),
        contamination="auto",
        max_samples=normalize_value(max_samples),
        max_features=float(max_features),
        bootstrap=bool(bootstrap),
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(x_fit)
    return -clf.decision_function(x_val), -clf.decision_function(x_test)


def score_knn(x_fit, x_val, x_test, k):
    k = min(int(k), len(x_fit))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    nn.fit(x_fit)
    val_dist, _ = nn.kneighbors(x_val)
    test_dist, _ = nn.kneighbors(x_test)
    return val_dist[:, -1], test_dist[:, -1]


def extract_wget_embeddings(checkpoint, device, pool_mode):
    dataset = load_batch_level_dataset("wget")
    model = load_model_from_checkpoint("wget", checkpoint, device)
    pooler = Pooling("mean")
    xs, ys = [], []
    with torch.no_grad():
        for i in dataset["full_index"]:
            graph, label = dataset["dataset"][i]
            graph = transform_graph(graph, dataset["n_feat"], dataset["e_feat"]).to(device)
            emb = model.embed(graph)
            if pool_mode == "typed":
                pooled = pooler(graph, emb, n_types=dataset["n_feat"]).cpu().numpy()
            else:
                pooled = pooler(graph, emb).cpu().numpy()
            xs.append(np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0))
            ys.append(label)
    return np.concatenate(xs, axis=0), np.asarray(ys)


def split_bqts(x, y, seed, fit_count, val_fraction):
    rng = np.random.default_rng(seed)
    benign = np.flatnonzero(y == 0).copy()
    attack = np.flatnonzero(y == 1).copy()
    rng.shuffle(benign)
    fit_count = min(int(fit_count), len(benign) - 2)
    fit_idx = benign[:fit_count]
    heldout = benign[fit_count:]
    val_count = max(1, min(len(heldout) - 1, int(round(len(heldout) * float(val_fraction)))))
    val_idx = heldout[:val_count]
    test_benign_idx = heldout[val_count:]
    test_idx = np.concatenate([test_benign_idx, attack])
    y_test = np.concatenate([np.zeros(len(test_benign_idx), dtype=np.int8), np.ones(len(attack), dtype=np.int8)])
    return x[fit_idx], x[val_idx], x[test_idx], y_test, len(fit_idx), len(val_idx), len(test_benign_idx), len(attack)


def summarize(raw, out_dir):
    metric_cols = ["auc", "precision", "recall", "f1", "fpr", "tp", "fp", "tn", "fn"]
    group_cols = [
        "variant",
        "pooling",
        "fit_count",
        "val_fraction",
        "scaler",
        "pca_dim",
        "detector",
        "detector_params",
        "threshold_method",
    ]
    summary = raw.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(map(str, c)).rstrip("_") for c in summary.columns]
    summary = summary.sort_values(["f1_mean", "fpr_mean", "precision_mean"], ascending=[False, True, False])
    summary.to_csv(out_dir / "wget_bqts_sweep_summary.csv", index=False)
    top = summary.head(30)
    top.to_csv(out_dir / "wget_bqts_sweep_top30.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out-dir", default="wget_bqts_sweep")
    parser.add_argument("--variants", default="gaussian_edge0,mask_edge0,none_edge1")
    parser.add_argument("--pooling", default="typed,global")
    parser.add_argument("--fit-counts", default="75,80,90,100,110")
    parser.add_argument("--val-fractions", default="0.4,0.5,0.6,0.7")
    parser.add_argument("--quantiles", default="0.80,0.85,0.90,0.925,0.95,0.975,0.99")
    parser.add_argument("--scalers", default="standard,robust,minmax")
    parser.add_argument("--pca-dims", default="0,16,32,64")
    parser.add_argument("--detectors", default="iforest,knn")
    parser.add_argument("--iforest-estimators", default="100,200,500")
    parser.add_argument("--iforest-max-samples", default="auto,0.5,0.75,1.0")
    parser.add_argument("--iforest-max-features", default="0.5,0.75,1.0")
    parser.add_argument("--iforest-bootstrap", default="false,true")
    parser.add_argument("--knn-k", default="1,3,5,10,15")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "wget_bqts_sweep_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    seeds = parse_list(args.seeds, int)
    variants_requested = set(parse_list(args.variants, str))
    pool_modes = parse_list(args.pooling, str)
    fit_counts = parse_list(args.fit_counts, int)
    val_fractions = parse_list(args.val_fractions, float)
    quantiles = parse_list(args.quantiles, float)
    scalers = parse_list(args.scalers, str)
    pca_dims = [None if int(x) == 0 else int(x) for x in parse_list(args.pca_dims, int)]
    detectors = set(parse_list(args.detectors, str))

    variant_paths = {
        "gaussian_edge0": "p2_experiments/checkpoints/wget_gaussian_edge0_seed{seed}.pt",
        "gaussian_edge1": "p2_experiments/checkpoints/wget_gaussian_edge1_seed{seed}.pt",
        "mask_edge0": "p2_experiments/checkpoints/wget_mask_edge0_seed{seed}.pt",
        "mask_edge1": "reviewer_experiments/checkpoints/wget_mask_seed{seed}.pt",
        "none_edge1": "reviewer_experiments/checkpoints/wget_none_seed{seed}.pt",
    }
    variant_paths = {k: v for k, v in variant_paths.items() if k in variants_requested}

    raw_path = out_dir / "wget_bqts_sweep_raw.csv"
    rows = []
    done = set()
    if raw_path.exists():
        old = pd.read_csv(raw_path)
        rows = old.to_dict("records")
        done = {
            (
                r["variant"], r["pooling"], int(r["seed"]), int(r["fit_count"]),
                float(r["val_fraction"]), r["scaler"], str(r["pca_dim"]),
                r["detector"], r["detector_params"], r["threshold_method"],
            )
            for r in rows
        }
        print(f"Resuming {len(rows)} rows from {raw_path}")

    emb_cache = {}
    for variant, pattern in variant_paths.items():
        for seed in seeds:
            ckpt = Path(pattern.format(seed=seed))
            if not ckpt.exists():
                print(f"[missing] {variant} seed={seed}: {ckpt}")
                continue
            for pool_mode in pool_modes:
                cache_key = (variant, seed, pool_mode)
                if cache_key not in emb_cache:
                    print(f"[embed] variant={variant} seed={seed} pooling={pool_mode}")
                    emb_cache[cache_key] = extract_wget_embeddings(ckpt, device, pool_mode)
                x, y = emb_cache[cache_key]
                for fit_count in fit_counts:
                    for val_fraction in val_fractions:
                        x_fit0, x_val0, x_test0, y_test, n_fit, n_val, n_test_benign, n_attack = split_bqts(
                            x, y, seed, fit_count, val_fraction
                        )
                        for scaler in scalers:
                            for pca_dim in pca_dims:
                                x_fit, x_val, x_test = fit_preprocess(
                                    x_fit0, x_val0, x_test0, scaler, pca_dim, seed
                                )
                                detector_runs = []
                                if "iforest" in detectors:
                                    for n_est in parse_list(args.iforest_estimators, int):
                                        for max_samples in parse_list(args.iforest_max_samples, str):
                                            for max_features in parse_list(args.iforest_max_features, float):
                                                for bootstrap in parse_list(args.iforest_bootstrap, str):
                                                    b = bootstrap.lower() == "true"
                                                    params = f"n={n_est};max_samples={max_samples};max_features={max_features};bootstrap={b}"
                                                    detector_runs.append(("iforest", params, (n_est, max_samples, max_features, b)))
                                if "knn" in detectors:
                                    for k in parse_list(args.knn_k, int):
                                        params = f"k={k}"
                                        detector_runs.append(("knn", params, (k,)))

                                for detector, params, param_values in detector_runs:
                                    if detector == "iforest":
                                        val_scores, test_scores = score_iforest(
                                            x_fit, x_val, x_test, seed, *param_values
                                        )
                                    else:
                                        val_scores, test_scores = score_knn(x_fit, x_val, x_test, *param_values)
                                    for q in quantiles:
                                        method = f"benign-q{q:g}"
                                        key = (
                                            variant, pool_mode, seed, fit_count, val_fraction,
                                            scaler, str(pca_dim), detector, params, method,
                                        )
                                        if key in done:
                                            continue
                                        threshold = float(np.quantile(val_scores, q))
                                        m = metric_at_threshold(y_test, test_scores, threshold)
                                        rows.append({
                                            "variant": variant,
                                            "pooling": pool_mode,
                                            "seed": seed,
                                            "fit_count": fit_count,
                                            "val_fraction": val_fraction,
                                            "n_fit": n_fit,
                                            "n_val": n_val,
                                            "n_test_benign": n_test_benign,
                                            "n_attack": n_attack,
                                            "scaler": scaler,
                                            "pca_dim": "none" if pca_dim is None else pca_dim,
                                            "detector": detector,
                                            "detector_params": params,
                                            "threshold_method": method,
                                            "quantile": q,
                                            **m,
                                        })
                pd.DataFrame(rows).to_csv(raw_path, index=False)
                print(f"[done] variant={variant} seed={seed} pooling={pool_mode}; rows={len(rows)}")

    raw = pd.DataFrame(rows)
    raw.to_csv(raw_path, index=False)
    summary = summarize(raw, out_dir)
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
