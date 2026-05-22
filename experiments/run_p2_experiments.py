"""
P2 Experiments:
  A) DARPA Trace corruption ablation (gaussian / mask / none) × 5 seeds
  B) Edge reconstruction on/off × 3 datasets × 5 seeds
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
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from nimble_core.model.autoencoder_graphsage_denosing import GMAEModel, build_model
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


# ── Corruption Ablation Model ────────────────────────────────────

class CorruptionAblationModel(GMAEModel):
    """Extends GMAEModel to support gaussian / mask / none corruption."""
    def __init__(self, *args, corruption_mode="gaussian", use_edge_recon=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._corruption_mode = corruption_mode
        self._use_edge_recon_flag = use_edge_recon
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.decoder.layers[-1].out_dim))

    def encoding_denoising(self, g, noise_rate=0.3, noise_std=0.1):
        new_g = g.clone()
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=g.device)
        num_noise_nodes = int(noise_rate * num_nodes)
        noise_nodes = perm[:num_noise_nodes]
        keep_nodes = perm[num_noise_nodes:]

        if self._corruption_mode == "gaussian":
            new_g.ndata["attr"][noise_nodes] = self.apply_gaussian_corruption(
                new_g.ndata["attr"][noise_nodes], noise_std,
            )
        elif self._corruption_mode == "mask":
            new_g.ndata["attr"][noise_nodes] = self.enc_mask_token
        elif self._corruption_mode == "none":
            noise_nodes = perm
            keep_nodes = perm[:0]
        else:
            raise ValueError(f"Unknown corruption mode: {self._corruption_mode}")

        return new_g, (noise_nodes, keep_nodes)

    def compute_loss(self, g):
        pre_use_g, (noise_nodes, keep_nodes) = self.encoding_denoising(
            g, self._noise_rate, self._noise_std
        )
        pre_use_x = pre_use_g.ndata['attr'].to(pre_use_g.device)
        enc_rep, all_hidden = self.encoder(pre_use_g, pre_use_x, return_hidden=True)
        enc_rep = torch.cat(all_hidden, dim=1)
        rep = self.encoder_to_decoder(enc_rep)
        recon = self.decoder(pre_use_g, rep)

        if self._corruption_mode == "none":
            loss = self.criterion(recon, g.ndata['attr'])
        elif len(noise_nodes) > 0:
            x_init = g.ndata['attr'][noise_nodes]
            x_rec = recon[noise_nodes]
            loss = self.criterion(x_rec, x_init)
        else:
            loss = self.criterion(recon, g.ndata['attr'])

        # Edge reconstruction
        if self._use_edge_recon_flag:
            threshold = min(5000, g.num_nodes())
            negative_edge_pairs = dgl.sampling.global_uniform_negative_sampling(g, threshold)
            positive_edge_pairs = random.sample(
                range(g.number_of_edges()), min(threshold, g.number_of_edges())
            )
            positive_edge_pairs = (
                g.edges()[0][positive_edge_pairs],
                g.edges()[1][positive_edge_pairs],
            )
            sample_src = enc_rep[torch.cat([positive_edge_pairs[0], negative_edge_pairs[0]])].to(g.device)
            sample_dst = enc_rep[torch.cat([positive_edge_pairs[1], negative_edge_pairs[1]])].to(g.device)
            y_pred = self.edge_recon_fc(torch.cat([sample_src, sample_dst], dim=-1)).squeeze(-1)
            y = torch.cat([
                torch.ones(len(positive_edge_pairs[0])),
                torch.zeros(len(negative_edge_pairs[0])),
            ]).to(g.device)
            loss += self._lambda_weight * self.recon_loss(y_pred, y)

        return loss


def build_ablation_model(dataset, n_dim, e_dim, corruption="gaussian",
                         use_edge_recon=True, **kw):
    if dataset in ("streamspot", "wget"):
        num_hidden = kw.get("num_hidden", 256)
        num_layers = kw.get("num_layers", 4)
    else:
        num_hidden = kw.get("num_hidden", 64)
        num_layers = kw.get("num_layers", 3)

    return CorruptionAblationModel(
        n_dim=n_dim,
        e_dim=e_dim,
        hidden_dim=num_hidden,
        n_layers=num_layers,
        n_heads=4,
        activation="prelu",
        feat_drop=0.1,
        negative_slope=0.2,
        residual=True,
        noise_rate=kw.get("mask_rate", 0.5),
        noise_std=kw.get("noise_std", 0.1),
        bounded_noise=kw.get("bounded_noise", False),
        renorm_noise=kw.get("renorm_noise", False),
        norm="BatchNorm",
        loss_fn="sce",
        alpha_l=kw.get("alpha_l", 3),
        lambda_weight=kw.get("lambda_weight", 0.5),
        aggregator=kw.get("aggregator", "mean"),
        corruption_mode=corruption,
        use_edge_recon=use_edge_recon,
    )


# ── metrics ──────────────────────────────────────────────────────

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
        tp=tp, fp=fp, tn=tn, fn=fn,
    )


# ── DARPA Trace training & evaluation ────────────────────────────

def train_trace_ablation(seed, corruption, use_edge_recon, device, out_dir,
                         max_epoch=50, force=False):
    tag = f"trace_{corruption}_edge{int(use_edge_recon)}_seed{seed}"
    ckpt = out_dir / "checkpoints" / f"{tag}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata("trace")
    n_dim = metadata["node_feature_dim"]
    e_dim = metadata["edge_feature_dim"]

    if ckpt.exists() and not force:
        model = build_ablation_model("trace", n_dim, e_dim,
                                     corruption=corruption,
                                     use_edge_recon=use_edge_recon)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device).eval()
        return model, metadata

    set_random_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_ablation_model("trace", n_dim, e_dim,
                                 corruption=corruption,
                                 use_edge_recon=use_edge_recon).to(device)
    optimizer = create_optimizer("adam", model, 0.001, 5e-4)

    for epoch in tqdm(range(max_epoch), desc=f"{tag}"):
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


def eval_trace(model, metadata, device, seed):
    model.eval()
    malicious, _ = metadata["malicious"]
    n_train, n_test = metadata["n_train"], metadata["n_test"]

    with torch.no_grad():
        x_train = np.concatenate([
            np.nan_to_num(model.embed(
                load_entity_level_dataset("trace", "train", i).to(device)
            ).cpu().numpy()) for i in range(n_train)
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
    test_idx = np.array([i for i in range(x_test_all.shape[0])
                         if i >= skip_benign or y_all[i] == 1])
    x_test = x_test_all[test_idx]
    y_test = y_all[test_idx]

    # iForest
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    clf = IsolationForest(n_estimators=200, contamination="auto",
                          random_state=seed, n_jobs=-1)
    rng = np.random.default_rng(seed)
    n_fit = min(50000, x_train_s.shape[0])
    fit_idx = rng.choice(x_train_s.shape[0], n_fit, replace=False)
    clf.fit(x_train_s[fit_idx])
    scores = -clf.decision_function(x_test_s)
    return metrics_from_scores(y_test, scores)


# ── Batch-level (StreamSpot / Wget) training & evaluation ────────

def train_batch_ablation(dataset_name, seed, corruption, use_edge_recon,
                         device, out_dir, max_epoch=5, force=False):
    tag = f"{dataset_name}_{corruption}_edge{int(use_edge_recon)}_seed{seed}"
    ckpt = out_dir / "checkpoints" / f"{tag}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    data = load_batch_level_dataset(dataset_name)
    n_dim, e_dim = data["n_feat"], data["e_feat"]

    if ckpt.exists() and not force:
        model = build_ablation_model(dataset_name, n_dim, e_dim,
                                     corruption=corruption,
                                     use_edge_recon=use_edge_recon)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device).eval()
        return model, data

    set_random_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_ablation_model(dataset_name, n_dim, e_dim,
                                 corruption=corruption,
                                 use_edge_recon=use_edge_recon).to(device)
    optimizer = create_optimizer("adam", model, 0.001, 5e-4)

    from torch.utils.data.sampler import SubsetRandomSampler
    from dgl.dataloading import GraphDataLoader
    train_idx = list(data["train_index"])
    random.shuffle(train_idx)
    batch_size = 12 if dataset_name == "streamspot" else 1
    loader = GraphDataLoader(train_idx, batch_size=batch_size,
                             sampler=SubsetRandomSampler(torch.arange(len(train_idx))))

    model = batch_level_train(model, data["dataset"], loader, optimizer,
                              max_epoch, device, n_dim, e_dim)
    torch.save(model.state_dict(), ckpt)
    model.eval()
    return model, data


def eval_batch(model, dataset_name, data, device, seed):
    model.eval()
    pooler = Pooling("mean")
    x_list, y_list = [], []
    pool_mode = "typed" if dataset_name == "wget" else "global"

    with torch.no_grad():
        for i in data["full_index"]:
            g = transform_graph(data["dataset"][i][0], data["n_feat"],
                                data["e_feat"]).to(device)
            emb = model.embed(g)
            if pool_mode == "typed":
                pooled = pooler(g, emb, n_types=data["n_feat"]).cpu().numpy()
            else:
                pooled = pooler(g, emb).cpu().numpy()
            x_list.append(np.nan_to_num(pooled))
            y_list.append(data["dataset"][i][1])

    x = np.concatenate(x_list, axis=0)
    y = np.asarray(y_list)

    # iForest evaluation
    train_count = 400 if dataset_name == "streamspot" else 100
    rng = np.random.default_rng(seed)
    benign_idx = np.where(y == 0)[0].copy()
    attack_idx = np.where(y == 1)[0].copy()
    rng.shuffle(benign_idx)

    x_train = x[benign_idx[:train_count]]
    x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
    y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    clf = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=seed, n_jobs=-1)
    clf.fit(x_train_s)
    scores = -clf.decision_function(x_test_s)
    return metrics_from_scores(y_test, scores)


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["corruption", "edge_recon", "all"],
                        default="all")
    parser.add_argument("--datasets", nargs="+", default=["trace"])
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out-dir", default="p2_experiments")
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "p2_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    raw_path = out_dir / "p2_raw.csv"
    all_rows = []
    done_keys = set()
    if raw_path.exists():
        old_df = pd.read_csv(raw_path)
        all_rows = old_df.to_dict("records")
        for r in all_rows:
            done_keys.add((
                r["experiment"], r["dataset"], r["corruption"],
                bool(r["edge_recon"]), int(r["seed"]),
            ))
        print(f"Resuming from {raw_path}: {len(all_rows)} completed rows")

    def write_raw():
        pd.DataFrame(all_rows).to_csv(raw_path, index=False)

    # ── A) Corruption ablation ───────────────────────────────────
    if args.experiment in ("corruption", "all"):
        corruptions = ["gaussian", "mask", "none"]
        for ds in args.datasets:
            for corruption in corruptions:
                for seed in seeds:
                    key = ("corruption", ds, corruption, True, seed)
                    if key in done_keys:
                        print(f"[skip] {key}")
                        continue
                    print(f"\n[corruption] {ds} {corruption} seed={seed}")
                    if ds == "trace":
                        model, metadata = train_trace_ablation(
                            seed, corruption, True, device, out_dir,
                            force=args.force_train,
                        )
                        m = eval_trace(model, metadata, device, seed)
                    else:
                        model, data = train_batch_ablation(
                            ds, seed, corruption, True, device, out_dir,
                            force=args.force_train,
                        )
                        m = eval_batch(model, ds, data, device, seed)

                    all_rows.append({
                        "experiment": "corruption",
                        "dataset": ds,
                        "corruption": corruption,
                        "edge_recon": True,
                        "seed": seed,
                        **m,
                    })
                    done_keys.add(key)
                    write_raw()

                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # ── B) Edge recon on/off ─────────────────────────────────────
    if args.experiment in ("edge_recon", "all"):
        for ds in args.datasets:
            for edge_recon in [True, False]:
                for seed in seeds:
                    key = ("edge_recon", ds, "gaussian", edge_recon, seed)
                    if key in done_keys:
                        print(f"[skip] {key}")
                        continue
                    print(f"\n[edge_recon] {ds} edge={edge_recon} seed={seed}")
                    if ds == "trace":
                        model, metadata = train_trace_ablation(
                            seed, "gaussian", edge_recon, device, out_dir,
                            force=args.force_train,
                        )
                        m = eval_trace(model, metadata, device, seed)
                    else:
                        model, data = train_batch_ablation(
                            ds, seed, "gaussian", edge_recon, device, out_dir,
                            force=args.force_train,
                        )
                        m = eval_batch(model, ds, data, device, seed)

                    all_rows.append({
                        "experiment": "edge_recon",
                        "dataset": ds,
                        "corruption": "gaussian",
                        "edge_recon": edge_recon,
                        "seed": seed,
                        **m,
                    })
                    done_keys.add(key)
                    write_raw()

                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(raw_path, index=False)

    metric_cols = ["auc", "precision", "recall", "f1", "fpr"]

    # Corruption summary
    corr_df = df[df["experiment"] == "corruption"]
    if not corr_df.empty:
        corr_summary = (
            corr_df.groupby(["dataset", "corruption"])[metric_cols]
            .agg(["mean", "std"]).reset_index()
        )
        corr_summary.columns = ["_".join(c).rstrip("_") for c in corr_summary.columns]
        corr_summary.to_csv(out_dir / "corruption_summary.csv", index=False)
        print("\n" + "=" * 60)
        print("Corruption Ablation Summary")
        print("=" * 60)
        print(corr_summary.to_string(index=False))

    # Edge recon summary
    edge_df = df[df["experiment"] == "edge_recon"]
    if not edge_df.empty:
        edge_summary = (
            edge_df.groupby(["dataset", "edge_recon"])[metric_cols]
            .agg(["mean", "std"]).reset_index()
        )
        edge_summary.columns = ["_".join(c).rstrip("_") for c in edge_summary.columns]
        edge_summary.to_csv(out_dir / "edge_recon_summary.csv", index=False)
        print("\n" + "=" * 60)
        print("Edge Recon On/Off Summary")
        print("=" * 60)
        print(edge_summary.to_string(index=False))

    print(f"\nAll done! Results in {out_dir}/")


if __name__ == "__main__":
    main()
