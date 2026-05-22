import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dgl.dataloading import GraphDataLoader
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import SubsetRandomSampler

from nimble_core.model.autoencoder_graphsage_denosing import GMAEModel
from nimble_core.model.train import batch_level_train
from nimble_core.utils.loaddata import load_batch_level_dataset, transform_graph
from nimble_core.utils.poolers import Pooling
from nimble_core.utils.utils import create_optimizer, set_random_seed


@dataclass
class ExperimentConfig:
    dataset: str
    corruption: str
    seed: int
    num_hidden: int = 256
    num_layers: int = 4
    max_epoch: int = 5
    mask_rate: float = 0.5
    noise_std: float = 0.1
    bounded_noise: bool = False
    renorm_noise: bool = False
    alpha_l: float = 3.0
    lambda_weight: float = 0.5
    aggregator: str = "mean"
    iforest_estimators: int = 100
    lr: float = 1e-3
    weight_decay: float = 5e-4
    device: int = 0


class CorruptionAblationModel(GMAEModel):
    def __init__(self, *args, corruption_mode="gaussian", **kwargs):
        super().__init__(*args, **kwargs)
        self._corruption_mode = corruption_mode
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
                new_g.ndata["attr"][noise_nodes],
                noise_std,
            )
        elif self._corruption_mode == "mask":
            new_g.ndata["attr"][noise_nodes] = self.enc_mask_token
        elif self._corruption_mode == "none":
            noise_nodes = perm
            keep_nodes = perm[:0]
        else:
            raise ValueError(f"Unknown corruption mode: {self._corruption_mode}")

        return new_g, (noise_nodes, keep_nodes)


def build_corruption_model(cfg: ExperimentConfig, n_dim: int, e_dim: int):
    return CorruptionAblationModel(
        n_dim=n_dim,
        e_dim=e_dim,
        hidden_dim=cfg.num_hidden,
        n_layers=cfg.num_layers,
        n_heads=4,
        activation="prelu",
        feat_drop=0.1,
        negative_slope=0.2,
        residual=True,
        noise_rate=cfg.mask_rate,
        noise_std=cfg.noise_std,
        bounded_noise=cfg.bounded_noise,
        renorm_noise=cfg.renorm_noise,
        norm="BatchNorm",
        loss_fn="sce",
        alpha_l=cfg.alpha_l,
        lambda_weight=cfg.lambda_weight,
        aggregator=cfg.aggregator,
        corruption_mode=cfg.corruption,
    )


def extract_dataloader(entries, batch_size, seed):
    rng = random.Random(seed)
    entries = list(entries)
    rng.shuffle(entries)
    train_idx = torch.arange(len(entries))
    return GraphDataLoader(entries, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))


def train_one(cfg: ExperimentConfig, out_dir: Path):
    set_random_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    dataset = load_batch_level_dataset(cfg.dataset)
    graphs = dataset["dataset"]
    train_index = dataset["train_index"]
    n_dim = dataset["n_feat"]
    e_dim = dataset["e_feat"]
    batch_size = 12 if cfg.dataset == "streamspot" else 1
    device = torch.device(cfg.device if cfg.device >= 0 and torch.cuda.is_available() else "cpu")

    model = build_corruption_model(cfg, n_dim, e_dim).to(device)
    optimizer = create_optimizer("adam", model, cfg.lr, cfg.weight_decay)
    loader = extract_dataloader(train_index, batch_size, cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    model = batch_level_train(model, graphs, loader, optimizer, cfg.max_epoch, device, n_dim, e_dim)
    train_time = time.time() - start
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else np.nan

    ckpt = out_dir / "checkpoints" / f"{cfg.dataset}_{cfg.corruption}_seed{cfg.seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    return model, dataset, str(ckpt), train_time, peak_mb


def pool_embedding(pool_mode, graph, feat, n_types):
    pooler = Pooling("mean")
    if pool_mode == "global":
        return pooler(graph, feat).cpu().numpy()
    if pool_mode == "typed":
        return pooler(graph, feat, n_types=n_types).cpu().numpy()
    raise ValueError(pool_mode)


def extract_graph_embeddings(model, dataset_name, dataset, pool_mode, device):
    model.eval()
    x_list, y_list = [], []
    graphs = dataset["dataset"]
    n_types = dataset["n_feat"]
    with torch.no_grad():
        for i in dataset["full_index"]:
            g = transform_graph(graphs[i][0], dataset["n_feat"], dataset["e_feat"]).to(device)
            out = model.embed(g)
            emb = pool_embedding(pool_mode, g, out, n_types)
            emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            x_list.append(emb)
            y_list.append(graphs[i][1])
    return np.concatenate(x_list, axis=0), np.asarray(y_list)


def eval_embeddings(x, y, dataset_name, seed, iforest_estimators=100):
    train_count = 400 if dataset_name == "streamspot" else 100
    rng = np.random.default_rng(seed)
    benign_idx = np.where(y == 0)[0].copy()
    attack_idx = np.where(y == 1)[0].copy()
    rng.shuffle(benign_idx)
    rng.shuffle(attack_idx)

    x_train = x[benign_idx[:train_count]]
    x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
    y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    clf = IsolationForest(n_estimators=iforest_estimators, contamination="auto", random_state=seed)
    clf.fit(x_train)
    scores = -clf.decision_function(x_test)
    auc = roc_auc_score(y_test, scores)
    prec, rec, threshold = precision_recall_curve(y_test, scores)
    f1 = 2 * prec * rec / (rec + prec + 1e-9)
    best = int(np.argmax(f1))
    th = threshold[min(best, len(threshold) - 1)]
    pred = (scores >= th).astype(int)
    tp = int(((y_test == 1) & (pred == 1)).sum())
    fp = int(((y_test == 0) & (pred == 1)).sum())
    tn = int(((y_test == 0) & (pred == 0)).sum())
    fn = int(((y_test == 1) & (pred == 0)).sum())
    return {
        "auc": auc,
        "precision": prec[best],
        "recall": rec[best],
        "f1": f1[best],
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def run_experiments(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seeds = [int(s) for s in args.seeds.split(",")]

    for dataset_name in args.datasets:
        max_epoch = args.max_epoch_streamspot if dataset_name == "streamspot" else args.max_epoch_wget
        for corruption in args.corruptions:
            for seed in seeds:
                cfg = ExperimentConfig(
                    dataset=dataset_name,
                    corruption=corruption,
                    seed=seed,
                    max_epoch=max_epoch,
                    mask_rate=args.mask_rate,
                    noise_std=args.noise_std,
                    bounded_noise=args.bounded_gaussian,
                    renorm_noise=args.renorm_gaussian,
                    alpha_l=args.alpha_l,
                    lambda_weight=args.lambda_weight,
                    aggregator=args.aggregator,
                    iforest_estimators=args.iforest_estimators,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    device=args.device,
                )
                print(
                    f"\n[reviewer-exp] dataset={dataset_name} corruption={corruption} "
                    f"seed={seed} epochs={max_epoch} noise_std={cfg.noise_std} "
                    f"bounded={cfg.bounded_noise} renorm={cfg.renorm_noise} "
                    f"aggregator={cfg.aggregator}"
                )
                model, dataset, ckpt, train_time, peak_mb = train_one(cfg, out_dir)
                device = torch.device(args.device if args.device >= 0 and torch.cuda.is_available() else "cpu")
                for pool_mode in args.pool_modes:
                    x, y = extract_graph_embeddings(model, dataset_name, dataset, pool_mode, device)
                    metrics = eval_embeddings(
                        x,
                        y,
                        dataset_name,
                        seed,
                        iforest_estimators=cfg.iforest_estimators,
                    )
                    rows.append({
                        "dataset": dataset_name,
                        "corruption": corruption,
                        "pooling": pool_mode,
                        "seed": seed,
                        "checkpoint": ckpt,
                        "train_time_s": train_time,
                        "peak_memory_mb": peak_mb,
                        "max_epoch": cfg.max_epoch,
                        "noise_std": cfg.noise_std,
                        "bounded_noise": cfg.bounded_noise,
                        "renorm_noise": cfg.renorm_noise,
                        "aggregator": cfg.aggregator,
                        "iforest_estimators": cfg.iforest_estimators,
                        **metrics,
                    })
                    pd.DataFrame(rows).to_csv(out_dir / "reviewer_experiment_raw.csv", index=False)

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "reviewer_experiment_raw.csv", index=False)
    summarize(raw, out_dir)


def summarize(raw, out_dir: Path):
    metrics = ["auc", "f1", "precision", "recall", "fpr", "train_time_s", "peak_memory_mb"]
    summary = (
        raw.groupby(["dataset", "corruption", "pooling"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    summary.to_csv(out_dir / "reviewer_experiment_summary.csv", index=False)

    table = summary.copy()
    for metric in ["auc", "f1", "precision", "recall", "fpr"]:
        table[f"{metric}_pct"] = table[f"{metric}_mean"] * 100
        table[f"{metric}_std_pct"] = table[f"{metric}_std"] * 100
    table.to_csv(out_dir / "reviewer_experiment_summary_percent.csv", index=False)

    latex_rows = []
    for _, r in table.iterrows():
        latex_rows.append({
            "Dataset": r["dataset"],
            "Corruption": r["corruption"],
            "Pooling": r["pooling"],
            "AUC": f"{r['auc_pct']:.2f} $\\pm$ {r['auc_std_pct']:.2f}",
            "F1": f"{r['f1_pct']:.2f} $\\pm$ {r['f1_std_pct']:.2f}",
            "Precision": f"{r['precision_pct']:.2f} $\\pm$ {r['precision_std_pct']:.2f}",
            "Recall": f"{r['recall_pct']:.2f} $\\pm$ {r['recall_std_pct']:.2f}",
            "FPR": f"{r['fpr_pct']:.2f} $\\pm$ {r['fpr_std_pct']:.2f}",
        })
    latex_df = pd.DataFrame(latex_rows)
    with open(out_dir / "reviewer_experiment_table.tex", "w", encoding="utf-8") as f:
        f.write(latex_df.to_latex(index=False, escape=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["streamspot", "wget"])
    parser.add_argument("--corruptions", nargs="+", default=["gaussian", "mask", "none"])
    parser.add_argument("--pool-modes", nargs="+", default=["global", "typed"])
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max-epoch-streamspot", type=int, default=5)
    parser.add_argument("--max-epoch-wget", type=int, default=5)
    parser.add_argument("--mask-rate", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--bounded-gaussian", action="store_true")
    parser.add_argument("--renorm-gaussian", action="store_true")
    parser.add_argument("--alpha-l", type=float, default=3.0)
    parser.add_argument("--lambda-weight", type=float, default=0.5)
    parser.add_argument("--aggregator", choices=["mean", "max", "sum"], default="mean")
    parser.add_argument("--iforest-estimators", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out-dir", default="reviewer_experiments")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiments(parse_args())
