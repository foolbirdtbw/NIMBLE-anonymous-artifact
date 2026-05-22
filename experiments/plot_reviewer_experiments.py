from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INFILE = ROOT / "reviewer_experiments" / "reviewer_experiment_summary_percent.csv"
OUTDIR = ROOT / "reviewer_experiments" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
    }
)


def style_axis(ax):
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, pad=2)


def add_panel_label(ax, label):
    ax.text(
        -0.16,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


df = pd.read_csv(INFILE)
df["dataset_label"] = df["dataset"].map({"streamspot": "StreamSpot", "wget": "Wget"})
df["corruption_label"] = df["corruption"].map(
    {"gaussian": "Gaussian", "mask": "Mask token", "none": "No corruption"}
)
df["pooling_label"] = df["pooling"].map({"global": "Global", "typed": "Type-aware"})

colors = {
    "Gaussian": "#386cb0",
    "Mask token": "#7fc97f",
    "No corruption": "#bf5b17",
    "Global": "#4d4d4d",
    "Type-aware": "#984ea3",
}

fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.6), constrained_layout=True)

datasets = ["StreamSpot", "Wget"]
corruptions = ["Gaussian", "Mask token", "No corruption"]
poolings = ["Global", "Type-aware"]

for ax, dataset in zip(axes[0], datasets):
    sub = df[(df["dataset_label"] == dataset)]
    x = np.arange(len(corruptions))
    width = 0.34
    for i, pooling in enumerate(poolings):
        vals = [
            sub[(sub["corruption_label"] == c) & (sub["pooling_label"] == pooling)][
                "f1_pct"
            ].iloc[0]
            for c in corruptions
        ]
        errs = [
            sub[(sub["corruption_label"] == c) & (sub["pooling_label"] == pooling)][
                "f1_std_pct"
            ].iloc[0]
            for c in corruptions
        ]
        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            color=colors[pooling],
            edgecolor="black",
            linewidth=0.35,
            label=pooling,
            yerr=errs,
            error_kw={"elinewidth": 0.6, "capsize": 2, "capthick": 0.6},
        )
    ax.set_title(f"{dataset}: detection F1")
    ax.set_xticks(x)
    ax.set_xticklabels(corruptions, rotation=20, ha="right")
    ax.set_ylabel("F1 (%)")
    ymin = 74 if dataset == "Wget" else 94
    ax.set_ylim(ymin, 100.8)
    style_axis(ax)

axes[0, 1].legend(loc="lower right", ncol=1, handlelength=1.2)

for ax, dataset in zip(axes[1], datasets):
    sub = df[(df["dataset_label"] == dataset)]
    x = np.arange(len(corruptions))
    width = 0.34
    for i, pooling in enumerate(poolings):
        vals = [
            sub[(sub["corruption_label"] == c) & (sub["pooling_label"] == pooling)][
                "fpr_pct"
            ].iloc[0]
            for c in corruptions
        ]
        errs = [
            sub[(sub["corruption_label"] == c) & (sub["pooling_label"] == pooling)][
                "fpr_std_pct"
            ].iloc[0]
            for c in corruptions
        ]
        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            color=colors[pooling],
            edgecolor="black",
            linewidth=0.35,
            yerr=errs,
            error_kw={"elinewidth": 0.6, "capsize": 2, "capthick": 0.6},
        )
    ax.set_title(f"{dataset}: false positive rate")
    ax.set_xticks(x)
    ax.set_xticklabels(corruptions, rotation=20, ha="right")
    ax.set_ylabel("FPR (%)")
    ax.set_ylim(0, 6 if dataset == "StreamSpot" else 55)
    style_axis(ax)

for label, ax in zip(["a", "b", "c", "d"], axes.ravel()):
    add_panel_label(ax, label)

caption = (
    "Reviewer-requested ablations over five random seeds. "
    "Bars show mean; whiskers show standard deviation."
)
fig.text(0.01, -0.01, caption, fontsize=7)

for ext, kwargs in {
    "pdf": {},
    "svg": {},
    "png": {"dpi": 600},
    "tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
}.items():
    fig.savefig(OUTDIR / f"reviewer_ablation.{ext}", bbox_inches="tight", **kwargs)

print(f"Saved figure files to {OUTDIR}")
