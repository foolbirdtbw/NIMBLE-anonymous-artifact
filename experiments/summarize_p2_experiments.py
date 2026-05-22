"""Summarize P2 reviewer-risk experiments.

Inputs:
  p2_experiments/p2_raw.csv

Outputs:
  p2_experiments/corruption_summary.csv
  p2_experiments/edge_recon_summary.csv
  p2_experiments/p2_tables.tex
"""

from pathlib import Path

import pandas as pd


METRICS = ["auc", "precision", "recall", "f1", "fpr"]


def pct(mean, std):
    return f"{mean * 100:.2f} $\\pm$ {std * 100:.2f}"


def summarize(df, group_cols):
    out = df.groupby(group_cols)[METRICS].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(c).rstrip("_") for c in out.columns]
    return out


def table_from_summary(summary, group_cols, caption, label):
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Dataset & Setting & AUC & Precision & Recall & F1 & FPR\\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        dataset = row["dataset"]
        if "corruption" in group_cols:
            setting = row["corruption"]
        else:
            setting = "with edge recon." if bool(row["edge_recon"]) else "without edge recon."
        lines.append(
            f"{dataset} & {setting} & "
            f"{pct(row['auc_mean'], row['auc_std'])} & "
            f"{pct(row['precision_mean'], row['precision_std'])} & "
            f"{pct(row['recall_mean'], row['recall_std'])} & "
            f"{pct(row['f1_mean'], row['f1_std'])} & "
            f"{pct(row['fpr_mean'], row['fpr_std'])}\\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main():
    out_dir = Path("p2_experiments")
    raw_path = out_dir / "p2_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    df = pd.read_csv(raw_path)
    tex_parts = []

    corr = df[df["experiment"] == "corruption"]
    if not corr.empty:
        corr_summary = summarize(corr, ["dataset", "corruption"])
        corr_summary.to_csv(out_dir / "corruption_summary.csv", index=False)
        tex_parts.append(
            table_from_summary(
                corr_summary,
                ["dataset", "corruption"],
                "Corruption-mode ablation under the P2 reviewer-risk protocol.",
                "tab:p2_corruption",
            )
        )

    edge = df[df["experiment"] == "edge_recon"]
    if not edge.empty:
        edge_summary = summarize(edge, ["dataset", "edge_recon"])
        edge_summary.to_csv(out_dir / "edge_recon_summary.csv", index=False)
        tex_parts.append(
            table_from_summary(
                edge_summary,
                ["dataset", "edge_recon"],
                "Edge-reconstruction on/off ablation under the P2 reviewer-risk protocol.",
                "tab:p2_edge_recon",
            )
        )

    (out_dir / "p2_tables.tex").write_text("\n".join(tex_parts), encoding="utf-8")

    print(f"Rows: {len(df)}")
    print(f"Wrote summaries to {out_dir}")


if __name__ == "__main__":
    main()
