# NIMBLE Anonymous Reviewer Artifact

This anonymized artifact contains the core NIMBLE implementation and compact audit outputs for the three datasets used in the revised manuscript: StreamSpot, Unicorn Wget, and DARPA E3 Trace. It is organized as a reviewer artifact rather than as a copy of the original development workspace.

The package intentionally excludes author-identifying manuscript files, author biographies, local machine paths, checkpoints, cached embeddings, raw provenance datasets, and third-party baseline implementations.

## Artifact Layout

```text
.
├── nimble_core/
│   ├── model/                 # denoising GraphSAGE autoencoder and used GNN backbones
│   └── utils/                 # loaders, pooling, and dataset utilities
├── experiments/               # executable scripts for manuscript/revision experiments
├── outputs/                   # compact CSV/TEX/JSON outputs already used in the revision
│   ├── label_free_bqts/       # StreamSpot/Wget/Trace benign-quantile threshold protocol
│   ├── reviewer_ablations/    # Gaussian vs mask vs no corruption; pooling checks
│   ├── macro_ablation_5seed/  # five-seed macro-architecture ablation
│   ├── p2_experiments/        # Trace corruption and edge-reconstruction checks
│   ├── magic_matched_compute/ # MAGIC-style vs NIMBLE matched-compute ablation
│   ├── trace_controls/        # Trace multiseed, iForest, cycle-vs-DAG checks
│   ├── trace_bqts/            # Trace BQTS precision sweeps
│   ├── scheme_b/              # revised default-configuration checks
│   └── wget_rescue/           # Wget robustness/rescue sweeps
├── reviewer_ablation_config.yaml
├── environment.yml
├── requirements.txt
└── data_README.md
```

The earlier generic `model/`, `utils/`, and `results/` top-level layout has been replaced by `nimble_core/`, `experiments/`, and `outputs/` to make the artifact structure specific to NIMBLE and easier to inspect.

## Core Code Kept

Only model modules required by the included experiments are kept:

- `autoencoder_graphsage_denosing.py`
- `autoencoder_graphsage_denosing_ablation.py`
- `autoencoder.py` (MAGIC-style masked graph autoencoder path used only for the matched-compute ablation)
- `graphsage.py`
- `gat.py`, `gin.py`, `gcnii.py`
- `loss_func.py`
- `train.py`

Unused development modules such as the graph transformer, PNA, MLP-only helper, and standalone evaluation script are not included.

## Environment

The experiments were run with:

- Python 3.8
- PyTorch 1.13.1
- DGL 0.9.1
- CUDA-enabled GPU for full graph experiments
- NumPy, pandas, scikit-learn, NetworkX, tqdm, matplotlib

Create a matching conda environment:

```bash
conda env create -f environment.yml
conda activate nimble-review
```

If your CUDA/PyTorch/DGL stack differs, install matching PyTorch and DGL wheels for your system, then install the remaining packages from `requirements.txt`.

## Data Layout

Place processed datasets under `data/` at the repository root:

```text
data/
├── streamspot/
├── wget/
└── trace/
```

For DARPA E3 Trace, the scripts expect:

```text
data/trace/metadata.json
data/trace/train*.pkl
data/trace/test*.pkl
data/trace/malicious.pkl
```

For StreamSpot and Unicorn Wget, the loaders use the processed graph/batch files produced by the preprocessing utilities in `nimble_core/utils/`.

## Main Reproduction Commands

Run commands from the repository root. The `experiments` package is invoked with `python -m` so that imports resolve through `nimble_core`.

### 1. Strict Label-Free BQTS Protocol

This experiment trains/fits only on benign data, selects the threshold from a held-out benign validation stream, and uses attack labels only for final reporting.

```bash
python -m experiments.run_labelfree_threshold \
  --datasets streamspot wget trace \
  --seeds 0,1,2,3,4 \
  --out-dir outputs/label_free_bqts_rerun
```

Bootstrap confidence intervals from saved prediction files:

```bash
python -m experiments.run_bootstrap_ci \
  --input-dir outputs/label_free_bqts_rerun \
  --out-dir outputs/label_free_bqts_rerun
```

### 2. Reviewer-Requested Corruption and Pooling Ablation

```bash
python -m experiments.reviewer_experiments \
  --datasets streamspot wget \
  --corruptions gaussian mask none \
  --pool-modes global typed \
  --seeds 0,1,2,3,4 \
  --out-dir outputs/reviewer_ablations_rerun
```

Optional plotting:

```bash
python -m experiments.plot_reviewer_experiments
```

### 3. Five-Seed Macro-Architecture Ablation

```bash
python -m experiments.run_macro_ablation_multiseed \
  --datasets streamspot wget trace \
  --variants full no_denoising no_graphsage baseline \
  --seeds 0,1,2,3,4 \
  --out-dir outputs/macro_ablation_5seed_rerun
```

### 4. Trace Controls: iForest, Cycle-vs-DAG, and Multiseed Trace

```bash
python -m experiments.reviewer_required_experiments \
  --out-dir outputs/trace_controls_rerun \
  --device 0 \
  --detector-seeds 0,1,2,3,4
```

Repeated Trace training run:

```bash
python -m experiments.reviewer_required_experiments \
  --out-dir outputs/trace_multiseed_rerun \
  --device 0 \
  --seeds 0,1,2,3,4 \
  --detector-seeds 0 \
  --trace-epochs 50 \
  --lambda-weight 5.0 \
  --iforest-estimators 200 \
  --max-samples 1.0 \
  --skip-iforest --skip-cycle-dag --skip-bootstrap --force-train
```

### 5. Trace BQTS Precision Sweep

```bash
python -m experiments.tune_trace_bqts_precision \
  --seeds 0,1,2,3,4 \
  --n-estimators 200,500 \
  --max-samples 1.0,0.5,auto \
  --quantiles 0.95,0.975,0.99,0.995,0.9975,0.999 \
  --out-dir outputs/trace_bqts_rerun
```

### 6. MAGIC-Style Matched-Compute Ablation

This experiment compares the MAGIC-style masked graph autoencoder path against NIMBLE under the same datasets, seeds, training budget, pooling granularity, and downstream detector choices. It is an internal matched-compute ablation, not a reproduction of the published MAGIC authors' private run artifacts.

```bash
python -m experiments.run_magic_matched_compute \
  --datasets streamspot wget trace \
  --seeds 0,1,2,3,4 \
  --pipelines magic_style nimble \
  --detectors iforest knn \
  --batch-epochs 5 \
  --trace-epochs 50 \
  --out-dir outputs/magic_matched_compute_rerun
```

## Output Files

The `outputs/` directory contains compact files already generated for the revision:

- Raw seed-level CSV files.
- Summary CSV files with mean and standard deviation.
- LaTeX snippets used for manuscript/response tables.
- Small JSON diagnostic files where relevant.

Large rerun byproducts such as checkpoints, `.npz` prediction files, embedding caches, and raw datasets are ignored by `.gitignore`.

## Interpretation Boundaries

The included outputs support audit of NIMBLE-side experiments and robustness checks on StreamSpot, Wget, and DARPA E3 Trace. They do not reproduce every cited baseline row from prior literature because third-party baseline code and per-instance baseline predictions are not included.

The BQTS experiments intentionally separate benign fitting, benign-only threshold selection, and held-out evaluation. Label-tuned rows are reported as upper-bound references, not as label-free deployment protocols.

The reviewer-requested ablations show that the best corruption and pooling settings are dataset-dependent. The revised manuscript therefore treats Gaussian corruption and edge reconstruction as practical configuration choices/regularizers rather than universal dominance claims.

The MAGIC-style matched-compute ablation uses the repository's masked-autoencoder path to isolate representation and detector effects under a shared codebase. It should be interpreted as a controlled internal ablation rather than as an official MAGIC reproduction.
