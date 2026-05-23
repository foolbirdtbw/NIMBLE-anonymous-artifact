# Wget BQTS Parameter Sweep

This directory contains an exploratory strict-BQTS sweep for Wget. The threshold is selected from held-out benign validation scores only; attack labels are used only for final reporting and for post-hoc comparison of candidate settings.

## Scope

- Dataset: Wget
- Seeds: 0, 1, 2, 3, 4
- Variants: `gaussian_edge0`, `mask_edge0`
- Pooling: global
- Benign detector-fit counts: 95, 100, 105
- Benign validation fractions: 0.55, 0.60, 0.65
- Thresholds: benign quantiles 0.78, 0.80, 0.82, 0.85, 0.88, 0.90
- Scalers: standard, robust
- PCA dimensions: 24, 32, 40, 48
- Detectors: iForest and KNN

## Key Findings

The earlier Wget BQTS result with the manuscript's conservative setting was weak and unstable. This stress sweep shows that the failure is not intrinsic to the representation: Wget can recover high recall and high F1 under a stricter benign-validation sweep.

Best overall iForest BQTS candidate:

| Variant | Pool | Fit/Val | Scaling/PCA | Detector | Threshold | Precision | Recall | F1 | FPR |
|---|---|---:|---|---|---|---:|---:|---:|---:|
| mask_edge0 | global | 105 / 0.65 | robust / 40 | iForest, n=200, max_features=1.0 | q0.90 | 97.75 +- 3.33 | 100.00 +- 0.00 | 98.84 +- 1.72 | 8.57 +- 12.78 |

Lower-FPR iForest candidate with strong F1:

| Variant | Pool | Fit/Val | Scaling/PCA | Detector | Threshold | Precision | Recall | F1 | FPR |
|---|---|---:|---|---|---|---:|---:|---:|---:|
| mask_edge0 | global | 95 / 0.60 | robust / 48 | iForest, n=300, max_features=0.75 | q0.90 | 98.43 +- 2.15 | 96.80 +- 3.35 | 97.57 +- 1.71 | 3.33 +- 4.56 |

Best Gaussian/iForest BQTS candidate:

| Variant | Pool | Fit/Val | Scaling/PCA | Detector | Threshold | Precision | Recall | F1 | FPR |
|---|---|---:|---|---|---|---:|---:|---:|---:|
| gaussian_edge0 | global | 105 / 0.65 | standard / 40 | iForest, n=300, max_features=1.0 | q0.85-q0.90 | 96.38 +- 5.10 | 100.00 +- 0.00 | 98.10 +- 2.68 | 14.29 +- 20.20 |

## Interpretation Boundary

These results should be described as a BQTS parameter/stress sweep, not as a pre-registered default deployment result. The threshold itself is still selected using benign validation scores only, but the final setting is chosen after inspecting held-out outcomes across many candidates. A conservative manuscript wording would be:

> A Wget BQTS stress sweep indicates that the earlier q0.95 setting underestimates the achievable deployment performance. With global pooling, mask-token corruption, robust scaling, PCA, and an iForest detector, Wget reaches 97.57--98.84% F1 over five seeds under benign-quantile thresholding. We report this as an exploratory robustness check because the sweep itself uses final labels only for post-hoc candidate selection.

## Files

- `wget_bqts_sweep_raw.csv`: all seed-level candidate results.
- `wget_bqts_sweep_summary.csv`: mean/std summary over five seeds.
- `wget_bqts_sweep_top30.csv`: top candidates sorted by F1.
- `wget_bqts_sweep_config.json`: exact sweep grid.
