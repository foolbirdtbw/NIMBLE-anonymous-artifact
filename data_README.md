# Dataset Placement

Raw and processed provenance datasets are not redistributed in this anonymous artifact.

Place processed datasets under `data/` at the repository root before running full experiments:

```text
data/
├── streamspot/
├── wget/
└── trace/
```

DARPA E3 Trace should contain `metadata.json`, `train*.pkl`, `test*.pkl`, and `malicious.pkl`. StreamSpot and Unicorn Wget should follow the processed graph/batch format used by `nimble_core/utils/loaddata.py`.
