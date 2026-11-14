# Final Part 1 → Part 2 → Part 3 integration

This patch **does not overwrite** previous files. It adds `src/avangrid/integration.py`, `src/13_integrate_workflows.py`, and `tests/test_gridflex_integration.py`.

## Install in the cloned GitHub repository branch

```bash
cd ~/Downloads/avangrid-github
unzip -o ~/Downloads/gridflex-workflows-integration-patch.zip -d .
PYTHONPATH=src python -m pytest tests/test_gridflex_integration.py -q
```

## Run on your actual project

The existing Part 1/2 scripts use a **separate cache** in `data_local/` (from `src/config.py`), whereas GridFlex uses `data/processed/` and `outputs/` (from `avangrid.config`). Therefore, classification and segmentation must be run on a consistent, correctly configured dataset snapshot before the integration can claim end-to-end coverage.

Required existing files in **the cloned GitHub folder**:

- `data_local/classification.parquet`: created by `src/07_classify_rules.py` (after the Part 1 pipeline)
- `data_local/segments.parquet`: created by `src/09_segment.py` (after classification)
- `outputs/part3_battery_sizing.csv`: created by `src/11_gridflex_sizing.py`

Once these exist:

```bash
PYTHONPATH=src python src/13_integrate_workflows.py --require-complete --matrix-customers data/processed/customers.npy
```

If Part 1/2 have *not* been run on the expanded 35,236-customer delivery, **do not interpret stale classifications as current**. Align the data pipelines, rerun Part 1/2, and regenerate Part 3 outputs first. **The old Part 1/2 `src/config.py` uses a separate external `Data/` directory and `data_local/` caches, so copying in the new ZIP or rerunning only `avangrid.pipeline` does NOT refresh their predictions.** Passing integration unit tests does not replace this data check.

The command writes (git-ignored):
- `outputs/part3_integrated_customer_recommendations.csv` (contains customer IDs; do not commit)
- `outputs/part3_integrated_segment_summary.csv` (aggregate)
- `outputs/part3_integration_audit.json` (coverage)

**Scope:** post-optimization explanation and reporting by segment/technology. GridFlex does not automatically change customer selection or dispatch based on classification and segmentation. This is intentional to preserve the independently validated economics. Do not call it an end-to-end decision engine until the cache/data alignment and full sample run are verified.
