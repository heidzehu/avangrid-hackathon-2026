"""Integrate Parts 1, 2 and 3 by customer ID without changing GridFlex decisions.

Run from repository root AFTER classification, segmentation and battery sizing:
PYTHONPATH=src python src/13_integrate_workflows.py
"""
import argparse
from pathlib import Path
from avangrid.config import Paths
from avangrid.integration import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", type=Path, default=Path("data_local/classification.parquet"))
    ap.add_argument("--segments", type=Path, default=Path("data_local/segments.parquet"))
    ap.add_argument("--sizing", type=Path, default=Path("outputs/part3_battery_sizing.csv"))
    ap.add_argument("--require-complete", action="store_true", help="Fail when any optimized customer lacks Part 1 or 2 outputs")
    ap.add_argument("--matrix-customers", type=Path, help="Check Parts 1/2 IDs against data/processed/customers.npy for full snapshot alignment")
    args = ap.parse_args()
    audit = run(args.classification, args.segments, args.sizing, Paths().ensure().outputs, args.require_complete, args.matrix_customers)
    print(f"Integrated {audit['optimized_customers']} optimized customers")
    print(f"Part 1 matched: {audit['classification_matches']}; Part 2 matched: {audit['segmentation_matches']}; both: {audit['fully_matched']}")
    print("Wrote outputs/part3_integrated_customer_recommendations.csv, part3_integrated_segment_summary.csv, part3_integration_audit.json")
    if audit['fully_matched'] < audit['optimized_customers']:
        print("WARNING: upstream data not fully aligned; rerun Parts 1 and 2 on the current dataset before calling this end-to-end validated.")


if __name__ == "__main__":
    main()
