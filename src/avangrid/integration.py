"""Join Part 1 classification, Part 2 segmentation, Part 3 optimization.

Descriptive post-optimization attribution only: segment/technology labels do not
change candidate screening, dispatch, or NPV. Never infer an actual appliance
or program enrollment from an AMI-based prediction.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np

CLASS_COLUMNS = ["customer", "f_ev", "f_heat_pump", "f_hp_candidate", "f_solar_pv"]


def _unique_customer(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if "customer" not in frame.columns:
        raise ValueError(f"{label} missing customer ID column 'customer'")
    frame = frame.copy()
    frame["customer"] = pd.to_numeric(frame["customer"], errors="raise")
    if frame["customer"].isna().any() or (frame["customer"] % 1 != 0).any():
        raise ValueError(f"{label} has invalid customer IDs")
    frame["customer"] = frame["customer"].astype("int64")
    if frame.customer.duplicated().any():
        raise ValueError(f"{label} contains duplicate customer IDs")
    return frame


def integrate(classification: pd.DataFrame, segmentation: pd.DataFrame,
              sizing: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return customer-level recommendations, segment aggregates and coverage audit.

    Expects Part 1's data_local/classification.parquet, Part 2's
    data_local/segments.parquet and Part 3's outputs/part3_battery_sizing.csv.
    Preserves missing classifications instead of inventing them.
    """
    cls = _unique_customer(classification, "classification")
    seg = _unique_customer(segmentation, "segmentation")
    missing_cls = set(CLASS_COLUMNS) - set(cls.columns)
    missing_seg = {"segment", "cluster"} - set(seg.columns)
    if missing_cls:
        raise ValueError(f"classification missing {sorted(missing_cls)}")
    if missing_seg:
        raise ValueError(f"segmentation missing {sorted(missing_seg)}")
    if not {"customer_number", "option", "npv_usd", "critical_hour_reduction_kw", "recommended"}.issubset(sizing):
        raise ValueError("sizing CSV is missing required recommendation/economic columns")
    recommended_mask = sizing.recommended.astype(str).str.lower().isin(["true", "1"])
    if sizing.loc[~recommended_mask, "customer_number"].nunique() + sizing.loc[recommended_mask, "customer_number"].nunique() < sizing["customer_number"].nunique():
        raise AssertionError("customer list inconsistent")
    selected_ids = set(sizing.loc[recommended_mask, "customer_number"])
    if selected_ids != set(sizing["customer_number"]):
        raise ValueError("some optimized customers have no recommended option")
    chosen = sizing.loc[recommended_mask].copy()
    if chosen.empty:
        raise ValueError("sizing CSV contains no recommended rows")
    chosen = chosen.rename(columns={"customer_number": "customer"})
    chosen = _unique_customer(chosen, "recommended sizing results")
    original_n = len(chosen)

    # Keep customer-level attributes to enable explanation/stratification;
    # do not use these predictors to change NPV or technology recommendations.
    cls_cols = CLASS_COLUMNS + [c for c in ("ev_conf", "hp_conf", "ev_prob", "hp_prob", "f_ac", "f_elec_heat") if c in cls.columns]
    combined = chosen.merge(cls[cls_cols], on="customer", how="left", validate="one_to_one", indicator="_cls")
    combined = combined.merge(seg[["customer", "segment", "cluster"]], on="customer", how="left", validate="one_to_one", indicator="_seg")
    combined["classification_available"] = combined.pop("_cls").eq("both")
    combined["segmentation_available"] = combined.pop("_seg").eq("both")
    combined["segment"] = combined["segment"].fillna("Unmatched / not classified")
    if len(combined) != original_n:
        raise AssertionError("one row per optimized customer was not preserved")

    summary = combined.groupby("segment", dropna=False).agg(
        customers=("customer", "size"),
        mean_best_npv_usd=("npv_usd", "mean"),
        total_best_npv_usd=("npv_usd", "sum"),
        mean_critical_hour_reduction_kw=("critical_hour_reduction_kw", "mean"),
        sum_customer_critical_reduction_kw=("critical_hour_reduction_kw", "sum"),
        with_classification=("classification_available", "sum"),
    ).reset_index()
    for tech in ("f_ev", "f_heat_pump", "f_hp_candidate", "f_solar_pv"):
        summary[f"{tech}_flag_count"] = combined.groupby("segment")[tech].apply(lambda s: int(s.eq(True).sum())).reindex(summary.segment).to_numpy()
    mix = combined.groupby(["segment", "option"], dropna=False).size().rename("count").reset_index()
    summary = summary.merge(mix.pivot(index="segment", columns="option", values="count").fillna(0).astype(int).add_prefix("n_option_").reset_index(), on="segment", how="left", validate="one_to_one")
    summary = summary.sort_values("customers", ascending=False).reset_index(drop=True)

    audit = {
        "optimized_customers": original_n,
        "classification_matches": int(combined.classification_available.sum()),
        "segmentation_matches": int(combined.segmentation_available.sum()),
        "fully_matched": int((combined.classification_available & combined.segmentation_available).sum()),
        "unmatched_customer_ids": combined.loc[~(combined.classification_available & combined.segmentation_available), "customer"].astype(int).tolist(),
        "classification_source_customers": len(cls),
        "segmentation_source_customers": len(seg),
        "warning": "Labels are predictions / screening flags; the join does not modify optimization or establish feeder reliability.",
    }
    return combined, summary, audit


def run(classification_path: Path, segments_path: Path, sizing_path: Path, output_dir: Path,
        require_complete: bool = False, matrix_customers: Path | None = None) -> dict:
    for path in (classification_path, segments_path, sizing_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run Part 1, Part 2 and Part 3 before integration.")
    cls = pd.read_parquet(classification_path)
    seg = pd.read_parquet(segments_path)
    if matrix_customers is not None:
        if not matrix_customers.is_file():
            raise FileNotFoundError(f"Missing customer matrix ID list: {matrix_customers}")
        expected = set(int(x) for x in np.load(matrix_customers))
        for label, frame in (("classification", cls), ("segmentation", seg)):
            found = set(_unique_customer(frame, label).customer)
            if found != expected:
                raise ValueError(f"{label} does not match the {len(expected)} customer IDs in {matrix_customers}: missing {len(expected - found)}, extra {len(found - expected)}. Rebuild this workflow on the same snapshot.")
    combined, summary, audit = integrate(cls, seg, pd.read_csv(sizing_path))
    if require_complete and audit["fully_matched"] != audit["optimized_customers"]:
        raise ValueError(f"Only {audit['fully_matched']}/{audit['optimized_customers']} optimized customers matched both upstream outputs. Rerun Parts 1 and 2 on the same data snapshot.")
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "part3_integrated_customer_recommendations.csv", index=False)
    summary.to_csv(output_dir / "part3_integrated_segment_summary.csv", index=False)
    (output_dir / "part3_integration_audit.json").write_text(json.dumps(audit, indent=2))
    return audit
