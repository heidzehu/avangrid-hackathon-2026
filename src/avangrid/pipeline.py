"""End-to-end data preparation: raw files -> clean matrix -> feature tables.

Run from the repo root:
    python -m avangrid.pipeline            # uses cached steps where possible
    python -m avangrid.pipeline --rebuild  # re-read everything

Or in a notebook:
    from avangrid.pipeline import run, load_project
    run()                 # once
    D = load_project()    # any time after: everything loaded, matrices memory-mapped
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import clean, features, io_ami, labels, weather
from .config import Paths, hour_index


def run(paths: Paths | None = None, rebuild: bool = False, verbose: bool = True) -> dict:
    P = (paths or Paths()).ensure()
    t0 = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    H = hour_index()

    log("1/6  Reading AMI deliveries ...")
    C, K, src_stats = io_ami.build_raw_matrix(P.raw, P.processed, rebuild=rebuild, verbose=verbose)
    raw_warnings = src_stats.pop("_warnings", [])
    log(f"     {len(C):,} customers x {K.shape[1]:,} hours from deliveries {', '.join(src_stats)}")

    log("2/6  Weather ...")
    wh = weather.load_weather(P.raw)
    wd = weather.daily_weather(wh)
    wh.merge(H[["slot", "ts"]], on="slot").to_csv(P.outputs / "weather_hourly_clean.csv", index=False)
    wd.to_csv(P.outputs / "weather_daily.csv", index=False)

    log("3/6  Cleaning + gap filling ...")
    raw_p, cln_p = P.processed / "kwh_raw.npy", P.processed / "kwh_clean.npy"
    q_p, z_p = P.processed / "quality.parquet", P.processed / "zero_slots.npy"
    meta_p = P.processed / "clean_meta.json"
    fresh = (not rebuild and all(p.exists() for p in (cln_p, q_p, z_p, meta_p))
             and json.loads(meta_p.read_text()).get("version") == clean.CLEAN_VERSION
             and cln_p.stat().st_mtime >= raw_p.stat().st_mtime
             and np.load(cln_p, mmap_mode="r").shape == K.shape)
    if fresh:  # reuse: also avoids rewriting a file another notebook may have open
        F, q, zslots = np.load(cln_p, mmap_mode="r"), pd.read_parquet(q_p), np.load(z_p)
        log("     using cached cleaned matrix (pass rebuild=True to redo)")
    else:
        F, q, zslots = clean.clean_matrix(K, H, cln_p, verbose=verbose)
        q.insert(0, "customer_number", C)
        q.to_parquet(q_p, index=False)
        np.save(z_p, zslots)
        meta_p.write_text(json.dumps({"version": clean.CLEAN_VERSION}))
    clean.zero_slot_table(K, H, zslots).to_csv(P.outputs / "anomalous_zero_hours.csv", index=False)

    log("4/6  System load ...")
    S = features.system_load(F, hours=H)
    S.to_csv(P.outputs / "system_load_hourly.csv", index=False)

    log("5/6  Features ...")
    feat = features.build_features(F, C, H, wh, wd, S, verbose=verbose)
    refs = labels.load_reference_customers(P.raw)
    lab = labels.label_columns(C, refs)
    table = lab.merge(q, on="customer_number").merge(feat, on="customer_number")
    table.to_parquet(P.outputs / "customer_features.parquet", index=False)
    table.to_csv(P.outputs / "customer_features.csv", index=False)
    fd = features.feature_dictionary(table)
    qdocs = {
        "ref_ev": ("bool", "On Avangrid's EV reference list (random ~25% sample of likely EV customers)"),
        "ref_hp": ("bool", "On the heat-pump reference list (random ~25% sample)"),
        "ref_pv": ("bool", "On the solar PV list (complete list)"),
        "hours_original_missing": ("hours", "Hours with no row in the AMI files"),
        "hours_zero_glitch_masked": ("hours", "Zero readings removed because they fell in a system-wide zero glitch/outage hour"),
        "hours_spike_masked": ("hours", "Isolated spikes removed (likely meter catch-up reads)"),
        "hours_filled_linear": ("hours", "Filled by linear interpolation (gaps <= 3 h)"),
        "hours_filled_template": ("hours", "Filled from the same hour/day type within +/-7 days"),
        "hours_filled_yearmean": ("hours", "Filled from the year-long same hour/day type mean"),
        "longest_gap_h": ("hours", "Longest run of missing hours after masking"),
        "whole_days_missing": ("days", "Days with no usable reading at all"),
        "first_reading_slot": ("slot", "First hour with data (0 = 9/1/2025 00:00)"),
        "last_reading_slot": ("slot", "Last hour with data (8759 = 8/31/2026 23:00)"),
        "zero_share": ("share", "Share of hours reading exactly 0 (before cleaning)"),
        "zero_share_10to16": ("share", "Share of hours 10-15 reading exactly 0 (before cleaning)"),
        "max_kwh_raw": ("kWh", "Largest raw hourly reading (before spike masking)"),
        "coverage": ("share", "Share of the 8760 hours with a usable original reading"),
        "filled_share": ("share", "Share of hours that were filled"),
        "flag_partial_year": ("bool", "Data starts >14 days after 9/1 or ends >14 days before 8/31 (move-in/out)"),
        "flag_low_coverage": ("bool", "coverage < 0.90"),
        "flag_mostly_zero": ("bool", "More than half of hours read 0 (vacant, disconnected or offset by solar)"),
        "analysis_ready": ("bool", "Not low coverage and not mostly zero"),
    }
    for k, (u, d) in qdocs.items():
        fd.loc[fd.feature == k, ["unit", "definition"]] = [u, d]
    fd.to_csv(P.outputs / "feature_dictionary.csv", index=False)

    log("6/6  Profile tables ...")
    features.seasonal_profiles(F, C, H).to_parquet(P.outputs / "seasonal_profiles.parquet", index=False)
    features.daily_kwh(F, C).to_parquet(P.outputs / "daily_kwh.parquet", index=False)

    summary = {
        "deliveries": {k: {kk: v[kk] for kk in ("rows", "customers_expected", "customers_present", "seconds")}
                       for k, v in src_stats.items()},
        "warnings": raw_warnings,
        "missing_customer_numbers": sum((v["customers_missing"] for v in src_stats.values()), []),
        "n_customers": int(len(C)),
        "weather_hours_filled": {c: int(wh[c].sum()) for c in wh if c.endswith("_filled")},
        "anomalous_zero_hours": int(zslots.sum()),
        "flags": {c: int(q[c].sum()) for c in q if c.startswith("flag_") or c == "analysis_ready"},
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (P.outputs / "run_summary.json").write_text(json.dumps(summary, indent=1))
    log(f"Done in {summary['minutes']} min. Outputs in {P.outputs}")
    for w in raw_warnings:
        log("WARNING:", w)
    return summary


@dataclass
class Project:
    """Everything the analysis needs, loaded in one call. Matrices are memory-mapped
    (float32, rows = customers in ``customers`` order, columns = 8760 hour slots)."""

    paths: Paths
    customers: np.ndarray
    kwh_raw: np.ndarray  # NaN = missing
    kwh: np.ndarray  # cleaned + gap-filled (use this one)
    hours: pd.DataFrame
    weather: pd.DataFrame
    weather_daily: pd.DataFrame
    system: pd.DataFrame
    table: pd.DataFrame  # labels + quality + features, one row per customer
    feature_dictionary: pd.DataFrame
    refs: dict

    def row(self, customer_number: int) -> int:
        """Matrix row index of a customer number."""
        i = np.searchsorted(self.customers, customer_number)
        if i >= len(self.customers) or self.customers[i] != customer_number:
            raise KeyError(customer_number)
        return int(i)

    def series(self, customer_number: int, raw: bool = False) -> np.ndarray:
        return np.asarray((self.kwh_raw if raw else self.kwh)[self.row(customer_number)])

    def matrix(self, customer_numbers) -> np.ndarray:
        idx = np.sort([self.row(c) for c in customer_numbers])
        return np.asarray(self.kwh[idx])


def load_project(paths: Paths | None = None) -> Project:
    P = paths or Paths()
    H = hour_index()
    wh = pd.read_csv(P.outputs / "weather_hourly_clean.csv", parse_dates=["ts"])
    return Project(
        paths=P,
        customers=np.load(P.processed / "customers.npy"),
        kwh_raw=np.load(P.processed / "kwh_raw.npy", mmap_mode="r"),
        kwh=np.load(P.processed / "kwh_clean.npy", mmap_mode="r"),
        hours=H,
        weather=wh,
        weather_daily=pd.read_csv(P.outputs / "weather_daily.csv", parse_dates=["date"]),
        system=pd.read_csv(P.outputs / "system_load_hourly.csv", parse_dates=["ts", "date"]),
        table=pd.read_parquet(P.outputs / "customer_features.parquet"),
        feature_dictionary=pd.read_csv(P.outputs / "feature_dictionary.csv"),
        refs=labels.load_reference_customers(P.raw),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true", help="ignore caches and re-read the raw zips")
    ap.add_argument("--root", type=Path, default=None, help="repo root (default: this repo)")
    a = ap.parse_args()
    run(Paths(a.root) if a.root else None, rebuild=a.rebuild)
