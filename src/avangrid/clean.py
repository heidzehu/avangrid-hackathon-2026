"""Data-quality checks, anomaly masking and gap filling for the AMI matrix.

Pipeline (all steps are mechanical and every changed value is counted):

1. ``anomalous_zero_slots``   hours where far more meters than usual for that hour of day
                              read exactly 0 (data glitches / outages). Those zeros become NaN for
                              customers who rarely read 0 (so solar and vacant homes keep
                              their genuine zeros).
2. ``isolated_spikes``        single hours far above the customer's own range and both
                              neighbours (meter catch-up reads). They become NaN.
3. ``fill_gaps``              short gaps (<= 3 h) -> linear interpolation; longer gaps ->
                              the customer's own mean for the same hour and day type
                              (weekday / weekend) within +/- 7 days; anything left ->
                              the customer's mean for that hour and day type over the year.
4. ``customer_quality``       per-customer coverage / fill / zero / spike statistics, with
                              boolean flags the team can use to filter.

Nothing here removes a customer. Low-quality customers are *flagged*, not dropped.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import N_DAYS, N_HOURS


CLEAN_VERSION = 2  # bump when cleaning logic changes so cached cleaned matrices are rebuilt


@dataclass
class CleaningRules:
    # step 1: an hour is anomalous if its share of meters reading 0 is more than BOTH
    # ratio x and +min_excess above the typical share for that hour of day (median over the year).
    # Time-of-day aware (solar raises midday zeros) and independent of how many customers are loaded.
    zero_slot_ratio: float = 2.0
    zero_slot_min_excess: float = 0.01  # 1 percentage point = ~360 meters at once with all 35.7k loaded
    # ...and only customers whose overall zero share is below this get those zeros masked
    zero_customer_max_share: float = 0.05
    # step 2: spike = value > abs_kwh AND > neighbour_ratio x both neighbours
    #         AND > own_ratio x the customer's own 99.5th percentile
    spike_abs_kwh: float = 10.0
    spike_neighbour_ratio: float = 4.0
    spike_own_ratio: float = 3.0
    # step 3
    max_linear_gap_h: int = 3
    template_window_days: int = 7
    # flags
    min_coverage: float = 0.90  # share of hours with an original (unfilled) reading
    mostly_zero_share: float = 0.50


RULES = CleaningRules()


# ----------------------------------------------------------------------- step 1
def _zero_share(K, chunk: int = 4000) -> np.ndarray:
    zeros = np.zeros(N_HOURS)
    valid = np.zeros(N_HOURS)
    for i in range(0, len(K), chunk):
        x = np.asarray(K[i:i + chunk])
        zeros += (x == 0).sum(0)
        valid += (~np.isnan(x)).sum(0)
    return zeros / np.maximum(valid, 1), zeros


def _zero_baseline(share: np.ndarray) -> np.ndarray:
    """Typical zero share for each hour slot = median share at that hour of day."""
    hod = np.arange(N_HOURS) % 24
    med = np.array([np.median(share[hod == h]) for h in range(24)])
    return med[hod]


def anomalous_zero_slots(K: np.ndarray, rules: CleaningRules = RULES, chunk: int = 4000) -> np.ndarray:
    """Boolean [8760]: hours where far more meters than usual for that time of day read exactly 0."""
    share, _ = _zero_share(K, chunk)
    base = _zero_baseline(share)
    return share > np.maximum(rules.zero_slot_ratio * base, base + rules.zero_slot_min_excess)


def zero_slot_table(K: np.ndarray, hours: pd.DataFrame, mask: np.ndarray, chunk: int = 4000) -> pd.DataFrame:
    """Describe each anomalous-zero hour (useful as a reliability / outage insight)."""
    share, zeros = _zero_share(K, chunk)
    base = _zero_baseline(share)
    t = hours.loc[mask, ["slot", "ts"]].copy()
    t["meters_reading_zero"] = zeros[mask].astype(int)
    t["share_reading_zero"] = share[mask].round(4)
    t["typical_share_this_hour"] = base[mask].round(4)
    t["excess_meters_est"] = np.round((share[mask] - base[mask]) * zeros[mask] / np.maximum(share[mask], 1e-9)).astype(int)
    return t.reset_index(drop=True)


# ----------------------------------------------------------------------- step 2
def isolated_spikes(x: np.ndarray, rules: CleaningRules = RULES) -> np.ndarray:
    """Boolean mask, same shape as x [n, 8760]."""
    prev = np.empty_like(x); prev[:, 0] = np.nan; prev[:, 1:] = x[:, :-1]
    nxt = np.empty_like(x); nxt[:, -1] = np.nan; nxt[:, :-1] = x[:, 1:]
    neigh = np.fmax(np.fmax(np.nan_to_num(prev, nan=0.0), np.nan_to_num(nxt, nan=0.0)), 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        own = np.nanpercentile(x, 99.5, axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        return ((x > rules.spike_abs_kwh) & (x > rules.spike_neighbour_ratio * neigh)
                & (x > rules.spike_own_ratio * own))


# ----------------------------------------------------------------------- step 3
def _nan_runs(m: np.ndarray) -> np.ndarray:
    """For a boolean [n, T] mask, return an int array giving the length of the True-run each
    True cell belongs to (0 for False cells)."""
    n, T = m.shape
    pad = np.zeros((n, T + 2), bool)
    pad[:, 1:-1] = m
    flat = pad.ravel()
    d = np.diff(flat.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    lengths = ends - starts
    out = np.zeros(flat.size - 1, dtype=np.int32)
    # mark run length at every cell of the run via a difference array
    acc = np.zeros(flat.size, dtype=np.int64)
    np.add.at(acc, starts + 1, lengths)
    np.add.at(acc, ends + 1, -lengths)
    out = np.cumsum(acc)[: flat.size].reshape(n, T + 2)[:, 1:-1]
    return np.where(m, out, 0).astype(np.int32)


def _window_mean(x3: np.ndarray, day_sel: np.ndarray, w: int) -> np.ndarray:
    """Mean over days d-w..d+w restricted to days where day_sel is True, per (customer, hour).
    x3: [n, 365, 24] with NaN. Returns [n, 365, 24]."""
    v = np.where(np.isnan(x3) | ~day_sel[None, :, None], 0.0, x3).astype(np.float64)
    c = (~np.isnan(x3) & day_sel[None, :, None]).astype(np.float64)
    cv = np.concatenate([np.zeros_like(v[:, :1]), np.cumsum(v, axis=1)], axis=1)
    cc = np.concatenate([np.zeros_like(c[:, :1]), np.cumsum(c, axis=1)], axis=1)
    d = np.arange(x3.shape[1])
    lo = np.clip(d - w, 0, x3.shape[1]); hi = np.clip(d + w + 1, 0, x3.shape[1])
    s = cv[:, hi] - cv[:, lo]
    k = cc[:, hi] - cc[:, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s / k).astype(np.float32)


def fill_gaps(x: np.ndarray, is_weekend_day: np.ndarray, rules: CleaningRules = RULES):
    """Fill NaN in a chunk x [n, 8760]. Returns (filled, method) where method is uint8:
    0 = original, 1 = linear (short gap), 2 = +/-7-day same-hour/day-type template,
    3 = year-long same-hour/day-type mean."""
    n = len(x)
    miss = np.isnan(x)
    run = _nan_runs(miss)
    method = np.zeros(x.shape, np.uint8)
    out = x.copy()
    x3 = x.reshape(n, N_DAYS, 24)
    # long gaps -> local template of the same day type
    long_ = run > rules.max_linear_gap_h
    if long_.any():
        tmpl = np.full_like(x3, np.nan)
        for sel in (is_weekend_day, ~is_weekend_day):
            m = _window_mean(x3, sel, rules.template_window_days)
            tmpl[:, sel, :] = m[:, sel, :]
        t = tmpl.reshape(n, N_HOURS)
        use = long_ & ~np.isnan(t)
        out[use] = t[use]
        method[use] = 2
    # short gaps (and any long-gap cell the template could not reach) -> linear
    still = np.isnan(out)
    short = still & (run <= rules.max_linear_gap_h)
    if short.any():
        rows = np.where(short.any(1))[0]
        df = pd.DataFrame(out[rows].T)
        lin = df.interpolate(method="linear", limit_area="inside", axis=0).bfill().ffill().to_numpy(np.float32).T
        sub = short[rows]
        o = out[rows]; o[sub] = lin[sub]; out[rows] = o
        mm = method[rows]; mm[sub] = 1; method[rows] = mm
    # anything left -> year-long mean of the same hour & day type
    still = np.isnan(out)
    if still.any():
        o3 = out.reshape(n, N_DAYS, 24)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices are expected here
            for sel in (is_weekend_day, ~is_weekend_day):
                prof = np.nanmean(x3[:, sel, :], axis=1)  # [n, 24]
                prof = np.where(np.isnan(prof), np.nanmean(x3.reshape(n, -1), axis=1, keepdims=True), prof)
                blk = o3[:, sel, :]
                o3[:, sel, :] = np.where(np.isnan(blk), prof[:, None, :], blk)
        out = o3.reshape(n, N_HOURS)
        m2 = still & ~np.isnan(out)
        method[m2] = 3
    out = np.nan_to_num(out, nan=0.0)  # customers with literally no data in a day type
    return out.astype(np.float32), method


# ------------------------------------------------------------------- full pass
def clean_matrix(K_raw, hours: pd.DataFrame, out_path, rules: CleaningRules = RULES,
                 chunk: int = 1000, verbose: bool = True):
    """Run steps 1-3 over the (memory-mapped) raw matrix in customer chunks.

    Writes the filled matrix to ``out_path`` (.npy, float32) and returns
    (filled_memmap, per-customer quality DataFrame (unindexed, same row order), zero-slot mask).
    """
    is_weekend_day = (hours.groupby("day")["is_weekend"].first() | hours.groupby("day")["is_holiday"].first()).to_numpy()
    zslots = anomalous_zero_slots(K_raw, rules)
    mid = hours["hour_beginning"].between(10, 15).to_numpy()
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=K_raw.shape)
    recs = []
    for i in range(0, len(K_raw), chunk):
        x = np.array(K_raw[i:i + chunk], dtype=np.float32)
        n = len(x)
        orig_missing = np.isnan(x)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            iszero = np.where(orig_missing, np.nan, (x == 0).astype(np.float32))
            zero_share = np.nanmean(iszero, axis=1)
            zero_share_mid = np.nanmean(iszero[:, mid], axis=1)
            max_raw = np.nanmax(x, axis=1)
        # step 1
        zmask = (x == 0) & zslots[None, :] & (zero_share < rules.zero_customer_max_share)[:, None]
        x[zmask] = np.nan
        # step 2
        smask = isolated_spikes(x, rules)
        x[smask] = np.nan
        # step 3
        filled, method = fill_gaps(x, is_weekend_day, rules)
        out[i:i + n] = filled
        miss = np.isnan(x)
        run = _nan_runs(miss)
        day_missing = miss.reshape(n, N_DAYS, 24).all(axis=2)
        valid_any = ~orig_missing
        first = np.argmax(valid_any, axis=1)
        last = N_HOURS - 1 - np.argmax(valid_any[:, ::-1], axis=1)
        recs.append(pd.DataFrame({
            "hours_original_missing": orig_missing.sum(1),
            "hours_zero_glitch_masked": zmask.sum(1),
            "hours_spike_masked": smask.sum(1),
            "hours_filled_linear": (method == 1).sum(1),
            "hours_filled_template": (method == 2).sum(1),
            "hours_filled_yearmean": (method == 3).sum(1),
            "longest_gap_h": run.max(1),
            "whole_days_missing": day_missing.sum(1),
            "first_reading_slot": first,
            "last_reading_slot": last,
            "zero_share": np.round(zero_share, 4),
            "zero_share_10to16": np.round(zero_share_mid, 4),
            "max_kwh_raw": max_raw,
        }))
        if verbose:
            print(f"  cleaned {i + n:>6,}/{len(K_raw):,}", end="\r")
    if verbose:
        print()
    out.flush()
    q = pd.concat(recs, ignore_index=True)
    q["coverage"] = (1 - (q[["hours_original_missing", "hours_zero_glitch_masked", "hours_spike_masked"]].sum(1)
                          / N_HOURS)).round(4)
    q["filled_share"] = ((q.hours_filled_linear + q.hours_filled_template + q.hours_filled_yearmean) / N_HOURS).round(4)
    q["flag_partial_year"] = (q.first_reading_slot > 14 * 24) | (q.last_reading_slot < N_HOURS - 14 * 24)
    q["flag_low_coverage"] = q.coverage < rules.min_coverage
    q["flag_mostly_zero"] = q.zero_share > rules.mostly_zero_share
    q["analysis_ready"] = ~(q.flag_low_coverage | q.flag_mostly_zero)
    return np.load(out_path, mmap_mode="r"), q, zslots
