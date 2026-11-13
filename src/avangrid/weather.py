"""Load, clean and align the hourly weather file to the 8760 AMI hour slots.

Alignment: weather TIME is the start of the hour, so TIME 0:00 is matched to AMI
HOUR_ENDING 1 (12:00-1:00 AM), i.e. ``slot = day * 24 + hour(TIME)``.

Issues found in Weather_Data_Avangrid_Hackathon.xlsx and how they are handled:
  * 11/2/2025 01:00 appears twice (DST fall-back) -> the two readings are averaged.
  * 3/8/2026 00:00 - 3/11/2026 ~22:00: ~94 consecutive hours missing (all three variables).
    Long gaps are filled with a "trend + diurnal shape" estimate: the daily mean is linearly
    interpolated between the last complete day before and the first complete day after the
    gap, plus the average hour-of-day deviation from the 7 days either side.
  * Short gaps (<= 3 h) are linearly interpolated.
Every filled hour is flagged in ``<var>_filled``.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import N_DAYS, N_HOURS, START_DATE

VARS = ["temp_f", "humidity_pct", "wind_mph"]


def find_weather_file(raw_dir: Path) -> Path:
    cands = [p for p in Path(raw_dir).glob("*.xlsx") if "weather" in p.name.lower()]
    cands += [p for p in Path(raw_dir).glob("*.csv") if "weather" in p.name.lower()]
    if not cands:
        raise FileNotFoundError(f"No weather file (name containing 'weather') in {raw_dir}")
    return sorted(cands)[0]


def _read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    date = pd.to_datetime(df["date"])
    t = df["time"].astype(str).str.strip()
    hour = pd.to_datetime(t, format="mixed").dt.hour if not t.str.match(r"^\d{1,2}$").all() else t.astype(int)
    out = pd.DataFrame({
        "slot": ((date.dt.normalize() - START_DATE).dt.days * 24 + hour).astype(int),
        "temp_f": pd.to_numeric(df["temp"], errors="coerce"),
        "humidity_pct": pd.to_numeric(df["humidity"], errors="coerce"),
        "wind_mph": pd.to_numeric(df["windspeed"], errors="coerce"),
    })
    return out


def _fill_long(s: np.ndarray, gap_mask: np.ndarray, w: int = 7) -> np.ndarray:
    """Trend + diurnal-shape fill for one variable (arrays of length 8760)."""
    x = s.reshape(N_DAYS, 24)
    g = gap_mask.reshape(N_DAYS, 24)
    out = x.copy()
    gap_days = np.where(g.any(1))[0]
    if len(gap_days) == 0:
        return s
    complete = ~np.isnan(x).any(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        daily = np.where(complete, np.nanmean(x, 1), np.nan)
    # group consecutive gap days into blocks
    blocks = np.split(gap_days, np.where(np.diff(gap_days) > 1)[0] + 1)
    for b in blocks:
        before = [d for d in range(b[0] - 1, -1, -1) if complete[d]][:1]
        after = [d for d in range(b[-1] + 1, N_DAYS) if complete[d]][:1]
        anchors = before + after
        ctx = [d for d in range(max(0, b[0] - w), min(N_DAYS, b[-1] + w + 1)) if complete[d] and d not in b]
        shape = np.nanmean(x[ctx] - daily[ctx, None], axis=0) if ctx else np.zeros(24)
        if len(anchors) == 2:
            dm = np.interp(b, anchors, daily[anchors])
        else:
            dm = np.full(len(b), daily[anchors[0]] if anchors else np.nanmean(daily))
        est = dm[:, None] + shape[None, :]
        out[b] = np.where(g[b], est, x[b])
    return out.ravel()


def load_weather(raw_dir: Path, max_linear_gap_h: int = 3) -> pd.DataFrame:
    """Hourly weather on the 8760-slot grid with fill flags. Columns: slot, temp_f,
    humidity_pct, wind_mph, <var>_filled, plus duplicate-row count in ``.attrs``."""
    raw = _read_raw(find_weather_file(raw_dir))
    n_dup = int(raw["slot"].duplicated().sum())
    raw = raw[(raw.slot >= 0) & (raw.slot < N_HOURS)].groupby("slot", as_index=True)[VARS].mean()
    w = pd.DataFrame(index=pd.RangeIndex(N_HOURS, name="slot")).join(raw)
    for v in VARS:
        s = w[v].to_numpy(float)
        miss = np.isnan(s)
        # run lengths
        idx = np.flatnonzero(np.diff(np.r_[0, miss.astype(int), 0]))
        runs = idx.reshape(-1, 2)
        long_mask = np.zeros(N_HOURS, bool)
        for a, b in runs:
            if b - a > max_linear_gap_h:
                long_mask[a:b] = True
        s2 = _fill_long(s, long_mask)
        s2 = pd.Series(s2).interpolate(limit_area="inside").bfill().ffill().to_numpy()
        w[v] = s2
        w[f"{v}_filled"] = miss
    w = w.reset_index()
    w.attrs["duplicate_rows_averaged"] = n_dup
    return w


def daily_weather(w: pd.DataFrame, hdd_base: float = 65.0, cdd_base: float = 65.0) -> pd.DataFrame:
    """Daily mean/min/max and degree days (from the daily mean temperature)."""
    d = w.assign(day=w["slot"] // 24).groupby("day").agg(
        temp_mean=("temp_f", "mean"), temp_min=("temp_f", "min"), temp_max=("temp_f", "max"),
        humidity_mean=("humidity_pct", "mean"), wind_mean=("wind_mph", "mean"),
        hours_filled=("temp_f_filled", "sum"))
    d["date"] = START_DATE + pd.to_timedelta(d.index, unit="D")
    d["hdd65"] = np.maximum(hdd_base - d["temp_mean"], 0)
    d["cdd65"] = np.maximum(d["temp_mean"] - cdd_base, 0)
    return d.reset_index()
