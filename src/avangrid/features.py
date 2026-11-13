"""Neutral, descriptive per-customer features built from the cleaned hourly matrix.

These describe *how* each customer uses energy. They deliberately contain no appliance
rules, scores or segment assignments; those are the team's analytical work.

Main outputs
  * ``build_features``      -> one row per customer (see FEATURE_DOCS for every column)
  * ``seasonal_profiles``   -> mean kW by season x day type x hour (192 columns per customer)
  * ``daily_kwh``           -> 365 daily totals per customer
  * ``system_load``         -> hourly sum over all loaded customers + top-hour ranks

Units: one hour of kWh == average kW over that hour, so hourly values are reported as kW.
Hours are "hour beginning" 0-23 (AMI HOUR_ENDING 1 == hour 0).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import N_DAYS, N_HOURS, SEASONS

TEMP_BIN_EDGES = [-np.inf, 10, 20, 30, 40, 50, 60, 70, 80, np.inf]
TEMP_BIN_NAMES = ["lt10", "10_20", "20_30", "30_40", "40_50", "50_60", "60_70", "70_80", "ge80"]
PERIODS = {"night_0to6": range(0, 6), "morning_6to10": range(6, 10), "midday_10to16": range(10, 16),
           "evening_16to21": range(16, 21), "late_21to24": range(21, 24)}
RAMP_THRESHOLDS = (2, 4, 6)
LEVEL_THRESHOLDS = (3, 5, 7, 9)

FEATURE_DOCS: dict[str, tuple[str, str]] = {
    "customer_number": ("id", "Anonymous customer identifier (joins every table)"),
    # size
    "annual_kwh": ("kWh", "Total energy over the 365 days (cleaned and gap-filled)"),
    "mean_kw": ("kW", "Mean hourly demand"),
    "median_kw": ("kW", "Median hourly demand"),
    "p95_kw": ("kW", "95th percentile hourly demand"),
    "p99_kw": ("kW", "99th percentile hourly demand"),
    "max_kw": ("kW", "Maximum hourly demand"),
    "load_factor": ("ratio", "mean_kw / max_kw"),
    "load_factor_p99": ("ratio", "mean_kw / p99_kw (less sensitive to one-off peaks)"),
    "base_kw": ("kW", "10th percentile of hours 0-5 (overnight always-on load)"),
    "min_kw_p1": ("kW", "1st percentile hourly demand"),
    "daily_kwh_mean": ("kWh/day", "Mean daily energy"),
    "daily_kwh_p95": ("kWh/day", "95th percentile daily energy"),
    "daily_kwh_max": ("kWh/day", "Maximum daily energy"),
    "daily_kwh_cv": ("ratio", "Std / mean of daily energy"),
    # time of day / week
    **{f"share_{k}": ("share", f"Share of annual energy used in hours {k.split('_', 1)[1].replace('to', '-')}")
       for k in PERIODS},
    "weekend_weekday_ratio": ("ratio", "Mean daily kWh on weekends+holidays / on weekdays"),
    "midday_to_shoulder_ratio": ("ratio", "Mean kW hours 10-15 / mean kW hours 7-9 and 16-19"),
    **{f"peak_hour_{s}": ("hour 0-23", f"Hour with the highest average demand in {s}") for s in SEASONS},
    "autocorr_24h": ("corr", "Correlation of the hourly series with itself 24 h later (daily routine)"),
    "autocorr_168h": ("corr", "Correlation with itself 168 h later (weekly routine)"),
    # season
    **{f"daily_kwh_{s}": ("kWh/day", f"Mean daily energy in {s} (Dec-Feb winter, Mar-May spring, Jun-Aug summer, Sep-Nov fall)") for s in SEASONS},
    "summer_winter_ratio": ("ratio", "daily_kwh_summer / daily_kwh_winter"),
    "monthly_kwh_cv": ("ratio", "Std / mean of the 12 monthly totals"),
    "min_month_share": ("share", "Smallest monthly total / mean monthly total"),
    # weather
    "cp_model": ("label", "Best change-point model of daily kWh vs daily mean temp: const / heat / cool / heat+cool (lowest BIC, slopes >= 0)"),
    "cp_base_kwh_day": ("kWh/day", "Change-point intercept: daily energy not explained by temperature"),
    "cp_heat_slope": ("kWh/day/°F", "Extra daily kWh per °F the daily mean is below cp_heat_balance_f"),
    "cp_heat_balance_f": ("°F", "Heating balance point (daily mean temp where heating load starts)"),
    "cp_cool_slope": ("kWh/day/°F", "Extra daily kWh per °F the daily mean is above cp_cool_balance_f"),
    "cp_cool_balance_f": ("°F", "Cooling balance point"),
    "cp_r2": ("R²", "R² of the chosen change-point model"),
    "corr_daily_hdd": ("corr", "Correlation of daily kWh with HDD65"),
    "corr_daily_cdd": ("corr", "Correlation of daily kWh with CDD65"),
    **{f"kw_temp_{b}": ("kW", f"Mean hourly demand in hours with outdoor temp {b.replace('lt', '< ').replace('ge', '>= ').replace('_', '-')} °F") for b in TEMP_BIN_NAMES},
    "cold10_to_mild_ratio": ("ratio", "Mean daily kWh on the 10 coldest days / on mild days (daily mean 55-65 °F)"),
    "hot10_to_mild_ratio": ("ratio", "Mean daily kWh on the 10 hottest days / on mild days"),
    # dynamics
    **{f"n_upramps_ge{t}kw": ("hours/yr", f"Hours where demand rose by >= {t} kW vs the previous hour") for t in RAMP_THRESHOLDS},
    "max_upramp_kw": ("kW", "Largest hour-to-hour increase"),
    "p99_abs_ramp_kw": ("kW", "99th percentile of |hour-to-hour change|"),
    **{f"hours_ge{t}kw": ("hours/yr", f"Hours with demand >= {t} kW") for t in LEVEL_THRESHOLDS},
    # zeros (solar shows up as midday zeros: the meter records grid imports only)
    "zero_share_summer_10to16": ("share", "Share of summer hours 10-15 reading exactly 0"),
    "zero_share_winter_10to16": ("share", "Share of winter hours 10-15 reading exactly 0"),
    "days_with_midday_zero_share": ("share", "Share of days with at least one 0 reading in hours 10-15"),
    # system coincidence
    "kw_sys_top100": ("kW", "Mean demand during the 100 highest system-load hours"),
    "kw_sys_top50_winter": ("kW", "Mean demand during the 50 highest system hours in Dec-Feb"),
    "kw_sys_top50_summer": ("kW", "Mean demand during the 50 highest system hours in Jun-Aug"),
    "coincidence_factor": ("ratio", "kw_sys_top100 / max_kw"),
    "share_of_sys_top100": ("share", "Customer's energy in the top-100 system hours / system energy in those hours"),
}


def _hour_arrays(hours: pd.DataFrame) -> dict:
    hb = hours["hour_beginning"].to_numpy()
    season = hours["season"].to_numpy()
    wkend = (hours["is_weekend"] | hours["is_holiday"]).to_numpy()
    day_season = season.reshape(N_DAYS, 24)[:, 0]
    day_wkend = wkend.reshape(N_DAYS, 24)[:, 0]
    return dict(hb=hb, season=season, wkend=wkend, month=hours["month"].to_numpy(),
                ym=hours["ts"].dt.strftime("%Y_%m").to_numpy(),
                day_season=day_season, day_wkend=day_wkend)


# ----------------------------------------------------------------- system load
def system_load(F, chunk: int = 4000, hours: pd.DataFrame | None = None) -> pd.DataFrame:
    """Hourly total demand of every loaded customer (kW) and its rank."""
    s = np.zeros(N_HOURS)
    for i in range(0, len(F), chunk):
        s += np.asarray(F[i:i + chunk], dtype=np.float64).sum(0)
    df = pd.DataFrame({"slot": np.arange(N_HOURS), "system_kw": s})
    df["rank"] = df["system_kw"].rank(ascending=False, method="first").astype(int)
    if hours is not None:
        df = hours.merge(df, on="slot")
        for sname in ("winter", "summer"):
            m = df["season"] == sname
            df[f"rank_{sname}"] = np.where(m, df["system_kw"].where(m).rank(ascending=False, method="first"), np.nan)
    return df


# --------------------------------------------------------- change-point model
def changepoint_fit(Y: np.ndarray, T: np.ndarray,
                    heat_grid=np.arange(40, 69, 2), cool_grid=np.arange(56, 77, 2)) -> pd.DataFrame:
    """Vectorised PRISM-style change-point regression for many customers at once.

    Y: [n, D] daily kWh, T: [D] daily mean temperature (°F). Candidate models:
      const:      E = b0
      heat:       E = b0 + bh * max(Th - T, 0)
      cool:       E = b0 + bc * max(T - Tc, 0)
      heat+cool:  E = b0 + bh * max(Th - T, 0) + bc * max(T - Tc, 0),  Th <= Tc
    over a grid of balance points; for each customer the model with the lowest BIC whose
    slopes are all >= 0 is kept (breakpoints count as parameters).
    """
    n, D = Y.shape
    Yt = Y.T.astype(np.float64)
    yy = (Yt ** 2).sum(0)
    sst = ((Yt - Yt.mean(0)) ** 2).sum(0)
    ones = np.ones(D)
    models = [("const", None, None)]
    models += [("heat", th, None) for th in heat_grid]
    models += [("cool", None, tc) for tc in cool_grid]
    models += [("heat+cool", th, tc) for th in heat_grid for tc in cool_grid if tc >= th]
    best_bic = np.full(n, np.inf)
    out = {k: np.full(n, np.nan) for k in ("b0", "hs", "hb", "cs", "cb", "sse")}
    kind = np.array(["const"] * n, dtype=object)
    for name, th, tc in models:
        cols = [ones]
        if th is not None:
            cols.append(np.maximum(th - T, 0))
        if tc is not None:
            cols.append(np.maximum(T - tc, 0))
        X = np.column_stack(cols)
        if np.linalg.matrix_rank(X) < X.shape[1]:
            continue
        XtY = X.T @ Yt
        beta = np.linalg.solve(X.T @ X, XtY)
        sse = np.maximum(yy - (beta * XtY).sum(0), 1e-9)
        valid = (beta[1:] >= 0).all(0) if X.shape[1] > 1 else np.ones(n, bool)
        k = X.shape[1] + (th is not None) + (tc is not None)
        bic = D * np.log(sse / D) + k * np.log(D)
        upd = valid & (bic < best_bic)
        if not upd.any():
            continue
        best_bic[upd] = bic[upd]
        kind[upd] = name
        out["b0"][upd] = beta[0, upd]
        out["sse"][upd] = sse[upd]
        j = 1
        if th is not None:
            out["hs"][upd] = beta[j, upd]; out["hb"][upd] = th; j += 1
        else:
            out["hs"][upd] = 0.0; out["hb"][upd] = np.nan
        if tc is not None:
            out["cs"][upd] = beta[j, upd]; out["cb"][upd] = tc
        else:
            out["cs"][upd] = 0.0; out["cb"][upd] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.where(sst > 0, 1 - out["sse"] / sst, np.nan)
    return pd.DataFrame({"cp_model": kind, "cp_base_kwh_day": out["b0"], "cp_heat_slope": out["hs"],
                         "cp_heat_balance_f": out["hb"], "cp_cool_slope": out["cs"],
                         "cp_cool_balance_f": out["cb"], "cp_r2": r2})


def _rowcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(1) / den, np.nan)


def _corr_with(Y: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _rowcorr(Y, np.broadcast_to(v, Y.shape).copy())


# ------------------------------------------------------------------ features
def _chunk_features(x: np.ndarray, A: dict, wh: pd.DataFrame, wd: pd.DataFrame, sysinfo: dict) -> pd.DataFrame:
    n = len(x)
    f: dict[str, np.ndarray] = {}
    x3 = x.reshape(n, N_DAYS, 24)
    daily = x3.sum(2)
    # size
    f["annual_kwh"] = x.sum(1)
    f["mean_kw"] = x.mean(1)
    q = np.percentile(x, [1, 50, 95, 99], axis=1)
    f["min_kw_p1"], f["median_kw"], f["p95_kw"], f["p99_kw"] = q
    f["max_kw"] = x.max(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f["load_factor"] = f["mean_kw"] / f["max_kw"]
        f["load_factor_p99"] = f["mean_kw"] / f["p99_kw"]
    f["base_kw"] = np.percentile(x[:, A["hb"] < 6], 10, axis=1)
    f["daily_kwh_mean"] = daily.mean(1)
    f["daily_kwh_p95"] = np.percentile(daily, 95, axis=1)
    f["daily_kwh_max"] = daily.max(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f["daily_kwh_cv"] = daily.std(1) / f["daily_kwh_mean"]
    # time of day / week
    tot = np.maximum(f["annual_kwh"], 1e-9)
    for k, hrs in PERIODS.items():
        f[f"share_{k}"] = x[:, np.isin(A["hb"], list(hrs))].sum(1) / tot
    with np.errstate(invalid="ignore", divide="ignore"):
        f["weekend_weekday_ratio"] = daily[:, A["day_wkend"]].mean(1) / daily[:, ~A["day_wkend"]].mean(1)
        mid = np.isin(A["hb"], range(10, 16)); sh = np.isin(A["hb"], [7, 8, 9, 16, 17, 18, 19])
        f["midday_to_shoulder_ratio"] = x[:, mid].mean(1) / x[:, sh].mean(1)
    for s in SEASONS:
        m = A["season"] == s
        prof = np.stack([x[:, m & (A["hb"] == h)].mean(1) for h in range(24)], 1)
        f[f"peak_hour_{s}"] = prof.argmax(1)
        f[f"daily_kwh_{s}"] = daily[:, A["day_season"] == s].mean(1)
    f["autocorr_24h"] = _rowcorr(x[:, :-24], x[:, 24:])
    f["autocorr_168h"] = _rowcorr(x[:, :-168], x[:, 168:])
    with np.errstate(invalid="ignore", divide="ignore"):
        f["summer_winter_ratio"] = f["daily_kwh_summer"] / f["daily_kwh_winter"]
    months = pd.unique(A["ym"])
    mk = np.stack([x[:, A["ym"] == m].sum(1) for m in months], 1)
    for j, m in enumerate(months):
        f[f"kwh_{m}"] = mk[:, j]
    with np.errstate(invalid="ignore", divide="ignore"):
        f["monthly_kwh_cv"] = mk.std(1) / mk.mean(1)
        f["min_month_share"] = mk.min(1) / mk.mean(1)
    # weather response
    cp = changepoint_fit(daily, wd["temp_mean"].to_numpy())
    f["corr_daily_hdd"] = _corr_with(daily, wd["hdd65"].to_numpy())
    f["corr_daily_cdd"] = _corr_with(daily, wd["cdd65"].to_numpy())
    tb = pd.cut(wh["temp_f"], TEMP_BIN_EDGES, labels=TEMP_BIN_NAMES, right=False).to_numpy()
    for b in TEMP_BIN_NAMES:
        m = tb == b
        f[f"kw_temp_{b}"] = x[:, m].mean(1) if m.any() else np.full(n, np.nan)
    tm = wd["temp_mean"].to_numpy()
    cold = np.argsort(tm)[:10]; hot = np.argsort(-tm)[:10]; mild = (tm >= 55) & (tm <= 65)
    with np.errstate(invalid="ignore", divide="ignore"):
        f["cold10_to_mild_ratio"] = daily[:, cold].mean(1) / daily[:, mild].mean(1)
        f["hot10_to_mild_ratio"] = daily[:, hot].mean(1) / daily[:, mild].mean(1)
    # dynamics
    d = np.diff(x, axis=1)
    for t in RAMP_THRESHOLDS:
        f[f"n_upramps_ge{t}kw"] = (d >= t).sum(1)
    f["max_upramp_kw"] = d.max(1)
    f["p99_abs_ramp_kw"] = np.percentile(np.abs(d), 99, axis=1)
    for t in LEVEL_THRESHOLDS:
        f[f"hours_ge{t}kw"] = (x >= t).sum(1)
    # zeros
    mid = np.isin(A["hb"], range(10, 16))
    f["zero_share_summer_10to16"] = (x[:, mid & (A["season"] == "summer")] == 0).mean(1)
    f["zero_share_winter_10to16"] = (x[:, mid & (A["season"] == "winter")] == 0).mean(1)
    f["days_with_midday_zero_share"] = (x3[:, :, 10:16] == 0).any(2).mean(1)
    # system coincidence
    f["kw_sys_top100"] = x[:, sysinfo["top100"]].mean(1)
    f["kw_sys_top50_winter"] = x[:, sysinfo["top50_winter"]].mean(1)
    f["kw_sys_top50_summer"] = x[:, sysinfo["top50_summer"]].mean(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f["coincidence_factor"] = f["kw_sys_top100"] / f["max_kw"]
    f["share_of_sys_top100"] = x[:, sysinfo["top100"]].sum(1) / sysinfo["sys_top100_kwh"]
    out = pd.DataFrame(f)
    return pd.concat([out, cp], axis=1)


def build_features(F, customers: np.ndarray, hours: pd.DataFrame, weather_hourly: pd.DataFrame,
                   weather_daily: pd.DataFrame, sysload: pd.DataFrame, chunk: int = 2000,
                   verbose: bool = True) -> pd.DataFrame:
    """Feature table for every customer in the cleaned matrix ``F`` (memory-mapped OK)."""
    A = _hour_arrays(hours)
    sysinfo = {
        "top100": np.sort(sysload.nsmallest(100, "rank")["slot"].to_numpy()),
        "top50_winter": np.sort(sysload.nsmallest(50, "rank_winter")["slot"].to_numpy()),
        "top50_summer": np.sort(sysload.nsmallest(50, "rank_summer")["slot"].to_numpy()),
    }
    sysinfo["sys_top100_kwh"] = sysload.set_index("slot").loc[sysinfo["top100"], "system_kw"].sum()
    parts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(0, len(F), chunk):
            x = np.asarray(F[i:i + chunk], dtype=np.float32)
            parts.append(_chunk_features(x, A, weather_hourly, weather_daily, sysinfo))
            if verbose:
                print(f"  features {i + len(x):>6,}/{len(F):,}", end="\r")
    if verbose:
        print()
    feat = pd.concat(parts, ignore_index=True)
    num = feat.select_dtypes("number").columns
    feat[num] = feat[num].replace([np.inf, -np.inf], np.nan)  # ratios for all-zero customers
    feat.insert(0, "customer_number", customers)
    return feat


def feature_dictionary(feat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in feat.columns:
        if c in FEATURE_DOCS:
            u, d = FEATURE_DOCS[c]
        elif c.startswith("kwh_20"):
            u, d = "kWh", f"Energy in month {c[4:].replace('_', '-')}"
        else:
            u, d = "", ""
        rows.append({"feature": c, "unit": u, "definition": d})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- profile tables
def seasonal_profiles(F, customers: np.ndarray, hours: pd.DataFrame, chunk: int = 4000) -> pd.DataFrame:
    """Mean kW by season x day type (wd = weekday, we = weekend/holiday) x hour:
    columns like ``winter_wd_h17``. Handy input for clustering / plotting."""
    A = _hour_arrays(hours)
    cols, parts = [], []
    keys = [(s, dt, h) for s in SEASONS for dt in ("wd", "we") for h in range(24)]
    masks = [(A["season"] == s) & ((A["wkend"]) if dt == "we" else (~A["wkend"])) & (A["hb"] == h) for s, dt, h in keys]
    cols = [f"{s}_{dt}_h{h:02d}" for s, dt, h in keys]
    for i in range(0, len(F), chunk):
        x = np.asarray(F[i:i + chunk], dtype=np.float32)
        parts.append(np.stack([x[:, m].mean(1) for m in masks], 1))
    df = pd.DataFrame(np.vstack(parts), columns=cols)
    df.insert(0, "customer_number", customers)
    return df


def daily_kwh(F, customers: np.ndarray, chunk: int = 4000) -> pd.DataFrame:
    from .config import START_DATE
    parts = [np.asarray(F[i:i + chunk], dtype=np.float32).reshape(-1, N_DAYS, 24).sum(2) for i in range(0, len(F), chunk)]
    cols = (START_DATE + pd.to_timedelta(np.arange(N_DAYS), unit="D")).strftime("%Y-%m-%d")
    df = pd.DataFrame(np.vstack(parts), columns=cols)
    df.insert(0, "customer_number", customers)
    return df


def runs_above(x: np.ndarray, threshold_kw: float, min_hours: int = 1) -> pd.DataFrame:
    """Utility (not used in the feature table): every run of consecutive hours with demand
    >= threshold for ONE customer's 8760 series. Returns start slot, length, mean kW."""
    m = np.r_[False, x >= threshold_kw, False]
    d = np.diff(m.astype(int))
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    keep = (en - st) >= min_hours
    st, en = st[keep], en[keep]
    return pd.DataFrame({"start_slot": st, "hours": en - st,
                         "mean_kw": [x[a:b].mean() for a, b in zip(st, en)]})
