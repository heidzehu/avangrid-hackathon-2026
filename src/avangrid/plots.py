"""Plotting helpers for exploring customers. Pure matplotlib, no styling dependencies.

The day x hour heatmap is the workhorse: each row is a day (Sep -> Aug, top to bottom),
each column an hour, colour = kW. Solar shows as a midday band of zeros, EV charging as
bright evening/overnight blocks, electric heat as a bright winter band, AC as a summer
afternoon band, a pool pump as a fixed-clock summer stripe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import N_DAYS, START_DATE, SEASONS

_MONTH_TICKS = [(START_DATE + pd.DateOffset(months=i)) for i in range(12)]


def _day_ticks(ax):
    days = [(m - START_DATE).days for m in _MONTH_TICKS]
    ax.set_yticks(days)
    ax.set_yticklabels([m.strftime("%b") for m in _MONTH_TICKS])


def heatmap(series_kw: np.ndarray, ax=None, title: str = "", vmax: float | None = None,
            daily_temp: np.ndarray | None = None, cmap: str = "magma", cbar_label: str = "kW"):
    """Day x hour heatmap of one customer's 8760 hourly kW. Optionally a daily-temperature
    strip on the right."""
    x = np.asarray(series_kw, dtype=float).reshape(N_DAYS, 24)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 6))
    vmax = vmax if vmax is not None else max(np.nanpercentile(x, 99.5), 0.5)
    im = ax.imshow(x, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
    _day_ticks(ax)
    ax.set_xticks([0, 6, 12, 18, 23]); ax.set_xticklabels(["0", "6", "12", "18", "23"])
    ax.set_xlabel("hour of day")
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.set_label(cbar_label, fontsize=8)
    if daily_temp is not None:
        ax2 = ax.inset_axes([1.0 + 0.32, 0, 0.12, 1])
        ax2.imshow(np.asarray(daily_temp)[:, None], aspect="auto", cmap="coolwarm")
        ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_title("°F", fontsize=7)
    return ax


def heatmap_grid(F, customers: np.ndarray, ids, titles=None, ncols: int = 4, vmax=None, suptitle: str = ""):
    """Small multiples of heatmaps for a list of customer numbers."""
    ids = list(ids)
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 5.2 * nrows), squeeze=False)
    pos = {c: i for i, c in enumerate(customers)}
    for k, ax in enumerate(axes.ravel()):
        if k >= len(ids):
            ax.axis("off"); continue
        c = ids[k]
        heatmap(np.asarray(F[pos[c]]), ax=ax, vmax=vmax, title=(titles[k] if titles else f"customer {c}"))
    if suptitle:
        fig.suptitle(suptitle, y=1.0)
    fig.tight_layout()
    return fig


def group_mean_heatmaps(F, customers: np.ndarray, groups: dict[str, np.ndarray], normalize: bool = True):
    """Average heatmap per group (e.g. labeled EV vs random customers). With ``normalize``,
    each customer is divided by their own mean first so shape, not size, is compared."""
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 5.6), squeeze=False)
    pos = pd.Series(np.arange(len(customers)), index=customers)
    for ax, (name, ids) in zip(axes[0], groups.items()):
        idx = pos.reindex(ids).dropna().astype(int).to_numpy()
        X = np.asarray(F[np.sort(idx)], dtype=np.float64)
        if normalize:
            X = X / np.maximum(X.mean(1, keepdims=True), 1e-6)
        heatmap(X.mean(0), ax=ax, title=f"{name} (n={len(idx)})" + (" - normalised" if normalize else ""),
                cbar_label="x customer's own mean" if normalize else "kW")
    fig.tight_layout()
    return fig


def seasonal_profile(series_kw: np.ndarray, hours: pd.DataFrame, ax=None, title: str = ""):
    """Average weekday hourly profile per season for one customer."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    x = np.asarray(series_kw, dtype=float)
    hb = hours["hour_beginning"].to_numpy(); se = hours["season"].to_numpy()
    wk = ~(hours["is_weekend"] | hours["is_holiday"]).to_numpy()
    for s in SEASONS:
        m = (se == s) & wk
        ax.plot(range(24), [x[m & (hb == h)].mean() for h in range(24)], label=s)
    ax.set_xlabel("hour of day (weekdays)"); ax.set_ylabel("kW"); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8); ax.set_xticks(range(0, 24, 3))
    return ax


def temperature_scatter(daily_kwh_row: np.ndarray, daily_temp: np.ndarray, fit: pd.Series | None = None,
                        ax=None, title: str = ""):
    """Daily kWh vs daily mean temperature, with the fitted change-point line if given
    (a row of the feature table)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(daily_temp, daily_kwh_row, s=6, alpha=0.5)
    if fit is not None:
        t = np.linspace(np.min(daily_temp), np.max(daily_temp), 200)
        y = fit["cp_base_kwh_day"] + 0 * t
        if fit["cp_heat_slope"] > 0 and not np.isnan(fit["cp_heat_balance_f"]):
            y = y + fit["cp_heat_slope"] * np.maximum(fit["cp_heat_balance_f"] - t, 0)
        if fit["cp_cool_slope"] > 0 and not np.isnan(fit["cp_cool_balance_f"]):
            y = y + fit["cp_cool_slope"] * np.maximum(t - fit["cp_cool_balance_f"], 0)
        ax.plot(t, y, color="k", lw=1.5, label=f"{fit['cp_model']}  R²={fit['cp_r2']:.2f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("daily mean temperature (°F)"); ax.set_ylabel("kWh/day"); ax.set_title(title, fontsize=10)
    return ax


def system_load_plot(sysload: pd.DataFrame, weather_hourly: pd.DataFrame | None = None, top_n: int = 100):
    """System demand over the year with the top-N hours marked."""
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(sysload["ts"], sysload["system_kw"] / 1000, lw=0.4, color="0.35")
    top = sysload.nsmallest(top_n, "rank")
    ax.scatter(top["ts"], top["system_kw"] / 1000, s=8, color="crimson", label=f"top {top_n} hours", zorder=3)
    ax.set_ylabel("MW (customers loaded)"); ax.legend(loc="upper left")
    if weather_hourly is not None:
        ax2 = ax.twinx()
        ax2.plot(sysload["ts"], weather_hourly["temp_f"], lw=0.3, color="tab:blue", alpha=0.4)
        ax2.set_ylabel("°F", color="tab:blue")
    fig.tight_layout()
    return fig
