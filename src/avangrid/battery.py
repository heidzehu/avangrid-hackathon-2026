"""Behind-the-meter battery simulator (physics only).

This module answers "if the battery is *asked* to do X each hour, what actually happens?"
It enforces the program unit's limits from the Challenge Statement:

    energy 20.16 kWh (2 x 10.08), continuous power 7.68 kW (2 x 3.84),
    round-trip efficiency 89.4% (split evenly: sqrt(0.894) = 0.9455 on charge and on discharge),
    hourly time step, no export to the grid unless allowed.

It deliberately does NOT decide *what* to ask the battery to do. Dispatch strategy (peak
shaving vs. time-of-use vs. solar self-consumption), candidate selection, backup reserve,
tariffs and valuation are the team's choices. ``cap_request`` and ``min_daily_cap`` are two
generic building blocks for peak-cap strategies; anything else can be expressed as a
``request_kw`` array.

Sign convention: battery power > 0 = discharging into the home, < 0 = charging.
Everything is vectorised over customers: arrays are [n_customers, n_hours].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import N_DAYS, PROGRAM_BATTERY, BatterySpec


@dataclass
class BatteryResult:
    load_kw: np.ndarray  # original demand
    net_kw: np.ndarray  # grid import after the battery
    batt_kw: np.ndarray  # + discharge to home, - charge from grid
    soc_kwh: np.ndarray  # stored energy at the END of each hour
    spec: BatterySpec
    reserve_kwh: float
    soc0_kwh: np.ndarray  # stored energy before the first hour

    def summary(self, eval_slots: np.ndarray | None = None) -> pd.DataFrame:
        """Per-customer operating summary. ``eval_slots``: optional hour slots (e.g. the
        system's top-100 hours) at which to report average demand before/after."""
        d = np.clip(self.batt_kw, 0, None); c = np.clip(-self.batt_kw, 0, None)
        usable = self.spec.energy_kwh - self.reserve_kwh
        out = pd.DataFrame({
            "peak_before_kw": self.load_kw.max(1),
            "peak_after_kw": self.net_kw.max(1),
            "kwh_discharged": d.sum(1),
            "kwh_charged": c.sum(1),
            # energy drawn - energy delivered - change in storage = conversion losses
            "losses_kwh": c.sum(1) - d.sum(1) - (self.soc_kwh[:, -1] - self.soc0_kwh),
            "equivalent_full_cycles": d.sum(1) / max(usable * self.spec.one_way_efficiency, 1e-9),
            "annual_kwh_before": self.load_kw.sum(1),
            "annual_kwh_after": self.net_kw.sum(1),
        })
        out["peak_reduction_kw"] = out.peak_before_kw - out.peak_after_kw
        if eval_slots is not None:
            out["kw_eval_before"] = self.load_kw[:, eval_slots].mean(1)
            out["kw_eval_after"] = self.net_kw[:, eval_slots].mean(1)
            out["kw_eval_reduction"] = out.kw_eval_before - out.kw_eval_after
        return out


def simulate(load_kw: np.ndarray, request_kw: np.ndarray, spec: BatterySpec = PROGRAM_BATTERY,
             reserve_frac: float = 0.0, soc_init_frac: float | None = None,
             allow_export: bool = False) -> BatteryResult:
    """Step the battery hour by hour.

    load_kw:     [n, T] (or [T]) hourly demand
    request_kw:  same shape: what the strategy WANTS the battery to do each hour
                 (+ discharge, - charge). The simulator delivers as much as the limits allow:
                   discharge <= continuous power, <= (stored - reserve) * eta,
                              <= home demand (unless allow_export)
                   charge    <= continuous power, <= (capacity - stored) / eta
    reserve_frac: share of capacity kept for backup and never used for daily dispatch
    soc_init_frac: starting state of charge (default: full usable)
    """
    L = np.atleast_2d(np.asarray(load_kw, dtype=np.float64))
    R = np.broadcast_to(np.atleast_2d(np.asarray(request_kw, dtype=np.float64)), L.shape)
    n, T = L.shape
    eta = spec.one_way_efficiency
    cap, pmax = spec.energy_kwh, spec.power_kw
    reserve = reserve_frac * cap
    soc = np.full(n, cap if soc_init_frac is None else soc_init_frac * cap, dtype=np.float64)
    soc = np.clip(soc, reserve, cap)
    soc0 = soc.copy()
    B = np.zeros((n, T)); S = np.zeros((n, T))
    for t in range(T):
        r = R[:, t]
        dis = np.minimum(np.clip(r, 0, None), np.minimum(pmax, (soc - reserve) * eta))
        if not allow_export:
            dis = np.minimum(dis, np.clip(L[:, t], 0, None))
        chg = np.minimum(np.clip(-r, 0, None), np.minimum(pmax, (cap - soc) / eta))
        soc = soc - dis / eta + chg * eta
        B[:, t] = dis - chg
        S[:, t] = soc
    return BatteryResult(L, L - B, B, S, spec, reserve, soc0)


# ----------------------------------------------------------- generic building blocks
def _expand_cap(cap_kw, n: int, T: int) -> np.ndarray:
    c = np.asarray(cap_kw, dtype=np.float64)
    if c.ndim == 0:
        return np.full((n, T), float(c))
    if c.ndim == 1:
        return np.repeat(c[:, None], T, axis=1)
    if c.shape == (n, N_DAYS) and T == N_DAYS * 24:
        return np.repeat(c, 24, axis=1)
    return np.broadcast_to(c, (n, T)).astype(np.float64)


def cap_request(load_kw: np.ndarray, cap_kw, charge_allowed: np.ndarray | None = None,
                discharge_allowed: np.ndarray | None = None) -> np.ndarray:
    """Request that keeps grid demand at or below ``cap_kw``: discharge the excess above the
    cap; when below it, charge with at most the headroom under the cap, so charging never
    creates a new peak.

    cap_kw: scalar, [n] per customer, [n, 365] per customer-day, or [n, T] per hour.
    charge_allowed / discharge_allowed: optional bool [T] or [n, T] masks (e.g. charge only
    overnight, discharge only in a peak window). Default: always allowed.
    """
    L = np.atleast_2d(np.asarray(load_kw, dtype=np.float64))
    n, T = L.shape
    req = L - _expand_cap(cap_kw, n, T)  # > 0: excess to discharge, < 0: headroom to charge
    if charge_allowed is not None:
        ok = np.broadcast_to(np.asarray(charge_allowed, bool), (n, T))
        req = np.where((req < 0) & ~ok, 0.0, req)
    if discharge_allowed is not None:
        ok = np.broadcast_to(np.asarray(discharge_allowed, bool), (n, T))
        req = np.where((req > 0) & ~ok, 0.0, req)
    return req


def min_daily_cap(load_kw: np.ndarray, spec: BatterySpec = PROGRAM_BATTERY, reserve_frac: float = 0.0,
                  discharge_hours: np.ndarray | None = None, charge_hours: np.ndarray | None = None,
                  require_recharge: bool = True, iters: int = 30) -> np.ndarray:
    """Lowest grid-demand cap the battery could hold on each day, with perfect foresight of
    that day's load. Returns [n, 365] kW. A cap C is feasible for a day if, over that day:

      * energy above C (in ``discharge_hours``) <= usable kWh x discharge efficiency
      * the largest excess above C <= continuous power
      * (``require_recharge``) the room under C in ``charge_hours``, at most the power limit
        each hour, can put that energy back: headroom x round-trip efficiency >= energy above C.
        This makes the cap repeatable day after day instead of draining the battery once.

    ``discharge_hours`` / ``charge_hours``: optional bool [24] hour-of-day masks (default all).
    Use the same masks in ``cap_request``. Days are treated independently, so check any
    strategy built on this with ``simulate``.
    """
    L = np.atleast_2d(np.asarray(load_kw, dtype=np.float64))
    n = len(L)
    X = L.reshape(n, N_DAYS, 24)
    dmask = np.ones(24, bool) if discharge_hours is None else np.asarray(discharge_hours, bool)
    cmask = np.ones(24, bool) if charge_hours is None else np.asarray(charge_hours, bool)
    Xd = np.where(dmask[None, None, :], X, -np.inf)
    Xc = X[:, :, cmask]
    energy = (1 - reserve_frac) * spec.energy_kwh * spec.one_way_efficiency
    pmax, rte = spec.power_kw, spec.round_trip_efficiency
    hi = Xd.max(2)
    lo = np.maximum(hi - pmax, 0.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        need = np.clip(Xd - mid[:, :, None], 0, None).sum(2)
        ok = need <= energy
        if require_recharge:
            room = np.minimum(np.clip(mid[:, :, None] - Xc, 0, None), pmax).sum(2)
            ok &= room * rte >= need - 1e-9
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    return hi


# ------------------------------------------------------------------ cost helpers
def energy_cost(net_kw: np.ndarray, price_per_kwh) -> np.ndarray:
    """Annual energy cost per customer. ``price_per_kwh``: scalar or [T] array (e.g. a TOU
    schedule built from ``config.hour_index()``)."""
    return (np.atleast_2d(net_kw) * np.asarray(price_per_kwh)).sum(1)


def monthly_peak_kw(net_kw: np.ndarray, hours: pd.DataFrame) -> pd.DataFrame:
    """Max hourly demand per customer per month (for demand-charge style tariffs)."""
    ym = hours["ts"].dt.strftime("%Y-%m").to_numpy()
    months = pd.unique(ym)
    X = np.atleast_2d(net_kw)
    return pd.DataFrame(np.stack([X[:, ym == m].max(1) for m in months], 1), columns=months)


def kva(kw, power_factor: float = 1.0):
    """Convert kW to kVA (residential loads are often assumed close to unity power factor)."""
    return np.asarray(kw) / power_factor
