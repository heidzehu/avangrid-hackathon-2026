"""Step 4b: event-level features — EV charging sessions and overnight heating response.

EV charging leaves a *rectangular* footprint: load jumps up by a fixed charger power, stays flat for
1-10 h, then drops. Heating/cooling ramps with temperature and is not flat. For each customer:
  elevation(t) = kWh(t) - that day's 20th-percentile hour            (removes baseload & slow HVAC drift)
  L2 session  = run of hours with elevation >= 3 kW, length 1-10 h, flat (max-min <= 0.4*median + 1)
  L1 session  = run of hours with 0.9 <= elevation < 3 kW, length 4-14 h, flat, starting 6pm-2am
Also: overnight (HE1-6) winter heating slope, kWh/h per °F, where occupant behavior adds least noise.
Appends columns to data_local/features.parquet.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import LOCAL

t0 = time.time()
H = np.load(LOCAL / "hourly.npy"); cust = np.load(LOCAL / "customers.npy")
W = pd.read_parquet(LOCAL / "weather.parquet"); temp = W.temp.to_numpy()
ts = pd.DatetimeIndex(W.ts); month = np.asarray(ts.month); hod = np.asarray(ts.hour)
C, NH = H.shape; ND = NH // 24
Hf = np.where(np.isnan(H), np.nanmean(H, 1, keepdims=True), H).astype(np.float32)
daybase = np.percentile(Hf.reshape(C, ND, 24), 20, axis=2)                 # [C, ND]
E = Hf - np.repeat(daybase, 24, axis=1)                                   # elevation above daily floor
tday = temp.reshape(ND, 24).mean(1)
cold_day = np.repeat(tday < 35, 24); winter = np.isin(month, [12, 1, 2])
n_days = ND; n_nonwinter_days = (~np.isin(month[::24], [12, 1, 2])).sum()

def runs(mask_row):
    m = np.concatenate([[False], mask_row, [False]])
    d = np.diff(m.astype(np.int8))
    return np.where(d == 1)[0], np.where(d == -1)[0]                      # start, end (exclusive)

out = {k: np.zeros(C, np.float32) for k in
       ["l2_sess_per30d", "l2_sess_nonwinter_per30d", "l2_sess_kw", "l2_sess_len_h", "l2_sess_night_start",
        "l2_sess_cold_share", "l2_weeks_active", "l1_sess_per30d", "l1_sess_len_h", "ev_night_kwh_excess"]}
week = np.arange(NH) // (24 * 7)
for i in range(C):
    e = E[i]
    # ---- Level 2 ----
    s, en = runs(e >= 3.0)
    L = en - s
    keep = (L >= 1) & (L <= 10)
    if keep.any():
        s, en, L = s[keep], en[keep], L[keep]
        mx = np.maximum.reduceat(e, s)[: len(s)] if len(s) else np.array([])
        # reduceat over [s_k, s_{k+1}) — compute per-run stats explicitly for correctness
        mx = np.array([e[a:b].max() for a, b in zip(s, en)]); mn = np.array([e[a:b].min() for a, b in zip(s, en)])
        med = np.array([np.median(e[a:b]) for a, b in zip(s, en)])
        flat = (mx - mn) <= 0.4 * med + 1.0
        ok = flat & (med <= 12)
        s, en, L, med = s[ok], en[ok], L[ok], med[ok]
        if len(s):
            out["l2_sess_per30d"][i] = len(s) / n_days * 30
            nw = ~winter[s]
            out["l2_sess_nonwinter_per30d"][i] = nw.sum() / n_nonwinter_days * 30
            out["l2_sess_kw"][i] = np.median(med); out["l2_sess_len_h"][i] = np.median(L)
            sh = hod[s]; out["l2_sess_night_start"][i] = ((sh >= 18) | (sh <= 2)).mean()
            out["l2_sess_cold_share"][i] = cold_day[s].mean()
            out["l2_weeks_active"][i] = len(np.unique(week[s])) / (NH / 168)
    # ---- Level 1 ----
    s, en = runs((e >= 0.9) & (e < 3.0))
    L = en - s
    keep = (L >= 4) & (L <= 14) & ((hod[np.minimum(s, NH - 1)] >= 18) | (hod[np.minimum(s, NH - 1)] <= 2))
    s, en, L = s[keep], en[keep], L[keep]
    if len(s):
        rng = np.array([np.ptp(e[a:b]) for a, b in zip(s, en)])
        ok = rng <= 1.2
        if ok.any():
            out["l1_sess_per30d"][i] = ok.sum() / n_days * 30; out["l1_sess_len_h"][i] = np.median(L[ok])
night = (hod <= 4) & ~winter                                               # HE1-5 outside winter
out["ev_night_kwh_excess"] = (E[:, night]).mean(1)                         # avg overnight elevation, no heating

# ---- overnight winter heating response (HE1-6, Nov-Mar) ----
nm = np.isin(month, [11, 12, 1, 2, 3]) & (hod <= 5)
x = temp[nm]; A = Hf[:, nm]
xc = x - x.mean(); slope = ((A - A.mean(1, keepdims=True)) @ xc) / (xc @ xc)
out["night_heat_slope"] = -slope                                           # kWh/h per °F colder
cold = nm & (temp < 15); mild = nm & (temp >= 35) & (temp < 50)
out["night_cold_minus_mild_kw"] = Hf[:, cold].mean(1) - Hf[:, mild].mean(1)
F = pd.read_parquet(LOCAL / "features.parquet")
assert (F.customer.to_numpy() == cust).all()
for k, v in out.items(): F[k] = v
# heating / cooling share of annual energy from the change-point model
W_d = pd.Series(tday)
F["heat_share"] = (F.heat_slope.clip(lower=0) * np.clip(F.heat_balance_F.to_numpy()[:, None] - tday[None, :], 0, None).sum(1)) / F.annual_kwh.clip(lower=1)
F["cool_share"] = (F.cool_slope.clip(lower=0) * np.clip(tday[None, :] - F.cool_balance_F.to_numpy()[:, None], 0, None).sum(1)) / F.annual_kwh.clip(lower=1)
F.to_parquet(LOCAL / "features.parquet", index=False)
print(f"added {len(out) + 2} features in {time.time()-t0:.0f}s; total columns {F.shape[1]}")
g = np.select([F.is_ev == 1, F.is_hp == 1, F.is_pv == 1], ["EV", "HP", "PV"], "Unl")
print(F.groupby(g)[list(out) + ["heat_share", "cool_share"]].median().T.round(3).to_string())
