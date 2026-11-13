"""Step 4: one row of interpretable features per customer.

Every feature is a plain-language measurement of the hourly load (and its relation to weather),
so each classification / segment / battery decision can be traced back to physical evidence.

Inputs : data_local/hourly.npy, customers.npy, weather.parquet, label workbook
Outputs: data_local/features.parquet  (customer, labels, ~45 features)
         data_local/profiles.npy      [n, 48] normalized 24-h shape, winter then summer (for clustering)
         data_local/monthly.npy       [n, 12] mean kWh/h by calendar month (Sep..Aug)
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import LOCAL, LABELS_XLSX

t0 = time.time()
H = np.load(LOCAL / "hourly.npy")                       # [C, 8760] kWh (NaN = missing)
cust = np.load(LOCAL / "customers.npy")
W = pd.read_parquet(LOCAL / "weather.parquet")
C, NH = H.shape; ND = NH // 24
temp = W.temp.to_numpy()
ts = pd.DatetimeIndex(W.ts)
month = np.asarray(ts.month); hod = np.asarray(ts.hour)          # hod 0 = hour-ending 1
weekend = np.asarray(ts.day_of_week) >= 5

rowmean = np.nanmean(H, 1, keepdims=True)
Hf = np.where(np.isnan(H), rowmean, H).astype(np.float32)       # gap-filled copy for vector ops
D = Hf.reshape(C, ND, 24)                                       # [C, day, hour]
dmonth = month[::24]; tday = temp.reshape(ND, 24).mean(1)       # per-day month / mean temp

F = pd.DataFrame({"customer": cust})
def per30(x, ndays): return x / ndays * 30

# ---------------- 1. size & peakiness ----------------
F["annual_kwh"] = np.nanmean(H, 1) * 8760
F["mean_kw"] = np.nanmean(H, 1)
F["peak_kw"] = np.nanmax(H, 1)
F["p99_kw"] = np.percentile(Hf, 99, axis=1)
F["load_factor"] = F.mean_kw / F.peak_kw.clip(lower=1e-3)
F["coverage"] = 1 - np.isnan(H).mean(1)

# ---------------- 2. weather response (change-point regression on daily kWh) ----------------
# daily_kWh = b0 + bH * max(Th - T, 0) + bC * max(T - Tc, 0); Th, Tc chosen per customer by least squares.
Y = D.sum(2)                                                    # [C, 365]
sst = ((Y - Y.mean(1, keepdims=True)) ** 2).sum(1)
best = dict(sse=np.full(C, np.inf), b=np.zeros((C, 3)), th=np.zeros(C), tc=np.zeros(C))
for th in range(45, 70, 5):
    for tc in range(55, 80, 5):
        if tc < th: continue
        X = np.column_stack([np.ones(ND), np.clip(th - tday, 0, None), np.clip(tday - tc, 0, None)])
        B = Y @ np.linalg.pinv(X).T                             # OLS for all customers at once
        sse = ((Y - B @ X.T) ** 2).sum(1)
        m = sse < best["sse"]
        best["sse"][m] = sse[m]; best["b"][m] = B[m]; best["th"][m] = th; best["tc"][m] = tc
F["base_kwh_day"] = best["b"][:, 0]
F["heat_slope"] = best["b"][:, 1]           # kWh per heating degree-day
F["cool_slope"] = best["b"][:, 2]           # kWh per cooling degree-day
F["heat_balance_F"] = best["th"]; F["cool_balance_F"] = best["tc"]
F["weather_r2"] = 1 - best["sse"] / np.maximum(sst, 1e-6)

def daymean(mask): return Y[:, mask].mean(1)
bins = {"vcold": tday < 20, "cold": (tday >= 20) & (tday < 35), "cool": (tday >= 35) & (tday < 50),
        "mild": (tday >= 50) & (tday < 65), "hot": tday >= 72}
bm = {k: daymean(v) for k, v in bins.items()}
F["kwh_vcold_day"] = bm["vcold"]; F["kwh_mild_day"] = bm["mild"]; F["kwh_hot_day"] = bm["hot"]
F["cold_ratio"] = bm["vcold"] / np.maximum(bm["mild"], 0.5)      # heating intensity vs shoulder days
F["hot_ratio"] = bm["hot"] / np.maximum(bm["mild"], 0.5)          # cooling intensity vs shoulder days
# convexity of heating: extra kWh per °F in the coldest range vs the moderate range.
# Heat pumps lose efficiency in deep cold (convex); resistance heat is close to linear.
s_cold = (bm["vcold"] - bm["cold"]) / (tday[bins["cold"]].mean() - tday[bins["vcold"]].mean())
s_mod = (bm["cold"] - bm["cool"]) / (tday[bins["cool"]].mean() - tday[bins["cold"]].mean())
F["heat_convexity"] = s_cold / np.maximum(np.abs(s_mod), 0.05) * np.sign(s_mod + 1e-9)

# hourly winter temperature correlation (strongly negative = electric heating)
def rowcorr(A, x):
    A = A - A.mean(1, keepdims=True); x = x - x.mean()
    return (A @ x) / (np.sqrt((A ** 2).sum(1)) * np.sqrt((x ** 2).sum()) + 1e-9)
win = np.isin(month, [12, 1, 2]); smr = np.isin(month, [6, 7, 8])
F["winter_temp_corr"] = rowcorr(Hf[:, win], temp[win])
F["summer_temp_corr"] = rowcorr(Hf[:, smr], temp[smr])

# ---------------- 3. daily shape ----------------
def seasonal_profile(mask_days):
    p = D[:, mask_days, :].mean(1)                              # [C, 24]
    return p / np.maximum(p.mean(1, keepdims=True), 1e-3)
dwin = np.isin(dmonth, [12, 1, 2]); dsmr = np.isin(dmonth, [6, 7, 8])
pw, ps = seasonal_profile(dwin), seasonal_profile(dsmr)
np.save(LOCAL / "profiles.npy", np.hstack([pw, ps]).astype(np.float32))
pa = D.mean(1); pa = pa / np.maximum(pa.sum(1, keepdims=True), 1e-6)    # annual shares by hour
F["share_night_HE1_6"] = pa[:, 0:6].sum(1)
F["share_morning_HE7_9"] = pa[:, 6:9].sum(1)
F["share_midday_HE10_15"] = pa[:, 9:15].sum(1)
F["share_peak_HE16_21"] = pa[:, 15:21].sum(1)
F["share_late_HE22_24"] = pa[:, 21:24].sum(1)
F["winter_flatness"] = pw.min(1) / np.maximum(pw.max(1), 1e-3)  # mini-splits run all day -> flat
F["weekend_ratio"] = Hf[:, weekend].mean(1) / np.maximum(Hf[:, ~weekend].mean(1), 1e-3)
F["peak_hour_HE"] = pa.argmax(1) + 1

# ---------------- 4. solar signature ----------------
sun = np.isin(month, [4, 5, 6, 7, 8, 9]) & (hod >= 9) & (hod <= 14)
F["solar_zero_frac"] = (Hf[:, sun] < 0.02).mean(1)              # daytime hours with ~zero net draw
F["summer_midday_ratio"] = ps[:, 9:14].mean(1) / np.maximum(ps[:, [5, 6, 18, 19]].mean(1), 1e-3)

# ---------------- 5. EV signatures (mild months only, to keep heating/cooling out) ----------------
dmild = np.isin(dmonth, [4, 5, 10]); nmild = dmild.sum()
Hm = D[:, dmild, :].reshape(C, -1)
base = np.median(Hm, 1, keepdims=True)
high = Hm > base + 3.0                                          # >= 3 kW above the home's own typical
starts = high & ~np.concatenate([np.zeros((C, 1), bool), high[:, :-1]], 1)
F["ev_hours_gt6_per30d"] = per30((Hm > 6).sum(1), nmild)
F["ev_blocks_per30d"] = per30(starts.sum(1), nmild)
F["ev_block_len_h"] = high.sum(1) / np.maximum(starts.sum(1), 1)
F["ev_block_kw"] = ((Hm - base) * high).sum(1) / np.maximum(high.sum(1), 1)
F["ev_upjumps_per30d"] = per30((np.diff(Hm, axis=1) >= 4).sum(1), nmild)
hodm = np.tile(np.arange(24), nmild)
nightish = (hodm >= 20) | (hodm <= 5)
F["ev_high_night_share"] = (high & nightish).sum(1) / np.maximum(high.sum(1), 1)
# Level 1: long overnight plateau ~1.2-2 kW above the home's normal overnight floor
night = D[:, dmild, 0:6].mean(2)                                # [C, mild days]
uplift = night - np.percentile(night, 20, axis=1, keepdims=True)
F["ev_l1_nights_frac"] = ((uplift > 0.9) & (uplift < 2.6)).mean(1)
# year-round consistency: big-load hours that are NOT in winter (EV) vs concentrated in winter (heat)
big = Hf > np.median(Hf, 1, keepdims=True) + 3.0
F["big_load_nonwinter_share"] = big[:, ~win].sum(1) / np.maximum(big.sum(1), 1)

# ---------------- 6. seasonal / pool / baseload ----------------
dspring = np.isin(dmonth, [4, 5])
daymin = D[:, :, 8:17].min(2)                                   # lowest daytime hour each day
nightmin = D[:, :, 0:6].min(2)
F["summer_daymin_uplift"] = np.median(daymin[:, dsmr], 1) - np.median(daymin[:, dspring], 1)
F["summer_daymin_temp_corr"] = rowcorr(daymin[:, dsmr], tday[dsmr])   # pool pump ~ uncorrelated
F["winter_nightmin_uplift"] = np.median(nightmin[:, dwin], 1) - np.median(nightmin[:, dspring], 1)
F["baseload_kw"] = np.percentile(nightmin, 10, axis=1)
mon = np.stack([Hf[:, month == m].mean(1) for m in [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]], 1)
np.save(LOCAL / "monthly.npy", mon.astype(np.float32))
F["winter_summer_ratio"] = mon[:, 3:6].mean(1) / np.maximum(mon[:, 9:12].mean(1), 1e-3)
F["min_max_month_ratio"] = mon.min(1) / np.maximum(mon.max(1), 1e-3)
F["n_vacant_months"] = (mon < 0.15).sum(1)

# ---------------- 7. grid impact ----------------
system = np.nansum(H, 0)
top = np.argsort(system)[-100:]
F["coincident_peak_kw"] = Hf[:, top].mean(1)
F["coincidence_factor"] = F.coincident_peak_kw / F.peak_kw.clip(lower=1e-3)
ts_top = ts[top]
print(f"system peak hours: {pd.Series(ts_top.month).value_counts().sort_index().to_dict()} (month: count), "
      f"HE {pd.Series(ts_top.hour + 1).value_counts().index[:4].tolist()}; system max {system.max()/1000:.1f} MW")

# ---------------- labels ----------------
for sheet, col in [("EV Customers", "is_ev"), ("HP Customers", "is_hp"), ("PV Customers", "is_pv")]:
    ids = set(pd.read_excel(LABELS_XLSX, sheet_name=sheet).CUSTOMER_NUMBER)
    F[col] = F.customer.isin(ids).astype(np.int8)

F = F.replace([np.inf, -np.inf], np.nan)
F.to_parquet(LOCAL / "features.parquet", index=False)
print(f"features: {F.shape[0]} customers x {F.shape[1]-4} features in {time.time()-t0:.0f}s")
print("day counts per temp bin:", {k: int(v.sum()) for k, v in bins.items()})
