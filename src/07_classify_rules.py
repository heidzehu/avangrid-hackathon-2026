"""Step 7: full appliance inventory = ML scores (EV, heat pump) + transparent physics rules (everything else).

Each rule turns physical evidence into a 0-1 confidence with a logistic ramp:
    ramp(x, x0, w) = 1 / (1 + exp(-(x - x0) / w))     x0 = 50% point, w = softness
and multiplies ramps when several conditions must hold together (AND).
A technology is flagged when its confidence >= 0.5.

Outputs: outputs/customer_classification.csv (one row per customer; git-ignored, contains event data)
         outputs/classification_summary.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import LOCAL

OUT = LOCAL.parent / "outputs"
F = pd.read_parquet(LOCAL / "features.parquet")
S = pd.read_parquet(LOCAL / "ml_scores.parquet")
H = np.load(LOCAL / "hourly.npy"); W = pd.read_parquet(LOCAL / "weather.parquet")
ts = pd.DatetimeIndex(W.ts); month = np.asarray(ts.month); hod = np.asarray(ts.hour)
C, NH = H.shape; ND = NH // 24
Hf = np.where(np.isnan(H), np.nanmean(H, 1, keepdims=True), H).astype(np.float32)

# ---- two extra event features computed here ----
E = Hf - np.repeat(np.percentile(Hf.reshape(C, ND, 24), 20, axis=2), 24, axis=1)
nonwinter = ~np.isin(month, [11, 12, 1, 2, 3])
prev = np.concatenate([E[:, :1], E[:, :-1]], 1); nxt = np.concatenate([E[:, 1:], E[:, -1:]], 1)
burst = (E >= 2.0) & (E < 5.5) & (prev < 1.5) & (nxt < 1.5)                # isolated 1-h burst
dhw_hours = np.isin(hod, [5, 6, 7, 8, 17, 18, 19, 20, 21])                  # HE6-9, HE18-22
F["wh_bursts_per_day"] = (burst & nonwinter & dhw_hours).sum(1) / (nonwinter.sum() / 24)
summer_eve = np.isin(month, [6, 7, 8]) & (hod >= 17) & (hod <= 21)
F["evening_zero_frac"] = (Hf[:, summer_eve] < 0.05).mean(1)

def ramp(x, x0, w): return 1 / (1 + np.exp(-np.clip((np.asarray(x, float) - x0) / w, -50, 50)))

R = F[["customer", "is_ev", "is_hp", "is_pv"]].copy()
R = R.merge(S[["customer", "ev_prob", "ev_flag", "ev_rank", "hp_prob", "hp_flag", "hp_rank",
               "dq_low_coverage", "dq_inactive", "large_nonres_like"]], on="customer")
R[["ev_prob", "hp_prob"]] = R[["ev_prob", "hp_prob"]].fillna(0)
R[["ev_flag", "hp_flag"]] = R[["ev_flag", "hp_flag"]].fillna(False).astype(bool)

# EV: model flag; labeled customers always count.  Level split by charger power evidence.
R["ev"] = R.ev_flag | (R.is_ev == 1)
R["ev_conf"] = np.maximum(R.ev_prob, R.is_ev)
l2_evidence = ramp(F.ev_block_kw, 4.5, 0.5) * ramp(F.ev_hours_gt6_per30d + F.l2_sess_per30d, 2.0, 0.7)
R["ev_level"] = np.where(~R.ev, "", np.where(l2_evidence >= 0.5, "L2", "L1/occasional"))

# Air conditioning: cooling slope + hourly load follows summer temperature
R["ac_conf"] = ramp(F.cool_slope, 0.35, 0.1) * ramp(F.summer_temp_corr, 0.25, 0.08)
# Electric space heating (any type)
R["eheat_conf"] = ramp(F.heat_share, 0.12, 0.03) * ramp(-F.winter_temp_corr, 0.25, 0.08)
# Heat pump.  Diagnostic finding (see README): the labeled heat-pump rate rises ~8x with heating intensity, and
# among heavy electric heaters labeled heat pumps and other homes have the same cooling, baseload and winter shape.
# Hourly AMI cannot reliably separate a whole-home heat pump from resistance heat, so the model's top-K list is
# only ~12% precise.  We therefore do NOT claim new heat pumps individually:
#   heat_pump       = labeled (confirmed) owners
#   hp_candidate    = unlabeled customers in the model's top-K (outreach list, ~1 in 8 expected to be a heat pump)
#   population size = labeled / 0.25 (organizer sampling rate)
R["hp"] = R.is_hp == 1
R["hp_candidate"] = R.hp_flag & (R.is_hp == 0)
R["hp_conf"] = np.maximum(R.hp_prob, R.is_hp)
R["resist_conf"] = R.eheat_conf * (1 - R.hp_conf) * ramp(F.night_cold_minus_mild_kw, 1.0, 0.3)
# Electric water heater: repeated isolated 2-5.5 kW one-hour bursts at morning/evening hot-water times, outside heating season
R["wh_conf"] = ramp(F.wh_bursts_per_day, 0.35, 0.08)
# Pool pump: summer daytime floor rises >=0.6 kW vs spring and does NOT follow temperature (timer-driven)
R["pool_conf"] = ramp(F.summer_daymin_uplift, 0.6, 0.15) * ramp(-F.summer_daymin_temp_corr, -0.45, 0.1)
# Hot tub / spa: overnight winter floor rises 0.4-2 kW without broad space-heating signature
R["spa_conf"] = ramp(F.winter_nightmin_uplift, 0.4, 0.1) * (1 - ramp(F.winter_nightmin_uplift, 2.0, 0.3)) \
                * (1 - R.eheat_conf)
# Solar PV: labeled list is complete per organizers, but flag look-alikes for data-quality follow-up.
# Occupied homes only (annual > 3,000 kWh, overnight floor > 0.15 kW): an empty house also reads ~0 at noon.
R["pv"] = R.is_pv == 1
occupied = (F.annual_kwh > 3000) & (F.baseload_kw > 0.15)
R["pv_lookalike_conf"] = np.where(R.pv | ~occupied, 0,
                                  ramp(F.solar_zero_frac, 0.12, 0.03) * ramp(-F.summer_midday_ratio, -0.5, 0.08))
# Solar + battery: solar homes that also import ~nothing on summer evenings (battery covering the peak)
R["pv_batt_conf"] = np.where(R.pv & (F.n_vacant_months == 0) & (F.annual_kwh > 3000),
                            ramp(F.evening_zero_frac, 0.3, 0.07), 0)
# Discovered segments
R["seasonal_home"] = F.n_vacant_months >= 2          # >= 2 calendar months averaging < 0.15 kW
R["large_nonres_like"] = R.large_nonres_like.fillna(False)

flags = {"ev": R.ev, "ev_L2": R.ev_level == "L2", "ev_L1": R.ev_level == "L1/occasional",
         "ac": R.ac_conf >= .5, "elec_heat": R.eheat_conf >= .5, "heat_pump": R.hp, "hp_candidate": R.hp_candidate,
         "resistance_heat": R.resist_conf >= .5, "water_heater": R.wh_conf >= .5, "pool": R.pool_conf >= .5,
         "hot_tub": R.spa_conf >= .5, "solar_pv": R.pv, "solar_battery": R.pv_batt_conf >= .5,
         "solar_lookalike": R.pv_lookalike_conf >= .5, "seasonal_home": R.seasonal_home,
         "large_nonres_like": R.large_nonres_like}
for k, v in flags.items(): R[f"f_{k}"] = v.astype(bool)
R["any_elec_heat"] = R.f_elec_heat | R.f_heat_pump | R.f_resistance_heat
R["n_electrified"] = R[["f_ev", "any_elec_heat", "f_water_heater", "f_pool", "f_hot_tub"]].sum(axis=1)
R = R.merge(F[["customer", "annual_kwh", "peak_kw", "coincident_peak_kw", "load_factor", "wh_bursts_per_day",
               "evening_zero_frac"]], on="customer")
R.to_csv(OUT / "customer_classification.csv", index=False)
R.to_parquet(LOCAL / "classification.parquet", index=False)

M = pd.read_csv(OUT / "model_metrics.csv").set_index("model")
# Population estimate: flagged count for rule-based technologies; labeled / 0.25 for the PU-modelled ones.
est = {"ev": M.loc["ev", "expected_owners"], "heat_pump": M.loc["hp", "expected_owners"]}
prec = {"ev": M.loc["ev", "precision_at_k"], "hp_candidate": M.loc["hp", "precision_at_k"]}
rows = []
for k in flags:
    m = R[f"f_{k}"]
    rows.append(dict(technology=k, customers=int(m.sum()), pct=m.mean() * 100,
                     estimated_population=est.get(k, int(m.sum())), est_list_precision=prec.get(k, np.nan),
                     median_annual_kwh=R.loc[m, "annual_kwh"].median(), median_peak_kw=R.loc[m, "peak_kw"].median(),
                     mean_coincident_peak_kw=R.loc[m, "coincident_peak_kw"].mean()))
rows.append(dict(technology="ALL customers", customers=len(R), pct=100, median_annual_kwh=R.annual_kwh.median(),
                 median_peak_kw=R.peak_kw.median(), mean_coincident_peak_kw=R.coincident_peak_kw.mean()))
T = pd.DataFrame(rows).round(2); T.to_csv(OUT / "classification_summary.csv", index=False)
print(T.to_string(index=False))
print("\nEV labeled present:", int(R.is_ev.sum()), " new EV found:", int((R.ev & (R.is_ev == 0)).sum()))
print("HP labeled present:", int(R.is_hp.sum()), " HP candidates (outreach list):", int(R.hp_candidate.sum()))
print("electrified-measure count distribution:", R.n_electrified.value_counts().sort_index().to_dict())
