"""Step 14: territory-wide BTM battery + demand-flexibility opportunity (Part 3), on the Part 1/2 customer base.

Transparent, vectorized screening model run for EVERY eligible residential home (complements the detailed
GridFlex LP optimizer in src/avangrid/gridflex.py, which is run on a short list).

Eligibility: residential homes only. Excludes large non-residential-like meters (p99 > 15 kW or > 50 MWh/yr),
seasonal/vacant homes and meters with < 80% coverage or < 500 kWh/yr.

Battery (challenge spec): Enphase IQ 10C. 1 unit = 10.08 kWh / 3.84 kW; 2 units = 20.16 kWh / 7.68 kW.
Round-trip 89.4% split evenly (94.55% each way); 10% reserve kept for backup.
Dispatch rule (every day): discharge during the 3-9 pm window to cover the home's own load, highest hours first,
limited by power and stored energy (no export); recharge overnight (midnight-6 am), or from rooftop solar where
midday imports are zero.
Sizing: 1 unit if the home's 3-9 pm energy on the 19 system-peak days fits in one unit's deliverable energy, else 2.

Demand flexibility (no hardware): smart-thermostat AC event for AC homes: 30% of the cooling component during
system-peak hours (cooling component = peak-hour load minus the same hours' load in mild months).

Prioritization: utility net cost per kW of peak relief = (capex - PV of the customer's bill savings net of O&M)
/ kW reduced at the system's top-100 hours, i.e. the incentive the utility would need to fund per kW if the customer
keeps the bill savings. Compared with the PV of the assumed capacity value ($200/kW-yr x 15 yr at 7% = ~$1,822/kW):
At today's assumed costs no home clears that bar, so tiers are relative: Tier 1 = the 1,000 lowest-cost homes (pilot),
Tier 2 = next 2,000, Tier 3 = other homes with persistent peaks (>= 2 kW in >= 40% of top-100 hours), else "not now".
The break-even battery cost (capex at which a home reaches the capacity value) is reported per tier.
Value (illustrative, stated on slides): capacity value $200/kW-yr on the reduction at the system's top-100 hours;
TOU bill savings at $0.28/kWh on-peak (3-9 pm) vs $0.11/kWh off-peak; capex $7,500 (1 unit) / $14,000 (2 units),
O&M $100/yr, 15 years, 7% discount rate.
3 kVA rule: a home can add a Level 2 charger (7.2 kW) without exceeding +3 kVA at peak if the battery can carry the
4.2 kW excess for 2 h (8.4 kWh) after its own peak duty.

Outputs: data_local/bess_customers.parquet (customer-level, git-ignored), outputs/bess_summary.json (aggregate).
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import LOCAL

OUT = LOCAL.parent / "outputs"
ETA = 0.894 ** 0.5
RESERVE = 0.10
UNITS = {1: (10.08, 3.84), 2: (20.16, 7.68)}
CAPEX = {1: 7500.0, 2: 14000.0}
OM, YEARS, DISC = 100.0, 15, 0.07
CAP_VALUE, P_ON, P_OFF = 200.0, 0.28, 0.11
WIN = list(range(15, 21))      # hour-beginning 15..20 = 3-9 pm
CHG = list(range(0, 6))        # midnight-6 am
ANN = (1 - (1 + DISC) ** -YEARS) / DISC

F = pd.read_parquet(LOCAL / "features.parquet")
R = pd.read_parquet(LOCAL / "classification.parquet")
S = pd.read_parquet(LOCAL / "segments.parquet")
assert (F.customer.values == R.customer.values).all() and (S.customer.values == F.customer.values).all()
H = np.load(LOCAL / "hourly.npy")
W = pd.read_parquet(LOCAL / "weather.parquet"); ts = pd.DatetimeIndex(W.ts)
C, NH = H.shape; ND = NH // 24
H = np.where(np.isnan(H), np.nanmean(H, 1, keepdims=True), H).astype(np.float32)
sysload = H.sum(0)
top = np.argsort(-sysload)[:100]
top_days = np.unique(top // 24)

elig = ~(R.large_nonres_like | R.seasonal_home | R.dq_low_coverage.fillna(False) | R.dq_inactive.fillna(False)).to_numpy()
idx = np.where(elig)[0]
X = H[idx].reshape(len(idx), ND, 24)
Lw = X[:, :, WIN]                                   # (n, days, 6)
need_peak = Lw[:, top_days, :].sum(2).mean(1)        # kWh 3-9 pm on system-peak days


def dispatch(E_nom, P):
    E = E_nom * (1 - RESERVE) * ETA                 # deliverable kWh per day
    order = np.argsort(-Lw, axis=2)
    Ls = np.take_along_axis(Lw, order, 2)
    d = np.minimum(Ls, P)
    prev = np.cumsum(d, 2) - d
    d = np.clip(np.minimum(d, E - prev), 0, None)
    out = np.empty_like(d); np.put_along_axis(out, order, d, 2)
    return out, E


res = {}
for u, (E_nom, P) in UNITS.items():
    d, E = dispatch(E_nom, P)
    res[u] = (d, E)
E1 = res[1][1]
units = np.where(need_peak <= E1, 1, 2)
D = np.where(units[:, None, None] == 1, res[1][0], res[2][0])
Edel = np.where(units == 1, res[1][1], res[2][1])
Pmax = np.where(units == 1, UNITS[1][1], UNITS[2][1])

# hourly discharge back onto the 8760 grid for system-peak metrics
disc = np.zeros((len(idx), ND, 24), np.float32); disc[:, :, WIN] = D
disc = disc.reshape(len(idx), NH)
cp_before = H[idx][:, top].mean(1)
cp_red = disc[:, top].mean(1)
shifted = D.sum((1, 2))                              # kWh/yr delivered in 3-9 pm
charged = shifted / ETA ** 2                        # kWh/yr drawn to recharge (round trip)

# solar homes: share of days with >= 3 zero-import midday hours (battery can charge from PV those days)
mid = X[:, :, 10:15]
solar_days = (np.sum(mid < 0.05, 2) >= 3).mean(1)
is_pv = R.f_solar_pv.to_numpy()[idx]
grid_charge = np.where(is_pv, charged * (1 - solar_days), charged)
bill = P_ON * shifted - P_OFF * grid_charge
cap_val = CAP_VALUE * cp_red
annual = cap_val + bill - OM
npv = annual * ANN - np.vectorize(CAPEX.get)(units)

# persistence: share of the top-100 hours where the home draws >= 2 kW
persist = (H[idx][:, top] >= 2.0).mean(1)

# demand flexibility: thermostat event on AC homes
mon = np.asarray(ts.month); hod = np.asarray(ts.hour)
mild = np.isin(mon, [4, 5, 10])
mild_by_hour = np.stack([H[idx][:, mild & (hod == h)].mean(1) for h in range(24)], 1)
cool_comp = np.clip(H[idx][:, top] - mild_by_hour[:, hod[top]], 0, None).mean(1)
is_ac = R.f_ac.to_numpy()[idx]
dr_kw = np.where(is_ac, 0.30 * cool_comp, 0.0)

# electrification headroom (3 kVA rule) for homes without an EV
spare_e = Edel - need_peak
can_add_l2 = (spare_e >= 8.4) & (Pmax - cp_red >= 4.2) & ~R.f_ev.to_numpy()[idx]

B = pd.DataFrame(dict(customer=F.customer.values[idx], segment=S.segment.values[idx], units=units,
                      cp_before_kw=cp_before, cp_red_kw=cp_red, kwh_shifted=shifted, bill_savings=bill,
                      capacity_value=cap_val, annual_value=annual, npv=npv, persistence=persist, dr_kw=dr_kw,
                      can_add_l2=can_add_l2, ev=R.f_ev.values[idx], pv=is_pv, ac=is_ac,
                      eheat=R.any_elec_heat.values[idx], pool=R.f_pool.values[idx]))
capex = np.vectorize(CAPEX.get)(units)
B["net_cost_per_kw"] = (capex - (bill - OM) * ANN) / np.maximum(cp_red, 0.05)
B["incentive_needed"] = np.maximum(capex - (bill - OM) * ANN, 0)
CAP_PV = CAP_VALUE * ANN
B["rank"] = B.net_cost_per_kw.rank(method="first").astype(int)
# No home clears CAP_PV at today's assumed costs, so tiers are relative (pilot first, lowest cost per kW):
B["tier"] = np.select([B["rank"] <= 1000, B["rank"] <= 3000, B.persistence >= 0.40],
                      ["Tier 1", "Tier 2", "Tier 3"], "Not now")
B["breakeven_capex"] = CAP_PV * cp_red + (bill - OM) * ANN
# electrification: with a 2-unit battery, can a home without an EV add a Level 2 charger and stay within +3 kVA?
spare2 = res[2][1] - need_peak
cp2 = disc2 = None
B["l2_ready_2u"] = (spare2 >= 8.4) & ~R.f_ev.to_numpy()[idx]
B.to_parquet(LOCAL / "bess_customers.parquet", index=False)

def agg(m):
    b = B[m]
    return dict(homes=int(len(b)), one_unit=int((b.units == 1).sum()), two_unit=int((b.units == 2).sum()),
                cp_red_mw=round(float(b.cp_red_kw.sum() / 1000), 2), cp_red_kw_avg=round(float(b.cp_red_kw.mean()), 2),
                cp_before_kw_avg=round(float(b.cp_before_kw.mean()), 2),
                mwh_shifted=round(float(b.kwh_shifted.sum() / 1000), 0), bill_avg=round(float(b.bill_savings.mean()), 0),
                capval_avg=round(float(b.capacity_value.mean()), 0), annual_avg=round(float(b.annual_value.mean()), 0),
                npv_avg=round(float(b.npv.mean()), 0), npv_pos=int((b.npv > 0).sum()),
                persist_avg=round(float(b.persistence.mean()), 2), dr_mw=round(float(b.dr_kw.sum() / 1000), 2),
                l2_ready=int(b.l2_ready_2u.sum()), ev=int(b.ev.sum()),
                incentive_avg=round(float(b.incentive_needed.mean()), 0),
                cost_per_kw_med=round(float(b.net_cost_per_kw.median()), 0),
                breakeven_med=round(float(b.breakeven_capex.median()), 0), pv=int(b.pv.sum()), pool=int(b.pool.sum()),
                eheat=int(b.eheat.sum()))
summary = dict(eligible=agg(np.ones(len(B), bool)), tiers={t: agg(B.tier == t) for t in ["Tier 1", "Tier 2", "Tier 3", "Not now"]},
               system_peak_mw=round(float(sysload.max() / 1000), 1),
               top100_avg_mw=round(float(sysload[top].mean() / 1000), 1),
               excluded=int((~elig).sum()),
               tier1_by_segment=B[B.tier == "Tier 1"].segment.value_counts().to_dict(),
               tier12_by_segment=B[B.tier.isin(["Tier 1", "Tier 2"])].segment.value_counts().to_dict(),
               dr_total_mw=round(float(B.dr_kw.sum() / 1000), 2), dr_homes=int((B.dr_kw > 0).sum()),
               dr_avg_kw=round(float(B.dr_kw[B.dr_kw > 0].mean()), 2),
               solar_days_avg=round(float(solar_days[is_pv].mean()), 2),
               solar_pv_homes=int(is_pv.sum()), cap_pv_per_kw=round(float(CAP_PV), 0),
               tier12_cost_effective_mw=round(float(B.cp_red_kw[B.tier.isin(["Tier 1"])].sum() / 1000), 2))
# average Tier 1 home on the system-peak day: load vs net load
pk_day = int(top[0] // 24)
t1 = (B.tier == "Tier 1").to_numpy()
load_day = X[t1, pk_day, :].mean(0)
dis_day = disc.reshape(len(idx), ND, 24)[t1, pk_day, :].mean(0)
chg_day = np.zeros(24); chg_day[CHG] = dis_day.sum() / ETA ** 2 / len(CHG)
summary["peak_day"] = dict(date=str(ts[pk_day * 24].date()), load=load_day.round(3).tolist(),
                           discharge=dis_day.round(3).tolist(), charge=chg_day.round(3).tolist())
summary["units_sizing"] = {"1": int((units == 1).sum()), "2": int((units == 2).sum())}
summary["assumptions"] = dict(capacity_value=CAP_VALUE, tou_on=P_ON, tou_off=P_OFF, capex=CAPEX, om=OM, years=YEARS,
                              disc=DISC, reserve=RESERVE, window="3-9 pm", dr_share=0.30)
json.dump(summary, open(OUT / "bess_summary.json", "w"), indent=1)
print(json.dumps({k: summary[k] for k in ["eligible", "tiers", "excluded", "dr_total_mw", "dr_homes", "dr_avg_kw",
                                          "solar_days_avg", "tier1_by_segment", "units_sizing", "system_peak_mw",
                                          "top100_avg_mw"]}, indent=1))
