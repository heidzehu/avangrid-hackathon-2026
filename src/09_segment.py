"""Step 9: load-behavior segmentation (Part 2).

Pre-assigned segments (rule-based, not clustered):
  Insufficient data (coverage < 80% or < 500 kWh/yr), Large non-residential-like, Seasonal / vacant home.
Everyone else is clustered with k-means on three blocks of standardized inputs:
  - daily shape    : 6 PCA components of the normalized 24-h winter + summer profiles (profiles.npy)
  - seasonal shape : 3 PCA components of monthly kWh normalized by the customer's own mean (monthly.npy)
  - magnitude/behavior: log annual kWh, load factor, heating share, cooling share, coincidence factor, log baseload
k is chosen from 4-9; silhouette is low for every k (residential load is a continuum, not islands), so k is set
for interpretability/actionability and the silhouette curve is reported.

Each cluster is then described (shape, season, size, peak contribution) and overlaid with the Part 1 technology flags.
Outputs: data_local/segments.parquet, outputs/segment_summary.csv, outputs/segment_k_selection.csv,
         outputs/figures/08_segment_profiles.png, 09_segment_overview.png
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import LOCAL, FIG

OUT = LOCAL.parent / "outputs"
K = 6
F = pd.read_parquet(LOCAL / "features.parquet")
R = pd.read_parquet(LOCAL / "classification.parquet")
assert (F.customer.values == R.customer.values).all()
P = np.load(LOCAL / "profiles.npy"); M = np.load(LOCAL / "monthly.npy")

insufficient = ((F.coverage < .8) | (F.annual_kwh < 500)).to_numpy()
nonres = R.large_nonres_like.to_numpy() & ~insufficient
seasonal = R.seasonal_home.to_numpy() & ~insufficient & ~nonres
clus = ~(insufficient | nonres | seasonal)

Mn = np.nan_to_num(M / M.mean(1, keepdims=True), nan=1.0)
shape = PCA(6, random_state=0).fit_transform(P[clus])
seas = PCA(3, random_state=0).fit_transform(Mn[clus])
mag = F.loc[clus, ["annual_kwh", "load_factor", "heat_share", "cool_share", "coincidence_factor", "baseload_kw"]].copy()
mag["annual_kwh"] = np.log(mag.annual_kwh); mag["baseload_kw"] = np.log(mag.baseload_kw.clip(lower=.02))
mag = mag.fillna(mag.median())
sc = lambda a: StandardScaler().fit_transform(a)
X = np.hstack([sc(shape) * 0.8, sc(seas), sc(mag)])

rng = np.random.default_rng(0); samp = rng.choice(len(X), 6000, replace=False)
ksel = []
for k in range(4, 10):
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(X)
    ksel.append(dict(k=k, silhouette=silhouette_score(X[samp], lab[samp]), inertia=None))
pd.DataFrame(ksel).drop(columns="inertia").round(3).to_csv(OUT / "segment_k_selection.csv", index=False)
km = KMeans(K, n_init=20, random_state=0).fit(X)
cl = np.full(len(F), -1); cl[clus] = km.labels_

# ---------- describe clusters ----------
D = F[["customer", "annual_kwh", "load_factor", "heat_share", "cool_share", "coincidence_factor", "baseload_kw",
       "coincident_peak_kw", "peak_kw", "winter_summer_ratio", "share_peak_HE16_21", "share_night_HE1_6",
       "share_midday_HE10_15"]].copy()
D["cluster"] = cl
for c in ["f_ev", "f_ac", "any_elec_heat", "f_heat_pump", "f_hp_candidate", "f_resistance_heat", "f_water_heater",
          "f_pool", "f_hot_tub", "f_solar_pv"]:
    D[c] = R[c].to_numpy()
pw, ps = P[:, :24], P[:, 24:]
D["evening_peakiness"] = ps[:, 16:21].mean(1) / ps.mean(1)          # summer HE17-21 vs daily mean
D["winter_morning"] = pw[:, 5:9].mean(1) / pw.mean(1)

agg = dict(n=("customer", "size"), annual_kwh=("annual_kwh", "median"), load_factor=("load_factor", "median"),
           heat_share=("heat_share", "median"), cool_share=("cool_share", "median"),
           winter_summer=("winter_summer_ratio", "median"), evening_peakiness=("evening_peakiness", "median"),
           night_share=("share_night_HE1_6", "median"), midday_share=("share_midday_HE10_15", "median"),
           coinc_peak_kw=("coincident_peak_kw", "mean"), peak_kw=("peak_kw", "median"),
           ev=("f_ev", "mean"), ac=("f_ac", "mean"), elec_heat=("any_elec_heat", "mean"), pool=("f_pool", "mean"),
           hot_tub=("f_hot_tub", "mean"), water_heater=("f_water_heater", "mean"), solar=("f_solar_pv", "mean"))
C = D[D.cluster >= 0].groupby("cluster").agg(**agg)


# ---------- name clusters from their centroid statistics (deterministic, rule-based) ----------
def name_clusters(C):
    names = {}
    left = list(C.index)
    def take(col, largest=True, label=""):
        s = C.loc[left, col]; i = s.idxmax() if largest else s.idxmin(); names[i] = label; left.remove(i)
    take("heat_share", True, "Winter-peaking electric heat")
    take("cool_share", True, "Summer-dominant heavy AC")
    take("annual_kwh", True, "High-use, year-round")
    take("annual_kwh", False, "Low-use, little AC")
    take("evening_peakiness", True, "Small homes, evening-peaky AC")
    for i in left: names[i] = "Steady moderate use"
    return names


names = name_clusters(C)
C.insert(0, "segment", [names[i] for i in C.index])
pre = []
for lbl, m in [("Seasonal / vacant home", seasonal), ("Large non-residential-like", nonres),
               ("Insufficient data", insufficient)]:
    sub = D[m]
    row = {k: (sub[v[0]].median() if v[1] == "median" else sub[v[0]].mean() if v[1] == "mean" else len(sub))
           for k, v in agg.items()}
    row["segment"] = lbl; pre.append(row)
code = {lbl: -(i + 2) for i, lbl in enumerate(["Seasonal / vacant home", "Large non-residential-like", "Insufficient data"])}
D.loc[seasonal, "cluster"] = code["Seasonal / vacant home"]
D.loc[nonres, "cluster"] = code["Large non-residential-like"]
D.loc[insufficient, "cluster"] = code["Insufficient data"]
S = pd.concat([C.reset_index(), pd.DataFrame(pre).assign(cluster=list(code.values()))], ignore_index=True)
S["share_pct"] = S.n / len(D) * 100
S["share_of_system_peak_pct"] = S.n * S.coinc_peak_kw / (D.coincident_peak_kw.sum()) * 100
D["segment"] = D.cluster.map(dict(zip(S.cluster, S.segment)))
D[["customer", "cluster", "segment"]].to_parquet(LOCAL / "segments.parquet", index=False)
S.round(3).to_csv(OUT / "segment_summary.csv", index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(ksel)[["k", "silhouette"]].round(3).to_string(index=False))
print(S[["cluster", "segment", "n", "share_pct", "annual_kwh", "load_factor", "heat_share", "cool_share", "winter_summer",
         "evening_peakiness", "night_share", "midday_share", "coinc_peak_kw", "share_of_system_peak_pct",
         "ev", "ac", "elec_heat", "pool", "hot_tub", "solar"]].round(2).to_string(index=False))

# ---------- charts ----------
order = list(C.sort_values("coinc_peak_kw", ascending=False).index)
cols = plt.cm.tab10(np.arange(10))
fig, axes = plt.subplots(2, 3, figsize=(17, 8.5), sharey=False)
mon_lbl = ["S", "O", "N", "D", "J", "F", "M", "A", "M", "J", "J", "A"]
H = np.load(LOCAL / "hourly.npy", mmap_mode="r")
for ax, ci in zip(axes.flat, order):
    m = cl == ci
    lw = (P[m, :24] * F.annual_kwh.to_numpy()[m, None] / 8760).mean(0)
    ls = (P[m, 24:] * F.annual_kwh.to_numpy()[m, None] / 8760).mean(0)
    ax.plot(range(1, 25), lw, color="#00798c", lw=2.2, label="Winter")
    ax.plot(range(1, 25), ls, color="#d1495b", lw=2.2, label="Summer")
    ax.axvspan(16.5, 20.5, color="orange", alpha=.12)
    r = C.loc[ci]
    ax.set_title(f"{r.segment}\n{int(r.n):,} homes ({r.n / len(D):.0%}) | {r.annual_kwh:,.0f} kWh/yr | "
                 f"{r.coinc_peak_kw:.1f} kW at system peak", fontsize=9.5, fontweight="bold")
    ax.set_xticks(range(1, 25, 3)); ax.grid(alpha=.3); ax.set_xlabel("Hour ending")
    ax.set_ylabel("Approx. avg kW")
    ins = ax.inset_axes([0.03, 0.62, 0.3, 0.33])
    ins.bar(range(12), np.nanmedian(Mn[m], 0), color="#888"); ins.set_xticks(range(12)); ins.set_xticklabels(mon_lbl, fontsize=6)
    ins.set_yticks([]); ins.set_title("monthly", fontsize=6)
axes[0, 0].legend(loc="upper right", fontsize=8)
fig.suptitle("Load-behavior segments: average daily shape by season (shaded = system-peak window HE17-20)",
             fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "08_segment_profiles.png", dpi=140); plt.close(fig)

S2 = S.sort_values("share_of_system_peak_pct", ascending=True)
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw=dict(width_ratios=[1, 1.25]))
ax = axes[0]
y = np.arange(len(S2))
ax.barh(y - .2, S2.share_pct, .4, color="#9e9e9e", label="% of customers")
ax.barh(y + .2, S2.share_of_system_peak_pct, .4, color="#d1495b", label="% of system-peak load")
ax.set_yticks(y); ax.set_yticklabels(S2.segment, fontsize=9); ax.legend(); ax.grid(alpha=.3, axis="x")
ax.set_title("Who drives the summer evening peak?", fontweight="bold"); ax.set_xlabel("%")
ax = axes[1]
tech = ["ev", "ac", "elec_heat", "water_heater", "pool", "hot_tub", "solar"]
Z = S2.set_index("segment")[tech] * 100
im = ax.imshow(Z.to_numpy(), cmap="Blues", aspect="auto")
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        v = Z.iat[i, j]; ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                                 color="white" if v > Z.to_numpy().max() * .6 else "black")
ax.set_xticks(range(len(tech))); ax.set_xticklabels(["EV", "AC", "Elec heat", "Water htr", "Pool", "Hot tub", "Solar"])
ax.set_yticks(range(len(Z))); ax.set_yticklabels(Z.index, fontsize=9)
ax.set_title("Technology penetration by segment (Part 1 overlay)", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "09_segment_overview.png", dpi=140); plt.close(fig)
print("saved 08_segment_profiles.png, 09_segment_overview.png")
