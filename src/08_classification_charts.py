"""Step 8: deck charts for the classification results.
  06_inventory.png            estimated appliance inventory (customers and % of population)
  07_new_vs_labeled.png       newly found EVs look like labeled EVs; why heat pumps cannot be confirmed individually
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import LOCAL, FIG

R = pd.read_parquet(LOCAL / "classification.parquet")
H = np.load(LOCAL / "hourly.npy"); W = pd.read_parquet(LOCAL / "weather.parquet")
ts = pd.DatetimeIndex(W.ts); month = np.asarray(ts.month); hod = np.asarray(ts.hour)
C, NH = H.shape; ND = NH // 24
Hf = np.where(np.isnan(H), np.nanmean(H, 1, keepdims=True), H).astype(np.float32)

# ---- inventory ----
inv = [("EV - Level 2", "f_ev_L2", "#d1495b"), ("EV - Level 1 / occasional", "f_ev_L1", "#e8838f"),
       ("Air conditioning", "f_ac", "#3d5a80"), ("Electric space heat (any)", "f_elec_heat", "#00798c"),
       ("  Heat pump (labeled)", "f_heat_pump", "#00a6b4"), ("  Heat-pump candidates", "f_hp_candidate", "#8fd3da"),
       ("  Resistance heat", "f_resistance_heat", "#66c2cc"),
       ("Electric water heater", "f_water_heater", "#8e7dbe"), ("Pool pump", "f_pool", "#2a9d8f"),
       ("Hot tub / spa", "f_hot_tub", "#90be6d"), ("Solar PV (known)", "f_solar_pv", "#edae49"),
       ("Solar + battery", "f_solar_battery", "#f4a259"), ("Seasonal / vacant home", "f_seasonal_home", "#9e9e9e"),
       ("Large non-residential-like", "f_large_nonres_like", "#555555")]
fig, ax = plt.subplots(figsize=(10, 6))
names = [i[0] for i in inv][::-1]; vals = [R[i[1]].sum() for i in inv][::-1]; cols = [i[2] for i in inv][::-1]
bars = ax.barh(names, vals, color=cols)
for b, v in zip(bars, vals):
    ax.text(b.get_width() * 1.15 + 1, b.get_y() + b.get_height() / 2, f"{v:,}  ({v / len(R):.1%})", va="center", fontsize=9)
M = pd.read_csv(LOCAL.parent / "outputs" / "model_metrics.csv").set_index("model")
note = (f"EV: est. {M.loc['ev','expected_owners']:,} owners in population; flagged list ~{M.loc['ev','precision_at_k']:.0%} precise\n"
        f"Heat pump: est. {M.loc['hp','expected_owners']:,} owners; candidate list ~{M.loc['hp','precision_at_k']:.0%} precise")
ax.text(0.98, 0.02, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#444",
        bbox=dict(fc="white", ec="#ccc"))
ax.set_xscale("log"); ax.set_xlim(1, 2e5); ax.set_xlabel("Customers (log scale)")
ax.set_title(f"Estimated appliance inventory across {len(R):,} residential meters", fontweight="bold")
ax.grid(alpha=.3, axis="x"); fig.tight_layout(); fig.savefig(FIG / "06_inventory.png", dpi=150); plt.close(fig)

# ---- new vs labeled ----
F = pd.read_parquet(LOCAL / "features.parquet").set_index("customer").loc[R.customer].reset_index()
lab_ev = (R.is_ev == 1).to_numpy(); new_ev = (R.f_ev & (R.is_ev == 0)).to_numpy()
lab_hp = (R.is_hp == 1).to_numpy()
hs = F.heat_slope.to_numpy()
whole = lab_hp & (hs >= 0.5); supp = lab_hp & (hs < 0.5)
resist = (R.f_resistance_heat & (R.is_hp == 0)).to_numpy()
rest = ~(R.f_ev | R.f_heat_pump | R.f_solar_pv | R.f_large_nonres_like | R.f_seasonal_home | R.any_elec_heat).to_numpy()
mild = np.isin(month, [4, 5, 10])
tday = W.temp.to_numpy().reshape(ND, 24).mean(1); Y = Hf.reshape(C, ND, 24).sum(2)
bins = np.arange(-5, 90, 5)
fig, axes = plt.subplots(1, 3, figsize=(19, 5), gridspec_kw=dict(width_ratios=[1.1, 1.1, 0.9]))
ax = axes[0]
for m, lbl, col, ls in [(lab_ev, f"Labeled EV (n={lab_ev.sum()})", "#d1495b", "-"),
                        (new_ev, f"Newly flagged EV (n={new_ev.sum()})", "#d1495b", "--"),
                        (rest, "Other customers", "#8c8c8c", "-")]:
    p = np.array([Hf[m][:, mild & (hod == h)].mean() for h in range(24)])
    ax.plot(range(1, 25), p, ls, color=col, lw=2.4, label=lbl)
ax.set_title("EV: newly flagged homes share the labeled\novernight-charging shape (Apr, May, Oct)", fontweight="bold")
ax.set_xlabel("Hour ending"); ax.set_ylabel("Average kWh per hour"); ax.set_xticks(range(1, 25, 2)); ax.grid(alpha=.3); ax.legend()
ax = axes[1]
idx = np.digitize(tday, bins)
for m, lbl, col, ls in [(whole, f"Labeled HP - whole-home (n={whole.sum()})", "#00798c", "-"),
                        (supp, f"Labeled HP - supplemental (n={supp.sum()})", "#00798c", ":"),
                        (resist, f"Resistance-heat homes (n={resist.sum():,})", "#e07a5f", "--"),
                        (rest, "Homes without electric heat", "#8c8c8c", "-")]:
    ym = np.median(Y[m], 0)
    xs = [bins[i - 1] + 2.5 for i in np.unique(idx)]; ys = [ym[idx == i].mean() for i in np.unique(idx)]
    ax.plot(xs, ys, ls, marker="o", ms=3, color=col, lw=2.4, label=lbl)
ax.set_title("Heat pumps come in two kinds: whole-home units\nheat like resistance homes; supplemental ones barely show", fontweight="bold")
ax.set_xlabel("Daily mean temperature (°F)"); ax.set_ylabel("Median daily kWh"); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax = axes[2]
ok = ~(R.dq_low_coverage.fillna(False) | R.dq_inactive.fillna(False) | R.large_nonres_like).to_numpy()
edges = [-9, 0.05, 0.2, 0.5, 1, 2, 99]; labels = ["<0.05", "0.05-0.2", "0.2-0.5", "0.5-1", "1-2", ">2"]
b = pd.cut(hs[ok], edges, labels=labels)
t = pd.DataFrame({"b": b, "hp": lab_hp[ok]}).groupby("b", observed=True).hp.agg(["mean", "size"])
ax.bar(t.index.astype(str), t["mean"] / 0.25 * 100, color="#00798c")
for i, (v, n) in enumerate(zip(t["mean"] / 0.25 * 100, t["size"])): ax.text(i, v + 0.2, f"n={n:,}", ha="center", fontsize=8)
ax.set_title("Implied heat-pump share rises with heating\nintensity, but never dominates", fontweight="bold")
ax.set_ylim(0, (t["mean"] / 0.25 * 100).max() * 1.12); ax.set_xlabel("Heating slope (kWh/day per °F below balance point)"); ax.set_ylabel("Implied % heat-pump owners (labeled / 0.25)")
ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(FIG / "07_new_vs_labeled.png", dpi=150); plt.close(fig)
print("saved 06_inventory.png, 07_new_vs_labeled.png")
