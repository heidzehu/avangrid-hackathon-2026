"""Step 5: which features separate labeled EV / heat-pump / solar customers from everyone else?

Univariate ROC-AUC per feature (0.5 = useless, 1.0 or 0.0 = perfect separation).
Note: 'unlabeled' still contains ~75% of the true EV/HP owners, so these AUCs understate real power.
Writes outputs/feature_auc.csv and outputs/figures/04_feature_separation.png
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import LOCAL, FIG

F = pd.read_parquet(LOCAL / "features.parquet")
feats = [c for c in F.columns if c not in ("customer", "is_ev", "is_hp", "is_pv", "coverage")]
unl = (F.is_ev + F.is_hp + F.is_pv) == 0
rows = []
for lab in ["is_ev", "is_hp", "is_pv"]:
    sub = F[(F[lab] == 1) | unl]
    for f in feats:
        x = sub[f].fillna(sub[f].median())
        rows.append((lab, f, roc_auc_score(sub[lab], x)))
A = pd.DataFrame(rows, columns=["label", "feature", "auc"]).pivot(index="feature", columns="label", values="auc")
A["best_sep"] = (A - 0.5).abs().max(1) + 0.5
A = A.sort_values("best_sep", ascending=False)
A.round(3).to_csv(LOCAL.parent / "outputs" / "feature_auc.csv")
pd.set_option("display.width", 140)
for lab in ["is_ev", "is_hp", "is_pv"]:
    s = (A[lab] - 0.5).abs().sort_values(ascending=False).head(8).index
    print(f"\nTop features for {lab}:  (AUC; <0.5 means lower in labeled group)")
    print(A.loc[s, lab].round(3).to_string())

# medians by group for the key features
grp = np.select([F.is_ev == 1, F.is_hp == 1, F.is_pv == 1], ["EV", "Heat pump", "Solar PV"], "Unlabeled")
F["grp"] = grp
key = ["annual_kwh", "heat_slope", "cool_slope", "cold_ratio", "winter_flatness", "ev_hours_gt6_per30d",
       "ev_blocks_per30d", "ev_block_kw", "big_load_nonwinter_share", "solar_zero_frac", "summer_midday_ratio",
       "coincident_peak_kw"]
print("\nGroup medians:"); print(F.groupby("grp")[key].median().T.round(2).to_string())

# figure
COLORS = {"EV": "#d1495b", "Heat pump": "#00798c", "Solar PV": "#edae49", "Unlabeled": "#8c8c8c"}
show = ["heat_slope", "cold_ratio", "winter_flatness", "cool_slope",
        "ev_hours_gt6_per30d", "ev_blocks_per30d", "big_load_nonwinter_share", "solar_zero_frac"]
titles = {"heat_slope": "Heating slope (kWh / degree-day)", "cold_ratio": "Cold-day ÷ mild-day kWh",
          "winter_flatness": "Winter profile flatness (min÷max)", "cool_slope": "Cooling slope (kWh / degree-day)",
          "ev_hours_gt6_per30d": "Hours > 6 kWh per 30 mild days", "ev_blocks_per30d": "High-load blocks per 30 mild days",
          "big_load_nonwinter_share": "Share of big loads outside winter", "solar_zero_frac": "Daytime hours ≈ 0 kWh (Apr-Sep)"}
fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
order = list(COLORS)
for ax, f in zip(axes.ravel(), show):
    data = [F.loc[F.grp == g, f].dropna().clip(*F[f].quantile([.005, .995])) for g in order]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=.6)
    for patch, g in zip(bp["boxes"], order): patch.set_facecolor(COLORS[g]); patch.set_alpha(.85)
    for m in bp["medians"]: m.set_color("k")
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(["EV", "HP", "PV", "Other"])
    auc = A.loc[f, ["is_ev", "is_hp", "is_pv"]]
    ax.set_title(f"{titles[f]}\nAUC  EV {auc.is_ev:.2f} | HP {auc.is_hp:.2f} | PV {auc.is_pv:.2f}", fontsize=9)
    ax.grid(alpha=.3, axis="y")
fig.suptitle("Engineered features separate the labeled technology groups", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "04_feature_separation.png", dpi=140); plt.close(fig)
print("\nsaved", FIG / "04_feature_separation.png")
