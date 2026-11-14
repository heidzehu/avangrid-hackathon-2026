"""Step 6: EV and heat-pump classification with positive-unlabeled (PU) learning.

Only ~25% of true EV / heat-pump owners are labeled, so "unlabeled" is a mix of negatives and hidden
positives. Approach (Elkan & Noto, 2008):
  1. Train LightGBM to separate labeled (s=1) from unlabeled (s=0) using 5-fold cross-validation,
     so every customer gets an out-of-fold score s_hat = P(s=1 | x) from a model that never saw it.
  2. c = P(labeled | owner) = 0.25, the organizers' stated sampling rate
  3. P(owner | x) = s_hat / c, clipped to [0, 1]   (sum of probabilities ~= expected owner count)
  4. Expected population = labeled_present / 0.25 (organizer's stated sampling rate) -> flag the top-K.
Validation: AUC (labeled vs unlabeled) and recall of held-out labeled owners within the top-K.

Outputs: data_local/ml_scores.parquet, outputs/model_metrics.csv, outputs/figures/05_*.png
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import LOCAL, FIG

OUT = LOCAL.parent / "outputs"
F = pd.read_parquet(LOCAL / "features.parquet")

# ---- data-quality flags ----
F["dq_low_coverage"] = F.coverage < 0.80
F["dq_inactive"] = F.annual_kwh < 500
F["large_nonres_like"] = (F.p99_kw > 15) | (F.annual_kwh > 50_000)
train_ok = ~(F.dq_low_coverage | F.dq_inactive | F.large_nonres_like)
print(f"customers {len(F)}, usable for training {train_ok.sum()}  "
      f"(low coverage {F.dq_low_coverage.sum()}, inactive {F.dq_inactive.sum()}, large/non-res-like {F.large_nonres_like.sum()})")

FEATS = [c for c in F.columns if c not in ("customer", "is_ev", "is_hp", "is_pv", "coverage")
         and not c.startswith(("dq_", "large_"))]
# Shallow, heavily regularized trees: chosen by out-of-fold AUC against logistic regression and deeper
# LightGBM (deeper trees memorized the ~140 / ~60 positives and generalized worse).
PARAMS = dict(objective="binary", learning_rate=0.03, n_estimators=150, num_leaves=4, min_child_samples=60,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, verbose=-1)
# Heat pump: settings from the nested-CV random search (src/06c_hyperparameter_search.py, full-data search winner).
# Nested CV: tuned beat the settings above in 7 of 10 outer folds, mean AUC gain +0.013 +/- 0.008 (SE), and found
# more held-out owners in the top-K, so it met the pre-set adoption rule.  EV: no gain, keeps PARAMS.
HP_PARAMS = dict(objective="binary", learning_rate=0.035, n_estimators=217, num_leaves=8, min_child_samples=146,
                 subsample=0.69, subsample_freq=1, colsample_bytree=0.36, reg_lambda=9.4, reg_alpha=0.52, verbose=-1)
SAMPLING_RATE = 0.25

def pu_model(label, name, seeds=(0, 1, 2, 3, 4), params=PARAMS):
    d = F[train_ok].reset_index(drop=True)
    y = d[label].to_numpy(); X = d[FEATS]
    oof = np.zeros(len(d)); imp = np.zeros(len(FEATS))
    for seed in seeds:                                              # repeated CV to stabilise
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            m = lgb.LGBMClassifier(**params, random_state=seed)
            m.fit(X.iloc[tr], y[tr])
            oof[te] += m.predict_proba(X.iloc[te])[:, 1] / len(seeds)
            imp += m.booster_.feature_importance("gain")
    s_hat = oof                                                      # unweighted model -> P(s=1|x)
    c = SAMPLING_RATE
    prob = np.clip(s_hat / c, 0, 1)
    n_lab = int(y.sum()); k = int(round(n_lab / SAMPLING_RATE))
    rank = (-oof).argsort().argsort()
    flag = rank < k
    auc = roc_auc_score(y, oof)
    recall_k = flag[y == 1].mean()
    lift = recall_k / (k / len(d))
    # Precision of the top-K list: labeled owners found / c = implied true owners in the list (unbiased under
    # random labeling), divided by K.  This is what a utility program manager would actually experience.
    precision_k = min(recall_k * n_lab / c / k, 1.0)
    print(f"\n[{name}] labeled={n_lab}  AUC={auc:.3f}  c={c:.3f}  expected owners K={k}  "
          f"recall@K={recall_k:.1%}  lift={lift:.1f}x  est. precision@K={precision_k:.0%}  sum(prob)={prob.sum():.0f}")
    res = pd.DataFrame({"customer": d.customer, f"{name}_score": oof, f"{name}_prob": prob, f"{name}_flag": flag,
                        f"{name}_rank": rank + 1})
    imp = pd.Series(imp / imp.sum(), index=FEATS).sort_values(ascending=False)
    # recall curve for plotting
    order = np.argsort(-oof); cum = np.cumsum(y[order]) / n_lab
    return res, dict(model=name, labeled=n_lab, auc=auc, c=c, expected_owners=k, recall_at_k=recall_k,
                     lift_at_k=lift, precision_at_k=precision_k, sum_prob=prob.sum()), imp, cum

ev, m_ev, imp_ev, cum_ev = pu_model("is_ev", "ev")
hp, m_hp, imp_hp, cum_hp = pu_model("is_hp", "hp", params=HP_PARAMS)
S = F[["customer", "is_ev", "is_hp", "is_pv", "dq_low_coverage", "dq_inactive", "large_nonres_like"]] \
    .merge(ev, on="customer", how="left").merge(hp, on="customer", how="left")
S.to_parquet(LOCAL / "ml_scores.parquet", index=False)
pd.DataFrame([m_ev, m_hp]).round(3).to_csv(OUT / "model_metrics.csv", index=False)
pd.concat({"ev": imp_ev, "hp": imp_hp}, axis=1).round(4).to_csv(OUT / "feature_importance.csv")
print("\nTop EV drivers:\n", imp_ev.head(8).round(3).to_string())
print("\nTop HP drivers:\n", imp_hp.head(8).round(3).to_string())

# ---- figures: importance + recall (gain) curves ----
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, imp, col, title in [(axes[0], imp_ev, "#d1495b", "EV model: top drivers"),
                            (axes[1], imp_hp, "#00798c", "Heat-pump model: top drivers")]:
    top = imp.head(10)[::-1]
    ax.barh(top.index, top.values, color=col); ax.set_title(title, fontweight="bold"); ax.set_xlabel("Share of total gain")
    ax.tick_params(axis="y", labelsize=8)
ax = axes[2]
n = len(cum_ev); x = np.arange(1, n + 1) / n
ax.plot(x * 100, cum_ev * 100, color="#d1495b", lw=2, label=f"EV (AUC {m_ev['auc']:.2f})")
ax.plot(x * 100, cum_hp * 100, color="#00798c", lw=2, label=f"Heat pump (AUC {m_hp['auc']:.2f})")
ax.plot([0, 100], [0, 100], "k--", lw=1, label="Random")
for m, col in [(m_ev, "#d1495b"), (m_hp, "#00798c")]:
    xk = m["expected_owners"] / n * 100
    ax.axvline(xk, color=col, ls=":", lw=1)
    ax.annotate(f"{m['recall_at_k']:.0%} found\nin top {m['expected_owners']}", (xk, m["recall_at_k"] * 100),
                xytext=(xk + 4, m["recall_at_k"] * 100 - 12), fontsize=8, color=col, arrowprops=dict(arrowstyle="->", color=col))
ax.set_xlim(0, 30); ax.set_xlabel("% of customers flagged (highest score first)")
ax.set_ylabel("% of held-out labeled owners found"); ax.set_title("Validation: cumulative gains (out-of-fold)", fontweight="bold")
ax.legend(loc="lower right"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "05_model_validation.png", dpi=140); plt.close(fig)
print("saved", FIG / "05_model_validation.png")
