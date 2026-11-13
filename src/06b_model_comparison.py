"""Step 6b: model comparison for the PU classifiers (EV, heat pump).

Same setup as 06_classify_ml.py (same customers, same features, 5-fold CV repeated over 5 seeds, out-of-fold
scores), so the numbers are directly comparable.  Candidates:
  - LightGBM (shallow)       : the production model
  - Random forest (2 settings): bagged deep-ish trees; min_samples_leaf controls overfitting
  - Logistic regression      : linear baseline on standardized features
Metrics: AUC (labeled vs unlabeled), recall@K and estimated precision@K (K = labeled / 0.25),
calibration check sum(prob) vs K.
Output: outputs/model_comparison.csv
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from config import LOCAL

OUT = LOCAL.parent / "outputs"
F = pd.read_parquet(LOCAL / "features.parquet")
ok = ~((F.coverage < .8) | (F.annual_kwh < 500) | (F.p99_kw > 15) | (F.annual_kwh > 50_000))
d = F[ok].reset_index(drop=True)
FEATS = [c for c in F.columns if c not in ("customer", "is_ev", "is_hp", "is_pv", "coverage")]
X = d[FEATS].fillna(d[FEATS].median())
C_RATE = 0.25

MODELS = {
    "LightGBM (shallow, production)": lambda s: lgb.LGBMClassifier(
        objective="binary", learning_rate=0.03, n_estimators=150, num_leaves=4, min_child_samples=60, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, verbose=-1, random_state=s),
    "Random forest (leaf>=20)": lambda s: RandomForestClassifier(
        n_estimators=500, min_samples_leaf=20, max_features="sqrt", n_jobs=-1, random_state=s),
    "Random forest (leaf>=50, balanced)": lambda s: RandomForestClassifier(
        n_estimators=500, min_samples_leaf=50, max_features="sqrt", class_weight="balanced_subsample",
        n_jobs=-1, random_state=s),
    "Logistic regression (C=0.05)": lambda s: make_pipeline(
        StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)),
}

rows = []
for label, name in [("is_ev", "EV"), ("is_hp", "Heat pump")]:
    y = d[label].to_numpy(); n_lab = int(y.sum()); k = int(round(n_lab / C_RATE))
    for mname, make in MODELS.items():
        t0 = time.time(); oof = np.zeros(len(d))
        for seed in range(5):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                m = make(seed).fit(X.iloc[tr], y[tr])
                oof[te] += m.predict_proba(X.iloc[te])[:, 1] / 5
        flag = (-oof).argsort().argsort() < k
        recall = flag[y == 1].mean()
        balanced = "balanced" in mname           # class-weighted scores are not P(s=1|x): no calibration
        rows.append(dict(target=name, model=mname, auc=roc_auc_score(y, oof), recall_at_k=recall,
                         est_precision_at_k=min(recall * n_lab / C_RATE / k, 1), expected_owners=k,
                         sum_prob=np.nan if balanced else np.clip(oof / C_RATE, 0, 1).sum(),
                         seconds=time.time() - t0))
        r = rows[-1]
        print(f"{name:10s} {mname:36s} AUC={r['auc']:.3f} recall@K={r['recall_at_k']:.1%} "
              f"precision@K={r['est_precision_at_k']:.0%} sum(prob)={r['sum_prob']:.0f} (K={k})  {r['seconds']:.0f}s")

pd.DataFrame(rows).round(3).to_csv(OUT / "model_comparison.csv", index=False)
print("saved", OUT / "model_comparison.csv")
