"""Step 6c: hyperparameter search for the PU LightGBM models, evaluated with nested cross-validation.

Why nested: with only ~136 EV / ~58 heat-pump labels, picking the best of many configurations and reporting its
cross-validated score is optimistically biased (some configuration wins by luck).  So:
  outer loop : 5-fold stratified CV x 2 repeats -> the honest performance estimate
  inner loop : on each outer-training set only, random search over N_ITER LightGBM configurations,
               each scored by 3-fold CV AUC; the best configuration is refit on the outer-training set and
               scored on the untouched outer-test fold.
The production configuration is scored on the same outer folds, so the comparison is paired.

Selection metric: AUC of labeled vs unlabeled.  Under the PU "selected completely at random" assumption this is a
monotone function of the true owner-vs-non-owner AUC, so maximizing it is valid without knowing hidden owners.

Decision rule (fixed before running): adopt the tuned configuration only if its nested AUC beats production by
more than one standard error of the paired fold differences AND recall@K is not worse; otherwise keep production.

A final (non-nested) search on all data gives the configuration to deploy if adopted.
Outputs: outputs/hp_search_trials.csv (every configuration tried), outputs/hp_search_nested.csv (per outer fold),
         outputs/hp_search_summary.csv, outputs/figures/10_hyperparameter_search.png
"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import StratifiedKFold, ParameterSampler
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import LOCAL, FIG

OUT = LOCAL.parent / "outputs"
N_ITER, INNER_K, OUTER_K, OUTER_REPEATS, C_RATE = 40, 3, 5, 2, 0.25

F = pd.read_parquet(LOCAL / "features.parquet")
ok = ~((F.coverage < .8) | (F.annual_kwh < 500) | (F.p99_kw > 15) | (F.annual_kwh > 50_000))
d = F[ok].reset_index(drop=True)
FEATS = [c for c in F.columns if c not in ("customer", "is_ev", "is_hp", "is_pv", "coverage")]
X = d[FEATS]

PRODUCTION = dict(learning_rate=0.03, n_estimators=150, num_leaves=4, min_child_samples=60, subsample=0.8,
                  colsample_bytree=0.6, reg_lambda=5.0, reg_alpha=0.0)
SPACE = dict(learning_rate=loguniform(0.01, 0.2), n_estimators=randint(50, 600), num_leaves=randint(2, 32),
             min_child_samples=randint(10, 200), subsample=uniform(0.5, 0.5), colsample_bytree=uniform(0.3, 0.7),
             reg_lambda=loguniform(0.01, 50), reg_alpha=loguniform(1e-3, 10))


def make(p, seed):
    return lgb.LGBMClassifier(objective="binary", subsample_freq=1, verbose=-1, random_state=seed, n_jobs=-1,
                              **{k: (int(v) if k in ("n_estimators", "num_leaves", "min_child_samples") else v)
                                 for k, v in p.items()})


def cv_auc(p, Xs, ys, seed, k=INNER_K):
    oof = np.zeros(len(ys))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(Xs, ys):
        oof[te] = make(p, seed).fit(Xs.iloc[tr], ys[tr]).predict_proba(Xs.iloc[te])[:, 1]
    return roc_auc_score(ys, oof)


def search(Xs, ys, seed):
    cands = [PRODUCTION] + list(ParameterSampler(SPACE, N_ITER, random_state=seed))
    res = [(cv_auc(p, Xs, ys, seed), p) for p in cands]
    return max(res, key=lambda r: r[0]), res


trials, nested, summary, deploy = [], [], [], {}
for label, name in [("is_ev", "EV"), ("is_hp", "Heat pump")]:
    y = d[label].to_numpy(); t0 = time.time()
    oof_tuned = np.zeros((OUTER_REPEATS, len(d))); oof_prod = np.zeros((OUTER_REPEATS, len(d)))
    for r in range(OUTER_REPEATS):
        for f, (tr, te) in enumerate(StratifiedKFold(OUTER_K, shuffle=True, random_state=100 + r).split(X, y)):
            (best_auc, best_p), _ = search(X.iloc[tr], y[tr], seed=1000 * r + f)
            pt = make(best_p, r).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
            pp = make(PRODUCTION, r).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
            oof_tuned[r, te], oof_prod[r, te] = pt, pp
            nested.append(dict(target=name, repeat=r, fold=f, inner_best_auc=best_auc,
                               outer_auc_tuned=roc_auc_score(y[te], pt), outer_auc_production=roc_auc_score(y[te], pp),
                               chosen_is_production=best_p is PRODUCTION,
                               **{f"chosen_{k}": v for k, v in best_p.items()}))
            print(f"{name} repeat {r} fold {f}: inner best {best_auc:.3f} | outer tuned "
                  f"{nested[-1]['outer_auc_tuned']:.3f} vs production {nested[-1]['outer_auc_production']:.3f}",
                  flush=True)

    n_lab = int(y.sum()); k = int(round(n_lab / C_RATE))
    def metrics(oof):
        aucs, recs = [], []
        for o in oof:
            aucs.append(roc_auc_score(y, o)); flag = (-o).argsort().argsort() < k; recs.append(flag[y == 1].mean())
        return np.mean(aucs), np.mean(recs)
    auc_t, rec_t = metrics(oof_tuned); auc_p, rec_p = metrics(oof_prod)
    nd = pd.DataFrame([n for n in nested if n["target"] == name])
    diff = nd.outer_auc_tuned - nd.outer_auc_production
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    adopt = bool(diff.mean() > se and rec_t >= rec_p)
    summary.append(dict(target=name, labeled=n_lab, K=k, nested_auc_production=auc_p, nested_auc_tuned=auc_t,
                        recall_at_k_production=rec_p, recall_at_k_tuned=rec_t,
                        precision_at_k_production=min(rec_p * n_lab / C_RATE / k, 1),
                        precision_at_k_tuned=min(rec_t * n_lab / C_RATE / k, 1),
                        mean_fold_auc_gain=diff.mean(), se_fold_auc_gain=se, adopt_tuned=adopt,
                        minutes=(time.time() - t0) / 60))

    # final search on all data: the configuration to deploy if adopted, and the full trial log
    (best_auc, best_p), res = search(X, y, seed=7)
    for a, p in res:
        trials.append(dict(target=name, cv_auc=a, is_production=p is PRODUCTION, **p))
    deploy[name] = dict(adopt=adopt, params={k: (int(v) if k in ("n_estimators", "num_leaves", "min_child_samples")
                                                  else float(v)) for k, v in best_p.items()}, cv_auc=best_auc)
    print(f"\n[{name}] nested AUC production {auc_p:.3f} vs tuned {auc_t:.3f} | recall@K {rec_p:.1%} vs {rec_t:.1%} | "
          f"fold gain {diff.mean():+.3f} ± {se:.3f} (SE) -> {'ADOPT tuned' if adopt else 'KEEP production'}\n")

T = pd.DataFrame(trials); T.round(4).to_csv(OUT / "hp_search_trials.csv", index=False)
pd.DataFrame(nested).round(4).to_csv(OUT / "hp_search_nested.csv", index=False)
S = pd.DataFrame(summary); S.round(4).to_csv(OUT / "hp_search_summary.csv", index=False)
(OUT / "hp_search_best.json").write_text(json.dumps(deploy, indent=2))
print(S.round(3).T.to_string())

# ---------- figure ----------
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
ND = pd.DataFrame(nested)
for ax, name, col in [(axes[0], "EV", "#d1495b"), (axes[1], "Heat pump", "#00798c")]:
    t = T[T.target == name]
    sc = ax.scatter(t.num_leaves, t.cv_auc, c=np.log10(t.min_child_samples), cmap="viridis", s=30)
    pr = t[t.is_production]
    ax.scatter(pr.num_leaves, pr.cv_auc, marker="*", s=300, color=col, edgecolor="k", zorder=5, label="Production")
    ax.set_xlabel("num_leaves (tree complexity)"); ax.set_ylabel("3-fold CV AUC")
    ax.set_title(f"{name}: {len(t)} configurations (random search)", fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=.3)
    cb = fig.colorbar(sc, ax=ax); cb.set_label("log10 min_child_samples")
ax = axes[2]
x = np.arange(len(S)); w = .35
ax.bar(x - w / 2, S.nested_auc_production, w, color="#9e9e9e", label="Production (hand-tuned)")
ax.bar(x + w / 2, S.nested_auc_tuned, w, color="#3d5a80", label="Random-search tuned")
for i, r in S.iterrows():
    ax.text(i, max(r.nested_auc_production, r.nested_auc_tuned) + .01,
            f"gain {r.mean_fold_auc_gain:+.3f} ± {r.se_fold_auc_gain:.3f}\n{'adopt' if r.adopt_tuned else 'keep production'}",
            ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(S.target); ax.set_ylim(0.5, 0.9); ax.set_ylabel("Nested-CV AUC (unbiased)")
ax.set_title("Nested CV: tuned vs production on identical folds", fontweight="bold"); ax.legend(loc="lower left")
ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(FIG / "10_hyperparameter_search.png", dpi=140); plt.close(fig)
print("saved", FIG / "10_hyperparameter_search.png")
