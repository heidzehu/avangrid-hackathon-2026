"""Reference customers (EV / heat pump / solar PV) and evaluation helpers.

What the workbook says (Description tab):
  * EV Customers  - random sample of 200 customers Avangrid *believes may* have an EV,
                    ~25% of the likely-EV population.
  * HP Customers  - random sample of 100 likely heat-pump customers, ~25% of that population.
  * PV Customers  - the full list of solar PV customers.

So EV/HP are *positive-unlabeled*: a customer not on the list may still have one
(~3 of every 4 do). Only PV is a complete list. The helpers below score any method the
team builds against these lists without assuming unlisted == negative.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import LABEL_SAMPLE_FRACTION

SHEETS = {"ev": "EV Customers", "hp": "HP Customers", "pv": "PV Customers"}


def find_label_file(raw_dir: Path) -> Path:
    for p in sorted(Path(raw_dir).glob("*.xlsx")):
        key = p.name.lower().replace(" ", "").replace("_", "")
        if "ev" in key and "heatpump" in key and "solar" in key:
            return p
    raise FileNotFoundError(f"Reference-customer workbook (EV_Heat Pump_Solar PV Customers.xlsx) not found in {raw_dir}")


def load_reference_customers(raw_dir: Path) -> dict[str, np.ndarray]:
    path = find_label_file(raw_dir)
    xl = pd.ExcelFile(path)
    out = {}
    for key, sheet in SHEETS.items():
        name = next((s for s in xl.sheet_names if s.strip().lower() == sheet.lower()), None)
        if name is None:
            raise ValueError(f"Sheet '{sheet}' missing from {path.name}: {xl.sheet_names}")
        df = xl.parse(name)
        col = next(c for c in df.columns if "customer" in str(c).lower())
        out[key] = np.sort(pd.to_numeric(df[col], errors="coerce").dropna().astype(int).unique())
    return out


def label_columns(customers: np.ndarray, refs: dict[str, np.ndarray]) -> pd.DataFrame:
    """Boolean columns ``ref_ev``, ``ref_hp``, ``ref_pv`` for the given customers."""
    df = pd.DataFrame({"customer_number": customers})
    for k, ids in refs.items():
        df[f"ref_{k}"] = np.isin(customers, ids)
    return df


def holdout_split(positives: np.ndarray, frac: float = 0.3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split the labeled positives into (train, holdout). Keep the holdout untouched while you
    tune rules/models, then report recall on it."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(np.asarray(positives))
    k = int(round(len(p) * frac))
    return np.sort(p[k:]), np.sort(p[:k])


def pu_report(scores: pd.Series, positives: np.ndarray, label_fraction: float,
              ks=(100, 250, 500, 1000, 2000), population: np.ndarray | None = None) -> pd.DataFrame:
    """Evaluate a ranking (higher score = more likely) against a *random sample* of positives.

    scores:          Series indexed by customer_number
    positives:       labeled positive customer numbers (e.g. a holdout)
    label_fraction:  share of all true positives that the labeled list represents
                     (0.25 for EV/HP; for a holdout of 30% of the EV list use 0.25 * 0.3)

    For each k: ``hits`` = labeled positives in the top k, ``recall`` = hits / labeled positives
    present, and ``est_precision`` = hits / label_fraction / k, which estimates the share of
    the top k that are true positives, because the labeled list is a random sample of them.
    Also reports the AUC of labeled positives vs all other customers (a lower bound on the
    true AUC when unlabeled customers include hidden positives).
    """
    s = scores.dropna()
    if population is not None:
        s = s[s.index.isin(population)]
    pos = np.intersect1d(np.asarray(positives), s.index.to_numpy())
    order = s.sort_values(ascending=False)
    is_pos = order.index.isin(pos)
    rows = []
    for k in ks:
        if k > len(order):
            continue
        h = int(is_pos[:k].sum())
        rows.append({"k": k, "hits": h, "recall": h / max(len(pos), 1),
                     "est_precision": min(1.0, h / label_fraction / k)})
    out = pd.DataFrame(rows)
    r = order.rank(ascending=True)
    n1, n0 = len(pos), len(order) - len(pos)
    auc = (r[is_pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
    out.attrs.update({"n_labeled_in_population": n1, "auc_labeled_vs_rest": auc,
                      "implied_total_positives": n1 / label_fraction})
    return out


def implied_prevalence(refs: dict[str, np.ndarray], n_customers: int) -> pd.DataFrame:
    """Population sizes implied by the sampling fractions in the workbook."""
    rows = []
    for k, ids in refs.items():
        frac = LABEL_SAMPLE_FRACTION[k]
        rows.append({"technology": k, "labeled": len(ids), "sample_fraction": frac,
                     "implied_total": round(len(ids) / frac), "implied_share": len(ids) / frac / n_customers})
    return pd.DataFrame(rows)
