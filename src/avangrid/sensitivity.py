"""GridFlex two-way sensitivity: re-dispatch for DR flexibility; revalue capacity.

Scenario assumptions only. Critical hours based on sample-wide system load,
not feeder constraints. Historical demand, perfect foresight and assumed DR.
"""
from __future__ import annotations
from dataclasses import replace
import math
import numpy as np
import pandas as pd
from .gridflex import GridFlexConfig, screen_customer_scores
from .sizing import SizingAssumptions, compare_sizes

DEFAULT_CAPACITY_VALUES = (100.0, 200.0, 300.0, 400.0)
DEFAULT_DR_FLEXIBILITY = (0.025, 0.05, 0.075, 0.10)


def evaluate_sensitivity(project, limit=5, critical_top=100,
                         capacity_values=DEFAULT_CAPACITY_VALUES,
                         dr_fractions=DEFAULT_DR_FLEXIBILITY,
                         cfg=GridFlexConfig(), costs=SizingAssumptions(),
                         screen_method='critical_mean'):
    """Return all options and best-choice rows for each sensitivity cell.

    Re-run DR and all battery dispatches for every DR fraction, because DR
    shifts the input load for battery+DR. Capacity value enters economics only
    (not the optimizer's dispatch objective), so NPV can be recomputed exactly
    for alternative $/kW-year values using the *same* dispatch outcomes.
    """
    if limit < 1 or critical_top < 1:
        raise ValueError('limit and critical_top must be positive')
    caps = tuple(float(x) for x in capacity_values)
    flexes = tuple(float(x) for x in dr_fractions)
    if not caps or not flexes or any(not math.isfinite(x) or x < 0 for x in caps):
        raise ValueError('capacity values must be finite and nonnegative')
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in flexes):
        raise ValueError('DR fractions must be between zero and one')
    if not (0 <= cfg.discount_rate) or cfg.years < 1:
        raise ValueError('discount rate must be nonnegative and years positive')
    costs.validate()
    scores, critical = screen_customer_scores(project, critical_top, screen_method)
    annuity = sum((1+cfg.discount_rate)**(-year) for year in range(1, cfg.years+1))
    all_rows = []
    for score, customer in scores[:limit]:
        load = project.series(customer)
        for fraction in flexes:
            flex_cfg = replace(cfg, dr_shift_fraction=fraction)
            base = compare_sizes(load, critical, project.hours, cfg=flex_cfg,
                                 assumptions=costs)
            for capacity_value in caps:
                table = base.copy()
                delta = capacity_value - cfg.capacity_value_per_kw_year
                table['npv_usd'] = (table['npv_usd'] + delta *
                                    table['critical_hour_reduction_kw'] * annuity)
                table['system_capacity_value_usd'] = (
                    table['critical_hour_reduction_kw'] * capacity_value)
                table['annual_net_benefit_usd'] = (
                    table['annual_net_benefit_usd'] + delta * table['critical_hour_reduction_kw'])
                table['breakeven_capex_usd'] = (
                    table['breakeven_capex_usd'] + delta * table['critical_hour_reduction_kw'] * annuity)
                table['capex_gap_usd'] = table['capex_usd'] - table['breakeven_capex_usd']
                table['simple_payback_years'] = np.where(
                    table['annual_net_benefit_usd'] > 0,
                    table['capex_usd'] / table['annual_net_benefit_usd'].clip(lower=1e-30),
                    np.inf)
                # Recompute choice with baseline-first tie-breaking.
                table['recommended'] = False
                order = table['option'].map({'baseline': 0, 'dr': 1}).fillna(2)
                best = table.assign(_tie=order).sort_values(
                    ['npv_usd', '_tie'], ascending=[False, True]).index[0]
                table.loc[best, 'recommended'] = True
                table.insert(0, 'customer_number', customer)
                table.insert(1, 'screen_score_kw', score)
                table.insert(2, 'dr_shift_fraction', fraction)
                table.insert(3, 'capacity_value_per_kw_year', capacity_value)
                all_rows.append(table)
    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()
    details = pd.concat(all_rows, ignore_index=True)
    best = details.loc[details.recommended].copy()
    if (best.groupby(['customer_number', 'dr_shift_fraction',
                       'capacity_value_per_kw_year']).size() != 1).any():
        raise AssertionError('every sensitivity cell must choose one option')
    return details, best


def summarize_best(best):
    """Count the winning investments, average NPV, and average reduction."""
    if best.empty:
        return pd.DataFrame()
    group = ['dr_shift_fraction', 'capacity_value_per_kw_year']
    summary = best.groupby(group, as_index=False).agg(
        customers=('customer_number','size'),
        average_best_npv_usd=('npv_usd', 'mean'),
        mean_critical_hour_reduction_kw=('critical_hour_reduction_kw','mean'))
    counts = (best.groupby(group+['option']).size().unstack(fill_value=0)
              .reset_index())
    return summary.merge(counts, on=group, how='left')
