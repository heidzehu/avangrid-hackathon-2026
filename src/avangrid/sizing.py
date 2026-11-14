"""Discrete 1-vs-2 battery sizing on GridFlex's critical-hour screening sample.

All $ inputs are scenario assumptions. Deployment must be confirmed on actual
feeder constraints and installation feasibility. Perfect-foresight dispatch is
an optimistic modeling bound, not achieved customer behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
import pandas as pd

from .config import BatterySpec
from .gridflex import GridFlexConfig, compare_scenarios, screen_customer_scores


@dataclass(frozen=True)
class SizingAssumptions:
    # Illustrative decomposition: two units cost $14,000 total; one $9,500.
    # NOT contractor prices, program tariff, or verified cost curve.
    fixed_install_usd: float = 5_000.0
    capex_per_unit_usd: float = 4_500.0
    annual_om_per_unit_usd: float = 50.0
    dr_enrollment_usd: float = 150.0
    units: tuple[int, ...] = (1, 2)

    def validate(self) -> None:
        if not self.units or len(set(self.units)) != len(self.units) or any(type(n) is not int or n < 1 for n in self.units):
            raise ValueError('units must be a nonempty sequence of distinct positive integers')
        if any(not np.isfinite(x) or x < 0 for x in (
            self.fixed_install_usd, self.capex_per_unit_usd,
            self.annual_om_per_unit_usd, self.dr_enrollment_usd)):
            raise ValueError('installation, operating and DR costs must be finite nonnegative')

    def battery_capex(self, units: int) -> float:
        return self.fixed_install_usd + units * self.capex_per_unit_usd


def compare_sizes(load, critical, hours, cfg=GridFlexConfig(),
                  assumptions=SizingAssumptions()) -> pd.DataFrame:
    """Return candidate interventions at all discrete battery sizes.

    Baseline and DR-only are computed once. Battery and battery+DR are
    compared for each allowed count of whole units. Among modeled scenarios,
    the recommendation is the maximum-NPV choice; baseline NPV is zero.
    """
    assumptions.validate()
    records = []
    for units in assumptions.units:
        spec = BatterySpec(units=units)
        unit_cfg = replace(cfg, capex=assumptions.battery_capex(units),
                           annual_om=units*assumptions.annual_om_per_unit_usd)
        scenarios = compare_scenarios(load, critical, hours, cfg=unit_cfg, spec=spec)
        for _, row in scenarios.iterrows():
            scenario = row['scenario']
            if scenario in ('baseline', 'dr') and units != assumptions.units[0]:
                continue
            r = row.to_dict()
            # Existing compare_scenarios prices DR at $150. Correct for
            # alternative DR enrollment cost using PV of upfront difference.
            if scenario in ('dr', 'battery_plus_dr'):
                correction = assumptions.dr_enrollment_usd - 150.0
                r['npv_usd'] -= correction
                r['capex_gap_usd'] += correction
                if r['critical_hour_reduction_kw'] > 1e-10:
                    annuity = sum(1/(1+cfg.discount_rate)**y for y in range(1,cfg.years+1))
                    r['breakeven_capacity_value_per_kw_year'] += (
                        correction / annuity / r['critical_hour_reduction_kw'])
                r['simple_payback_years'] = (
                    (assumptions.dr_enrollment_usd + (assumptions.battery_capex(units) if scenario == 'battery_plus_dr' else 0))
                    / r['annual_net_benefit_usd'] if r['annual_net_benefit_usd'] > 0 else np.inf
                )
                # Updated gross CAPEX for the variant; economic break-even
                # operating benefit is unchanged.
                r['capex_usd'] = assumptions.dr_enrollment_usd + (
                    assumptions.battery_capex(units) if scenario == 'battery_plus_dr' else 0)
            else:
                r['capex_usd'] = (assumptions.battery_capex(units) if scenario == 'battery' else 0.0)
            if scenario == 'baseline':
                size = 0
            elif scenario == 'dr':
                size = 0
            else:
                size = units
            r.update(battery_units=size, battery_kwh=size*spec.energy_kwh_per_unit,
                     battery_kw=size*spec.continuous_kw_per_unit,
                     annual_om_usd=size*assumptions.annual_om_per_unit_usd)
            records.append(r)
    result = pd.DataFrame(records)
    result['option'] = result.apply(lambda r: (
        r['scenario'] if r['battery_units'] == 0 else
        f"{r['scenario']}_{int(r['battery_units'])}unit"), axis=1)
    result['recommended'] = False
    # Prefer baseline on exact ties; never choose negative-NPV deployment.
    order = {'baseline': 0, 'dr': 1}
    result['_tie'] = result.option.map(order).fillna(2)
    best = result.sort_values(['npv_usd', '_tie'], ascending=[False, True]).index[0]
    result.loc[best, 'recommended'] = True
    return result.drop(columns=['_tie'])


def rank_sized_customers(project, limit=5, cfg=GridFlexConfig(), critical_top=100,
                         screen_method='critical_mean',
                         assumptions=SizingAssumptions()) -> pd.DataFrame:
    """Score candidates on *system* critical hours, compare discrete sizes.

    Screen is a proxy, not a feeder study; selected customer lists can overlap.
    """
    if limit < 1:
        raise ValueError('limit must be >= 1')
    scores, critical = screen_customer_scores(project, critical_top, screen_method)
    frames = []
    for score, customer in scores[:limit]:
        rows = compare_sizes(project.series(customer), critical, project.hours, cfg,
                             assumptions)
        rows.insert(0, 'customer_number', customer)
        rows.insert(1, 'screen_method', screen_method)
        rows.insert(2, 'screen_score_kw', score)
        frames.append(rows)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result['rank_npv'] = result.groupby('option').npv_usd.rank(ascending=False, method='min')
    return result.sort_values(['customer_number', 'recommended', 'npv_usd'], ascending=[True, False, False])
