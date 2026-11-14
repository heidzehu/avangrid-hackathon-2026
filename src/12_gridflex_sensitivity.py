"""Two-way grid value × feasible DR shift sensitivity, using prepared AMI data.

From repository root:
PYTHONPATH=src python src/12_gridflex_sensitivity.py --limit 5
"""
import argparse
from avangrid.config import Paths
from avangrid.gridflex import GridFlexConfig
from avangrid.pipeline import load_project
from avangrid.sizing import SizingAssumptions
from avangrid.sensitivity import (evaluate_sensitivity, summarize_best,
                                  DEFAULT_CAPACITY_VALUES, DEFAULT_DR_FLEXIBILITY)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--critical-top', type=int, default=100)
    parser.add_argument('--screen-method', choices=['critical_mean','individual_peak'],
                        default='critical_mean')
    parser.add_argument('--capacity-values', nargs='+', type=float,
                        default=DEFAULT_CAPACITY_VALUES)
    parser.add_argument('--dr-fractions', nargs='+', type=float,
                        default=DEFAULT_DR_FLEXIBILITY,
                        help='Fraction of usage shifted at eligible critical hours: 0.025=2.5%%')
    parser.add_argument('--fixed-install', type=float, default=5000)
    parser.add_argument('--per-unit-capex', type=float, default=4500)
    parser.add_argument('--per-unit-om', type=float, default=50)
    parser.add_argument('--dr-cost', type=float, default=150)
    args = parser.parse_args()
    paths = Paths().ensure()
    data = load_project(paths)
    cfg = GridFlexConfig()
    costs = SizingAssumptions(fixed_install_usd=args.fixed_install,
                              capex_per_unit_usd=args.per_unit_capex,
                              annual_om_per_unit_usd=args.per_unit_om,
                              dr_enrollment_usd=args.dr_cost)
    detail, best = evaluate_sensitivity(
        data, limit=args.limit, critical_top=args.critical_top,
        capacity_values=args.capacity_values, dr_fractions=args.dr_fractions,
        cfg=cfg, costs=costs, screen_method=args.screen_method)
    if detail.empty:
        print('No eligible customers'); return
    summary = summarize_best(best)
    for filename, table in [
        ('part3_sensitivity_all_options.csv', detail),
        ('part3_sensitivity_best_options.csv', best),
        ('part3_sensitivity_summary.csv', summary),
    ]:
        target = paths.outputs / filename
        table.to_csv(target, index=False)
        print(f'Saved {len(table)} rows: {target}')
    print('\nSENSITIVITY MATRIX (best options per customer):')
    print(summary.round(2).to_string(index=False))
    print('\nNOTE: Illustrative utility value and DR flexibility; not feeder-validated.')
    print('DR fractions change assumed shiftable load, not proven customer enrollment.')
    print('Capacity value changes cash flows only; dispatch objective stays unchanged.')


if __name__ == '__main__':
    main()
