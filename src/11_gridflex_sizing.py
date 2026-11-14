"""Run optional discrete battery-size comparison on prepared AMI data.

From repository root:
PYTHONPATH=src python src/11_gridflex_sizing.py --limit 5
"""
import argparse
from avangrid.config import Paths
from avangrid.gridflex import GridFlexConfig
from avangrid.pipeline import load_project
from avangrid.sizing import SizingAssumptions, rank_sized_customers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=5)
    ap.add_argument('--critical-top', type=int, default=100)
    ap.add_argument('--screen-method', choices=['critical_mean', 'individual_peak'], default='critical_mean')
    ap.add_argument('--capacity-value', type=float, default=200.0)
    ap.add_argument('--fixed-install', type=float, default=5000.0)
    ap.add_argument('--per-unit-capex', type=float, default=4500.0)
    ap.add_argument('--per-unit-om', type=float, default=50.0)
    ap.add_argument('--dr-cost', type=float, default=150.0)
    args = ap.parse_args()
    cfg = GridFlexConfig(capacity_value_per_kw_year=args.capacity_value)
    cost = SizingAssumptions(fixed_install_usd=args.fixed_install,
                             capex_per_unit_usd=args.per_unit_capex,
                             annual_om_per_unit_usd=args.per_unit_om,
                             dr_enrollment_usd=args.dr_cost)
    P = Paths().ensure()
    project = load_project(P)
    result = rank_sized_customers(project, limit=args.limit, critical_top=args.critical_top,
                                  screen_method=args.screen_method, cfg=cfg, assumptions=cost)
    if result.empty:
        print('No analysis-ready customers found')
        return
    output = P.outputs / 'part3_battery_sizing.csv'
    result.to_csv(output, index=False)
    summary = result[result.recommended].sort_values('customer_number')
    print(f'Saved {len(result)} scenarios for {len(summary)} customers to {output}')
    print(summary[['customer_number','option','battery_units','battery_kw','battery_kwh',
                   'peak_reduction_kw','critical_hour_reduction_kw','capex_usd',
                   'npv_usd','breakeven_capex_usd']].round(2).to_string(index=False))
    print('NOTE: all prices/capacity value are assumptions. Baseline can be optimal.')


if __name__ == '__main__':
    main()
