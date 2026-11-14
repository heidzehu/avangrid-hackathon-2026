"""Part 3: run GridFlex on cleaned AMI profiles; no raw data in repository ZIP.

From repo root: PYTHONPATH=src python src/10_gridflex.py --limit 30
"""
import argparse
from avangrid.config import Paths
from avangrid.pipeline import load_project
from avangrid.gridflex import GridFlexConfig, rank_customers

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--limit',type=int,default=30)
    ap.add_argument('--screen-method',choices=['critical_mean','individual_peak'],default='critical_mean')
    ap.add_argument('--critical-top',type=int,default=100)
    ap.add_argument('--capacity-value',type=float,default=200)
    ap.add_argument('--capex',type=float,default=14000)
    args=ap.parse_args()
    P=Paths().ensure()
    D=load_project(P)
    cfg=GridFlexConfig(capex=args.capex,capacity_value_per_kw_year=args.capacity_value)
    result=rank_customers(D,args.limit,cfg,args.critical_top,args.screen_method)
    path=P.outputs/'part3_bess_candidates.csv'
    result.to_csv(path,index=False)
    print(f'Wrote {len(result)} scenario rows to {path}; screening={args.screen_method}')
    print(result[result.scenario=='battery_plus_dr'].head(10).to_string(index=False))

if __name__=='__main__':main()
