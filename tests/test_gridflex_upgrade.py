import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from avangrid.gridflex import (GridFlexConfig, screen_customer_scores, economics,
                              rank_customers)

class SyntheticProject:
    def __init__(self):
        self.hours=pd.DataFrame({'ts':pd.date_range('2025-01-01',periods=24,freq='h')})
        self.table=pd.DataFrame({'customer_number':[101,202], 'analysis_ready':[True,True]})
        self.system=pd.DataFrame({'slot':np.arange(24),
                                  'rank':np.where(np.arange(24)==18,0,np.arange(24)+1)})
        self.loads={101:np.array([20.]+[1.]*23),202:np.array([2.]*18+[10.]+[2.]*5)}
    def series(self,customer_id):return self.loads[customer_id]

def test_critical_screen_changes_candidate():
    project=SyntheticProject()
    a,_=screen_customer_scores(project,critical_top=1,method='critical_mean')
    b,_=screen_customer_scores(project,critical_top=1,method='individual_peak')
    assert a[0][1]==202
    assert b[0][1]==101

def test_breakeven_economics_identity():
    l=np.ones(24);l[18]=10
    net=l.copy();net[18]=8
    h=pd.DataFrame({'ts':pd.date_range('2025-01-01',periods=24,freq='h')})
    c=np.zeros(24,bool);c[18]=True
    cfg=GridFlexConfig(annual_om=100,capacity_value_per_kw_year=200,capex=14000)
    result=economics(l,net,c,h,cfg)
    assert np.isclose(result['npv_usd'],-result['capex_gap_usd'])
    assert np.isfinite(result['breakeven_capacity_value_per_kw_year'])
    cfg2=GridFlexConfig(capacity_value_per_kw_year=result['breakeven_capacity_value_per_kw_year'])
    assert abs(economics(l,net,c,h,cfg2)['npv_usd'])<1e-7

def test_no_critical_reduction_threshold_nan():
    l=np.ones(24);h=pd.DataFrame({'ts':pd.date_range('2025-01-01',periods=24,freq='h')})
    c=np.zeros(24,bool);c[18]=True
    result=economics(l,l,c,h)
    assert np.isnan(result['breakeven_capacity_value_per_kw_year'])

def test_ranked_screen_and_new_columns():
    p=SyntheticProject()
    ranked=rank_customers(p,limit=1,critical_top=1)
    assert len(ranked)==4
    assert set(ranked.customer_number)=={202}
    assert set(ranked.screen_method)=={'critical_mean'}
    assert {'rank_npv','breakeven_capex_usd','breakeven_capacity_value_per_kw_year','capex_gap_usd'}<=set(ranked.columns)
