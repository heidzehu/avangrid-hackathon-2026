import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from avangrid.gridflex import GridFlexConfig, optimize_day, dispatch_year, demand_response, compare_scenarios
from avangrid.config import PROGRAM_BATTERY

def profile():
    l=np.ones(24);l[18:21]=8
    critical=np.zeros(24,bool);critical[18:21]=True
    return l,critical

def test_optimizer_physical_and_cyclic():
    l,c=profile();r=optimize_day(l,c)
    assert np.all(r['net_kw']>=-1e-7)
    assert np.max(abs(r['battery_kw']))<=PROGRAM_BATTERY.power_kw+1e-7
    assert np.min(r['soc_kwh'])>=PROGRAM_BATTERY.energy_kwh*.1-1e-7
    assert np.max(r['soc_kwh'])<=PROGRAM_BATTERY.energy_kwh+1e-7
    assert abs(r['soc_kwh'][-1]-PROGRAM_BATTERY.energy_kwh*.5)<1e-6
    assert r['net_kw'].max()<l.max()
    assert abs((r['charge_kw'].sum()*PROGRAM_BATTERY.one_way_efficiency)-
               r['discharge_kw'].sum()/PROGRAM_BATTERY.one_way_efficiency)<1e-6

def test_dr_energy_neutral_and_no_new_peak():
    l,c=profile();net=demand_response(l,c)
    assert np.isclose(net.sum(),l.sum())
    assert net.max()<=l.max()+1e-8
    assert net[c].mean()<l[c].mean()

def test_scenario_outputs():
    l,c=profile();h=pd.DataFrame({'ts':pd.date_range('2025-09-01',periods=24,freq='h')})
    r=compare_scenarios(l,c,h)
    assert set(r.scenario)=={'baseline','dr','battery','battery_plus_dr'}
    assert r.loc[r.scenario=='baseline','npv_usd'].iloc[0]==0
    assert r.loc[r.scenario=='battery','peak_reduction_kw'].iloc[0]>0

def test_multiple_days():
    l,c=profile();r=dispatch_year(np.tile(l,2),np.tile(c,2))
    assert len(r['net_kw'])==48
    assert abs(r['soc_kwh'][23]-r['soc_kwh'][47])<1e-7
