import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
import pytest
from avangrid.sizing import SizingAssumptions, compare_sizes, rank_sized_customers
from avangrid.gridflex import GridFlexConfig


def sample():
    l=np.ones(24)*2
    l[18:21]=9
    critical=np.zeros(24,bool);critical[18:21]=True
    hours=pd.DataFrame({'ts':pd.date_range('2025-09-01',periods=24,freq='h')})
    return l,critical,hours


def test_two_sizes_and_alternatives():
    l,c,h=sample()
    d=compare_sizes(l,c,h)
    assert set(d.option)=={'baseline','dr','battery_1unit','battery_plus_dr_1unit',
                          'battery_2unit','battery_plus_dr_2unit'}
    assert d.loc[d.option=='battery_1unit','battery_kwh'].iloc[0]==10.08
    assert d.loc[d.option=='battery_2unit','battery_kw'].iloc[0]==7.68
    assert d.loc[d.option=='battery_1unit','capex_usd'].iloc[0]==9500
    assert d.loc[d.option=='battery_2unit','capex_usd'].iloc[0]==14000
    assert d.recommended.sum()==1
    assert d.loc[d.recommended,'npv_usd'].iloc[0]>=0


def test_very_high_capex_makes_baseline_or_dr_preferred():
    l,c,h=sample()
    d=compare_sizes(l,c,h,assumptions=SizingAssumptions(fixed_install_usd=1e7))
    assert d.loc[d.recommended,'battery_units'].iloc[0]==0


def test_invalid_assumptions():
    l,c,h=sample()
    with pytest.raises(ValueError):
        compare_sizes(l,c,h,assumptions=SizingAssumptions(units=()))
    with pytest.raises(ValueError):
        compare_sizes(l,c,h,assumptions=SizingAssumptions(capex_per_unit_usd=-5))


def test_alternative_dr_price_adjusts_npv():
    l,c,h=sample()
    a=compare_sizes(l,c,h,assumptions=SizingAssumptions(dr_enrollment_usd=150))
    b=compare_sizes(l,c,h,assumptions=SizingAssumptions(dr_enrollment_usd=350))
    for option in ['dr','battery_plus_dr_1unit','battery_plus_dr_2unit']:
        assert np.isclose(b.set_index('option').loc[option,'npv_usd'],
                          a.set_index('option').loc[option,'npv_usd']-200)


def test_rank_sized_on_synthetic_project():
    l,c,h=sample()
    class P:
        hours=h
        system=pd.DataFrame({'slot':np.arange(24), 'rank':np.arange(24)})
        table=pd.DataFrame({'customer_number':[99], 'analysis_ready':[True]})
        def series(self, customer):return l
    r=rank_sized_customers(P(),limit=1,critical_top=2)
    assert len(r)==6 and r.recommended.sum()==1
    assert set(r.customer_number)=={99}
