import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
import pytest
from avangrid.gridflex import GridFlexConfig
from avangrid.sensitivity import evaluate_sensitivity, summarize_best


class Project:
    def __init__(self):
        self.hours=pd.DataFrame({'ts':pd.date_range('2025-01-01',periods=24,freq='h')})
        self.system=pd.DataFrame({'slot':list(range(24)), 'rank':list(range(24))})
        self.table=pd.DataFrame({'customer_number':[1], 'analysis_ready':[True]})
    def series(self, customer):
        load=np.ones(24)*2
        load[:4]=8
        return load


def test_sensitivity_row_count_and_single_winner():
    detail,best=evaluate_sensitivity(Project(),limit=1,critical_top=4,
                    capacity_values=(100,200),dr_fractions=(0.0,0.1))
    assert len(detail)==24 and len(best)==4
    assert best.groupby(['dr_shift_fraction','capacity_value_per_kw_year']).size().eq(1).all()
    assert set(detail.option)=={'baseline','dr','battery_1unit','battery_2unit',
             'battery_plus_dr_1unit','battery_plus_dr_2unit'}
    assert len(summarize_best(best))==4


def test_capacity_revaluation_is_exact_and_no_dispatch_change():
    detail,_=evaluate_sensitivity(Project(),capacity_values=(100,300),dr_fractions=(0.1,))
    lo=detail[detail.capacity_value_per_kw_year==100].set_index('option')
    hi=detail[detail.capacity_value_per_kw_year==300].set_index('option')
    ann=sum(1/1.07**y for y in range(1,16))
    np.testing.assert_allclose((hi.npv_usd-lo.npv_usd).to_numpy(),
           200*lo.critical_hour_reduction_kw.to_numpy()*ann)
    np.testing.assert_allclose(lo.critical_hour_reduction_kw.to_numpy(),
                               hi.critical_hour_reduction_kw.to_numpy())


def test_dr_flex_changes_dispatch():
    detail,_=evaluate_sensitivity(Project(),critical_top=4,capacity_values=(200,),dr_fractions=(0.0,0.1))
    a=detail[(detail.option=='dr') & (detail.dr_shift_fraction==0)].iloc[0]
    b=detail[(detail.option=='dr') & (detail.dr_shift_fraction==0.1)].iloc[0]
    assert a.critical_hour_reduction_kw==0
    assert b.critical_hour_reduction_kw>a.critical_hour_reduction_kw


@pytest.mark.parametrize('vals,fracs', [((-1,),(.1,)),((100,),(-.1,)),((100,),(1.1,)),((),(.1,))])
def test_invalid_inputs(vals,fracs):
    with pytest.raises(ValueError):
        evaluate_sensitivity(Project(),capacity_values=vals,dr_fractions=fracs)
