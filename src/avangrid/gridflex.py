"""GridFlex: auditable, day-ahead BTM dispatch with cyclic daily SOC.

Decision model, not a reproduction of observed customer behavior. Hourly AMI
imports are treated as average kW over the hour. No PV-export inference.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from .config import PROGRAM_BATTERY, BatterySpec

@dataclass(frozen=True)
class GridFlexConfig:
    reserve_frac: float = 0.10
    start_soc_frac: float = 0.50
    energy_price: float = 0.20  # illustrative $/kWh; replace with actual tariff
    peak_price: float = 1.0     # $/kW per *daily* peak proxy; not an actual demand tariff
    critical_hour_weight: float = 0.50  # $/kWh additional objective weight
    throughput_penalty: float = 0.002  # $/kWh of charge or discharge, regularization
    dr_shift_fraction: float = 0.10  # assumed fraction of hourly usage transferable
    dr_max_hours: int = 4
    capex: float = 14000.0
    annual_om: float = 100.0
    capacity_value_per_kw_year: float = 200.0  # scenario assumption, not a tariff
    discount_rate: float = 0.07
    years: int = 15
    demand_charge_per_kw_month: float = 0.0


def optimize_day(load, critical=None, price=None, spec=PROGRAM_BATTERY,
                 cfg=GridFlexConfig()):
    """Solve LP: charge c, discharge d, end-of-hour SOC s, daily peak p.

    SOC is cyclic at cfg.start_soc_frac, preventing one-time depletion.
    No exports; positive grid imports. Throughput penalty discourages simultaneous
    charge/discharge, but the relaxed LP is not a binary on/off MILP.
    """
    L = np.asarray(load, float).ravel()
    T = len(L)
    if T == 0 or not np.isfinite(L).all() or (L < 0).any():
        raise ValueError('load must be nonnegative finite and nonempty')
    if not (0 <= cfg.reserve_frac <= cfg.start_soc_frac <= 1):
        raise ValueError('invalid reserve/initial SOC fractions')
    crit = np.zeros(T) if critical is None else np.broadcast_to(np.asarray(critical, float), (T,))
    rate = np.broadcast_to(np.asarray(cfg.energy_price if price is None else price, float), (T,))
    if (crit < 0).any() or not np.isfinite(rate).all():
        raise ValueError('critical weights must be nonnegative; prices finite')
    eta = spec.one_way_efficiency
    # x = [c(T), d(T), s(T), p]
    obj = np.zeros(3*T+1)
    obj[:T] = rate + cfg.critical_hour_weight*crit + cfg.throughput_penalty
    obj[T:2*T] = -rate - cfg.critical_hour_weight*crit + cfg.throughput_penalty
    obj[-1] = cfg.peak_price
    Aeq = lil_matrix((T+1,3*T+1))
    beq = np.zeros(T+1)
    for t in range(T):
        Aeq[t,t] = -eta
        Aeq[t,T+t] = 1/eta
        Aeq[t,2*T+t] = 1
        if t: Aeq[t,2*T+t-1] = -1
        else: beq[t] = cfg.start_soc_frac*spec.energy_kwh
    Aeq[T, 3*T-1] = 1
    beq[T] = cfg.start_soc_frac*spec.energy_kwh
    Aub = lil_matrix((2*T,3*T+1))
    bub = np.zeros(2*T)
    for t in range(T):
        # d - c <= L (no export), L + c - d <= peak
        Aub[t,t] = -1; Aub[t,T+t] = 1; bub[t] = L[t]
        Aub[T+t,t] = 1; Aub[T+t,T+t] = -1; Aub[T+t,-1] = -1
        bub[T+t] = -L[t]
    bounds = ([(0,spec.power_kw)]*(2*T) +
              [(cfg.reserve_frac*spec.energy_kwh,spec.energy_kwh)]*T +
              [(0,None)])
    solution = linprog(obj, A_ub=Aub.tocsr(), b_ub=bub,
                       A_eq=Aeq.tocsr(), b_eq=beq, bounds=bounds, method='highs')
    if not solution.success:
        raise RuntimeError(f'Battery LP failed: {solution.message}')
    c,d,s = (solution.x[i*T:(i+1)*T] for i in range(3))
    net = np.maximum(0, L+c-d)
    return {'net_kw': net, 'battery_kw': d-c, 'soc_kwh': s,
            'charge_kw': c, 'discharge_kw': d, 'daily_peak_kw': solution.x[-1]}


def dispatch_year(load, critical=None, price=None, spec=PROGRAM_BATTERY,
                  cfg=GridFlexConfig()):
    """Independent cyclic 24h plans (transparent day-ahead perfect-foresight bound)."""
    L = np.asarray(load,float).ravel()
    if len(L)%24: raise ValueError('hourly series must contain complete 24-hour days')
    crit = np.zeros(len(L)) if critical is None else np.broadcast_to(np.asarray(critical,float),L.shape)
    rate = np.broadcast_to(np.asarray(cfg.energy_price if price is None else price,float),L.shape)
    out = {k:np.empty(len(L)) for k in ('net_kw','battery_kw','soc_kwh','charge_kw','discharge_kw')}
    for i in range(0,len(L),24):
        day = optimize_day(L[i:i+24],crit[i:i+24],rate[i:i+24],spec,cfg)
        for k in out: out[k][i:i+24] = day[k]
    return out


def demand_response(load, critical, cfg=GridFlexConfig()):
    """Illustrative daily energy-neutral DR: shift critical-hour consumption to
    off-peak hours with the greatest headroom, never exceeding pre-DR daily peak.
    Not an estimated adoption/participation model or verified flexible end use.
    """
    L=np.asarray(load,float).ravel()
    crit=np.broadcast_to(np.asarray(critical,bool),L.shape)
    if len(L)%24: raise ValueError('DR needs complete days')
    net=L.copy()
    for i in range(0,len(L),24):
        day=net[i:i+24]; mask=crit[i:i+24]
        source=np.where(mask)[0]
        source=source[np.argsort(-day[source])][:cfg.dr_max_hours]
        target=np.where(~mask)[0]
        for s in source:
            moving=day[s]*cfg.dr_shift_fraction
            # Conserve energy and respect peak; only transfer what fits
            for t in target[np.argsort(day[target])]:
                amount=min(moving,max(0, L[i:i+24].max()-day[t]))
                day[s]-=amount;day[t]+=amount;moving-=amount
                if moving<1e-9:break
    return net


def economics(load, net, critical, hours, cfg=GridFlexConfig(),
              capex=None, annual_om=None):
    """Project-level benefit accounting. Capacity value and demand-charge savings
    are distinct mechanisms; avoid counting both if they reward the same service.
    """
    L=np.asarray(load,float).ravel(); N=np.asarray(net,float).ravel()
    crit=np.broadcast_to(np.asarray(critical,bool),L.shape)
    if len(L)!=len(N) or len(hours)!=len(L): raise ValueError('hour index mismatch')
    rate=np.broadcast_to(np.asarray(cfg.energy_price,float),L.shape)
    energy_saving=float(np.dot(rate,L-N))
    months=pd.to_datetime(hours['ts']).dt.to_period('M')
    peak_saved=[]
    for m in months.unique():
        idx=(months==m).to_numpy()
        peak_saved.append(max(0,float(L[idx].max()-N[idx].max())))
    bill_demand_saving=cfg.demand_charge_per_kw_month*sum(peak_saved)
    critical_reduction=max(0,float(np.mean(L[crit]-N[crit]))) if crit.any() else 0.0
    capacity_value=critical_reduction*cfg.capacity_value_per_kw_year
    gross=energy_saving+bill_demand_saving+capacity_value
    operating=gross-(cfg.annual_om if annual_om is None else annual_om)
    capital=cfg.capex if capex is None else capex
    annuity=sum(1/(1+cfg.discount_rate)**y for y in range(1,cfg.years+1))
    npv=-capital+operating*annuity
    # Break-even CAPEX is the maximum up-front investment supported by the
    # modeled annual net benefit (before paying for the battery). Negative
    # values mean there is no nonnegative profitable CAPEX under this case.
    breakeven_capex=operating*annuity
    om=cfg.annual_om if annual_om is None else annual_om
    # Solve NPV=0 for the $/kW-year capacity value. This is an implied
    # threshold, NOT a forecast of actual utility remuneration.
    if critical_reduction > 1e-10:
        breakeven_capacity_value=(capital/annuity - energy_saving - bill_demand_saving + om)/critical_reduction
    else:
        breakeven_capacity_value=float('nan')
    return {'energy_bill_savings_usd':energy_saving,
            'demand_charge_savings_usd':bill_demand_saving,
            'system_capacity_value_usd':capacity_value,
            'critical_hour_reduction_kw':critical_reduction,
            'annual_net_benefit_usd':operating,'npv_usd':npv,
            'breakeven_capex_usd':breakeven_capex,
            'breakeven_capacity_value_per_kw_year':breakeven_capacity_value,
            'capex_gap_usd':capital-breakeven_capex,
            'simple_payback_years':capital/operating if operating>0 else np.inf,
            'monthly_peak_reduction_kw_mean':float(np.mean(peak_saved)),
            'annual_energy_shift_net_kwh':float((L-N).sum())}


def compare_scenarios(load, critical, hours, cfg=GridFlexConfig(),spec=PROGRAM_BATTERY):
    L=np.asarray(load,float).ravel()
    dr=demand_response(L,critical,cfg)
    battery=dispatch_year(L,critical,cfg=cfg,spec=spec)['net_kw']
    combo=dispatch_year(dr,critical,cfg=cfg,spec=spec)['net_kw']
    records=[]
    for name,net,capital,om in [('baseline',L,0,0),('dr',dr,150,0),
                                 ('battery',battery,cfg.capex,cfg.annual_om),
                                 ('battery_plus_dr',combo,cfg.capex+150,cfg.annual_om)]:
        stats=economics(L,net,critical,hours,cfg,capital,om)
        records.append({'scenario':name,'peak_before_kw':float(max(L)),
                        'peak_after_kw':float(max(net)),
                        'peak_reduction_kw':float(max(L)-max(net)),**stats})
    return pd.DataFrame(records)


def screen_customer_scores(project, critical_top=100, method='critical_mean'):
    """Screen analysis-ready customers without running expensive LP dispatch.

    critical_mean: average customer load at the highest-ranked *system* hours,
                   a coincidence proxy (not a feeder-specific constraint).
    individual_peak: legacy customer maximum kW screening benchmark.
    Returns descending (score, customer ID) pairs and a critical-hour mask.
    """
    if method not in ('critical_mean', 'individual_peak'):
        raise ValueError('screen method must be critical_mean or individual_peak')
    if critical_top < 1:
        raise ValueError('critical_top must be >= 1')
    T=project.table
    if 'analysis_ready' not in T:
        raise KeyError('analysis_ready missing; run pipeline first')
    ids=T.loc[T.analysis_ready,'customer_number'].tolist()
    slots=project.system.nsmallest(min(critical_top,len(project.system)), 'rank')['slot'].to_numpy(dtype=int)
    critical=np.zeros(len(project.hours),dtype=bool)
    critical[slots]=True
    scores=[]
    for customer_id in ids:
        load=np.asarray(project.series(customer_id),dtype=float)
        if len(load)!=len(critical):
            raise ValueError('customer load length does not match system hours')
        score=float(load[critical].mean()) if method=='critical_mean' else float(load.max())
        if np.isfinite(score):
            scores.append((score,customer_id))
    scores.sort(key=lambda item:item[0],reverse=True)
    return scores,critical


def rank_customers(project, limit=30, cfg=GridFlexConfig(),critical_top=100,
                   screen_method='critical_mean'):
    """Screen by coincident critical-hour load (or legacy individual peak).

    The NPV rank is computed *within scenario* only for screened candidates.
    Neither screen is a feeder constraint or an installation approval. For
    final deployment, obtain feeder-specific coincidence and engineering data.
    """
    if limit < 1:
        raise ValueError('limit must be >= 1')
    scores,critical=screen_customer_scores(project,critical_top,screen_method)
    results=[]
    for score,customer_id in scores[:limit]:
        load=project.series(customer_id)
        for _,r in compare_scenarios(load,critical,project.hours,cfg).iterrows():
            results.append({'customer_number':customer_id,
                            'screen_method':screen_method,
                            'screen_score_kw':score,**r.to_dict()})
    if not results:
        return pd.DataFrame()
    ranked=pd.DataFrame(results)
    ranked['rank_npv']=ranked.groupby('scenario')['npv_usd'].rank(ascending=False,method='min')
    return ranked.sort_values(['scenario','rank_npv'])
