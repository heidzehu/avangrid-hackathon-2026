"""Step 2: first look — how do labeled EV / heat-pump / solar customers differ from everyone else?

Outputs (outputs/figures/):
  01_group_daily_profiles.png  average 24-h load shape, winter vs summer, per group
  02_temperature_response.png  daily kWh vs daily mean temperature, per group
  03_example_weeks.png         raw hourly load of sample customers, a winter and a summer week
"""
import sys
from pathlib import Path
import duckdb, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import LABELS_XLSX, WEATHER_XLSX, LOCAL, FIG

FIG.mkdir(parents=True, exist_ok=True)
AMI = str(LOCAL / "ami_clean.parquet")
COLORS = {"EV": "#d1495b", "Heat pump": "#00798c", "Solar PV": "#edae49", "Unlabeled": "#5c5c5c"}

# ---- labels -------------------------------------------------------------------------------
lab = []
for sheet, g in [("EV Customers", "EV"), ("HP Customers", "Heat pump"), ("PV Customers", "Solar PV")]:
    lab.append(pd.DataFrame({"customer": pd.read_excel(LABELS_XLSX, sheet_name=sheet).CUSTOMER_NUMBER, "grp": g}))
lab = pd.concat(lab).drop_duplicates("customer")       # a customer in 2 lists keeps the first

# ---- weather (hourly -> HE = hour+1; daily mean) -------------------------------------------
w = pd.read_excel(WEATHER_XLSX)
w["date"] = pd.to_datetime(w["date"]).dt.normalize()
w["he"] = w["time"].map(lambda t: t.hour) + 1
wd = w.groupby("date", as_index=False)["temp"].mean().rename(columns={"temp": "tavg"})
print(f"weather: {len(w)} hours, temp {w.temp.min()}..{w.temp.max()} F")

con = duckdb.connect(); con.execute("SET enable_progress_bar=false")
con.register("lab", lab)
grp_sql = "coalesce(l.grp, 'Unlabeled')"

# ---- 1. group daily profiles ---------------------------------------------------------------
prof = con.sql(f"""
  select {grp_sql} grp,
         case when month(a.date) in (12,1,2) then 'Winter (Dec-Feb)'
              when month(a.date) in (6,7,8) then 'Summer (Jun-Aug)' end season,
         a.he, avg(a.kwh) kwh
  from '{AMI}' a left join lab l using(customer)
  where month(a.date) in (12,1,2,6,7,8)
  group by all""").df()
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, season in zip(axes, ["Winter (Dec-Feb)", "Summer (Jun-Aug)"]):
    for g, col in COLORS.items():
        d = prof[(prof.grp == g) & (prof.season == season)].sort_values("he")
        ax.plot(d.he, d.kwh, color=col, lw=2.2, label=g)
    ax.set_title(season); ax.set_xlabel("Hour ending"); ax.set_xticks(range(1, 25, 2)); ax.grid(alpha=.3)
axes[0].set_ylabel("Average kWh per hour"); axes[0].legend()
fig.suptitle("Average daily load shape by known technology group", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "01_group_daily_profiles.png", dpi=150); plt.close(fig)

# ---- 2. temperature response ---------------------------------------------------------------
daily = con.sql(f"""
  select {grp_sql} grp, a.customer, a.date, sum(a.kwh) kwh
  from '{AMI}' a left join lab l using(customer) group by all""").df()
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.merge(wd, on="date")
daily["tbin"] = (daily.tavg // 5) * 5 + 2.5
tr = daily.groupby(["grp", "tbin"]).kwh.median().reset_index()
fig, ax = plt.subplots(figsize=(8, 5))
for g, col in COLORS.items():
    d = tr[tr.grp == g]; ax.plot(d.tbin, d.kwh, "o-", color=col, lw=2, ms=4, label=g)
ax.set_xlabel("Daily mean outdoor temperature (°F)"); ax.set_ylabel("Median daily kWh per customer")
ax.set_title("Temperature response by group (heating left, cooling right)", fontweight="bold")
ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
fig.savefig(FIG / "02_temperature_response.png", dpi=150); plt.close(fig)

# ---- 3. example customer weeks -------------------------------------------------------------
full = set(con.sql(f"select customer from '{AMI}' group by 1 having count(*) >= 8700").df().customer)
rng = np.random.default_rng(7)
picks = []
for g in ["EV", "Heat pump", "Solar PV"]:
    ids = sorted(set(lab[lab.grp == g].customer) & full)
    picks += [(g, c) for c in rng.choice(ids, 2, replace=False)]
unl = sorted(full - set(lab.customer))
picks += [("Unlabeled", c) for c in rng.choice(unl, 2, replace=False)]
weeks = {"Winter week (Jan 12-18, 2026)": ("2026-01-12", "2026-01-18"),
         "Summer week (Jul 13-19, 2026)": ("2026-07-13", "2026-07-19")}
ids_sql = ",".join(str(int(c)) for _, c in picks)
ex = con.sql(f"select * from '{AMI}' where customer in ({ids_sql}) order by customer, date, he").df()
ex["ts"] = pd.to_datetime(ex.date) + pd.to_timedelta(ex.he - 1, unit="h")
w["ts"] = pd.to_datetime(w.date) + pd.to_timedelta(w.he - 1, unit="h")

fig, axes = plt.subplots(len(picks), 2, figsize=(15, 2.0 * len(picks)), sharey="row")
for r, (g, c) in enumerate(picks):
    for k, (title, (s, e)) in enumerate(weeks.items()):
        ax = axes[r, k]
        d = ex[(ex.customer == c) & (ex.ts >= s) & (ex.ts < pd.Timestamp(e) + pd.Timedelta(days=1))]
        ax.fill_between(d.ts, d.kwh, color=COLORS[g], alpha=.85, step="post", lw=0)
        ww = w[(w.ts >= s) & (w.ts < pd.Timestamp(e) + pd.Timedelta(days=1))]
        ax2 = ax.twinx(); ax2.plot(ww.ts, ww.temp, color="k", lw=.8, alpha=.5)
        ax2.set_yticks([]) if k == 0 else ax2.set_ylabel("°F", fontsize=7)
        if r == 0: ax.set_title(title, fontweight="bold")
        if k == 0: ax.set_ylabel(f"{g}\nexample {r % 2 + 1}\nkWh", fontsize=8)   # no customer IDs (event rules)
        ax.tick_params(labelsize=7); ax.grid(alpha=.25)
fig.suptitle("Hourly load of sample customers (colored) with outdoor temperature (black line)", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "03_example_weeks.png", dpi=130); plt.close(fig)

# ---- quick numbers -------------------------------------------------------------------------
summ = daily.groupby(["grp", "customer"]).kwh.sum().groupby("grp").agg(["count", "median"])
summ.columns = ["customers", "median annual kWh"]
pk = con.sql(f"select {grp_sql} grp, median(p) med_peak_kw from (select customer, max(kwh) p from '{AMI}' group by 1) x left join lab l using(customer) group by 1").df()
print(summ.join(pk.set_index("grp")).round(1).to_string())
print("figures written to", FIG)
