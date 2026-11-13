"""Step 3a: reshape clean AMI into a dense customer x hour matrix aligned with weather.

The year 2025-09-01 .. 2026-08-31 is a fixed grid of 365 days x 24 hours = 8,760 columns.
Missing hours (e.g. the DST spring-forward hour) stay NaN.

Writes to data_local/:
  hourly.npy      float32 [n_customers, 8760] kWh
  customers.npy   int32   [n_customers]        customer number for each row
  weather.parquet hourly temp/humidity/wind on the same 8,760-hour grid
"""
import sys, time
from pathlib import Path
import duckdb, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import LOCAL, WEATHER_XLSX

START = pd.Timestamp("2025-09-01"); NDAY = 365; NH = NDAY * 24

t0 = time.time()
con = duckdb.connect(); con.execute("SET enable_progress_bar=false")
df = con.sql(f"""
  select customer, datediff('day', DATE '2025-09-01', date) * 24 + (he - 1) as t, kwh
  from '{LOCAL / "ami_clean.parquet"}' where he between 1 and 24""").fetchnumpy()
cust = np.unique(df["customer"]).astype(np.int32)
row = np.searchsorted(cust, df["customer"])
t = df["t"].astype(np.int64)
ok = (t >= 0) & (t < NH)
M = np.full((len(cust), NH), np.nan, dtype=np.float32)
M[row[ok], t[ok]] = df["kwh"][ok]
np.save(LOCAL / "hourly.npy", M); np.save(LOCAL / "customers.npy", cust)
print(f"matrix {M.shape}, {np.isnan(M).mean():.2%} missing, {time.time()-t0:.0f}s")

# weather on the same grid: reading at hh:00 is assigned to hour-ending hh+1
w = pd.read_excel(WEATHER_XLSX)
w["t"] = (pd.to_datetime(w["date"]).dt.normalize() - START).dt.days * 24 + w["time"].map(lambda x: x.hour)
w = w.groupby("t")[["temp", "humidity", "windspeed"]].mean().reindex(range(NH)).interpolate(limit_direction="both")
w.index.name = "t"
w["ts"] = START + pd.to_timedelta(w.index, unit="h")
w.reset_index().to_parquet(LOCAL / "weather.parquet")
print(f"weather grid {len(w)} hours, temp {w.temp.min():.0f}..{w.temp.max():.0f} F")
