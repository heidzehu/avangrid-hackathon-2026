"""Quick look at one AMI part-file without unzipping the archive."""
import sys, zipfile, io
import pandas as pd
sys.path.insert(0, "src")
from config import AMI_ZIPS

z = zipfile.ZipFile(AMI_ZIPS[0])
names = [n for n in z.namelist() if n.endswith(".csv")]
print(AMI_ZIPS[0].name, "->", len(names), "csv parts")
with z.open(names[0]) as f:
    df = pd.read_csv(f)
    df.columns = df.columns.str.upper()
print(df.head(5).to_string())
print("\nrows:", len(df), " customers in file:", df.CUSTOMER_NUMBER.nunique())
print("dtypes:\n", df.dtypes.to_string())
print("\nkWh describe:\n", df.HOURLY_KWH.describe().to_string())
print("negative kWh rows:", (df.HOURLY_KWH < 0).sum())
print("IS_COMPLETE_HOUR counts:", df.IS_COMPLETE_HOUR.value_counts().to_dict())
print("INTERVAL_COUNT counts:", df.INTERVAL_COUNT.value_counts().to_dict())
print("INTERVAL_TYPE:", df.INTERVAL_TYPE.unique()[:5])
print("HOUR_ENDING range:", df.HOUR_ENDING.min(), df.HOUR_ENDING.max())
print("date range:", df.USAGE_DATE.min(), "..", df.USAGE_DATE.max())
per = df.groupby("CUSTOMER_NUMBER").size()
print("rows per customer: min", per.min(), "median", per.median(), "max", per.max())
print("customer ids sample:", sorted(df.CUSTOMER_NUMBER.unique())[:10])
