from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT.parent / "Data"          # provided zips + Excel (outside the repo)
LOCAL = ROOT / "data_local"              # derived data (git-ignored)
FIG = ROOT / "outputs" / "figures"

LABELS_XLSX = DATA_DIR / "EV_Heat Pump_Solar PV Customers.xlsx"
WEATHER_XLSX = DATA_DIR / "Weather Data_Avangrid_Hackathon .xlsx"
_all = sorted(DATA_DIR.glob("2026 Avangrid Hackathon Data_*.zip"))
# A corrected delivery "<name>_v2.zip" replaces "<name>.zip" (the original 10001_20000 zip was a copy of 1_10000).
_replaced = {z.with_name(z.name.replace("_v2.zip", ".zip")) for z in _all if z.name.endswith("_v2.zip")}
AMI_ZIPS = [z for z in _all if z not in _replaced]
