from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT.parent / "Data"          # provided zips + Excel (outside the repo)
LOCAL = ROOT / "data_local"              # derived data (git-ignored)
FIG = ROOT / "outputs" / "figures"

LABELS_XLSX = DATA_DIR / "EV_Heat Pump_Solar PV Customers.xlsx"
WEATHER_XLSX = DATA_DIR / "Weather Data_Avangrid_Hackathon .xlsx"
AMI_ZIPS = sorted(DATA_DIR.glob("2026 Avangrid Hackathon Data_*.zip"))
