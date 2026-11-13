"""Project-wide paths, calendar and fixed constants.

Everything that is a *given* of the challenge (data period, battery spec, service
limits) lives here. Analytical choices that the team will tune (thresholds,
detector rules, tariff, candidate criteria) deliberately do NOT live here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Paths:
    """Folder layout. Override `root` if you keep the data somewhere else."""

    root: Path = REPO_ROOT

    @property
    def raw(self) -> Path:  # put the 4 AMI zips (or unzipped folders) + 2 xlsx here
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:  # generated caches (large, git-ignored)
        return self.root / "data" / "processed"

    @property
    def outputs(self) -> Path:  # tables + figures for the deck
        return self.root / "outputs"

    @property
    def figures(self) -> Path:
        return self.outputs / "figures"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.processed, self.outputs, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self


# ------------------------------------------------------------------------ calendar
# Data period from the Data Dictionary: 9/1/2025 - 8/31/2026, hourly.
START_DATE = pd.Timestamp("2025-09-01")
END_DATE = pd.Timestamp("2026-08-31")
N_DAYS = (END_DATE - START_DATE).days + 1  # 365
N_HOURS = N_DAYS * 24  # 8760

# Hour slots are LOCAL WALL-CLOCK hours: slot = day_index * 24 + (HOUR_ENDING - 1).
# The AMI files skip one slot on each DST day (11/2/2025 HE2, 3/8/2026 HE3); those
# slots are treated as missing and filled during cleaning.
DST_FALL_BACK = pd.Timestamp("2025-11-02")
DST_SPRING_FORWARD = pd.Timestamp("2026-03-08")


def hour_index() -> pd.DataFrame:
    """One row per hour slot (8760) with calendar attributes."""
    ts = pd.date_range(START_DATE, periods=N_HOURS, freq="h")  # hour-beginning, local
    df = pd.DataFrame({"slot": np.arange(N_HOURS), "ts": ts})
    df["date"] = df["ts"].dt.normalize()
    df["day"] = df["slot"] // 24
    df["hour_beginning"] = df["ts"].dt.hour  # 0-23
    df["hour_ending"] = df["hour_beginning"] + 1  # 1-24, matches AMI HOUR_ENDING
    df["month"] = df["ts"].dt.month
    df["dow"] = df["ts"].dt.dayofweek  # Mon=0
    df["is_weekend"] = df["dow"] >= 5
    from pandas.tseries.holiday import USFederalHolidayCalendar

    hol = USFederalHolidayCalendar().holidays(START_DATE, END_DATE)
    df["is_holiday"] = df["date"].isin(hol)
    df["season"] = df["month"].map(MONTH_TO_SEASON)
    return df


# Meteorological seasons
MONTH_TO_SEASON = {12: "winter", 1: "winter", 2: "winter",
                   3: "spring", 4: "spring", 5: "spring",
                   6: "summer", 7: "summer", 8: "summer",
                   9: "fall", 10: "fall", 11: "fall"}
SEASONS = ["winter", "spring", "summer", "fall"]

# --------------------------------------------------------------- challenge givens
@dataclass(frozen=True)
class BatterySpec:
    """Enphase IQ Battery 10C, 2-unit program setup (from the Challenge Statement)."""

    units: int = 2
    energy_kwh_per_unit: float = 10.08
    continuous_kw_per_unit: float = 3.84  # 7.68 kW total / 2
    peak_kw_per_unit: float = 5.76  # 11.52 kW total / 2, 10-second transient only
    round_trip_efficiency: float = 0.894
    warranty_years: int = 15

    @property
    def energy_kwh(self) -> float:
        return self.units * self.energy_kwh_per_unit

    @property
    def power_kw(self) -> float:
        return self.units * self.continuous_kw_per_unit

    @property
    def peak_kw(self) -> float:
        return self.units * self.peak_kw_per_unit

    @property
    def one_way_efficiency(self) -> float:
        """Round-trip loss split evenly between charge and discharge."""
        return float(np.sqrt(self.round_trip_efficiency))


PROGRAM_BATTERY = BatterySpec()  # 2 units: 20.16 kWh, 7.68 kW

# Residential service (Challenge Statement assumption)
SERVICE_AMPS = 200
SERVICE_VOLTS = 240
SERVICE_KVA = SERVICE_AMPS * SERVICE_VOLTS / 1000  # 48 kVA nameplate
INCREMENTAL_LOAD_LIMIT_KVA = 3.0  # "should not exceed ~3 kVA additional load without further evaluation"

# Reference-customer lists (from the EV/HP/PV workbook "Description" tab)
LABEL_SAMPLE_FRACTION = {"ev": 0.25, "hp": 0.25, "pv": 1.0}
