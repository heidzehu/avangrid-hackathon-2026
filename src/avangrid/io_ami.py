"""Read the hourly AMI files into a customer x hour matrix.

The four AMI deliveries are named like ``2026_Avangrid_Hackathon_Data_1_10000.zip``
(the Data Dictionary writes them with spaces: ``2026 Avangrid Hackathon Data_1_10000.zip``).
Each is a Spark export: a zip of ``part-000NN-....csv`` files. A customer's rows can be
split across several part files, so rows are written straight into a pre-allocated
matrix instead of being concatenated in pandas (which needs >6 GB for one 5.7k-customer zip).

Accepted layouts inside ``data/raw`` (mix and match):
  * the original zip:            2026_Avangrid_Hackathon_Data_10001_20000.zip
  * an unzipped folder:          2026_Avangrid_Hackathon_Data_10001_20000/part-*.csv
  * a folder you split yourself: any *.csv files somewhere under that folder
Spaces, underscores and hyphens in the name are all fine.
"""
from __future__ import annotations

import io
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from .config import N_HOURS, START_DATE

SOURCE_RE = re.compile(r"hackathon[\s_\-]*data[\s_\-]*(\d+)[\s_\-]+(\d+)", re.IGNORECASE)

# canonical (lower-case) column names used in this repo
COLS = ["customer_number", "usage_date", "hour_ending", "day_type", "hourly_kwh",
        "interval_count", "interval_type", "is_complete_hour"]


@dataclass
class AmiSource:
    start: int  # first customer number covered (from the file name)
    end: int  # last customer number covered
    path: Path
    kind: str  # "zip" or "dir"

    @property
    def tag(self) -> str:
        return f"{self.start}_{self.end}"

    @property
    def n_slots(self) -> int:
        return self.end - self.start + 1


def find_ami_sources(raw_dir: Path) -> list[AmiSource]:
    """Find AMI deliveries in ``raw_dir`` by the naming convention. An unzipped folder
    wins over a zip of the same range (it is faster to read)."""
    found: dict[tuple[int, int], AmiSource] = {}
    for p in sorted(Path(raw_dir).iterdir()):
        m = SOURCE_RE.search(p.name)
        if not m:
            continue
        start, end = int(m.group(1)), int(m.group(2))
        if p.is_dir() and any(p.rglob("*.csv")):
            found[(start, end)] = AmiSource(start, end, p, "dir")
        elif p.suffix.lower() == ".zip" and (start, end) not in found:
            found[(start, end)] = AmiSource(start, end, p, "zip")
    return [found[k] for k in sorted(found)]


EXPECTED_LAST_CUSTOMER = 35725  # from the Data Dictionary


def diagnose_raw_dir(raw_dir: Path, sources: list[AmiSource] | None = None) -> list[str]:
    """Human-readable warnings about AMI deliveries that look present but will NOT be read,
    and about customer-number ranges with no delivery at all."""
    raw_dir = Path(raw_dir)
    sources = find_ami_sources(raw_dir) if sources is None else sources
    used = {s.path.resolve() for s in sources}
    warn = []
    for p in sorted(raw_dir.iterdir()):
        name = p.name.lower()
        if "hackathon" not in name or p.resolve() in used or "weather" in name:
            continue
        if p.is_dir():
            inner = [q.name for q in p.rglob("*") if q.is_file() and not q.name.startswith(".")][:5]
            if any(q.lower().endswith(".zip") for q in inner):
                warn.append(f"'{p.name}' is a folder holding a zip, not CSVs ({inner}). Unzip it there, or move the zip up into data/raw.")
            elif not SOURCE_RE.search(p.name):
                warn.append(f"'{p.name}' does not follow the '..._Data_<first>_<last>' naming, so its customer range is unknown. Rename it.")
            else:
                warn.append(f"'{p.name}' has no .csv files inside (found: {inner or 'nothing'}). Is the unzip finished?")
        elif name.endswith(".cpgz"):
            warn.append(f"'{p.name}' is what macOS Archive Utility leaves when an unzip FAILS. Delete it and put the original .zip in data/raw (the pipeline reads zips directly).")
        elif name.endswith((".crdownload", ".part", ".download", ".partial")):
            warn.append(f"'{p.name}' is an unfinished download.")
        elif p.suffix.lower() == ".zip" and not SOURCE_RE.search(p.name):
            warn.append(f"'{p.name}' does not follow the '..._Data_<first>_<last>' naming. Rename it.")
    # gaps in customer-number coverage
    nxt = 1
    for s in sorted(sources, key=lambda s: s.start):
        if s.start > nxt:
            warn.append(f"No delivery covers customers {nxt:,}-{s.start - 1:,}.")
        nxt = max(nxt, s.end + 1)
    if nxt <= EXPECTED_LAST_CUSTOMER:
        warn.append(f"No delivery covers customers {nxt:,}-{EXPECTED_LAST_CUSTOMER:,}.")
    return warn


def _iter_csv_bytes(src: AmiSource) -> Iterator[tuple[str, bytes]]:
    if src.kind == "zip":
        with zipfile.ZipFile(src.path) as z:
            for info in sorted(z.infolist(), key=lambda i: i.filename):
                base = info.filename.rsplit("/", 1)[-1]
                if info.filename.lower().endswith(".csv") and "__MACOSX" not in info.filename and not base.startswith("._"):
                    yield info.filename, z.read(info)
    else:
        for f in sorted(src.path.rglob("*.csv")):
            if f.name.startswith("._") or "__MACOSX" in f.parts:  # macOS unzip metadata, not data
                continue
            yield str(f.relative_to(src.path)), f.read_bytes()


def _read_part(raw: bytes) -> pd.DataFrame:
    """Parse one CSV part into a small, typed frame with canonical column names."""
    header = raw[: raw.find(b"\n")].decode("utf-8-sig").strip().split(",")
    names = {h: h.strip().lower() for h in header}
    types = {h: t for h in header for key, t in [
        ("customer_number", pa.int32()), ("hour_ending", pa.int16()), ("hourly_kwh", pa.float32()),
        ("interval_count", pa.int16()), ("is_complete_hour", pa.int16())] if names[h] == key}
    tbl = pacsv.read_csv(
        io.BytesIO(raw),
        read_options=pacsv.ReadOptions(use_threads=True),
        convert_options=pacsv.ConvertOptions(
            column_types=types,
            include_columns=[h for h in header if names[h] in
                             ("customer_number", "usage_date", "hour_ending", "hourly_kwh",
                              "interval_count", "is_complete_hour", "interval_type")],
            strings_can_be_null=True, auto_dict_encode=True),
    )
    df = tbl.to_pandas()
    df.columns = [names[c] for c in df.columns]
    # usage_date: ISO in the actual files, MM/DD/YYYY in the dictionary -> accept both
    d = df["usage_date"]
    if not pd.api.types.is_datetime64_any_dtype(d) and not isinstance(d.iloc[0], (pd.Timestamp,)):
        d = d.astype(str)
        fmt = "%Y-%m-%d" if re.match(r"\d{4}-\d{2}-\d{2}", d.iloc[0]) else "%m/%d/%Y"
        d = pd.to_datetime(d, format=fmt)
    df["usage_date"] = pd.to_datetime(d)
    return df


@dataclass
class SourceMatrix:
    """Hourly kWh for one delivery. Row i is customer ``customers[i]``; NaN = no reading."""

    tag: str
    customers: np.ndarray  # int32, only customers that appear in the files
    kwh: np.ndarray  # float32 [n_customers, 8760]
    stats: dict = field(default_factory=dict)


def read_source(src: AmiSource, drop_incomplete: bool = True, verbose: bool = True) -> SourceMatrix:
    """Stream every CSV part of one delivery into a [customer x 8760] matrix.

    * duplicate (customer, date, hour) rows are averaged (none seen in 30001-35725)
    * hours flagged incomplete (IS_COMPLETE_HOUR = 0 or INTERVAL_COUNT < 4) become NaN
      when ``drop_incomplete`` (none seen in 30001-35725)
    * rows outside the 9/1/2025-8/31/2026 window or the file's customer range are counted
      and dropped
    """
    t0 = time.time()
    n = src.n_slots
    ksum = np.zeros((n, N_HOURS), dtype=np.float32)
    kcnt = np.zeros((n, N_HOURS), dtype=np.uint8)
    st = dict(parts=0, rows=0, rows_out_of_range=0, rows_incomplete=0, rows_negative=0,
              rows_nan=0, interval_types=set(), customers_in_multiple_parts=0)
    seen_in: dict[int, int] = {}
    for name, raw in _iter_csv_bytes(src):
        df = _read_part(raw)
        st["parts"] += 1
        st["rows"] += len(df)
        if "interval_type" in df:
            st["interval_types"].update(map(str, df["interval_type"].dropna().unique()))
        day = (df["usage_date"].values.astype("datetime64[D]") - np.datetime64(START_DATE.date(), "D")).astype(np.int64)
        he = df["hour_ending"].to_numpy(np.int64)
        slot = day * 24 + (he - 1)
        ci = df["customer_number"].to_numpy(np.int64) - src.start
        kwh = df["hourly_kwh"].to_numpy(np.float32)
        ok = (slot >= 0) & (slot < N_HOURS) & (he >= 1) & (he <= 24) & (ci >= 0) & (ci < n)
        st["rows_out_of_range"] += int((~ok).sum())
        if drop_incomplete:
            inc = np.zeros(len(df), bool)
            if "is_complete_hour" in df:
                inc |= df["is_complete_hour"].fillna(0).to_numpy() == 0
            if "interval_count" in df:
                inc |= df["interval_count"].fillna(0).to_numpy() < 4
            st["rows_incomplete"] += int((inc & ok).sum())
            ok &= ~inc
        st["rows_nan"] += int((np.isnan(kwh) & ok).sum())
        st["rows_negative"] += int(((kwh < 0) & ok).sum())
        ok &= ~np.isnan(kwh)
        np.add.at(ksum, (ci[ok], slot[ok]), kwh[ok])
        np.add.at(kcnt, (ci[ok], slot[ok]), 1)
        for c in np.unique(ci[ok]):
            seen_in[c] = seen_in.get(c, 0) + 1
        if verbose:
            print(f"  [{src.tag}] {name[:40]:<40} rows={len(df):>9,}  t={time.time() - t0:5.1f}s", end="\r")
    if verbose:
        print()
    st["customers_in_multiple_parts"] = sum(v > 1 for v in seen_in.values())
    st["duplicate_hours"] = int((kcnt > 1).sum())
    present = np.where(kcnt.any(axis=1))[0]
    np.divide(ksum, kcnt, out=ksum, where=kcnt > 0)  # in place: average any duplicates
    ksum[kcnt == 0] = np.nan
    del kcnt
    kwh = ksum if len(present) == n else ksum[present]
    del ksum
    st["customers_expected"] = n
    st["customers_present"] = int(len(present))
    st["customers_missing"] = [int(x + src.start) for x in np.setdiff1d(np.arange(n), present)]
    st["interval_types"] = sorted(st["interval_types"])
    st["seconds"] = round(time.time() - t0, 1)
    return SourceMatrix(src.tag, (present + src.start).astype(np.int32), kwh, st)


# ----------------------------------------------------------------- cached loading
def _cache_files(processed: Path, tag: str) -> tuple[Path, Path, Path]:
    return (processed / f"ami_{tag}_kwh.npy", processed / f"ami_{tag}_customers.npy",
            processed / f"ami_{tag}_stats.json")


def load_source_cached(src: AmiSource, processed: Path, rebuild: bool = False, **kw) -> SourceMatrix:
    import json

    fk, fc, fs = _cache_files(processed, src.tag)
    if not rebuild and fk.exists() and fc.exists() and fs.exists():
        return SourceMatrix(src.tag, np.load(fc), np.load(fk, mmap_mode="r"), json.loads(fs.read_text()))
    sm = read_source(src, **kw)
    processed.mkdir(parents=True, exist_ok=True)
    np.save(fk, sm.kwh)
    np.save(fc, sm.customers)
    fs.write_text(json.dumps(sm.stats, indent=1))
    sm.kwh = np.load(fk, mmap_mode="r")  # release RAM before the next delivery is read
    return sm


def build_raw_matrix(raw_dir: Path, processed: Path, rebuild: bool = False, verbose: bool = True):
    """Read every delivery found in ``raw_dir`` (cached per delivery) and stack them into
    ``processed/kwh_raw.npy`` (float32, NaN = missing) + ``processed/customers.npy``.

    Returns (customers, kwh_memmap, per_source_stats). Re-runs only re-read deliveries whose
    cache is missing, so adding the other three zips later just appends them.
    """
    import json

    sources = find_ami_sources(raw_dir)
    warnings_ = diagnose_raw_dir(raw_dir, sources)
    if verbose and warnings_:
        print("!" * 78)
        for w in warnings_:
            print("  WARNING:", w)
        print("!" * 78)
    if not sources:
        raise FileNotFoundError(
            f"No AMI deliveries found in {raw_dir}. Expected names like "
            "'2026_Avangrid_Hackathon_Data_1_10000.zip' (or an unzipped folder of the same name).")
    mats, stats = [], {"_warnings": warnings_}
    for s in sources:
        if verbose:
            print(f"Delivery {s.tag} ({s.kind}): {s.path.name}")
        sm = load_source_cached(s, processed, rebuild=rebuild, verbose=verbose)
        mats.append(sm)
        stats[s.tag] = sm.stats
    customers = np.concatenate([m.customers for m in mats]).astype(np.int32)
    order = np.argsort(customers, kind="stable")
    customers = customers[order]
    out_path = processed / "kwh_raw.npy"
    manifest = processed / "kwh_raw_manifest.json"
    tags = [m.tag for m in mats]
    if (not rebuild and out_path.exists() and manifest.exists()
            and json.loads(manifest.read_text()).get("tags") == tags):
        return customers, np.load(out_path, mmap_mode="r"), stats
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(len(customers), N_HOURS))
    offset = 0
    for m in mats:  # deliveries cover disjoint, increasing customer ranges -> already ordered
        k = len(m.customers)
        out[offset:offset + k] = m.kwh[np.argsort(m.customers, kind="stable")]
        offset += k
    out.flush()
    del out
    np.save(processed / "customers.npy", customers)
    manifest.write_text(json.dumps({"tags": tags, "n_customers": int(len(customers))}))
    return customers, np.load(out_path, mmap_mode="r"), stats
