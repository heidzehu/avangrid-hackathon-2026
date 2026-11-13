"""Step 1: convert the provided AMI zips (Spark CSV part-files) into compact Parquet.

Streams each CSV member straight out of its zip (no unzip to disk), normalises column
names, downcasts types, and writes one Parquet file per part to data_local/ami/.
Run:  python src/01_ingest.py
"""
import sys, zipfile, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pyarrow as pa, pyarrow.csv as pcsv, pyarrow.compute as pc, pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from config import AMI_ZIPS, LOCAL

OUT = LOCAL / "ami"

def convert(zip_path: str, member: str, idx: int) -> tuple[str, int]:
    out = OUT / f"part_{idx:04d}.parquet"
    if out.exists():
        return out.name, pq.read_metadata(out).num_rows
    with zipfile.ZipFile(zip_path) as z, z.open(member) as f:
        t = pcsv.read_csv(f)
    t = t.rename_columns([c.lower() for c in t.column_names])
    t = pa.table({
        "customer": pc.cast(t["customer_number"], pa.int32()),
        "date": pc.cast(t["usage_date"], pa.date32()),
        "he": pc.cast(t["hour_ending"], pa.int8()),
        "weekend": pc.equal(t["day_type"], "Weekend"),
        "kwh": pc.cast(t["hourly_kwh"], pa.float32()),
        "n_int": pc.cast(t["interval_count"], pa.int8()),
    })
    pq.write_table(t, out, compression="zstd")
    return out.name, t.num_rows

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs, i = [], 0
    for zp in AMI_ZIPS:
        for m in zipfile.ZipFile(zp).namelist():
            if m.endswith(".csv"):
                jobs.append((str(zp), m, i)); i += 1
    print(f"{len(jobs)} parts to convert")
    t0, total = time.time(), 0
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(convert, *j) for j in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            name, n = fu.result(); total += n
            if k % 20 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} parts, {total:,} rows, {time.time()-t0:.0f}s", flush=True)
    print("done")
