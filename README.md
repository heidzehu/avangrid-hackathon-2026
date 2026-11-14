# Avangrid 2026 AMI Hackathon: Customer Classification, Segmentation & BTM Battery Opportunity

> **Status:** the data pipeline, feature table and battery simulator are done (see below).
> The analysis sections (Parts 1–3) are the team's and are filled in as the work lands.

## What this repo does

A reproducible tool that takes Avangrid's hourly AMI data (35,725 residential meters, 9/1/2025–8/31/2026) plus hourly weather and:

1. **Prepares the data:** streams the raw files into a customer × 8,760-hour matrix, cleans it, fills gaps and flags low-quality meters.
2. **Describes every customer:** ~100 interpretable features (size, timing, seasonality, temperature response, ramps, zero readings, coincidence with the system peak).
3. **Classifies technologies (Part 1):** *team section, see below*
4. **Segments customers (Part 2):** *team section*
5. **Finds and sizes behind-the-meter battery / demand-flexibility candidates (Part 3):** *team section*, built on the battery simulator in `src/avangrid/battery.py`.

## Repository layout

```
├── data/
│   ├── raw/           <- put the hackathon files here (git-ignored, never commit)
│   └── processed/     <- generated matrices and caches (git-ignored)
├── notebooks/
│   ├── 01_data_pipeline.ipynb          run first: builds everything, data-quality report
│   ├── 10_part1_classification.ipynb   Part 1 workspace
│   ├── 20_part2_segmentation.ipynb     Part 2 workspace
│   └── 30_part3_battery.ipynb          Part 3 workspace
├── outputs/           <- generated tables and figures (git-ignored by default)
├── src/avangrid/
│   ├── config.py      paths, calendar, challenge givens (battery spec, 200 A / 3 kVA rule)
│   ├── io_ami.py      finds and streams the AMI deliveries into a matrix
│   ├── weather.py     weather cleaning and alignment, degree days
│   ├── clean.py       anomaly masking, gap filling, quality flags
│   ├── features.py    feature table, system load, profile tables
│   ├── labels.py      EV / heat pump / PV reference lists and evaluation helpers
│   ├── battery.py     battery simulator (physics only) and peak-cap helpers
│   ├── plots.py       heatmaps, profiles, temperature scatter, system load
│   └── pipeline.py    runs everything; load_project() to load results anywhere
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <repo-url> && cd <repo>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Data.** Copy into `data/raw/`:

| File | Notes |
|---|---|
| `2026_Avangrid_Hackathon_Data_1_10000.zip` | Customers 1–10,000 |
| `2026_Avangrid_Hackathon_Data_10001_20000.zip` | 10,001–20,000 |
| `2026_Avangrid_Hackathon_Data_20001_30000.zip` | 20,001–30,000 |
| `2026_Avangrid_Hackathon_Data_30001_35725.zip` | 30,001–35,725 |
| `Weather_Data_Avangrid_Hackathon.xlsx` | any file name containing "weather" |
| `EV_Heat Pump_Solar PV Customers.xlsx` | spaces or underscores both fine |

AMI deliveries can be the original zips **or** unzipped folders with the same name (any nesting of `*.csv` inside). Spaces, underscores and hyphens in the names are all recognised. You can start with one delivery and add the others later: only new ones are read. The pipeline prints a **WARNING** for anything that looks like a delivery but can't be read (failed macOS unzip `.cpgz`, a folder holding a zip, unfinished downloads) and for any customer range with no delivery.

## Run

```bash
cd src && python -m avangrid.pipeline          # or: open notebooks/01_data_pipeline.ipynb and run all
```
Then in any notebook:
```python
from avangrid.pipeline import load_project
D = load_project()
D.table           # one row per customer: reference labels + quality flags + features
D.kwh             # cleaned [customers x 8760] matrix, memory-mapped (rows follow D.customers)
D.series(30001)   # one customer's hourly kW
```

**Performance (measured on 2 CPUs / 8 GB RAM):** reading the 5,725-customer zip takes 76 s (≈ 2.2 min per 10,000 customers, once, then cached); cleaning + features + profiles take ≈ 20 s per 5,700 customers; re-runs with caches take seconds. Peak RAM 1.4 GB on that zip (expect ≈ 2 GB for a 10,000-customer zip). Disk for caches ≈ 3.8 GB for the full dataset. Results are identical on pandas 2.2 and 3.0. Everything runs in customer chunks over memory-mapped arrays, so cost grows linearly with customer count (see *Scaling*).

## Outputs

| File | Contents |
|---|---|
| `outputs/customer_features.parquet` / `.csv` | One row per customer: `ref_ev/ref_hp/ref_pv`, data-quality columns and flags, all features |
| `outputs/feature_dictionary.csv` | Unit and definition of every column |
| `outputs/seasonal_profiles.parquet` | Mean kW by season × weekday/weekend × hour (192 columns) |
| `outputs/daily_kwh.parquet` | 365 daily totals per customer |
| `outputs/system_load_hourly.csv` | Hourly sum of all loaded customers, overall / winter / summer ranks |
| `outputs/weather_hourly_clean.csv`, `weather_daily.csv` | Aligned, gap-filled weather; daily HDD65 / CDD65 |
| `outputs/anomalous_zero_hours.csv` | Hours where an unusual share of meters read 0 (possible outages / data glitches) |
| `outputs/run_summary.json` | Counts from the last run |
| `data/processed/kwh_raw.npy`, `kwh_clean.npy`, `customers.npy` | The matrices (float32, NaN = missing in raw) |

## Methodology: data preparation

### Data findings
Checked on the 30001–35725 delivery (5,704 customers, 49.6 M rows); re-checked automatically for every delivery.
- **All rows are complete hours** (`INTERVAL_COUNT = 4`, `IS_COMPLETE_HOUR = 1`). Missing data shows up as *missing rows*.
- **No negative readings:** the channel is grid *imports* only. Solar customers appear as **zero** readings at midday; export volumes are not observable.
- **21 customer numbers have no data at all**; ~60 customers start ~6 weeks late (move-ins / meter installs); 4 dates have system-wide whole-day gaps (3/10/2026: 329 customers, 12/8/2025: 206, 6/3/2026: 149, 12/21/2025: 69).
- **DST:** each DST day is missing one hour (11/2/2025 HE2, 3/8/2026 HE3).
- `USAGE_DATE` is ISO in the files (the dictionary says `MM/DD/YYYY`); both formats are accepted.
- A customer's rows can be split across `part-*.csv` files, so the reader writes into a pre-allocated matrix instead of concatenating.
- **Weather:** one duplicated hour on 11/2/2025 (DST) and a 94-hour gap 3/8–3/11/2026. Climate: 7,112 HDD65 vs 383 CDD65 (heating-dominated), yet this subset's system peak is in **summer** (7/2/2026 6 pm).

### Cleaning steps (`src/avangrid/clean.py`)
1. **Time grid:** local wall-clock hours, `slot = day × 24 + (HOUR_ENDING − 1)`, 8,760 slots. Weather `TIME` is hour-beginning, so 0:00 aligns with HE1.
2. **Zero-glitch hours:** an hour is anomalous if the share of meters reading exactly 0 is both 2× and 1 percentage point above the median share *for that hour of day* (time-of-day aware because solar raises midday zeros, and independent of how many customers are loaded). These are outages or data glitches hitting hundreds of meters at once (e.g. 30% of the 30001–35725 meters at 11 pm on 10/27/2025). Those zeros are set to missing **only for customers whose overall zero share < 5%**, so solar and vacant homes keep their genuine zeros. `outputs/anomalous_zero_hours.csv` lists every flagged hour with an estimate of the meters affected.
3. **Catch-up spikes:** a reading is set to missing if it is > 10 kWh, > 4× both neighbouring hours *and* > 3× the customer's own 99.5th percentile (example: a 39.8 kWh hour on a home whose 99th percentile is 1.2 kWh). This is deliberately strict so real EV or heating peaks are never removed.
4. **Gap filling:** ≤ 3 h → linear interpolation; longer → the customer's own mean for the same hour and day type (weekday vs weekend/holiday) within ±7 days; anything left → the customer's year-long mean for that hour and day type. Every filled hour is counted per customer and method.
5. **Flags (customers are never dropped):** `flag_low_coverage` (< 90% usable original hours), `flag_partial_year` (starts/ends > 14 days inside the period), `flag_mostly_zero` (> 50% zero hours), `analysis_ready` = neither low coverage nor mostly zero.
6. **Weather gaps:** ≤ 3 h linear; longer gaps use the daily mean interpolated between the complete days either side, plus the average hour-of-day shape from ±7 days. Filled hours are flagged.

### Features (`src/avangrid/features.py`; definitions in `outputs/feature_dictionary.csv`)
All features describe behaviour; none encodes an appliance rule.
- **Size:** annual kWh, mean/median/p95/p99/max kW, load factor, overnight base load, daily kWh stats.
- **Timing:** energy share by period of day, weekend/weekday ratio, midday-to-shoulder ratio, seasonal peak hour, 24 h / 168 h autocorrelation.
- **Seasonality:** monthly kWh, seasonal daily kWh, summer/winter ratio, monthly variability.
- **Weather response:** a PRISM-style change-point regression of daily kWh on daily mean temperature, fitted for every customer at once. It picks the best of constant / heating / cooling / heating + cooling models by BIC, with non-negative slopes, and reports base load, heating and cooling slopes (kWh/day/°F), balance points and R². Also mean kW by outdoor-temperature bin, and cold-day / hot-day vs mild-day ratios.
- **Dynamics:** counts of hour-to-hour up-ramps ≥ 2/4/6 kW, max ramp, hours above 3/5/7/9 kW.
- **Zero readings:** midday zero share in summer vs winter, share of days with a midday zero.
- **System coincidence:** demand during the system's top-100 hours and top-50 winter / summer hours, coincidence factor, share of system peak.

### Reference customers (`src/avangrid/labels.py`)
EV (200) and heat-pump (100) lists are **random ~25% samples** of the likely populations (implying ~800 EV and ~400 HP homes in total); the PV list (434) is complete. EV/HP are therefore *positive-unlabeled*. `pu_report` scores any ranking with recall@k, an estimated precision corrected for the 25% sampling (`hits / 0.25 / k`), and AUC vs all other customers. `holdout_split` keeps part of each list out of model development.

### Battery simulator (`src/avangrid/battery.py`)
Program unit (Challenge Statement): 2 × Enphase IQ Battery 10C = **20.16 kWh, 7.68 kW continuous** (11.52 kW 10-s transient, not used for sustained dispatch), **89.4 % round-trip**, split evenly as √0.894 = 94.55 % on charge and on discharge. Hourly step; discharge is limited by power, by stored energy above any backup reserve, and by home demand (no export by default); charge is limited by power and free capacity. Unit-tested: 19.06 kWh deliverable per full discharge, measured round-trip 0.894. `min_daily_cap` finds, per customer-day, the lowest demand cap the battery can both hold and recharge within the day (perfect-foresight bisection). `cap_request` turns any cap into an hourly dispatch request. Strategy, reserve, tariff and candidate choice are left to Part 3.

## Methodology: analysis *(team)*

### Part 1 – Customer classification
Code: `src/01_ingest.py` … `src/08_classification_charts.py` (run in order; see *Running the model pipeline* below).

**Approach: machine learning where labels exist, physics-based rules where they don't.**

*EV and heat pump: positive-unlabeled (PU) learning* (`src/06_classify_ml.py`)
- Only ~25% of true owners are labeled, so "unlabeled" mixes non-owners with hidden owners. Following Elkan & Noto (2008), a classifier learns P(labeled | features) and the owner probability is `P(owner) = P(labeled) / 0.25` (clipped to 1). Probabilities sum to ≈ the expected owner count (EV 703 vs 724; HP 303 vs 340), so they are calibrated.
- Model: regularized LightGBM: EV 150 trees × 4 leaves (hand-tuned); heat pump 217 trees × 8 leaves (from the nested-CV search below). All scores are out-of-fold (5-fold CV × 5 seeds), so no customer is scored by a model that saw it.
- Key engineered features (`src/04_features.py`, `src/04b_session_features.py`): EV charging sessions detected as flat, rectangular 1–10 h elevations above the daily baseline (Level 2: ≥ 3 kW; Level 1: 0.9–3 kW, overnight 4–14 h); change-point heating/cooling slopes; overnight cold-weather response; coincidence with the top-100 system hours.
- Data-quality exclusions from training: coverage < 80%, < 500 kWh/yr, or non-residential-like (p99 > 15 kW or > 50,000 kWh/yr).

| Model | Labeled | AUC | Recall in top-K | Est. precision of top-K | K (= labeled / 0.25) |
|---|---|---|---|---|---|
| EV | 181 | 0.80 | 37% (17× random) | ~36% | 724 |
| Heat pump | 85 | 0.77 | 9% (9× random) | ~9% | 340 |

- **Model comparison** (`src/06b_model_comparison.py`): random forest (2 settings) and logistic regression under the identical CV give EV AUC 0.79–0.80 and HP AUC 0.75–0.77, all within noise of the original LightGBM (0.80 / 0.76). The signal comes from the engineered features, not the algorithm, so the faster LightGBM model is kept.
- **Hyperparameter search with nested cross-validation** (`src/06c_hyperparameter_search.py`): random search over 40 LightGBM configurations (8 hyperparameters: trees, leaves, learning rate, min leaf size, row/feature sampling, L1/L2) plus the production configuration. Outer loop 5-fold × 2 repeats; the search runs only on each outer-training set (3-fold inner CV, AUC), and the winner is scored on the untouched outer fold alongside production. Decision rule fixed in advance: adopt only if the paired AUC gain exceeds one standard error and recall@K is not worse.

  | Target | Mean outer-fold AUC gain (tuned − production) | Folds tuned wins | Decision |
  |---|---|---|---|
  | EV | +0.001 ± 0.006 | – | keep hand-tuned |
  | Heat pump | +0.013 ± 0.008 | 7 of 10 | adopt tuned (217 trees, 8 leaves, stronger feature subsampling and L1/L2) |

  Configurations ranged from 0.71 to 0.80 AUC (EV) and 0.67 to 0.77 (heat pump), so settings matter, but the hand-tuned EV model is already near the top. The gain is judged per outer fold, because each fold may pick a different configuration and their scores are not on a common scale (pooling them understates the tuned model). The heat-pump gain is modest (~1.6 SE) and the deployed model's top-K recall moved by two customers (12% → 9%), within noise. Conclusion: tuning adds little; the limit is the information in hourly data.
- **Heat-pump limitation:** the labeled heat-pump rate rises ~8× with heating intensity, and among heavy electric heaters labeled heat pumps match other homes on cooling, baseload and winter shape. Hourly AMI cannot reliably separate whole-home heat pumps from resistance heat, so heat pumps are reported as *confirmed* (labeled) plus a ranked *candidate* list (~1 in 10 expected to be a heat pump), never as individually confirmed new finds.

*Other technologies: physics-based rules with 0–1 confidence* (`src/07_classify_rules.py`; logistic "ramps" multiplied for AND conditions, flagged at ≥ 0.5)

| Technology | Rule (evidence in hourly load) |
|---|---|
| EV level | Level 2 if typical session block ≥ 4.5 kW and ≥ 2 high-power sessions per 30 days, otherwise Level 1 / occasional |
| Air conditioning | cooling slope ≥ 0.35 kWh/day/°F and summer hourly load tracks temperature |
| Electric space heat | heating share of annual kWh ≥ 12% and winter load anti-correlated with temperature |
| Resistance heat | electric heat, not heat-pump-like, and ≥ 1 kW overnight rise on cold vs mild nights |
| Electric water heater | repeated isolated 2–5.5 kW one-hour bursts at hot-water times outside heating season |
| Pool pump | summer daytime floor rises ≥ 0.6 kW vs spring and does not follow temperature (timer-driven) |
| Hot tub / spa | overnight winter floor rises 0.4–2 kW without a space-heating signature |
| Solar + battery | known solar home that also imports ~nothing on summer evenings |
| Discovered segments | seasonal / vacant homes (≥ 2 months averaging < 0.15 kW); large non-residential-like meters |

Outputs: `outputs/customer_classification.csv` (one row per customer: probabilities, confidences, flags), `outputs/classification_summary.csv`, `outputs/model_metrics.csv`, figures `05`–`07` in `outputs/figures/`.

#### Running the model pipeline
The model scripts read the hackathon files from a `Data/` folder next to the repo (set in `src/config.py`) and cache to `data_local/` (git-ignored).
```bash
pip install -r requirements.txt
for s in 01_ingest 03_matrix 04_features 04b_session_features 06_classify_ml 07_classify_rules 08_classification_charts; do python src/$s.py; done
```
`01_ingest` also de-duplicates customer-hours (`DISTINCT ON (customer, date, he)`), as a safeguard: the original `10001_20000` zip was an exact copy of `1_10000`. `src/config.py` automatically uses a corrected `*_v2.zip` in place of its original.

**Data coverage:** The original model pipeline processed 34,238 customers. After incorporating the supplemental AMI delivery, the updated pipeline processes 35,236 unique customers. The original delivery had customer-range gaps; the supplemental delivery recovered additional customer records. The earlier labeled counts and results above refer to the original run unless explicitly updated.

### Part 2 – Segmentation and load insights
Implemented in `src/09_segment.py`. (a first-pass k-means segmentation on the model pipeline's features is in `src/09_segment.py`, figures `08`–`09`.)

### Part 3 – Demand flexibility and BTM battery opportunity
Implemented with GridFlex battery dispatch, demand response, battery sizing, economic sensitivity and integrated customer recommendations.

### Part 3 – Territory-wide battery & demand-flexibility screening
`src/14_bess_opportunity.py` simulates a battery (1 or 2 Enphase IQ 10C units) and a smart-thermostat event in **every eligible residential home** on the Part 1/2 customer base (34,238 customers; business-sized, seasonal and short-data meters excluded). It complements the GridFlex optimizer (`src/10–13`), which solves detailed dispatch for a short list.
- Dispatch: discharge 3–9 pm (biggest hours first, no export, 10% reserve, 89.4% round trip); recharge midnight–7 am.
- Sizing: 1 unit if the home's 3–9 pm energy on the 19 system-peak days fits in one unit, else 2.
- Ranking: utility cost per kW of peak relief = (battery cost − PV of the customer's bill savings) ÷ kW cut during the system's top-100 hours; Tier 1 = best 1,000 homes (pilot), Tier 2 = next 2,000, Tier 3 = other persistent-peak homes.
- Results (illustrative costs: $200/kW-yr capacity value, TOU $0.28/$0.11, $7,500/$14,000 per system): Tiers 1–2 = 6.4 MW of peak relief; median Tier 1 cost ≈ $3,400/kW vs. ≈ $1,800/kW capacity value; thermostat events ≈ 10 MW across ~22,000 AC homes; 12,445 homes could add a Level 2 charger within the 3 kVA guideline with a 2-unit battery.
- Outputs: `data_local/bess_customers.parquet` (customer-level, git-ignored), `outputs/bess_summary.json` (aggregate, git-ignored).

## Slide deck
`deck/06_DNH Energy_Final Submission.pptx` is the submission deck (24 slides, Team 6 · DNH Energy). `deck/build_deck.py` (with `deck/build_extra.py`) rebuilds it from the aggregate numbers in `deck/slide_data.json`: `cd deck && python3 build_deck.py` (needs `python-pptx`, `Pillow`, `lxml`).

## Scaling to millions of customers
The pipeline is chunked and memory-mapped end to end: each delivery is streamed file by file into a fixed-size matrix, and cleaning and features run over blocks of 2,000 customers. Runtime and memory per customer are constant, so the same code scales linearly. At utility scale the per-customer steps parallelise trivially by customer, e.g. one task per partition in Spark or Dask, matching the `part-*.csv` layout the data already arrives in.

## Assumptions and limitations (data preparation)
- Hourly data (aggregated from 15-min), so short appliance cycles (water-heater elements, hot-tub heaters, Level 1 vs. Level 2 EV) blur together.
- Import-only channel: solar generation and exports are not observable, only their effect (zero imports).
- One weather station for all customers.
- Gap-filled values are estimates. Use `filled_share` / `analysis_ready` to exclude heavily filled customers from model fitting.
- "System load" is the sum of the customers loaded in a run, not Avangrid's actual system or feeder load.

## Data handling
The hackathon data is for use during the event only. `data/` and `outputs/` are git-ignored; keep the repository private to the team and the judges.


## Final integrated GridFlex workflow

The expanded AMI processing run contains 35,236 unique customers.

### Part 1 — Classification
Estimates customer technology indicators using machine-learning scores and rule-based evidence.

### Part 2 — Segmentation
Groups customers by electricity consumption patterns and seasonal demand behavior.

### Part 3 — GridFlex optimization
Screens customers for critical-hour load reduction and evaluates baseline, demand response, battery and combined scenarios.

### Integrated decision support
`src/13_integrate_workflows.py` joins the technology classifications, segment labels and optimized recommendations using customer IDs.

The current analysis optimizes 30 screened customers, not the entire 35,236-customer population.

Classification and segmentation provide post-optimization explanations; they do not directly control battery dispatch.

### Validation
- 27 automated GridFlex and integration tests passed in the development environment.
- The actual-data integration matched all 30 optimized customers.
- Full customer-snapshot consistency must be verified separately.

### Limitations
Economic values, customer flexibility and battery costs are illustrative assumptions. System critical hours are not feeder-specific constraints. AMI predictions do not establish actual appliance ownership or demand response enrollment. The AMI merge audit's newly-filled counter includes original deliveries, not just the supplemental ZIP.

See `INTEGRATION_GUIDE.md` for integration instructions.
