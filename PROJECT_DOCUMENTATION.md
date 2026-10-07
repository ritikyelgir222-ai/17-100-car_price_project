# Project Documentation: Used Car Price Prediction
### Technical Documentation & Handover — Phase 12 of the SDLC

Companion to `SDLC_DOCUMENTATION.md` and `README.md`.

---

## 1. Project Summary

| | |
|---|---|
| **Objective** | Suggest a fair, data-driven starting price for a used Audi listing |
| **Client context** | Automotive marketplace |
| **Data source** | 100,000 UK Used Car Data Set, Audi file — [Kaggle](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes) |
| **Dataset size** | 10,668 listings → 10,565 after cleaning |
| **Final model** | XGBoost Regressor |
| **Test R²** | 0.938 |
| **Business framing** | % of predictions within a trustworthy tolerance of actual price |

---

## 2. Architecture

```
data/audi.csv (10,668 real scraped listings)
        │
        ▼
data_loader.py ──────► loads raw data, flags 3 real quality issues
        │
        ▼
clean_and_engineer.py ─► strips whitespace, drops 103 duplicates,
        │                imputes 57 zero-engineSize rows with
        │                MODEL-SPECIFIC median, engineers age_years
        │                and mileage_per_year
        ▼
   split.py ───────────► plain random split (no time axis, no class
        │                 imbalance -- regression, not classification)
        ▼
train_model.py ───────► Linear Regression (baseline, EXPECTED to
        │                underperform given EDA's confirmed non-
        │                linearity) → Random Forest → XGBoost,
        │                trained on log(price), evaluated in real £
        ▼
outputs/car_price_model.joblib
        │
        ├──────────────► app.py ─── FastAPI pricing service
        │
        └──────────────► monitor.py ─ monthly/quarterly market-drift check
```

---

## 3. Data Dictionary

| Column | Type | Description | Treatment |
|---|---|---|---|
| `model` | string | Audi model (A1-A8, Q2-Q8, R8, RS/S variants, TT) | Whitespace-stripped, one-hot encoded |
| `year` | int | Manufacture year | Converted to `age_years` |
| `price` | int | Sale price (£) | **Target**, log-transformed for training |
| `transmission` | string | Manual/Automatic/Semi-Auto | One-hot encoded |
| `mileage` | int | Odometer reading | Kept as-is, also used for `mileage_per_year` |
| `fuelType` | string | Petrol/Diesel/Hybrid | One-hot encoded |
| `tax` | int | Annual road tax (£) | Kept as-is |
| `mpg` | float | Fuel economy | Kept as-is |
| `engineSize` | float | Engine displacement (L) | 57 zero values imputed with model-specific median |

**Engineered features:**

| Feature | Formula | Rationale |
|---|---|---|
| `age_years` | 2020 − year | Raw manufacture year has no direct price relationship; age does |
| `mileage_per_year` | mileage / max(age_years, 1) | Captures usage intensity independent of age |

---

## 4. Key EDA Findings

| Finding | Detail |
|---|---|
| **Depreciation curve confirmed non-linear** | £5,710 drop age 0→1, only £1,241 drop age 5→6 — ~4.6x steeper early depreciation |
| Price by model | Q7 highest (£44,707 avg), A1 lowest among common models (£14,275) |
| Price by transmission | Automatic (£28,157) and Semi-Auto (£27,134) well above Manual (£16,024) |
| mileage_per_year correlation with price | -0.301 — real but moderate negative relationship |

---

## 5. Modeling Results

### Experiment log (validation set, real £)

| Model | RMSE | MAE | R² | Within 10% |
|---|---|---|---|---|
| Linear Regression (baseline) | £2,935 | £1,987 | 0.943 | 65.6% |
| Random Forest | £2,542 | £1,589 | 0.957 | 75.1% |
| **XGBoost** | **£2,396** | **£1,554** | **0.962** | **76.7%** |

**A clean, EXPECTED win for XGBoost** — unlike Days 1, 7, and 9's
near-ties or upsets, this result matches what Phase 6's EDA predicted
in advance (the confirmed non-linear depreciation curve is something a
linear model structurally cannot represent). The baseline's relative
weakness here is a validated prediction, not a surprise explained away
after the fact.

### Test set (final, honest read)

| Metric | Value |
|---|---|
| Test RMSE | £3,005 |
| Test MAE | £1,632 |
| Test R² | 0.938 |
| **Within ±5% of actual** | 45.2% |
| **Within ±10% of actual** | 75.2% |

### What drives the model

Top features: `age_years` (0.265), `transmission_Manual` (0.181),
`engineSize` (0.164), `model_A1` (0.053), `tax` (0.052) — age dominates,
consistent with the EDA's depreciation-curve finding being the
strongest single signal in the data.

---

## 6. API Reference

**Base URL (local):** `http://127.0.0.1:8000`

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Browser test form |
| `/docs` | GET | Interactive Swagger UI |
| `/health` | GET | Health check |
| `/price` | POST | Get a price estimate for a car listing |

**Response:**
```json
{"predicted_price_gbp": 45040.12, "top_factors": ["age_years", "transmission_Manual", "engineSize"]}
```

Verified with two constructed listings: a nearly-new, low-mileage Q7
(2019, 8,000 miles) priced at £45,040; a 15-year-old, high-mileage A1
(2005, 150,000 miles) priced at £2,659 — both plausible real-world prices.

---

## 7. Monitoring & Maintenance Plan

`monitor.py` checks PSI drift on `mileage`, `age_years`, and `engineSize`,
plus MAE decay against the stored baseline.

**Recommended cadence: monthly-to-quarterly** — a used-car market shifts
with fuel prices, interest rates, new-model-year releases, and seasonal
demand: gradual market drift, not adversarial (Day 2) or academic-
calendar-paced (Day 9). Closer in spirit to Day 5's quarterly
segmentation cadence. A model trained on one point-in-time snapshot
should be treated as reflecting *that* market, not assumed current
indefinitely.

---

## 8. Known Limitations

1. **Single brand (Audi only)** — a real, disclosed scope limit; the source dataset splits listings by manufacturer and only this file was reliably available as a complete raw mirror.
2. **UK market, one point in time** — no validation against other countries or economic periods.
3. **Only 45.2% of predictions land within ±5%** of actual price, despite 75.2% landing within ±10% — a pricing team needing tighter precision would need a more refined model or additional features (condition, accident history, color, optional equipment — none of which exist in this dataset).
4. **No fairness/bias check needed** in the demographic sense (no such features exist here), but "fair pricing" in this project means *consistent, accurate* pricing across car types, which the within-tolerance metric directly measures.

---

## 9. File Map

| File | Phase | Purpose |
|---|---|---|
| `data_loader.py` | 5 | Load raw data, flag 3 quality issues |
| `eda.py` | 6 | Confirm non-linear depreciation curve |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Whitespace/duplicate/engineSize fixes, feature engineering |
| `split.py` | 7 | Random split |
| `train_model.py` | 8–9 | Linear vs. RF vs. XGBoost, fair-pricing metric |
| `app.py` | 10 | FastAPI pricing service |
| `monitor.py` | 13 | Market-drift monitoring |
| `README.md` | 12 | Setup instructions |
| `PROJECT_DOCUMENTATION.md` (this file) | 12 | Technical documentation |
| `SDLC_DOCUMENTATION.md` | 1–14 | Full 14-phase narrative |
