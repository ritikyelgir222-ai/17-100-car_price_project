# Used Car Price Prediction — Day 17 (Automotive Marketplace, ML)

Full project code for Day 17 of the 100-day series. Uses the real
**100,000 UK Used Car Data Set** (Kaggle: adityadesai13/used-car-dataset-ford-and-mercedes),
Audi listings: 10,668 real, scraped used-car listings.

## Three real data quality issues in this scraped dataset

Unlike a clean benchmark dataset, this is real scraped marketplace data,
and it shows:

1. **Leading/trailing whitespace** on string columns (`" A1"` instead of `"A1"`) — would silently split one model's signal across two one-hot columns if not stripped
2. **103 exact duplicate rows** — likely the same listing scraped more than once — dropped to avoid over-weighting whatever happened to get scraped twice
3. **57 rows with `engineSize == 0`** for Petrol/Diesel cars — not a real 0.0L engine, almost certainly a scraping gap — imputed with the **model-specific** median (a TT and a Q7 have very different typical engine sizes, so a single global median would be a poor estimate)

Full reasoning for each fix is in `clean_and_engineer.py`.

## The real, confirmed depreciation curve

EDA confirmed the classic non-linear used-car depreciation pattern
directly in this data: price drops **£5,710** from age 0→1, but only
**£1,241** from age 5→6 — roughly 4.6x steeper early depreciation. This
non-linearity is *why* a tree-based model was expected to beat linear
regression here, stated before training, not discovered as a surprise
afterward.

## Results

| Model | Val RMSE | Val MAE | Val R² | Val Within 10% |
|---|---|---|---|---|
| Linear Regression (baseline) | £2,935 | £1,987 | 0.943 | 65.6% |
| Random Forest | £2,542 | £1,589 | 0.957 | 75.1% |
| **XGBoost** | £2,396 | £1,554 | **0.962** | **76.7%** |

**Test set:** R² = 0.938, MAE = £1,632, **75.2% of predictions within
±10% of the actual price** — the business-relevant metric for a "fair,
dynamic pricing engine," which doesn't need dollar-exact predictions,
just consistently close ones.

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python data_loader.py          # Phase 5 — loads real data, flags the 3 quality issues
python eda.py                   # Phase 6 — confirms the non-linear depreciation curve
python clean_and_engineer.py   # Phase 5 (fixes) + 7 — whitespace/duplicate/engineSize fixes, feature engineering
python train_model.py          # Phase 8-9 — linear vs. RF vs. XGBoost, fair-pricing business metric
python monitor.py              # Phase 13 — monthly/quarterly market-drift monitoring
```

## Serve the model (Phase 10)

```bash
uvicorn app:app --reload
```

Open **http://127.0.0.1:8000/** for a price-estimate test form.

## File map

| File | SDLC Phase | Purpose |
|---|---|---|
| `data_loader.py` | 5 | Loads real data, flags 3 quality issues |
| `eda.py` | 6 | Confirms non-linear depreciation curve |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Whitespace/duplicate/engineSize fixes, feature engineering |
| `split.py` | 7 | Random split (no time axis, no class imbalance) |
| `train_model.py` | 8–9 | Linear vs. RF vs. XGBoost, fair-pricing business metric |
| `app.py` | 10 | FastAPI pricing service |
| `monitor.py` | 13 | Market-drift monitoring (monthly/quarterly cadence) |

## Known limitations (stated honestly)

- **Single brand (Audi only)** — the source dataset splits listings by manufacturer; only this file was reliably available as a complete mirror.
- **UK market, one point in time** — prices for other countries or a different economic period (fuel prices, interest rates, demand all shift) need separate validation.
- **Only 45.2% of predictions land within ±5%** of actual price, even though 75.2% land within ±10% — a pricing team needing tighter accuracy than 10% would need a more refined model.
