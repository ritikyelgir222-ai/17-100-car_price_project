# SDLC Documentation: Used Car Price Prediction
### Automotive Marketplace Industry | ML | Day 17 of the 100-Day Series
### Filled against the master 14-phase SDLC template, using the real project we built

Every number below comes from the actual pipeline run on the real UK
Used Car (Audi) dataset.

---

## Phase 1: Discovery & Stakeholder Requirement Gathering
**Owner:** Business Analyst / Data Scientist | **Output:** Meeting notes, stakeholder map

- **Stakeholders identified:** marketplace pricing lead (problem owner), sellers listing cars (end users of the price suggestion), buyers (indirect beneficiaries of consistent, fair pricing)
- **Discovery findings:**
  - Current process (assumed for this exercise): sellers set their own asking price with no data-driven benchmark, leading to inconsistent pricing for comparable cars
  - Decision this project informs: what starting price should the marketplace suggest to a seller listing a given used Audi?
  - Critical framing: "fair" pricing here means CONSISTENT and ACCURATE across car types, not demographic fairness — this dataset has no demographic features at all, so the fairness question is purely about pricing accuracy across different models/ages/conditions
- **Deliverable — Problem Statement:** "Suggest a data-driven starting price for a used Audi listing, accurate enough that both buyers and sellers can trust it as a fair reference point."

---

## Phase 2: Business Requirement Document (BRD)
**Owner:** Business Analyst | **Output:** BRD

- **Business objective:** minimize the share of price suggestions far enough from actual market value to undermine seller/buyer trust
- **Scope:** in-scope: Audi listings only; out-of-scope: other manufacturers (a real, disclosed scope limit — see Phase 5)
- **Success metrics / KPIs:** % of predictions within ±10% of actual sale price, not just RMSE/MAE in isolation
- **Assumptions & constraints:** single point-in-time UK market snapshot; no condition/accident-history/equipment data available, a real ceiling on achievable accuracy
- **Sign-off:** N/A — self-directed portfolio project

---

## Phase 3: Functional & Technical Requirement Document (FRD/TRD)
**Owner:** Data Scientist / Tech Lead | **Output:** FRD/TRD

- **Functional requirements:** given a car's specifications, return a price estimate and the top factors driving it
- **Non-functional requirements:** sub-second response time (confirmed in testing)
- **Data sources:** a single static CSV in this exercise; a real system would need a live feed of sale prices to stay current with the market
- **Integration points:** none in this version — a production version would integrate with the marketplace's listing-creation flow

---

## Phase 4: Project Planning
**Owner:** Project/Delivery Manager | **Output:** Project plan, risk register

- **Build sequence used:** data loading (flagging quality issues) → EDA (confirming the depreciation curve) → cleaning/feature engineering → split → model comparison → API → monitoring → documentation
- **Real risks encountered during the build:**
  - Risk: only one manufacturer's file (Audi) was reliably available as a complete raw GitHub mirror, despite the source dataset covering many brands — resolved by proceeding with a single-brand scope and disclosing this explicitly, rather than fabricating or guessing data for other brands
  - Risk: three separate real data quality issues in the scraped data (whitespace, duplicates, zero engine sizes) — each required a DIFFERENT fix and different reasoning, not a single generic "clean the data" step

---

## Phase 5: Data Collection & Data Understanding
**Owner:** Data Engineer / Data Scientist | **Output:** Data dictionary, data quality report

- **Data inventory:** 10,668 real Audi listings, 9 raw columns, scraped UK marketplace data
- **Data quality report — three distinct real issues found:**
  1. Leading/trailing whitespace on string columns (e.g. `" A1"`)
  2. 103 exact duplicate rows
  3. 57 rows with `engineSize == 0` for Petrol/Diesel cars — a scraping gap, not a real value
- **Access & governance:** public, already-anonymized dataset (no personal seller/buyer information); no additional governance required

---

## Phase 6: Exploratory Data Analysis (EDA)
**Owner:** Data Scientist | **Output:** `eda.py`, `outputs/eda_summary.png`

**Actual findings:**

- **The depreciation curve is confirmed genuinely non-linear**: £5,710 price drop from age 0→1, but only £1,241 from age 5→6 — roughly 4.6x steeper early depreciation, the textbook real-world pattern
- Price by model: Q7 highest (£44,707 average), A1 lowest among common models (£14,275)
- Price by transmission: Automatic/Semi-Auto well above Manual
- mileage_per_year correlation with price: -0.301, real but moderate

**Why this EDA finding matters more than any other in this project:** it directly PREDICTS, before any model is trained, that a linear model will structurally underperform a tree-based model — Phase 8's results confirm this prediction rather than discovering it as a surprise.

---

## Phase 7: Data Preprocessing & Feature Engineering
**Owner:** Data Scientist / ML Engineer | **Output:** `clean_and_engineer.py`, `split.py`

- **Three distinct cleaning fixes, each with different reasoning:** whitespace stripping (prevents silently splitting one model's signal across two one-hot columns), duplicate removal (prevents over-weighting re-scraped listings), and model-specific median imputation for zero engine sizes (a single global median would be a poor estimate given how much engine size varies by model)
- **Feature engineering:** `age_years` (raw year has no direct price relationship; age does) and `mileage_per_year` (captures usage intensity independent of age)
- **Split strategy:** plain random split — no time axis exists in this dataset (year is a feature, not a timestamp) and no class imbalance to stratify on (this is regression, not classification) — the simplest correct choice, not a default applied without thought

---

## Phase 8: Model Development
**Owner:** Data Scientist / ML Engineer | **Output:** `train_model.py`, `outputs/experiment_log.csv`

**Actual experiment log (validation set, real £):**

| Model | RMSE | MAE | R² | Within 10% |
|---|---|---|---|---|
| Linear Regression (baseline) | £2,935 | £1,987 | 0.943 | 65.6% |
| Random Forest | £2,542 | £1,589 | 0.957 | 75.1% |
| **XGBoost** | **£2,396** | **£1,554** | **0.962** | **76.7%** |

**A clean, PREDICTED win, unlike Days 1, 7, and 9's near-ties:** Phase
6's EDA already confirmed the non-linear depreciation curve a linear
model can't represent — this result validates that prediction rather
than surprising anyone, a meaningfully different kind of "honest
reporting" than the close-call comparisons earlier in the series (here,
the honesty is in stating the expectation BEFORE seeing results, not in
disclosing an unexpected tie afterward).

---

## Phase 9: Model Evaluation & Business Validation
**Owner:** Data Scientist + Business Stakeholder | **Output:** `outputs/business_validation.json`

- **Technical metrics (test set):** RMSE £3,005, MAE £1,632, R² 0.938
- **Business validation — the "fair pricing" framing made concrete:** 75.2% of predictions fall within ±10% of actual price; only 45.2% fall within the tighter ±5% band — both numbers reported, not just the more flattering one
- **Bias/fairness check:** no demographic features exist in this dataset, so "fairness" here is interpreted as pricing CONSISTENCY and ACCURACY across car types (models, ages, transmissions) — directly measured by the within-tolerance metric, not a separate demographic audit as in Day 1 or Day 7
- **Explainability:** `age_years` dominates feature importance (0.265), directly matching the EDA's depreciation-curve finding — the model learned exactly the pattern EDA had already surfaced

---

## Phase 10: MLOps & Deployment
**Owner:** ML Engineer | **Output:** `app.py`

- **Packaging:** FastAPI service (`/price`, `/health`, browser test form)
- **Verified with 2 constructed listings:** a nearly-new, low-mileage Q7 priced at £45,040; a 15-year-old, high-mileage A1 priced at £2,659 — both plausible, sensible real-world estimates

---

## Phase 11: Testing
**Owner:** QA / Data Scientist | **Output:** manual test log

- **What was actually tested:** full pipeline run end-to-end; API tested via FastAPI's `TestClient` with two constructed listings spanning the value spectrum, both producing sensible prices
- **What was NOT done:** no formal unit test suite; no UAT with actual marketplace sellers; no test against a genuinely held-out FUTURE time period (this dataset has no timestamp finer than manufacture year, so a true out-of-time validation isn't possible with this data alone)

---

## Phase 12: Documentation & Handover
**Owner:** Data Scientist | **Output:** `README.md`, `PROJECT_DOCUMENTATION.md`, this file

- `README.md` leads with the three real data quality issues and the confirmed depreciation curve, since both are the most instructive, reusable lessons from this specific project
- Both within-tolerance figures (45.2% at ±5%, 75.2% at ±10%) are reported together, not just the more impressive-looking one

---

## Phase 13: Monitoring & Maintenance
**Owner:** MLOps / Data Scientist | **Output:** `monitor.py`

- **What's monitored:** PSI drift on mileage/age/engineSize, and MAE decay against baseline
- **Recommended cadence: monthly-to-quarterly** — a used-car market shifts with fuel prices, interest rates, new-model-year releases, and seasonal demand: gradual market drift, closer in spirit to Day 5's quarterly segmentation cadence than Day 2's adversarial fraud pace or Day 9's academic-calendar pace — a FIFTH distinct monitoring-cadence justification now established across this series, each tied to a different real-world mechanism

---

## Phase 14: Project Closure & Delivery
**Owner:** N/A (self-directed project) | **Output:** this document, `PROJECT_DOCUMENTATION.md`

- **Closure against original objective:** a working pricing model was built and validated, achieving 75.2% of predictions within a defensible ±10% tolerance on real held-out data, with the model's core behavior (age dominating price) directly matching the depreciation curve EDA had already confirmed
- **Retrospective:**
  - What went well: predicting the linear baseline's weakness FROM the EDA, before training anything, and then having Phase 8's results confirm that prediction, is a cleaner demonstration of "EDA actually informing modeling choices" than a project where the connection is asserted only after the fact
  - What to improve next time: track down complete raw mirrors for the other manufacturer files (BMW, Ford, Mercedes, etc.) to build a genuinely multi-brand pricing engine, since brand-to-brand pricing dynamics likely differ in ways a single-brand model can't capture
- **Handover:** packaged as a complete, runnable project (code + real data + trained model + documentation), consistent with every prior day in the series

---

## Mapping back to the Day 17 LinkedIn post

| LinkedIn section | Pulled from phases | Real number used |
|---|---|---|
| Client & Problem | 1–2 | Automotive marketplace, inconsistent seller pricing |
| Requirements & Data | 3, 5 | Real UK Audi dataset, 3 real scraped-data quality issues |
| Approach | 6–8 | Confirmed non-linear depreciation curve → XGBoost, a predicted win |
| Result & Business Impact | 9 | 75.2% of predictions within ±10% of actual price |
| Path to Production | 10–13 | Pricing API + monthly/quarterly market-drift monitoring |
