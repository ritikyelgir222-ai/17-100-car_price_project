"""
Phase 5: Data Collection & Data Understanding
------------------------------------------------
DATA SOURCE
-----------
This project uses the Audi file from the "100,000 UK Used Car Data Set"
(Kaggle: adityadesai13/used-car-dataset-ford-and-mercedes), a real,
scraped collection of UK used-car listings split by manufacturer:
    https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes

10,668 real Audi listings: model, year, price, transmission, mileage,
fuel type, road tax, mpg, and engine size.

WHY THIS DATASET FOR DAY 17 (Used Car Price Prediction, Automotive)
--------------------------------------------------------------------
- It's real, scraped marketplace data (not a synthetic price formula),
  so it carries the genuine data quality issues of scraped web data --
  see clean_and_engineer.py for three real ones found and fixed here.
- A used car's price is driven by well-understood, non-linear real-world
  dynamics (depreciation curves flatten with age, mileage matters more
  for younger cars than older ones) that a naive linear model handles
  poorly -- a genuinely different kind of non-linearity than any prior
  regression problem in this series (Day 10's demand forecasting used
  lag features on a time axis; this is non-linear on an AGE axis instead).
- WHY ONLY ONE BRAND (Audi), STATED UPFRONT: the source dataset splits
  listings by manufacturer into separate files; only the Audi file was
  reliably available as a complete raw mirror at the time this project
  was built. This is a real, disclosed scope limitation (a pricing
  engine for one brand, not the whole market), not something to gloss
  over -- see the Known Limitations section in the documentation.

If you're following along on Kaggle: download "audi.csv" from the link
above and place it at `data/audi.csv` -- identical schema to the file
already included here.
"""

import pandas as pd

RAW_DATA_PATH = "data/audi.csv"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} Audi listings, {len(df.columns)} columns from {RAW_DATA_PATH}")
    print(f"Price range: £{df['price'].min():,} to £{df['price'].max():,}, mean £{df['price'].mean():,.0f}")
    print(f"Year range: {df['year'].min()} to {df['year'].max()}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print(f"Rows with engineSize == 0: {(df['engineSize']==0).sum()}")
    print(f"Model column has leading/trailing whitespace: {(df['model'] != df['model'].str.strip()).any()}")
