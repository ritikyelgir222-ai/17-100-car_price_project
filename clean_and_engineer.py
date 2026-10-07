"""
Phases 5 (data quality) & 7 (feature engineering)
------------------------------------------------------
THREE REAL DATA QUALITY ISSUES FOUND IN THIS SCRAPED DATASET, each with
a different fix and a different reason for that fix -- a useful case
study since scraped marketplace data rarely arrives clean.
"""

import numpy as np
import pandas as pd

CURRENT_YEAR = 2020  # the dataset's own max year, used as "now" for age calculations


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # -----------------------------------------------------------------
    # Fix 1: leading/trailing whitespace on string columns (e.g. " A1"
    # instead of "A1"). WHY THIS MATTERS, NOT JUST COSMETIC: without
    # stripping this, one-hot encoding would silently create TWO
    # separate dummy columns for what should be the same model (" A1"
    # and "A1" if both ever appeared), splitting that model's signal
    # across two columns instead of concentrating it in one -- a subtle
    # bug that wouldn't raise any error, just quietly weaken the model.
    # -----------------------------------------------------------------
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="str").columns:
        df[col] = df[col].str.strip()

    # -----------------------------------------------------------------
    # Fix 2: exact duplicate rows (103 found). WHY DROPPED: a scraper
    # re-visiting the same listing (e.g. across multiple scrape runs, or
    # a listing appearing on more than one page) produces an identical
    # row -- keeping duplicates would let the model see the same real
    # car multiple times, silently over-weighting whatever price/feature
    # combination happened to get scraped more than once, for a reason
    # that has nothing to do with how representative that combination
    # actually is of the used car market.
    # -----------------------------------------------------------------
    df = df.drop_duplicates()

    # -----------------------------------------------------------------
    # Fix 3: engineSize == 0 for Petrol/Diesel cars (57 rows). WHY THIS
    # IS AN ERROR, NOT A REAL VALUE: a 0.0-litre combustion engine
    # doesn't exist -- these are Petrol and Diesel cars (verified: zero
    # of the 57 are Hybrid, none are a genuine electric variant), so this
    # is almost certainly a scraping gap (the engine size field wasn't
    # populated on the source listing) rather than a real "0". WHY
    # IMPUTE WITH THE MODEL'S MEDIAN, NOT THE DATASET-WIDE MEDIAN: engine
    # size varies enormously by model (a TT and a Q7 are not remotely
    # comparable), so a single global median would be a poor estimate
    # for a specific model. The model-specific median is a defensible,
    # like-for-like estimate.
    # -----------------------------------------------------------------
    model_median_engine = df[df["engineSize"] > 0].groupby("model")["engineSize"].median()
    zero_engine_mask = df["engineSize"] == 0
    df.loc[zero_engine_mask, "engineSize"] = df.loc[zero_engine_mask, "model"].map(model_median_engine)
    df["engineSize"] = df["engineSize"].fillna(df["engineSize"].median())

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -----------------------------------------------------------------
    # Engineered feature 1: age_years
    # BUSINESS LOGIC: raw `year` (e.g. 2017) has no direct relationship
    # to price on its own -- what matters is how OLD the car is relative
    # to now. Using the dataset's own max year (2020) as "now" is more
    # defensible than an arbitrary external "today," since it's the most
    # recent point this specific dataset can actually speak to.
    # -----------------------------------------------------------------
    df["age_years"] = CURRENT_YEAR - df["year"]

    # -----------------------------------------------------------------
    # Engineered feature 2: mileage_per_year
    # BUSINESS LOGIC: 60,000 miles means something very different for a
    # 2-year-old car (30,000 miles/year, heavily used) than a 10-year-old
    # car (6,000 miles/year, lightly used) -- raw mileage alone conflates
    # these. This ratio directly captures usage INTENSITY, a standard
    # real-world used-car valuation signal, independent of age itself.
    # -----------------------------------------------------------------
    df["mileage_per_year"] = df["mileage"] / df["age_years"].replace(0, 1)

    return df


def get_feature_columns() -> tuple:
    categorical_cols = ["model", "transmission", "fuelType"]
    numeric_cols = ["age_years", "mileage", "mileage_per_year", "tax", "mpg", "engineSize"]
    return categorical_cols, numeric_cols


if __name__ == "__main__":
    from data_loader import load_raw_data

    raw = load_raw_data()
    cleaned = clean_data(raw)
    engineered = engineer_features(cleaned)
    cat_cols, num_cols = get_feature_columns()

    print(f"Raw rows: {len(raw)}  ->  Cleaned rows: {len(cleaned)}")
    print(f"Remaining engineSize == 0 after imputation: {(engineered['engineSize']==0).sum()}")
    print(f"\nCategorical features: {cat_cols}")
    print(f"Numeric features: {num_cols}")

    engineered.to_csv("outputs/engineered_data.csv", index=False)
    print("\nSaved -> outputs/engineered_data.csv")
