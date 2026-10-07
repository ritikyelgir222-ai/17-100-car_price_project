"""
Phase 13: Monitoring & Maintenance
--------------------------------------
WHY THE RIGHT CADENCE HERE IS DIFFERENT FROM EVERY PRIOR DAY: a used-car
pricing model doesn't decay because of adversarial behavior (Day 2) or
slow-moving customer habits (Day 5) -- it decays because the MARKET
itself moves: fuel prices, interest rates, new-model releases, and
seasonal demand all shift what a given car is actually worth, on a
timescale of weeks to months, not adversarial hours or academic terms.
"""

import numpy as np
import pandas as pd
import joblib
import json
from sklearn.metrics import mean_absolute_error

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features, get_feature_columns
from split import split_data

MAE_INCREASE_THRESHOLD_PCT = 0.20


def population_stability_index(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def run_monitoring_check():
    model = joblib.load("outputs/car_price_model.joblib")

    raw = load_raw_data()
    engineered = engineer_features(clean_data(raw))
    cat_cols, num_cols = get_feature_columns()
    feature_cols = cat_cols + num_cols
    X = engineered[feature_cols]
    y_log = np.log1p(engineered["price"])
    y_raw = engineered["price"]

    X_train, X_val, X_test, y_log_train, y_log_val, y_log_test = split_data(X, y_log)
    y_raw_test = y_raw.loc[y_log_test.index]

    print("=== Feature Drift (PSI): mileage, age, engineSize ===")
    for feat in ["mileage", "age_years", "engineSize"]:
        psi = population_stability_index(X_train[feat], X_test[feat])
        flag = "SIGNIFICANT DRIFT" if psi > 0.25 else ("moderate" if psi > 0.1 else "ok")
        print(f"  {feat}: PSI={psi:.4f} [{flag}]")

    test_pred = np.expm1(model.predict(X_test))
    current_mae = mean_absolute_error(y_raw_test, test_pred)
    print(f"\n=== Performance on test batch ===")
    print(f"MAE: £{current_mae:.2f}")

    with open("outputs/business_validation.json") as f:
        baseline = json.load(f)
    baseline_mae = baseline["test_mae_gbp"]
    pct_change = (current_mae - baseline_mae) / baseline_mae

    if pct_change > MAE_INCREASE_THRESHOLD_PCT:
        print(f"\n⚠️  RETRAIN TRIGGERED: MAE rose {pct_change:+.1%} (threshold: {MAE_INCREASE_THRESHOLD_PCT:.0%})")
    else:
        print(f"\n✅ No retrain needed (MAE change: {pct_change:+.1%}, threshold: {MAE_INCREASE_THRESHOLD_PCT:.0%})")

    print("\n=== Recommended monitoring cadence ===")
    print("Monthly-to-quarterly: a used-car market shifts with fuel prices, interest")
    print("rates, new-model-year releases, and seasonal demand -- gradual market drift,")
    print("not adversarial or academic-calendar-paced, closer in spirit to Day 5's")
    print("quarterly segmentation cadence than Day 2's near-real-time fraud monitoring.")
    print("A model trained on one point-in-time market snapshot (like this project's")
    print("data) should be treated as reflecting THAT market, not assumed current.")


if __name__ == "__main__":
    run_monitoring_check()
