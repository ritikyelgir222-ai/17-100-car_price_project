"""
Phase 8: Model Development & Phase 9: Evaluation & Business Validation
---------------------------------------------------------------------------
WHY A LINEAR BASELINE IS EXPECTED TO STRUGGLE HERE, STATED UPFRONT (not
just discovered after the fact): Phase 6's EDA already confirmed price
vs. age is non-linear (steep early depreciation, flattening later) --
a linear model can't represent that curve no matter how it's tuned,
so the baseline's relative weakness here is a predicted, explained
outcome, not a surprise to report after the fact.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features, get_feature_columns
from split import split_data


def build_preprocessor(cat_cols, num_cols):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])


def within_tolerance_pct(y_true, y_pred, tolerance=0.10):
    """
    BUSINESS-RELEVANT METRIC for a "fair, dynamic pricing engine": a
    marketplace doesn't need every prediction to be dollar-exact -- it
    needs predictions CLOSE ENOUGH to be a trustworthy starting point for
    a listing price. This answers "what fraction of predictions land
    within +/-10% of the actual sale price?" -- a directly interpretable
    number for a pricing-team stakeholder, unlike RMSE alone.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pct_error = np.abs(y_pred - y_true) / y_true
    return float((pct_error <= tolerance).mean())


def train_and_evaluate():
    raw = load_raw_data()
    engineered = engineer_features(clean_data(raw))
    cat_cols, num_cols = get_feature_columns()
    feature_cols = cat_cols + num_cols

    X = engineered[feature_cols]
    y_log = np.log1p(engineered["price"])
    y_raw = engineered["price"]

    X_train, X_val, X_test, y_log_train, y_log_val, y_log_test = split_data(X, y_log)
    y_raw_train = y_raw.loc[y_log_train.index]
    y_raw_val = y_raw.loc[y_log_val.index]
    y_raw_test = y_raw.loc[y_log_test.index]

    preprocessor = build_preprocessor(cat_cols, num_cols)

    linreg_pipe = Pipeline([("prep", preprocessor), ("reg", LinearRegression())])
    linreg_pipe.fit(X_train, y_log_train)
    linreg_val_pred = np.expm1(linreg_pipe.predict(X_val))

    rf_pipe = Pipeline([("prep", build_preprocessor(cat_cols, num_cols)),
                         ("reg", RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42))])
    rf_pipe.fit(X_train, y_log_train)
    rf_val_pred = np.expm1(rf_pipe.predict(X_val))

    xgb_pipe = Pipeline([("prep", build_preprocessor(cat_cols, num_cols)),
                          ("reg", XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                                 subsample=0.8, colsample_bytree=0.8, random_state=42))])
    xgb_pipe.fit(X_train, y_log_train)
    xgb_val_pred = np.expm1(xgb_pipe.predict(X_val))

    experiment_log = []
    for name, pred in [("linear_regression (baseline)", linreg_val_pred),
                         ("random_forest", rf_val_pred),
                         ("xgboost", xgb_val_pred)]:
        rmse = np.sqrt(mean_squared_error(y_raw_val, pred))
        mae = mean_absolute_error(y_raw_val, pred)
        r2 = r2_score(y_raw_val, pred)
        within_10pct = within_tolerance_pct(y_raw_val, pred, 0.10)
        experiment_log.append({
            "model": name, "val_rmse": round(rmse, 2), "val_mae": round(mae, 2),
            "val_r2": round(r2, 4), "val_within_10pct": round(within_10pct, 4),
        })

    log_df = pd.DataFrame(experiment_log)
    print("=== Experiment Log (validation set, real GBP) ===")
    print(log_df.to_string(index=False))
    log_df.to_csv("outputs/experiment_log.csv", index=False)

    best_model_name = log_df.loc[log_df["val_mae"].idxmin(), "model"]
    print(f"\nModel selected by lowest validation MAE: {best_model_name}")
    final_pipe = {"linear_regression (baseline)": linreg_pipe, "random_forest": rf_pipe, "xgboost": xgb_pipe}[best_model_name]

    test_pred = np.expm1(final_pipe.predict(X_test))
    test_rmse = np.sqrt(mean_squared_error(y_raw_test, test_pred))
    test_mae = mean_absolute_error(y_raw_test, test_pred)
    test_r2 = r2_score(y_raw_test, test_pred)
    test_within_5pct = within_tolerance_pct(y_raw_test, test_pred, 0.05)
    test_within_10pct = within_tolerance_pct(y_raw_test, test_pred, 0.10)

    business_summary = {
        "selected_model": best_model_name,
        "test_rmse_gbp": round(test_rmse, 2),
        "test_mae_gbp": round(test_mae, 2),
        "test_r2": round(test_r2, 4),
        "test_within_5pct_of_actual": round(test_within_5pct, 4),
        "test_within_10pct_of_actual": round(test_within_10pct, 4),
        "n_test_listings": len(y_raw_test),
        "note": (
            "within_Xpct answers: what share of predicted prices would a "
            "pricing team consider 'close enough to trust as a starting "
            "listing price'? A fair, useful pricing engine doesn't need "
            "dollar-exact predictions, just consistently close ones -- "
            "this is the direct business translation of the model's error, "
            "not just its raw RMSE/MAE in isolation."
        ),
    }

    print("\n=== Business Validation Summary (Phase 9) ===")
    print(json.dumps(business_summary, indent=2))
    with open("outputs/business_validation.json", "w") as f:
        json.dump(business_summary, f, indent=2)

    ohe_names = final_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols)
    all_names = list(ohe_names) + num_cols
    reg = final_pipe.named_steps["reg"]
    if hasattr(reg, "feature_importances_"):
        importances = pd.Series(reg.feature_importances_, index=all_names).sort_values(ascending=False)
        print("\n=== Feature Importance (top 10) ===")
        print(importances.head(10).round(4).to_string())
        importances.to_csv("outputs/feature_importance.csv", header=["importance"])
    else:
        coefs = pd.Series(reg.coef_, index=all_names).sort_values(key=abs, ascending=False)
        print("\n=== Linear Regression Coefficients (top 10 by magnitude, log-price scale) ===")
        print(coefs.head(10).round(4).to_string())
        coefs.to_csv("outputs/feature_importance.csv", header=["coefficient"])

    joblib.dump(final_pipe, "outputs/car_price_model.joblib")
    joblib.dump({"cat_cols": cat_cols, "num_cols": num_cols}, "outputs/model_config.joblib")

    with open("outputs/model_card.json", "w") as f:
        json.dump({
            "model_type": best_model_name,
            "data_source": "100,000 UK Used Car Data Set, Audi listings (Kaggle: adityadesai13/used-car-dataset-ford-and-mercedes)",
            "n_features": len(feature_cols),
            "categorical_features": cat_cols,
            "numeric_features": num_cols,
            "training_rows": len(X_train),
            "test_mae_gbp": round(test_mae, 2),
            "test_within_10pct_of_actual": round(test_within_10pct, 4),
            "intended_use": "Suggest a starting listing/valuation price for a used Audi given its specifications.",
            "known_limitations": (
                "Single-brand (Audi only) model, UK market, scraped listing "
                "data from one point in time -- prices for other "
                "manufacturers, other countries, or a different economic "
                "period (fuel prices, interest rates, and used-car demand "
                "all shift over time) would need separate validation, not "
                "an assumption that this model transfers directly."
            ),
        }, f, indent=2)

    print("\nSaved model -> outputs/car_price_model.joblib")
    print("Saved model card -> outputs/model_card.json")


if __name__ == "__main__":
    train_and_evaluate()
