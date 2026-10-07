"""
Phase 6: Exploratory Data Analysis
------------------------------------
WHY THIS SPECIFIC CHECK COMES FIRST: a used car's depreciation curve is
famously NON-LINEAR (steep early depreciation, flattening with age) --
checking whether that shows up in this real data, rather than assuming
it, directly informs whether a plain linear model can capture price vs.
age at all, or whether a tree-based model's ability to learn non-linear
splits is actually necessary here (not just a default choice).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features


def run_eda(out_dir: str = "outputs"):
    raw = load_raw_data()
    df = engineer_features(clean_data(raw))

    print("=== Average price by age (the depreciation curve) ===")
    price_by_age = df.groupby("age_years")["price"].mean().round(0)
    print(price_by_age.head(10))
    # WHY WE PRINT THE FIRST 10 YEARS SPECIFICALLY: this is where the
    # steepest drop-off should be visible if the depreciation curve is
    # genuinely non-linear, per the standard real-world pattern.

    year1_drop = price_by_age.iloc[0] - price_by_age.iloc[1]
    later_drop = price_by_age.iloc[5] - price_by_age.iloc[6] if len(price_by_age) > 6 else None
    print(f"\nPrice drop from age 0->1: £{year1_drop:,.0f}")
    if later_drop is not None:
        print(f"Price drop from age 5->6: £{later_drop:,.0f}")
    print("If the first gap is much larger than the second, that confirms the")
    print("classic non-linear ('steep early, flattening later') depreciation shape.")

    print("\n=== Price by model (top 10 by count) ===")
    top_models = df["model"].value_counts().head(10).index
    print(df[df["model"].isin(top_models)].groupby("model")["price"].mean().round(0).sort_values(ascending=False))

    print("\n=== Price by transmission ===")
    print(df.groupby("transmission")["price"].mean().round(0))

    print("\n=== Correlation: mileage_per_year vs. price ===")
    corr = df[["mileage_per_year", "price"]].corr().iloc[0, 1]
    print(f"Pearson correlation: {corr:.3f}")

    # ---- Chart ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    price_by_age.plot(kind="line", marker="o", ax=axes[0], color="#4C72B0", title="Avg price by age (depreciation curve)")
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Avg price (£)")

    df[df["model"].isin(top_models)].groupby("model")["price"].mean().sort_values().plot(
        kind="barh", ax=axes[1], color="#55A868", title="Avg price by model (top 10 by count)"
    )

    axes[2].scatter(df["mileage"], df["price"], alpha=0.15, s=8, c=df["age_years"], cmap="viridis")
    axes[2].set_title("Price vs. mileage (colored by age)")
    axes[2].set_xlabel("Mileage")
    axes[2].set_ylabel("Price (£)")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_summary.png", dpi=120)
    print(f"\nSaved chart -> {out_dir}/eda_summary.png")


if __name__ == "__main__":
    run_eda()
