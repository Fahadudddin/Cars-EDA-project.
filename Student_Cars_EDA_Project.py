"""
Cars Exploratory Data Analysis (EDA) Project
Run: python Student_Cars_EDA_Project.py
Optional: python Student_Cars_EDA_Project.py path/to/cars.csv

IMPORTANT:
- The built-in demo data is synthetic and is used only to make this file runnable.
- For final college submission, replace it with a mentor-provided or public real cars dataset.
- Update COLUMN_MAP below if your dataset uses different column names.
- The report supplied with this project describes the demo-data run; regenerate it after
  using a real dataset so all results and charts match your own data.
Dependencies: pandas, numpy, matplotlib
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RANDOM_SEED = 42
OUTPUT_DIR = Path("cars_eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def make_demo_dataset(n=500, seed=RANDOM_SEED):
    """Create deterministic synthetic car records for a runnable demonstration."""
    rng = np.random.default_rng(seed)
    brands = np.array(["Toyota", "Honda", "Hyundai", "Ford", "BMW",
                       "Maruti", "Kia", "Mercedes", "Volkswagen", "Tata"])
    brand = rng.choice(brands, n, p=[.13, .11, .12, .10, .07, .14, .08, .05, .09, .11])
    year = rng.integers(2005, 2026, n)
    fuel = rng.choice(["Petrol", "Diesel", "Hybrid", "Electric"], n,
                      p=[.48, .31, .13, .08])
    transmission = rng.choice(["Manual", "Automatic"], n, p=[.56, .44])
    mileage = np.clip(rng.normal(16, 4, n) +
                      np.where(fuel == "Electric", 55,
                      np.where(fuel == "Hybrid", 10, 0)), 5, 100)
    engine = np.clip(rng.normal(1.8, .65, n), .7, 5).round(1)
    horsepower = np.clip(engine * 65 + rng.normal(25, 35, n) +
                         np.where(brand == "BMW", 60, 0) +
                         np.where(brand == "Mercedes", 75, 0), 45, 500).round().astype(int)
    price = (5 - (2026 - year) * .32 + engine * 3 + horsepower * .035 +
             np.where(transmission == "Automatic", 1.4, 0) +
             np.where(fuel == "Electric", 7, np.where(fuel == "Hybrid", 3, 0)) +
             np.where(brand == "BMW", 8, 0) + np.where(brand == "Mercedes", 10, 0) +
             rng.normal(0, 3, n))
    price = np.clip(price, 1.2, 55).round(2)
    data = pd.DataFrame({
        "brand": brand, "year": year, "price_lakh": price, "fuel_type": fuel,
        "mileage_kmpl": mileage.round(1), "engine_l": engine,
        "transmission": transmission, "horsepower": horsepower
    })
    # Small, deliberate quality issues for demonstrating the cleaning workflow.
    for col in ["mileage_kmpl", "engine_l", "fuel_type"]:
        data.loc[rng.choice(data.index, 5, replace=False), col] = np.nan
    data.loc[0, "brand"] = " toyota "
    data.loc[1, "fuel_type"] = "petrol"
    data.loc[2] = data.loc[3]
    return data

def clean_data(data):
    """Normalize labels, remove exact duplicate rows, and impute missing values."""
    data = data.copy()
    before_shape = data.shape
    missing_before = int(data.isna().sum().sum())
    data.columns = [str(c).strip().lower().replace(" ", "_") for c in data.columns]
    for col in ["brand", "fuel_type", "transmission"]:
        if col in data:
            data[col] = data[col].astype("string").str.strip().str.title()
    numeric_cols = ["year", "price_lakh", "mileage_kmpl", "engine_l", "horsepower"]
    for col in numeric_cols:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    duplicates = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()
    missing_after_drop = int(data.isna().sum().sum())
    for col in numeric_cols:
        if col in data and data[col].isna().any():
            data[col] = data[col].fillna(data[col].median())
    for col in ["brand", "fuel_type", "transmission"]:
        if col in data and data[col].isna().any():
            mode = data[col].mode()
            if not mode.empty:
                data[col] = data[col].fillna(mode.iloc[0])
    report = {
        "shape_before": before_shape, "shape_after": data.shape,
        "duplicates_removed": duplicates, "missing_before": missing_before,
        "missing_after_imputation": int(data.isna().sum().sum()),
        "missing_after_duplicate_drop": missing_after_drop
    }
    return data, report

def chart_and_observation(data):
    """Save six labeled charts; observations are printed to the terminal."""
    plt.figure(figsize=(8, 5))
    plt.hist(data["price_lakh"], bins=24, edgecolor="black")
    plt.title("Distribution of Car Prices"); plt.xlabel("Price (₹ lakh)"); plt.ylabel("Records")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "01_price_distribution.png", dpi=150); plt.close()

    data["brand"].value_counts().plot(kind="bar", figsize=(8, 5), edgecolor="black")
    plt.title("Cars by Brand"); plt.xlabel("Brand"); plt.ylabel("Count")
    plt.xticks(rotation=35, ha="right"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_brand_counts.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(data["year"], data["price_lakh"], alpha=.55)
    plt.title("Model Year vs Price"); plt.xlabel("Model year"); plt.ylabel("Price (₹ lakh)")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "03_year_vs_price.png", dpi=150); plt.close()

    data.boxplot(column="price_lakh", by="transmission", figsize=(8, 5))
    plt.suptitle(""); plt.title("Price by Transmission Type")
    plt.xlabel("Transmission"); plt.ylabel("Price (₹ lakh)")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "04_price_by_transmission.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(data["mileage_kmpl"], data["price_lakh"], alpha=.5)
    plt.title("Mileage vs Price"); plt.xlabel("Mileage (km/l or equivalent)")
    plt.ylabel("Price (₹ lakh)"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_mileage_vs_price.png", dpi=150); plt.close()

    numeric = data.select_dtypes(include="number")
    corr = numeric.corr()
    plt.figure(figsize=(8, 5))
    plt.imshow(corr, aspect="auto", vmin=-1, vmax=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=35, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar(label="Correlation"); plt.title("Correlation Heatmap")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "06_correlation_heatmap.png", dpi=150); plt.close()

    print("\\nChart observations:")
    print("1. Price histogram shows the spread and concentration of listed prices.")
    print("2. Brand count chart shows which brands are most represented in this dataset.")
    print("3. Year-price scatter plot helps inspect the association between age and price.")
    print("4. Box plot compares price distributions across transmission categories.")
    print("5. Mileage-price scatter plot shows whether a visible relationship exists.")
    print("6. Correlation heatmap summarizes linear relationships among numeric variables.")

def main():
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        raw = pd.read_csv(csv_path)
        print(f"Loaded dataset: {csv_path}")
    else:
        raw = make_demo_dataset()
        print("No CSV supplied: running the synthetic demonstration dataset.")
        print("Use a real cars dataset for final submission.")
    print("\\n--- Raw dataset inspection ---")
    print("Shape:", raw.shape)
    print("Columns:", raw.columns.tolist())
    print("\\nSample rows:\\n", raw.head())
    print("\\nData types:\\n", raw.dtypes)
    print("\\nMissing values by column:\\n", raw.isna().sum())
    print("Duplicate rows:", int(raw.duplicated().sum()))

    data, cleaning = clean_data(raw)
    print("\\n--- Cleaning summary ---")
    for key, value in cleaning.items():
        print(f"{key}: {value}")
    print("\\n--- Descriptive statistics ---")
    print(data.describe(include="all").T)
    if "brand" in data and "price_lakh" in data:
        print("\\nMean price by brand (₹ lakh):\\n",
              data.groupby("brand")["price_lakh"].mean().sort_values(ascending=False))
    if "fuel_type" in data:
        print("\\nFuel type counts:\\n", data["fuel_type"].value_counts())
    if "year" in data and "price_lakh" in data:
        print("\\nCorrelation year-price:", round(data["year"].corr(data["price_lakh"]), 3))
    if "mileage_kmpl" in data and "price_lakh" in data:
        print("Correlation mileage-price:", round(data["mileage_kmpl"].corr(data["price_lakh"]), 3))

    # IQR-based outlier count is reported, not automatically deleted.
    if "price_lakh" in data:
        q1, q3 = data["price_lakh"].quantile([.25, .75])
        iqr = q3 - q1
        outliers = data[(data["price_lakh"] < q1 - 1.5 * iqr) |
                        (data["price_lakh"] > q3 + 1.5 * iqr)]
        print(f"Potential price outliers by IQR rule: {len(outliers)}")
        print("Outliers are retained unless domain evidence justifies removal.")
    chart_and_observation(data)
    data.to_csv(OUTPUT_DIR / "cleaned_cars_data.csv", index=False)
    print(f"\\nCleaned data and six charts saved in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
