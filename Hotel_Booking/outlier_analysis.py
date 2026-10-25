from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    PROJECT_DIR
    / "results"
    / "hotel_bookings_deduplicated.csv"
)

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("NUMERICAL DATA ANALYSIS")
print("=" * 60)

print("\n1. Statistical summary:")
print(df.describe().T.round(2))

print("\n2. Rows with children greater than 3:")
print(df[df["children"] > 3][
    ["hotel", "is_canceled", "adults", "children", "babies"]
])

print("\n3. Rows with zero adults:")
zero_adults = df[df["adults"] == 0]
print("Number of rows:", len(zero_adults))
print(zero_adults[
    ["hotel", "is_canceled", "adults", "children", "babies"]
].head(10))

df["total_nights"] = (
    df["stays_in_weekend_nights"]
    + df["stays_in_week_nights"]
)

print("\n4. Stay duration:")
print("Minimum nights:", df["total_nights"].min())
print("Maximum nights:", df["total_nights"].max())
print("Rows with zero nights:", (df["total_nights"] == 0).sum())
print("Rows with more than 30 nights:",
      (df["total_nights"] > 30).sum())

print("\nOutlier analysis completed.")