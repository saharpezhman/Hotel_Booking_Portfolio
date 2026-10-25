from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    PROJECT_DIR
    / "results"
    / "hotel_bookings_deduplicated.csv"
)

df = pd.read_csv(DATA_PATH)

df["total_nights"] = (
    df["stays_in_weekend_nights"]
    + df["stays_in_week_nights"]
)

columns_to_show = [
    "hotel",
    "is_canceled",
    "adults",
    "children",
    "babies",
    "adr",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "total_nights",
    "market_segment",
    "customer_type",
]

print("=" * 60)
print("DETAILED OUTLIER INSPECTION")
print("=" * 60)

print("\n1. Highest ADR values:")
print(
    df.nlargest(10, "adr")[columns_to_show]
    .to_string(index=False)
)

print("\n2. Zero-night bookings:")
print("Total:", (df["total_nights"] == 0).sum())
print(
    df[df["total_nights"] == 0][columns_to_show]
    .head(10)
    .to_string(index=False)
)

print("\n3. Bookings longer than 30 nights:")
long_stays = df[df["total_nights"] > 30]
print("Total:", len(long_stays))
print(
    long_stays[columns_to_show]
    .sort_values("total_nights", ascending=False)
    .to_string(index=False)
)

print("\n4. Bookings with zero adults:")
zero_adults = df[df["adults"] == 0]
print("Total:", len(zero_adults))
print(
    zero_adults[columns_to_show]
    .head(10)
    .to_string(index=False)
)

print("\nDetailed inspection completed.")