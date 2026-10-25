from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    PROJECT_DIR
    / "results"
    / "hotel_bookings_deduplicated.csv"
)

df = pd.read_csv(DATA_PATH)

columns = [
    "hotel",
    "is_canceled",
    "lead_time",
    "adults",
    "children",
    "babies",
    "adr",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "market_segment",
    "customer_type",
    "reservation_status",
]

print("\n1. Negative ADR:")
print(df.loc[df["adr"] < 0, columns].to_string(index=False))


print("\n2. ADR above 1000:")
print(df.loc[df["adr"] > 1000, columns].to_string(index=False))

print("\n3. Babies above 3:")
print(df.loc[df["babies"] > 3, columns].to_string(index=False))

print("\n4. No adults, children, or babies:")
mask = (
    (df["adults"] == 0)
    & (df["children"].fillna(0) == 0)
    & (df["babies"] == 0)
)
print("Number of rows:", mask.sum())
print(df.loc[mask, columns].head(15).to_string(index=False))

print("\nInspection completed.")