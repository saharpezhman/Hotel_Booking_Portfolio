from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    PROJECT_DIR
    / "results"
    / "hotel_bookings_deduplicated.csv"
)
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

df["total_nights"] = (
    df["stays_in_weekend_nights"]
    + df["stays_in_week_nights"]
)

df["flag_negative_adr"] = df["adr"] < 0
df["flag_high_adr"] = df["adr"] > 1000
df["flag_zero_adults"] = df["adults"] == 0
df["flag_zero_nights"] = df["total_nights"] == 0
df["flag_many_babies"] = df["babies"] > 3
df["flag_many_children"] = df["children"] > 3

# گزارش تعداد موارد علامت‌گذاری‌شده
flag_columns = [
    "flag_negative_adr",
    "flag_high_adr",
    "flag_zero_adults",
    "flag_zero_nights",
    "flag_many_babies",
    "flag_many_children",
]

print("=" * 60)
print("PREPARING ANALYSIS DATA")
print("=" * 60)

for column in flag_columns:
    print(f"{column}: {df[column].sum()}")

output_path = RESULTS_DIR / "hotel_bookings_analysis_ready.csv"
df.to_csv(output_path, index=False)

print("\nRows preserved:", len(df))
print("Columns after adding analysis fields:", len(df.columns))
print("Saved to:")
print(output_path)
print("\nPreparation completed successfully.")