from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "hotel_bookings.csv"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATA CLEANING - HOTEL BOOKING DEMAND")
print("=" * 60)

print("\nOriginal dataset shape:", df.shape)

missing_before = df.isnull().sum()
missing_before = missing_before[missing_before > 0]

print("\nMissing values before cleaning:")
print(missing_before)

duplicate_count = df.duplicated().sum()

print("\nDuplicate rows found:", duplicate_count)

df_clean = df.drop_duplicates().copy()

print("\nRows removed:", len(df) - len(df_clean))
print("Cleaned dataset shape:", df_clean.shape)

missing_after = df_clean.isnull().sum()
missing_after = missing_after[missing_after > 0]

print("\nMissing values after duplicate removal:")
print(missing_after)


output_path = RESULTS_DIR / "hotel_bookings_deduplicated.csv"

df_clean.to_csv(output_path, index=False)

print("\nCleaned dataset saved to:")
print(output_path)

report = pd.DataFrame({
    "Metric": [
        "Original rows",
        "Original columns",
        "Duplicate rows removed",
        "Rows after deduplication",
        "Columns after deduplication",
    ],
    "Value": [
        len(df),
        len(df.columns),
        duplicate_count,
        len(df_clean),
        len(df_clean.columns),
    ],
})

report_path = RESULTS_DIR / "cleaning_report.csv"
report.to_csv(report_path, index=False)

print("\nCleaning report saved to:")
print(report_path)

print("\nStage 1 completed successfully.")