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

print("=" * 60)
print("MISSING VALUES ANALYSIS")
print("=" * 60)


missing_count = df.isnull().sum()

missing_percent = (
    df.isnull().sum() / len(df) * 100
).round(2)

missing_report = pd.DataFrame({
    "Missing_Count": missing_count,
    "Missing_Percent": missing_percent
})

missing_report = missing_report[
    missing_report["Missing_Count"] > 0
].sort_values("Missing_Percent", ascending=False)

print("\n1. Missing values report:")
print(missing_report)

print("\n2. Rows with missing children:")
print(df[df["children"].isnull()])

print("\n3. Children value distribution:")
print(df["children"].value_counts(dropna=False).sort_index())

report_path = RESULTS_DIR / "missing_values_report.csv"
missing_report.to_csv(report_path)

print("\nReport saved to:")
print(report_path)

print("\nMissing values analysis completed.")