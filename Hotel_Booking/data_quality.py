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


checks = {
    "Rows": len(df),
    "Missing children": df["children"].isna().sum(),
    "Zero adults": (df["adults"] == 0).sum(),
    "Zero-night bookings": (df["total_nights"] == 0).sum(),
    "Bookings over 30 nights": (df["total_nights"] > 30).sum(),
    "ADR equal to zero": (df["adr"] == 0).sum(),
    "ADR below zero": (df["adr"] < 0).sum(),
    "ADR above 1000": (df["adr"] > 1000).sum(),
    "Children above 3": (df["children"] > 3).sum(),
    "Babies above 3": (df["babies"] > 3).sum(),
}

report = pd.DataFrame(
    list(checks.items()),
    columns=["Check", "Count"]
)

print("=" * 50)
print("DATA QUALITY REPORT")
print("=" * 50)
print(report.to_string(index=False))

report_path = RESULTS_DIR / "data_quality_report.csv"
report.to_csv(report_path, index=False)

print("\nReport saved to:")
print(report_path)