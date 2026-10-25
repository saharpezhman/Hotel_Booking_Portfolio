from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    PROJECT_DIR
    / "results"
    / "hotel_bookings_analysis_ready.csv"
)

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("MISSING DATA PATTERN ANALYSIS")
print("=" * 60)

columns = ["company", "agent", "country", "children"]

for column in columns:
    print(f"\n--- {column} ---")

    df["missing_group"] = df[column].isna()

    summary = df.groupby("missing_group")["is_canceled"].agg(
        bookings="count",
        cancellation_rate="mean"
    )

    summary["cancellation_rate"] = (
        summary["cancellation_rate"] * 100
    ).round(2)

    summary = summary.rename(
        index={False: "Available", True: "Missing"}
    )

    print(summary)

df.drop(columns=["missing_group"], inplace=True)

print("\nAnalysis completed successfully.")