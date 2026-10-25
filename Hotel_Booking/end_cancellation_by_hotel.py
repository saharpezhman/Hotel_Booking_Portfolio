from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    PROJECT_DIR
    / "results"
    / "hotel_bookings_analysis_ready.csv"
)
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("EDA: CANCELLATION RATE BY HOTEL TYPE")
print("=" * 60)

summary = df.groupby("hotel")["is_canceled"].agg(
    total_bookings="count",
    canceled_bookings="sum",
    cancellation_rate="mean"
)

summary["cancellation_rate"] = (
    summary["cancellation_rate"] * 100
).round(2)

print("\nCancellation summary:")
print(summary)


report_path = RESULTS_DIR / "cancellation_by_hotel.csv"
summary.to_csv(report_path)

plt.figure(figsize=(8, 5))

bars = plt.bar(
    summary.index,
    summary["cancellation_rate"]
)

plt.title("Cancellation Rate by Hotel Type")
plt.xlabel("Hotel Type")
plt.ylabel("Cancellation Rate (%)")
plt.ylim(0, 100)


for bar, value in zip(bars, summary["cancellation_rate"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

chart_path = RESULTS_DIR / "cancellation_by_hotel.png"
plt.savefig(chart_path, dpi=150)
plt.show()

print("\nReport saved to:", report_path)
print("Chart saved to:", chart_path)
print("\nAnalysis completed successfully.")