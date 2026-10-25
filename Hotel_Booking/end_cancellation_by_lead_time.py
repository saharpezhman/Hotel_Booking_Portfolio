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


bins = [-1, 7, 30, 90, 180, 365, float("inf")]
labels = [
    "0-7 days",
    "8-30 days",
    "31-90 days",
    "91-180 days",
    "181-365 days",
    "Over 365 days",
]

df["lead_time_group"] = pd.cut(
    df["lead_time"],
    bins=bins,
    labels=labels
)


summary = df.groupby(
    "lead_time_group",
    observed=False
)["is_canceled"].agg(
    total_bookings="count",
    canceled_bookings="sum",
    cancellation_rate="mean"
)

summary["cancellation_rate"] = (
    summary["cancellation_rate"] * 100
).round(2)

print("\nCancellation rate by lead time:")
print(summary)

report_path = RESULTS_DIR / "cancellation_by_lead_time.csv"
summary.to_csv(report_path)


plt.figure(figsize=(10, 5))

bars = plt.bar(
    summary.index.astype(str),
    summary["cancellation_rate"]
)

plt.title("Cancellation Rate by Lead Time")
plt.xlabel("Days Between Booking and Arrival")
plt.ylabel("Cancellation Rate (%)")
plt.xticks(rotation=25)
plt.ylim(0, 100)

for bar, value in zip(bars, summary["cancellation_rate"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.1f}%",
        ha="center",
        fontsize=9
    )

plt.tight_layout()

chart_path = RESULTS_DIR / "cancellation_by_lead_time.png"
plt.savefig(chart_path, dpi=150)
plt.show()

print("\nReport saved to:", report_path)
print("Chart saved to:", chart_path)