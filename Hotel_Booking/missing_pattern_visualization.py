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

columns = ["company", "agent", "country", "children"]
results = []

for column in columns:
    for missing, label in [
        (False, "Available"),
        (True, "Missing"),
    ]:
        group = df[df[column].isna() == missing]

        if len(group) == 0:
            continue

        results.append({
            "Column": column,
            "Group": label,
            "Bookings": len(group),
            "Cancellation_Rate": (
                group["is_canceled"].mean() * 100
            ),
        })

report = pd.DataFrame(results)

print(report.round(2).to_string(index=False))

pivot = report.pivot(
    index="Column",
    columns="Group",
    values="Cancellation_Rate"
)

ax = pivot.plot(kind="bar", figsize=(10, 6))
ax.set_title("Cancellation Rate by Missing Data Status")
ax.set_xlabel("Column")
ax.set_ylabel("Cancellation Rate (%)")
ax.legend(title="Data Status")
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()

chart_path = RESULTS_DIR / "missing_pattern_cancellation.png"
plt.savefig(chart_path, dpi=150)
plt.show()

print("\nChart saved to:")
print(chart_path)