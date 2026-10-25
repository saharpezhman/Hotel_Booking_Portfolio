from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# 1. Find the prepared dataset
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

possible_files = [
    BASE_DIR / "hotel_bookings_analysis_ready.csv",
    BASE_DIR / "results" / "hotel_bookings_analysis_ready.csv",
]

DATA_FILE = next(
    (path for path in possible_files if path.exists()),
    None
)

if DATA_FILE is None:
    raise FileNotFoundError(
        "Dataset not found. Check the location of "
        "hotel_bookings_analysis_ready.csv"
    )

df = pd.read_csv(DATA_FILE)

# --------------------------------------------------
# 2. Check required columns
# --------------------------------------------------

required_columns = ["hotel", "lead_time", "is_canceled"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df = df[required_columns].copy()

df["lead_time"] = pd.to_numeric(
    df["lead_time"], errors="coerce"
)

df["is_canceled"] = pd.to_numeric(
    df["is_canceled"], errors="coerce"
)

df = df.dropna(subset=required_columns)

df = df[
    df["is_canceled"].isin([0, 1])
    & (df["lead_time"] >= 0)
]

# --------------------------------------------------
# 3. Create lead-time groups
# --------------------------------------------------

bins = [-1, 7, 30, 90, 180, 365, np.inf]

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
    labels=labels,
    right=True
)

# --------------------------------------------------
# 4. Calculate cancellation rates
# --------------------------------------------------

summary = (
    df.groupby(
        ["hotel", "lead_time_group"],
        observed=False
    )
    .agg(
        total_bookings=("is_canceled", "size"),
        canceled_bookings=("is_canceled", "sum"),
        cancellation_rate=("is_canceled", "mean"),
    )
    .reset_index()
)

summary["cancellation_rate"] *= 100

print("\nCancellation analysis:")
print(summary.to_string(index=False))

# --------------------------------------------------
# 5. Prepare results folder
# --------------------------------------------------

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

summary.to_csv(
    RESULTS_DIR / "lead_time_cancellation_summary.csv",
    index=False
)

# --------------------------------------------------
# 6. Create comparison chart
# --------------------------------------------------

pivot = summary.pivot(
    index="lead_time_group",
    columns="hotel",
    values="cancellation_rate"
).reindex(labels)

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(labels))
bar_width = 0.36

for i, hotel_name in enumerate(
    ["City Hotel", "Resort Hotel"]
):
    if hotel_name not in pivot.columns:
        continue

    values = pivot[hotel_name].to_numpy(dtype=float)

    bars = ax.bar(
        x + (i - 0.5) * bar_width,
        values,
        bar_width,
        label=hotel_name
    )

    ax.bar_label(
        bars,
        fmt="%.1f%%",
        padding=3,
        fontsize=8
    )

ax.set_title(
    "Hotel Cancellation Rate by Lead Time",
    fontsize=15
)

ax.set_xlabel("Lead Time Before Arrival")
ax.set_ylabel("Cancellation Rate (%)")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15)

ax.set_ylim(
    0,
    min(100, max(60, np.nanmax(pivot.to_numpy()) + 12))
)

ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend()
fig.tight_layout()

chart_path = (
    RESULTS_DIR / "lead_time_cancellation_comparison.png"
)

fig.savefig(chart_path, dpi=300, bbox_inches="tight")

plt.show()

print(f"\nChart saved to: {chart_path}")
print(
    "Summary saved to: "
    f"{RESULTS_DIR / 'lead_time_cancellation_summary.csv'}"
)