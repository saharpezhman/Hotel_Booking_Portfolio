from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "hotel_booking_project_workflow.png"

steps = [
    ("1. Raw Data", "Hotel Booking Demand CSV"),
    ("2. Data Cleaning", "Missing values, duplicates, quality checks"),
    ("3. Exploratory Analysis", "Cancellation patterns and visualizations"),
    ("4. Temporal Evaluation", "Estimated booking dates and feature audit"),
    ("5. Predictive Modeling", "Training, testing and model comparison"),
    ("6. Business Evaluation", "Thresholds, recall and follow-up capacity"),
    ("7. Final Product", "Interactive dashboard, report and GitHub"),
]

fig, ax = plt.subplots(figsize=(12, 15))
ax.set_xlim(0, 12)
ax.set_ylim(0, 16)
ax.axis("off")

box_width = 9
box_height = 1.25
x = (12 - box_width) / 2
start_y = 14.2
gap = 0.85

for i, (title, subtitle) in enumerate(steps):
    y = start_y - i * (box_height + gap)

    box = FancyBboxPatch(
        (x, y),
        box_width,
        box_height,
        boxstyle="round,pad=0.04,rounding_size=0.15",
        linewidth=1.5,
        edgecolor="#315A7D",
        facecolor="#EAF2F8" if i < 6 else "#DFF2E1",
    )
    ax.add_patch(box)

    ax.text(
        6,
        y + 0.82,
        title,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#17324D",
    )
    ax.text(
        6,
        y + 0.37,
        subtitle,
        ha="center",
        va="center",
        fontsize=9,
        color="#34495E",
    )

        

fig.suptitle(
    "Hotel Booking Cancellation Prediction\nProject Workflow",
    fontsize=17,
    fontweight="bold",
    y=0.97,
)

fig.text(
    0.5,
    0.025,
    "Note: Booking dates are estimated from arrival date and lead time.",
    ha="center",
    fontsize=9,
    color="#555555",
)

plt.tight_layout(rect=[0.03, 0.05, 0.97, 0.94])
fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Project workflow diagram created successfully.")
print(f"Saved to: {OUTPUT_FILE}")