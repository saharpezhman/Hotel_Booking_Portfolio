import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import os

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "hotel_booking_project_architecture.png"
)

# --------------------------------------------------
# 2. Define architecture components
# --------------------------------------------------

components = [
    (
        "Raw Dataset",
        "hotel_bookings.csv"
    ),
    (
        "Data Preparation",
        "Data cleaning and preparation"
    ),
    (
        "Prepared Dataset",
        "hotel_bookings_analysis_ready.csv"
    ),
    (
        "Exploratory Analysis",
        "Cancellation patterns and visualizations"
    ),
    (
        "Temporal Evaluation",
        "Estimated booking date and feature audit"
    ),
    (
        "Clustering",
        "Customer and booking group analysis"
    ),
    (
        "Prediction Models",
        "Model training, comparison and evaluation"
    ),
    (
        "Business Evaluation",
        "Thresholds, recall and follow-up capacity"
    ),
    (
        "Final Deliverables",
        "Results, visualizations, dashboard and GitHub documentation"
    ),
]

# --------------------------------------------------
# 3. Create the figure
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 15))

fig.patch.set_facecolor("#F5F7FA")
ax.set_facecolor("#F5F7FA")

ax.set_xlim(0, 13)
ax.set_ylim(0, 17)
ax.axis("off")

# --------------------------------------------------
# 4. Main title
# --------------------------------------------------

ax.text(
    6.5,
    16.25,
    "Hotel Booking Cancellation Prediction",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color="#203864"
)

ax.text(
    6.5,
    15.75,
    "Project Architecture Overview",
    ha="center",
    va="center",
    fontsize=13,
    color="#526780"
)

# --------------------------------------------------
# 5. Box layout settings
# --------------------------------------------------

box_x = 1.5
box_width = 10
box_height = 1.15

start_y = 14.8
vertical_gap = 1.55

# --------------------------------------------------
# 6. Draw numbered architecture components
# --------------------------------------------------

for i, (title, description) in enumerate(components):

    # Calculate the position of the current box
    y = start_y - i * vertical_gap - box_height

    # Draw the component box
    box = FancyBboxPatch(
        (box_x, y),
        box_width,
        box_height,
        boxstyle="round,pad=0.06,rounding_size=0.15",
        linewidth=1.5,
        edgecolor="#315A7D",
        facecolor="white"
    )

    ax.add_patch(box)

    # Draw a small numbered circle inside the box
    number_circle = Circle(
        (box_x + 0.48, y + box_height / 2),
        radius=0.23,
        facecolor="#315A7D",
        edgecolor="#315A7D",
        linewidth=1
    )

    ax.add_patch(number_circle)

    # Write the stage number
    ax.text(
        box_x + 0.48,
        y + box_height / 2,
        f"{i + 1:02d}",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="white"
    )

    # Write the component title
    ax.text(
        box_x + 0.95,
        y + 0.73,
        title,
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#203864"
    )

    # Write the component description
    ax.text(
        box_x + 0.95,
        y + 0.34,
        description,
        ha="left",
        va="center",
        fontsize=9,
        color="#526780",
        wrap=True
    )

# --------------------------------------------------
# 7. Architecture note
# --------------------------------------------------

ax.text(
    6.5,
    0.35,
    "Note: This diagram representsthe project workflow and components, "
    "not a deployed production system.",
    ha="center",
    va="center",
    fontsize=8,
    color="#6B7280",
    style="italic",
    wrap=True
)

# --------------------------------------------------
# 8. Save the image
# --------------------------------------------------

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)

plt.close(fig)

# --------------------------------------------------
# 9. Print the result
# --------------------------------------------------

print("Architecture diagram created successfully.")
print(f"Saved to: {OUTPUT_FILE}")
print("Number of numbered components:", len(components))