from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
INPUT_FILE = RESULTS_DIR / "feature_group_ablation.csv"

RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)

model_order = [
    "All 19 features",
    "Without customer history",
    "Without ADR and agent",
    "Without all unverified features",
]

df["model"] = pd.Categorical(
    df["model"],
    categories=model_order,
    ordered=True
)
df = df.sort_values("model")

labels = [
    "All 19 features",
    "Without customer\nhistory",
    "Without ADR and\nagent",
    "Conservative 8\nfeatures",
]

metrics = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
}

selected_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

x = np.arange(len(df))
width = 0.15

fig, ax = plt.subplots(figsize=(15, 7))

for i, metric in enumerate(selected_metrics):
    values = df[metric].astype(float).to_numpy()
    ax.bar(
        x + (i - 2) * width,
        values,
        width,
        label=metrics[metric]
    )

ax.set_title(
    "Hotel Cancellation Prediction - Feature Group Comparison",
    fontsize=14
)
ax.set_ylabel("Score")
ax.set_xlabel("Feature set")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)
ax.legend(ncol=3)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "feature_group_metrics_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 6))

values = df["roc_auc"].astype(float).to_numpy()
bars = ax.bar(labels, values)

ax.set_title("ROC-AUC Comparison Across Feature Sets", fontsize=14)
ax.set_ylabel("ROC-AUC")
ax.set_ylim(0, 1)
ax.grid(axis="y", alpha=0.25)

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "feature_group_roc_auc_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("Charts created successfully.")
print(f"Saved: {RESULTS_DIR / 'feature_group_metrics_comparison.png'}")
print(f"Saved: {RESULTS_DIR / 'feature_group_roc_auc_comparison.png'}")