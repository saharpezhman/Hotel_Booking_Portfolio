from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

file_path = RESULTS_DIR / "cancellation_models_comparison.csv"

if not file_path.exists():
    raise FileNotFoundError(
        "فایل cancellation_models_comparison.csv "
        "در پوشه results پیدا نشد."
    )

df = pd.read_csv(file_path)

print("ستون‌های فایل:")
print(df.columns.tolist())
print("\nنتایج موجود:")
print(df.to_string(index=False))


possible_metrics = [
    "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"
]

column_map = {}
for col in df.columns:
    normalized = col.strip().lower().replace("_", "-")
    if normalized in ["accuracy"]:
        column_map["Accuracy"] = col
    elif normalized in ["precision"]:
        column_map["Precision"] = col
    elif normalized in ["recall"]:
        column_map["Recall"] = col
    elif normalized in ["f1", "f1-score", "f1 score"]:
        column_map["F1-score"] = col
    elif normalized in ["roc-auc", "roc auc", "roc_auc"]:
        column_map["ROC-AUC"] = col

if len(column_map) < 4:
    print(
        "\nساختار فایل با قالب مورد انتظار متفاوت است. "
        "ستون‌های بالا را بررسی کن و خروجی را بفرست."
    )
else:
    metrics = list(column_map.keys())
    x = np.arange(len(metrics))
    width = 0.35

    # تشخیص ستون نام مدل
    model_col = next(
        (c for c in df.columns
         if c.strip().lower() in ["model", "model_name", "name"]),
        None
    )

    if model_col is None:
        raise ValueError(
            "ستون نام مدل پیدا نشد؛ نام ستون‌های چاپ‌شده را بررسی کن."
        )

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (_, row) in enumerate(df.iterrows()):
        values = [
            float(row[column_map[m]]) if m in column_map else 0
            for m in metrics
        ]
        ax.bar(
            x + (i - (len(df) - 1) / 2) * width,
            values,
            width,
            label=str(row[model_col])
        )

    ax.set_title("Comparison of Hotel Cancellation Prediction Models")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    output_path = RESULTS_DIR / "final_model_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nنمودار ذخیره شد: {output_path}")