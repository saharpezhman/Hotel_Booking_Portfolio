import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

file_path = RESULTS_DIR / "hotel_bookings_clustered.csv"

if not file_path.exists():
    file_path = BASE_DIR / "hotel_bookings_clustered.csv"

if not file_path.exists():
    raise FileNotFoundError(
        "فایل hotel_bookings_clustered.csv پیدا نشد. "
        "مطمئن شو اسکریپت خوشه‌بندی را اجرا کرده‌ای."
    )


df = pd.read_csv(file_path)


required_columns = ["cluster", "is_canceled"]
missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"ستون‌های مورد نیاز پیدا نشدند: {missing_columns}"
    )

df = df.dropna(subset=required_columns).copy()

df["is_canceled"] = pd.to_numeric(
    df["is_canceled"], errors="coerce"
)
df = df.dropna(subset=["is_canceled"])
df = df[df["is_canceled"].isin([0, 1])]


contingency_table = pd.crosstab(
    df["cluster"], df["is_canceled"]
)

contingency_table = contingency_table.reindex(
    columns=[0, 1], fill_value=0
)

print("=" * 60)
print("آزمون کای دو برای بررسی ارتباط خوشه و لغو رزرو")
print("=" * 60)

print("\nجدول فراوانی:")
print(contingency_table.rename(
    columns={0: "Not Canceled", 1: "Canceled"}
))


cluster_rates = df.groupby("cluster").agg(
    total_bookings=("is_canceled", "size"),
    canceled_bookings=("is_canceled", "sum")
)

cluster_rates["cancellation_rate_percent"] = (
    cluster_rates["canceled_bookings"]
    / cluster_rates["total_bookings"] * 100
)

print("\nآمار هر خوشه:")
print(cluster_rates.round(2))


chi2, p_value, degrees_of_freedom, expected = (
    chi2_contingency(contingency_table)
)

n = contingency_table.to_numpy().sum()
min_dimension = min(contingency_table.shape)

if n > 0 and min_dimension > 1:
    cramers_v = np.sqrt(
        chi2 / (n * (min_dimension - 1))
    )
else:
    cramers_v = np.nan

print("\nنتیجه آزمون کای دو:")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"Degrees of freedom: {degrees_of_freedom}")
print(f"P-value: {p_value:.6g}")
print(f"Cramer's V: {cramers_v:.4f}")

alpha = 0.05

if p_value < alpha:
    print(
        "\nنتیجه: تفاوت نرخ لغو بین خوشه‌ها "
        "از نظر آماری معنادار است (p < 0.05)."
    )
else:
    print(
        "\nنتیجه: شواهد کافی برای معنادار دانستن "
        "تفاوت نرخ لغو بین خوشه‌ها وجود ندارد."
    )


RESULTS_DIR.mkdir(parents=True, exist_ok=True)

cluster_rates.to_csv(
    RESULTS_DIR / "cluster_cancellation_rates.csv",
    encoding="utf-8-sig"
)

summary = pd.DataFrame([{
    "chi_square": chi2,
    "degrees_of_freedom": degrees_of_freedom,
    "p_value": p_value,
    "cramers_v": cramers_v,
    "significant_at_0.05": p_value < alpha
}])

summary.to_csv(
    RESULTS_DIR / "cluster_cancellation_chi_square.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nفایل‌های نتایج در پوشه results ذخیره شدند.")