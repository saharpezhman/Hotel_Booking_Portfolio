from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "hotel_bookings.csv"


df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("HOTEL BOOKING DATA ANALYSIS")
print("=" * 50)


print("\n1. Dataset Shape:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


print("\n2. First Five Records:")
print(df.head())

print("\n3. Column Data Types:")
print(df.dtypes)


print("\n4. Missing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0])


print("\n5. Duplicate Rows:")
print(df.duplicated().sum())

print("\n6. Numerical Statistics:")
print(df.describe().round(2))

print("\nAnalysis completed successfully.")
