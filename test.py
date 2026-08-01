import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')
data = pd.read_csv("car_prices.csv")
print(data.shape)
data.info()    
print(data.describe())
cols_to_keep = ["make", "model", "year", "body", "transmission", "odometer", "color", "condition", "sellingprice"]
data = data[cols_to_keep]
print(data.shape)
print(data.duplicated().sum())

# Drop duplicates 
data.drop_duplicates(inplace=True)
print("Missing Values Count")
print(data.isnull().sum())

print("\n Missing Values %")
print(data.isnull().sum() / len(data) * 100)

print("\n Are there nulls?")
print(data.isnull().any())

for col in ["make", "model", "body", "transmission", "color"]:
    data[col] = data[col].str.strip().str.lower().str.title()
for col in ["make", "model", "body", "transmission", "color"]:
    print(f"\n{col}:")
    print(sorted(data[col].unique()))
print(data[data['transmission'] == 'Sedan'])


# data = data[["condition", "sellingprice", "year", "odometer"]].copy()
# raw = pd.read_csv("car_prices.csv")[["condition", "sellingprice"]].dropna()
# coarse = raw[raw["condition"] <= 9]
# fine = raw[raw["condition"] > 9]

# print(coarse.groupby("condition")["sellingprice"].mean())
# print()
# print(fine.groupby("condition")["sellingprice"].mean())

# # نفصل المجموعتين الأصليتين قبل أي تطبيع
# data["scale_group"] = np.where(data["condition"] > 9, "fine_scale (11-49)", "coarse_scale (1-5)")

# print(data.groupby("scale_group")["sellingprice"].agg(["mean", "median", "count"]))
# print()
# print(data.groupby("scale_group")[["year", "odometer"]].mean())