import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import json
import os
import re

sys.stdout.reconfigure(encoding='utf-8')


def normalize_text(value: str) -> str:
    if value is None:
        return value
    v = str(value).strip().lower()
    v = re.sub(r"[^a-z0-9]", "", v)
    return v.capitalize()


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
    data[col] = data[col].apply(lambda x: normalize_text(x) if pd.notnull(x) else x)
data["condition"] = np.where(data["condition"] > 9, data["condition"] / 10, data["condition"])


data["odometer"] = data["odometer"].fillna(data["odometer"].median())
data["condition"] = data["condition"].fillna(data["condition"].median())

data.dropna(subset=["sellingprice"], inplace=True)


data.dropna(subset=["odometer", "color"], inplace=True)
data.dropna(subset=["make", "model"], inplace=True)
data["transmission"] = data["transmission"].fillna("Automatic")
body_distribution = data["body"].value_counts(normalize=True)
missing_mask = data["body"].isnull()


p = body_distribution.values 
p = p / p.sum()  
np.random.seed(42)
data.loc[missing_mask, "body"] = np.random.choice(
    body_distribution.index,
    size=missing_mask.sum(),
    p=p  
)



print(data.isnull().sum())
print("Remaining rows:", len(data))
data.info()    
#data.describe() 

print(data["body"].value_counts())

body_mapping = {
    "Crewcab": "Truck",
    "Kingcab": "Truck",
    "Accesscab": "Truck",
    "Regularcab": "Truck",
    "Crewmaxcab": "Truck",
    "Doublecab": "Truck",
    "Supercrew": "Truck",
    "Extendedcab": "Truck",
    "Ramvan": "Van",
    "Minivan": "Van",
    "Gsedan": "Sedan",
    "Gcoupe": "Coupe",
}

data["body"] = data["body"].replace(body_mapping)
print(data["body"].value_counts())

# Drop  rare ones (less than 1% of data)
top_bodies = data["body"].value_counts()
top_bodies = top_bodies[top_bodies / len(data) > 0.01].index
data = data[data["body"].isin(top_bodies)]


low  = data["sellingprice"].quantile(0.01)
high = data["sellingprice"].quantile(0.9999)
data = data[(data["sellingprice"] >= low) & (data["sellingprice"] <= high)]
print(f"sellingprice range: ${low:,.0f}  ${high:,.0f}")

# Keep IQR for odometer only
Q1 = data["odometer"].quantile(0.25)
Q3 = data["odometer"].quantile(0.75)
IQR = Q3 - Q1
lower = max(0, Q1 - 1.5 * IQR)
upper = Q3 + 1.5 * IQR
data = data[(data["odometer"] >= lower) & (data["odometer"] <= upper)]
print(f"odometer range: {lower:,.0f} – {upper:,.0f}")

# One-hot encode low cardinality columns
#Encode BEFORE splitting
le_make  = LabelEncoder()
le_model = LabelEncoder()
data["make"]  = le_make.fit_transform(data["make"])   
data["model"] = le_model.fit_transform(data["model"]) 



make_map  = {label: idx for idx, label in enumerate(le_make.classes_)}
model_map = {label: idx for idx, label in enumerate(le_model.classes_)}
# One-hot encode low cardinality columns
data = pd.get_dummies(data, columns=["body", "transmission", "color"])

X = data.drop(columns=["sellingprice"])
y = data["sellingprice"]

#split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", len(X_train))
print("Test size: ", len(X_test))
print("Train size:", len(X_train))
print("Test size: ", len(X_test))
# Train

mono = {col: 0 for col in X.columns}
mono["condition"] = 1
mono["year"] = 1
mono["odometer"] = -1


model = XGBRegressor(
    n_estimators=200,     
    learning_rate=0.1,    
    max_depth=6,          
    random_state=42,
    n_jobs=-1,         # use all CPU cores
    monotone_constraints=mono
)

model.fit(X_train, y_train)
print("Model trained!")


# Predict
y_pred = model.predict(X_test)

# Evaluate
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"R²   : {r2:.4f}  → explains {r2*100:.1f}% of price variance")
print(f"MAE  : ${mae:,.0f}   → average error in dollars")
print(f"RMSE : ${rmse:,.0f}")


def predict_price(make, model_name, year, body, transmission, odometer, color, condition):
    input_df = pd.DataFrame([{col: 0 for col in X.columns}])
    make = normalize_text(make)
    model_name = normalize_text(model_name)
    if make not in make_map:
        print(f"Unknown make: {make}")
        input_df["make"] = np.nan
    else:
        input_df["make"] = make_map[make]
    if model_name not in model_map:
        print(f"Unknown model: {model_name}")
        input_df["model"] = np.nan
    else:
        input_df["model"] = model_map[model_name]
        
    input_df["year"]      = year
    input_df["odometer"]  = odometer
    input_df["condition"] = condition  

    for col, val in [("body", body), ("transmission", transmission), ("color", color)]:
        col_name = f"{col}_{normalize_text(val)}"
        if col_name in input_df.columns:
            input_df[col_name] = 1
    if input_df.isnull().any().any():
        print("Warning: Some input values are unknown. The prediction may be inaccurate.")

    return model.predict(input_df)[0]


os.makedirs("model_artifacts", exist_ok=True)
model.save_model("model_artifacts/car_price_model.json")
with open("model_artifacts/encoders.json", "w", encoding="utf-8") as f:
    json.dump({
        "make_map": make_map,
        "model_map": model_map,
        "columns": list(X.columns)
    }, f, ensure_ascii=False)


aliases = {
    "make": {
        # "Alias": "CanonicalMakeKey"
    },
    "model": {
        "Cerato": "Forte",
    },
}


with open("model_artifacts/aliases.json", "w", encoding="utf-8") as f:
    json.dump(aliases, f, ensure_ascii=False, indent=2)

print("Artifacts saved to model_artifacts/")
base = dict(
    make="Kia",             
    model_name="Forte",
    year=2015,
    body="Sedan",
    transmission="Automatic",
    odometer=68000,
    color="Red",
)

results = []
for c in np.arange(1.0, 5.01, 0.1):
    price = predict_price(**base, condition=round(c, 1))
    results.append((round(c, 1), price))
    print(f"{c:.1f}  ->  ${price:,.0f}")
for i in range(1, len(results)):
    c, p = results[i]
    diff = p - results[i-1][1]
    flag = "  <-- تراجع!" if diff < 0 else ""
    print(f"{c:.1f}: {diff:+,.0f}{flag}")

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"R²   : {r2:.4f}  → explains {r2*100:.1f}% of price variance")
print(f"MAE  : ${mae:,.0f}   → average error in dollars")
print(f"RMSE : ${rmse:,.0f}")






