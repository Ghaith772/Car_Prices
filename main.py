import re
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, ConfigDict
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Car Price Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_artifacts" / "car_price_model.json"
ENCODERS_PATH = BASE_DIR / "model_artifacts" / "encoders.json"

try:
    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))
except Exception as e:
    raise RuntimeError(f"فشل تحميل الموديل من {MODEL_PATH}: {e}") from e

try:
    with open(ENCODERS_PATH, "r", encoding="utf-8") as f:
        artifacts = json.load(f)
except Exception as e:
    raise RuntimeError(f"فشل تحميل ملف encoders من {ENCODERS_PATH}: {e}") from e

ALIASES_PATH = BASE_DIR / "model_artifacts" / "aliases.json"
try:
    with open(ALIASES_PATH, "r", encoding="utf-8") as f:
        aliases = json.load(f)
except Exception as e:
    raise RuntimeError(f"فشل تحميل ملف aliases من {ALIASES_PATH}: {e}") from e

make_aliases  = aliases.get("make", {})
model_aliases = aliases.get("model", {})

make_map  = artifacts["make_map"]
model_map = artifacts["model_map"]
columns   = artifacts["columns"]

KM_TO_MILE = 0.621371

VALID_BODIES = {col.split("body_", 1)[1] for col in columns if col.startswith("body_")}
VALID_COLORS = {col.split("color_", 1)[1] for col in columns if col.startswith("color_")}


def normalize_text(value: str) -> str:
    v = re.sub(r"[^a-z0-9]", "", value.strip().lower())
    return v.capitalize()


class CarInput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    make: str
    model_name: str
    year: int
    body: str
    transmission: str
    odometer: float
    color: str
    condition: float

    @field_validator("make", "model_name", "body", "transmission", "color", mode="before")
    @classmethod
    def apply_normalize_text(cls, v):
        if not isinstance(v, str):
            raise ValueError("يجب أن تكون القيمة نصية")
        return normalize_text(v)

    @field_validator("year", mode="before")
    @classmethod
    def check_year_type(cls, v):
        if isinstance(v, bool):
            raise ValueError("سنة الصنع يجب أن تكون رقماً صحيحاً")
        if isinstance(v, float) and not v.is_integer():
            raise ValueError("سنة الصنع يجب أن تكون رقماً صحيحاً بدون كسور")
        try:
            return int(v)
        except (TypeError, ValueError):
            raise ValueError("سنة الصنع يجب أن تكون رقماً صحيحاً")

    @field_validator("condition", "odometer", mode="before")
    @classmethod
    def check_numeric_type(cls, v):
        if isinstance(v, bool):
            raise ValueError("يجب أن تكون القيمة رقماً")
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError("يجب أن تكون القيمة رقماً")

    @field_validator("make")
    @classmethod
    def validate_make(cls, v):

        if v not in make_map:
            v = make_aliases.get(v, v)
        if v not in make_map:
            raise ValueError(f"اسم الشركة المصنعة '{v}' غير معروف أو غير واقعي")
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model(cls, v):
        # Alias layer: e.g. "Cerato" -> "Forte" before the model_map lookup.
        if v not in model_map:
            v = model_aliases.get(v, v)
        if v not in model_map:
            raise ValueError(f"موديل السيارة '{v}' غير معروف أو غير واقعي")
        return v

    @field_validator("body")
    @classmethod
    def validate_body(cls, v):
        if v not in VALID_BODIES:
            raise ValueError(f"نوع الهيكل (body) '{v}' غير معروف أو غير واقعي")
        return v

    @field_validator("color")
    @classmethod
    def validate_color(cls, v):
        if v not in VALID_COLORS:
            raise ValueError(f"اللون '{v}' غير معروف أو غير واقعي")
        return v

    @field_validator("year")
    @classmethod
    def validate_year(cls, v):
        if not (1900 <= v <= 2026):
            raise ValueError("سنة الصنع يجب أن تكون بين 1900 و2026")
        return v

    @field_validator("transmission")
    @classmethod
    def validate_transmission(cls, v):
        if v not in ("Automatic", "Manual"):
            raise ValueError("ناقل الحركة يجب أن يكون Automatic أو Manual فقط")
        return v

    @field_validator("condition")
    @classmethod
    def validate_condition(cls, v):
        if not (1 <= v <= 5):
            raise ValueError("قيمة condition يجب أن تكون بين 1 و5")
        return v

    @field_validator("odometer")
    @classmethod
    def convert_odometer(cls, v):
        if v < 0:
            raise ValueError("قيمة odometer يجب ألا تكون سالبة")
        return round(v * KM_TO_MILE, 2)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    errors = [
        {"field": e["loc"][-1], "message": e["msg"].replace("Value error, ", "")}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"errors": errors})


def predict_price(make, model_name, year, body, transmission, odometer, color, condition):
    input_df = pd.DataFrame([{col: 0 for col in columns}])
    input_df["make"]      = make_map[make]
    input_df["model"]     = model_map[model_name]
    input_df["year"]      = year
    input_df["odometer"]  = odometer
    input_df["condition"] = condition

    for col, val in [("body", body), ("transmission", transmission), ("color", color)]:
        col_name = f"{col}_{val}"
        if col_name in input_df.columns:
            input_df[col_name] = 1

    if(year>2015):
        change=(year-2015)*0.06
        return model.predict(input_df)[0]*(1+change)
    else:
        return model.predict(input_df)[0]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(car: CarInput):

    try:
        price = predict_price(
            make         = car.make,
            model_name   = car.model_name,
            year         = car.year,
            body         = car.body,
            transmission = car.transmission,
            odometer     = car.odometer,
            color        = car.color,
            condition    = car.condition
        )
    except Exception:
        logger.exception("فشل حساب السعر المتوقع")
        raise HTTPException(status_code=500, detail="حدث خطأ أثناء حساب السعر المتوقع، حاول لاحقاً")

    return {"predicted_price": round(float(price), 2)}