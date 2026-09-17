# Car Price Prediction API

A FastAPI backend that predicts used-car selling prices from a trained XGBoost regression model. Built as part of my university graduation project (Software Engineering, Latakia University), as the ML/prediction service behind a car sales marketplace.

> This model was trained on a US-centric dataset and doesn't cover Chinese car brands, which are common in the Syrian market. In the full marketplace platform, if this model can't recognize the submitted car (e.g. an unrecognized make/model), the request is forwarded to a second model — [Car_Prices_2](https://github.com/Ghaith772/Car_Prices_2) — trained specifically on Chinese-brand-inclusive data. If neither model can recognize the car, the API rejects the request with an error instead of returning an unreliable prediction.

## Features

- **`POST /predict`** — accepts car details (make, model, year, body type, transmission, odometer, color, condition) and returns a predicted selling price.
- **`GET /health`** — simple health check endpoint.
- Full input validation with Pydantic: type checks, text normalization, and alias resolution for common make/model name variants (e.g. alternate spellings).
- Descriptive, localized error messages for invalid input (e.g. unknown make, out-of-range year).
- CORS configured for local frontend development (`localhost:3000`, `localhost:5173`).

## Tech Stack

- **API:** Python, FastAPI, Pydantic, Uvicorn
- **Modeling:** XGBoost (`XGBRegressor`), scikit-learn (encoding, train/test split, evaluation metrics)
- **Data processing:** Pandas, NumPy
- **Deployment:** Railway (via `Procfile`)

## Model & Data Pipeline (`train.py`)

The training script builds the model from raw sales data:

1. **Cleaning** — drops duplicate records, imputes missing `odometer`/`condition` values with the median , and fills missing `body` types by sampling from the observed body-type distribution.
2. **Category consolidation** — merges rare/overlapping body types (e.g. multiple truck cab variants) into consistent categories, then drops categories that make up less than 1% of the data.
3. **Outlier removal** — clips `sellingprice` at the 1st/99.99th percentiles and filters `odometer` using the IQR method.
4. **Encoding** — `LabelEncoder` for `make`/`model`, one-hot encoding for `body`/`transmission`/`color`.
5. **Training** — an `XGBRegressor` with **monotonic constraints**, so predictions stay logically consistent (price rises with `condition` and `year`, falls with `odometer`).
6. **Evaluation** — reports R², MAE, and RMSE on a held-out test set.
7. **Artifacts** — saves the trained model and its encoders/aliases to `model_artifacts/`, which `main.py` loads at startup to serve predictions.

## Data Source

Trained on the [Vehicle Sales Data](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data) dataset from Kaggle.

## Project Structure

```
.
├── main.py              # FastAPI app: /predict and /health endpoints
├── train.py              # Data cleaning, feature engineering, and model training
├── model_artifacts/      # Saved model + encoder/alias JSON files (used at inference time)
├── requirements.txt       # Python dependencies
└── Procfile               # Deployment entrypoint (Railway)
```

## Running Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`. Send a `POST /predict` request with a JSON body matching the `CarInput` schema:

```json
{
  "make": "Kia",
  "model_name": "Forte",
  "year": 2015,
  "body": "Sedan",
  "transmission": "Automatic",
  "odometer": 68000,
  "color": "Red",
  "condition": 3.5
}
```

## Retraining

The training dataset (`car_prices.csv`) is not included in this repository. To retrain the model, place your own dataset (with `make`, `model`, `year`, `body`, `transmission`, `odometer`, `color`, `condition`, and `sellingprice` columns) in the project root as `car_prices.csv` and run:

```bash
python train.py
```

This regenerates the artifacts in `model_artifacts/`.
