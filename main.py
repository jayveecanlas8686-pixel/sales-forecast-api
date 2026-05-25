from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from prophet import Prophet
from lightgbm import LGBMRegressor

import pandas as pd
import numpy as np

from datetime import timedelta

app = FastAPI(title="Enterprise Sales Forecast API")


# =========================
# Request Schema
# =========================

class ForecastPayload(BaseModel):
    run_id: str
    forecast_date: str
    forecast_horizon_days: int
    sales_history: List[Dict[str, Any]]
    external_factors: Dict[str, Any] = {}


# =========================
# Root
# =========================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Enterprise Forecast API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# =========================
# Prepare Data
# =========================

def prepare_df(sales_history):

    df = pd.DataFrame(sales_history)

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="sales_history is empty"
        )

    required = [
        "date",
        "location",
        "product",
        "sales",
        "quantity"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing columns: {missing}"
        )

    df["date"] = pd.to_datetime(df["date"])

    df["sales"] = pd.to_numeric(
        df["sales"],
        errors="coerce"
    ).fillna(0)

    df["quantity"] = pd.to_numeric(
        df["quantity"],
        errors="coerce"
    ).fillna(0)

    return df


# =========================
# Prophet Forecast
# =========================

def prophet_forecast(group, horizon):

    if len(group) < 30:
        return None

    try:

        prophet_df = group[["date", "sales"]].rename(
            columns={
                "date": "ds",
                "sales": "y"
            }
        )

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )

        model.fit(prophet_df)

        future = model.make_future_dataframe(
            periods=horizon
        )

        forecast = model.predict(future)

        preds = (
            forecast.tail(horizon)["yhat"]
            .clip(lower=0)
            .tolist()
        )

        return preds

    except Exception:
        return None


# =========================
# LightGBM Forecast
# =========================

def lightgbm_forecast(group, horizon):

    if len(group) < 60:
        return None

    try:

        df = group.copy()

        df = df.sort_values("date")

        df["dayofweek"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day

        df["lag_1"] = df["sales"].shift(1)
        df["lag_7"] = df["sales"].shift(7)

        df["rolling_7"] = (
            df["sales"]
            .shift(1)
            .rolling(7)
            .mean()
        )

        df = df.fillna(0)

        features = [
            "dayofweek",
            "month",
            "day",
            "lag_1",
            "lag_7",
            "rolling_7"
        ]

        X = df[features]
        y = df["sales"]

        model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )

        model.fit(X, y)

        predictions = []

        history = df.copy()

        for _ in range(horizon):

            next_date = (
                history["date"].max()
                + timedelta(days=1)
            )

            next_row = {
                "date": next_date
            }

            temp = pd.concat([
                history,
                pd.DataFrame([next_row])
            ])

            temp["dayofweek"] = (
                temp["date"].dt.dayofweek
            )

            temp["month"] = (
                temp["date"].dt.month
            )

            temp["day"] = (
                temp["date"].dt.day
            )

            temp["lag_1"] = (
                temp["sales"]
                .shift(1)
            )

            temp["lag_7"] = (
                temp["sales"]
                .shift(7)
            )

            temp["rolling_7"] = (
                temp["sales"]
                .shift(1)
                .rolling(7)
                .mean()
            )

            temp = temp.fillna(0)

            latest = temp.tail(1)

            pred = model.predict(
                latest[features]
            )[0]

            pred = max(0, float(pred))

            predictions.append(pred)

            history = pd.concat([
                history,
                pd.DataFrame([{
                    "date": next_date,
                    "sales": pred
                }])
            ])

        return predictions

    except Exception:
        return None


# =========================
# Baseline Fallback
# =========================

def baseline_forecast(group, horizon):

    avg_sales = (
        group["sales"]
        .tail(30)
        .mean()
    )

    avg_sales = max(0, avg_sales)

    return [avg_sales] * horizon


# =========================
# Main Forecast Endpoint
# =========================

@app.post("/forecast")
def forecast(payload: ForecastPayload):

    df = prepare_df(
        payload.sales_history
    )

    horizon = payload.forecast_horizon_days

    forecast_rows = []

    grouped = df.groupby(
        ["location", "product"]
    )

    for (location, product), group in grouped:

        group = group.sort_values("date")

        if len(group) < 14:
            continue

        last_date = group["date"].max()

        # Try LightGBM first
        preds = lightgbm_forecast(
            group,
            horizon
        )

        model_used = "LightGBM"

        # Fallback to Prophet
        if preds is None:

            preds = prophet_forecast(
                group,
                horizon
            )

            model_used = "Prophet"

        # Final fallback
        if preds is None:

            preds = baseline_forecast(
                group,
                horizon
            )

            model_used = "Baseline"

        recent_qty = (
            group["quantity"]
            .tail(30)
            .mean()
        )

        recent_sales = (
            group["sales"]
            .tail(30)
            .mean()
        )

        qty_ratio = 0

        if recent_sales > 0:
            qty_ratio = (
                recent_qty /
                recent_sales
            )

        for i, pred_sales in enumerate(preds):

            target_date = (
                last_date +
                timedelta(days=i + 1)
            )

            lower = pred_sales * 0.85
            upper = pred_sales * 1.15

            confidence = 0.85

            if model_used == "Prophet":
                confidence = 0.75

            if model_used == "Baseline":
                confidence = 0.60

            forecast_rows.append({

                "target_date":
                    target_date.strftime("%Y-%m-%d"),

                "location":
                    location,

                "product":
                    product,

                "forecast_sales":
                    round(float(pred_sales), 2),

                "forecast_quantity":
                    round(float(
                        pred_sales * qty_ratio
                    ), 2),

                "lower_bound":
                    round(float(lower), 2),

                "upper_bound":
                    round(float(upper), 2),

                "confidence_score":
                    confidence,

                "model_used":
                    model_used,

                "drivers": [
                    "historical sales trend",
                    "seasonality",
                    "holiday effect",
                    "recent momentum"
                ],

                "inventory_recommendation":
                    (
                        "Increase safety stock"
                        if pred_sales >
                        recent_sales * 1.15
                        else "Maintain stock level"
                    )
            })

    return {

        "run_id":
            payload.run_id,

        "model_used":
            "LightGBM_Prophet_Fallback",

        "forecast_count":
            len(forecast_rows),

        "forecast":
            forecast_rows
    }
