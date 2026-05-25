from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import lightgbm as lgb
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

app = FastAPI()

class ForecastRequest(BaseModel):
    sales_history: list

@app.post("/forecast")
def forecast(data: ForecastRequest):

    df = pd.DataFrame(data.sales_history)

    # ---------------------------
    # CLEAN DATA
    # ---------------------------

    df["Date"] = pd.to_datetime(df["Date"])
    df["Total Sales"] = pd.to_numeric(df["Total Sales"])

    # ---------------------------
    # FEATURE ENGINEERING
    # ---------------------------

    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day

    df["lag_7"] = df["Total Sales"].shift(7)
    df["lag_30"] = df["Total Sales"].shift(30)

    df["rolling_avg_7"] = df["Total Sales"].rolling(7).mean()
    df["rolling_avg_30"] = df["Total Sales"].rolling(30).mean()

    df = df.dropna()

    # ---------------------------
    # TRAIN MODEL
    # ---------------------------

    features = [
        "day_of_week",
        "month",
        "day",
        "lag_7",
        "lag_30",
        "rolling_avg_7",
        "rolling_avg_30"
    ]

    X = df[features]
    y = df["Total Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    model = lgb.LGBMRegressor()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, predictions)

    # ---------------------------
    # FUTURE FORECAST
    # ---------------------------

    latest = df.iloc[-1]

    future_data = pd.DataFrame([{
        "day_of_week": latest["day_of_week"],
        "month": latest["month"],
        "day": latest["day"],
        "lag_7": latest["lag_7"],
        "lag_30": latest["lag_30"],
        "rolling_avg_7": latest["rolling_avg_7"],
        "rolling_avg_30": latest["rolling_avg_30"]
    }])

    future_forecast = model.predict(future_data)[0]

    return {
        "model_used": "LightGBM",
        "accuracy": {
            "mape": round(float(mape), 4)
        },
        "forecast": {
            "forecast_sales": round(float(future_forecast), 2),
            "confidence_score": round(float(1 - mape), 2)
        }
    }
