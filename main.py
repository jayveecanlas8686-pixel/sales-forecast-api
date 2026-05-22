from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from datetime import timedelta

app = FastAPI()

# ==========================================
# REQUEST MODEL
# ==========================================

class ForecastRequest(BaseModel):
    sales_history: list
    forecast_horizon_days: int = 30


# ==========================================
# ROOT
# ==========================================

@app.get("/")
def root():
    return {
        "message": "Sales Forecast API Running"
    }


# ==========================================
# FORECAST ENDPOINT
# ==========================================

@app.post("/forecast")
def forecast(data: ForecastRequest):

    # ==========================================
    # LOAD DATA
    # ==========================================

    df = pd.DataFrame(data.sales_history)

    # ==========================================
    # CLEAN DATA
    # ==========================================

    df["Date"] = pd.to_datetime(df["Date"])

    df["Total Sales"] = (
        pd.to_numeric(df["Total Sales"], errors="coerce")
        .fillna(0)
    )

    df["Quantity"] = (
        pd.to_numeric(df["Quantity"], errors="coerce")
        .fillna(0)
    )

    # ==========================================
    # DAILY AGGREGATION
    # ==========================================

    daily = (
        df.groupby(
            ["Date", "ProductName", "LocationName"],
            as_index=False
        )
        .agg({
            "Total Sales": "sum",
            "Quantity": "sum",
            "Discount": "mean",
            "UnitPrice": "mean"
        })
    )

    # ==========================================
    # FEATURE ENGINEERING
    # ==========================================

    daily["day_of_week"] = daily["Date"].dt.dayofweek
    daily["month"] = daily["Date"].dt.month
    daily["day"] = daily["Date"].dt.day
    daily["week"] = daily["Date"].dt.isocalendar().week.astype(int)
    daily["quarter"] = daily["Date"].dt.quarter
    daily["is_weekend"] = (
        daily["day_of_week"].isin([5, 6]).astype(int)
    )

    # ==========================================
    # LAG FEATURES
    # ==========================================

    daily = daily.sort_values("Date")

    daily["lag_1"] = (
        daily.groupby(
            ["ProductName", "LocationName"]
        )["Total Sales"]
        .shift(1)
    )

    daily["lag_7"] = (
        daily.groupby(
            ["ProductName", "LocationName"]
        )["Total Sales"]
        .shift(7)
    )

    daily["lag_30"] = (
        daily.groupby(
            ["ProductName", "LocationName"]
        )["Total Sales"]
        .shift(30)
    )

    # ==========================================
    # ROLLING FEATURES
    # ==========================================

    daily["rolling_avg_7"] = (
        daily.groupby(
            ["ProductName", "LocationName"]
        )["Total Sales"]
        .transform(
            lambda x: x.rolling(7).mean()
        )
    )

    daily["rolling_avg_30"] = (
        daily.groupby(
            ["ProductName", "LocationName"]
        )["Total Sales"]
        .transform(
            lambda x: x.rolling(30).mean()
        )
    )

    daily["rolling_std_30"] = (
        daily.groupby(
            ["ProductName", "LocationName"]
        )["Total Sales"]
        .transform(
            lambda x: x.rolling(30).std()
        )
    )

    # ==========================================
    # REMOVE NULLS
    # ==========================================

    daily = daily.dropna()

    # ==========================================
    # FEATURES
    # ==========================================

    features = [
        "day_of_week",
        "month",
        "day",
        "week",
        "quarter",
        "is_weekend",
        "lag_1",
        "lag_7",
        "lag_30",
        "rolling_avg_7",
        "rolling_avg_30",
        "rolling_std_30",
        "Discount",
        "UnitPrice",
        "Quantity"
    ]

    target = "Total Sales"

    X = daily[features]
    y = daily[target]

    # ==========================================
    # TRAIN TEST SPLIT
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    # ==========================================
    # MODEL
    # ==========================================

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ==========================================
    # EVALUATION
    # ==========================================

    predictions = model.predict(X_test)

    mape = mean_absolute_percentage_error(
        y_test,
        predictions
    )

    confidence_score = max(
        0,
        round((1 - mape), 2)
    )

    # ==========================================
    # FUTURE FORECAST
    # ==========================================

    latest_rows = (
        daily.sort_values("Date")
        .groupby(
            ["ProductName", "LocationName"]
        )
        .tail(1)
    )

    forecast_results = []

    for _, row in latest_rows.iterrows():

        future_date = row["Date"] + timedelta(days=1)

        future_input = pd.DataFrame([{
            "day_of_week": future_date.dayofweek,
            "month": future_date.month,
            "day": future_date.day,
            "week": future_date.isocalendar()[1],
            "quarter": (future_date.month - 1) // 3 + 1,
            "is_weekend": 1 if future_date.dayofweek >= 5 else 0,
            "lag_1": row["Total Sales"],
            "lag_7": row["lag_7"],
            "lag_30": row["lag_30"],
            "rolling_avg_7": row["rolling_avg_7"],
            "rolling_avg_30": row["rolling_avg_30"],
            "rolling_std_30": row["rolling_std_30"],
            "Discount": row["Discount"],
            "UnitPrice": row["UnitPrice"],
            "Quantity": row["Quantity"]
        }])

        forecast_sales = float(
            model.predict(future_input)[0]
        )

        lower_bound = forecast_sales * 0.90
        upper_bound = forecast_sales * 1.10

        forecast_results.append({
            "target_date": str(future_date.date()),
            "ProductName": row["ProductName"],
            "LocationName": row["LocationName"],
            "forecast_sales": round(forecast_sales, 2),
            "forecast_quantity": round(
                row["Quantity"], 2
            ),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "confidence_score": confidence_score,
            "drivers": [
                "historical sales trend",
                "weekly sales pattern",
                "rolling sales average",
                "pricing behavior",
                "discount effect"
            ]
        })

    # ==========================================
    # RESPONSE
    # ==========================================

    return {
        "status": "success",
        "model_used": "LightGBM Enterprise Forecast",
        "accuracy": {
            "mape": round(float(mape), 4),
            "confidence_score": confidence_score
        },
        "forecast_count": len(forecast_results),
        "forecast": forecast_results
    }
