from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from datetime import timedelta

app = FastAPI()

class ForecastRequest(BaseModel):
    sales_history: list
    forecast_horizon_days: int = 30

@app.get("/")
def root():
    return {"message": "Sales Forecast API Running"}

@app.post("/forecast")
def forecast(data: ForecastRequest):
    df = pd.DataFrame(data.sales_history)

    if df.empty:
        return {"status": "error", "message": "No sales history received"}

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Total Sales"] = pd.to_numeric(df["Total Sales"], errors="coerce").fillna(0)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["UnitPrice"] = pd.to_numeric(df.get("UnitPrice", 0), errors="coerce").fillna(0)
    df["Discount"] = pd.to_numeric(df.get("Discount", 0), errors="coerce").fillna(0)

    df = df.dropna(subset=["Date"])

    daily = (
        df.groupby(["Date", "ProductName", "LocationName"], as_index=False)
        .agg({
            "Total Sales": "sum",
            "Quantity": "sum",
            "UnitPrice": "mean",
            "Discount": "mean"
        })
    )

    forecast_results = []

    for (product, location), group in daily.groupby(["ProductName", "LocationName"]):
        group = group.sort_values("Date")

        last_date = group["Date"].max()
        avg_7 = group["Total Sales"].tail(7).mean()
        avg_30 = group["Total Sales"].tail(30).mean()
        avg_qty_30 = group["Quantity"].tail(30).mean()

        if pd.isna(avg_7):
            avg_7 = 0
        if pd.isna(avg_30):
            avg_30 = avg_7
        if pd.isna(avg_qty_30):
            avg_qty_30 = 0

        base_forecast = (avg_7 * 0.60) + (avg_30 * 0.40)

        for i in range(1, data.forecast_horizon_days + 1):
            target_date = last_date + timedelta(days=i)

            weekend_factor = 0.95 if target_date.weekday() >= 5 else 1.00
            payday_factor = 1.08 if target_date.day in [15, 30, 31] else 1.00

            forecast_sales = base_forecast * weekend_factor * payday_factor

            forecast_results.append({
                "target_date": str(target_date.date()),
                "ProductName": product,
                "LocationName": location,
                "forecast_sales": round(float(forecast_sales), 2),
                "forecast_quantity": round(float(avg_qty_30), 2),
                "lower_bound": round(float(forecast_sales * 0.85), 2),
                "upper_bound": round(float(forecast_sales * 1.15), 2),
                "confidence_score": 0.75,
                "drivers": [
                    "7-day moving average",
                    "30-day moving average",
                    "payday adjustment",
                    "weekend adjustment"
                ]
            })

    return {
        "status": "success",
        "model_used": "Fast Moving Average Forecast",
        "forecast_count": len(forecast_results),
        "forecast": forecast_results[:1000]
    }
