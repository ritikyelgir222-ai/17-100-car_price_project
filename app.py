"""
Phase 10: MLOps & Deployment
--------------------------------
Run with:  uvicorn app:app --reload
"""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Used Audi Pricing Engine", version="1.0")

MODEL = joblib.load("outputs/car_price_model.joblib")
CONFIG = joblib.load("outputs/model_config.joblib")

CURRENT_YEAR = 2020  # matches clean_and_engineer.py's CURRENT_YEAR


class CarListing(BaseModel):
    model: str
    year: int
    transmission: str  # "Manual", "Automatic", or "Semi-Auto"
    mileage: int
    fuelType: str  # "Petrol", "Diesel", or "Hybrid"
    tax: int
    mpg: float
    engineSize: float


class PriceResponse(BaseModel):
    predicted_price_gbp: float
    top_factors: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/price", response_model=PriceResponse)
def price_car(listing: CarListing):
    row = listing.model_dump()
    row["age_years"] = CURRENT_YEAR - row["year"]
    row["mileage_per_year"] = row["mileage"] / max(row["age_years"], 1)

    X = pd.DataFrame([row])[CONFIG["cat_cols"] + CONFIG["num_cols"]]
    predicted_log_price = MODEL.predict(X)[0]
    predicted_price = float(np.expm1(predicted_log_price))

    prep = MODEL.named_steps["prep"]
    reg = MODEL.named_steps["reg"]
    ohe_names = prep.named_transformers_["cat"].get_feature_names_out(CONFIG["cat_cols"])
    all_names = list(ohe_names) + CONFIG["num_cols"]
    importances = pd.Series(reg.feature_importances_, index=all_names)
    top_factors = importances.sort_values(ascending=False).head(3).index.tolist()

    return PriceResponse(
        predicted_price_gbp=round(predicted_price, 2),
        top_factors=top_factors,
    )


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head><title>Used Audi Pricing Engine</title></head>
    <body style="font-family: sans-serif; max-width: 640px; margin: 40px auto;">
        <h2>Used Audi Pricing Engine</h2>
        <p>Full docs at <a href="/docs">/docs</a>.</p>
        <form id="priceForm">
            <label>Model:
                <select name="model">
                    <option>A1</option><option>A3</option><option>A4</option><option>A5</option>
                    <option>A6</option><option>Q3</option><option>Q5</option><option>Q7</option>
                    <option>TT</option><option>R8</option>
                </select>
            </label><br><br>
            <label>Year: <input name="year" type="number" value="2017"></label><br><br>
            <label>Transmission:
                <select name="transmission"><option>Manual</option><option>Automatic</option><option>Semi-Auto</option></select>
            </label><br><br>
            <label>Mileage: <input name="mileage" type="number" value="25000"></label><br><br>
            <label>Fuel type:
                <select name="fuelType"><option>Petrol</option><option>Diesel</option><option>Hybrid</option></select>
            </label><br><br>
            <label>Road tax (£): <input name="tax" type="number" value="145"></label><br><br>
            <label>MPG: <input name="mpg" type="number" step="0.1" value="55"></label><br><br>
            <label>Engine size (L): <input name="engineSize" type="number" step="0.1" value="1.4"></label><br><br>
            <button type="submit">Get price estimate</button>
        </form>
        <h3 id="result"></h3>
        <script>
        document.getElementById("priceForm").addEventListener("submit", async function(e) {
            e.preventDefault();
            const form = new FormData(e.target);
            const payload = {
                model: form.get("model"),
                year: parseInt(form.get("year")),
                transmission: form.get("transmission"),
                mileage: parseInt(form.get("mileage")),
                fuelType: form.get("fuelType"),
                tax: parseInt(form.get("tax")),
                mpg: parseFloat(form.get("mpg")),
                engineSize: parseFloat(form.get("engineSize"))
            };
            const res = await fetch("/price", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            document.getElementById("result").innerText =
                "Estimated price: £" + data.predicted_price_gbp.toLocaleString() +
                " | Top factors: " + data.top_factors.join(", ");
        });
        </script>
    </body>
    </html>
    """
