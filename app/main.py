import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
import pickle
import pandas as pd
from functools import lru_cache

app = FastAPI(
    title="Auto Subscription Prediction API",
    description="API for predicting auto subscription likelihood",
    version="1.0.0",
    docs_url="/",
)

model_path = "lgbm_model.pkl"


@lru_cache(maxsize=1)
def get_model():
    """Загружает модель с использованием кэширования."""
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"Модель успешно загружена из {model_path}")
        return model
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        raise


class SessionFeatures(BaseModel):
    # Numeric features
    client_id: str = Field(..., description="Client identifier")
    visit_number: int = Field(..., description="Number of visits by the user")
    day_of_week_num: int = Field(..., description="Day of week (0-6)")
    month: int = Field(..., description="Month (1-12)")
    day: int = Field(..., description="Day of month (1-31)")
    utm_source_encoded: int = Field(..., description="Encoded UTM source")
    utm_campaign_encoded: int = Field(..., description="Encoded UTM campaign")
    first_hit_number: int = Field(..., description="First hit number in the session")
    last_hit_number: int = Field(..., description="Last hit number in the session")
    total_hits: int = Field(..., description="Total number of hits in the session")
    total_time: int = Field(..., description="Total time of the session")
    main_referer: int = Field(..., description="Main referrer (encoded)")
    main_label: int = Field(..., description="Main event label (encoded)")
    geo_country: int = Field(
        ..., description="Country code (1 for Russia, 0 for others)"
    )

    # Categorical features
    utm_medium: str = Field(..., description="UTM medium")
    device_category: Literal["mobile", "desktop", "tablet"] = Field(
        ..., description="Device category"
    )
    device_brand: str = Field(..., description="Device brand")
    geo_city_grouped: str = Field(..., description="Grouped city name")
    entry_page: str = Field(..., description="Entry page URL")
    main_category_grouped: str = Field(..., description="Grouped main event category")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "client_id": "123.456",
                    "visit_number": 1,
                    "day_of_week_num": 3,
                    "month": 10,
                    "day": 15,
                    "utm_source_encoded": 42,
                    "utm_campaign_encoded": 120,
                    "first_hit_number": 1,
                    "last_hit_number": 10,
                    "total_hits": 10,
                    "total_time": 350,
                    "main_referer": 5,
                    "main_label": 20,
                    "geo_country": 1,
                    "utm_medium": "banner",
                    "device_category": "mobile",
                    "device_brand": "Apple",
                    "geo_city_grouped": "Moscow",
                    "entry_page": "podpiska.sberauto.com/",
                    "main_category_grouped": "sub_page_view",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    probability: float


class ErrorResponse(BaseModel):
    error: str

    model_config = {
        "json_schema_extra": {
            "examples": [{"error": "Ошибка при предсказании: неверный формат данных"}]
        }
    }


@app.post(
    "/infer",
    response_model=PredictionResponse,
    responses={
        200: {"model": PredictionResponse, "description": "Успешное предсказание"},
        400: {"model": ErrorResponse, "description": "Ошибка в данных запроса"},
        500: {"model": ErrorResponse, "description": "Внутренняя ошибка сервера"},
    },
)
async def infer(features: SessionFeatures):
    try:
        model = get_model()

        features_dict = features.model_dump()
        df = pd.DataFrame([features_dict])
        input_data = df.reindex(columns=model.feature_names_in_)

        probability = float(model.predict_proba(input_data)[0, 1])
        prediction = int(probability > 0.5)

        return PredictionResponse(prediction=prediction, probability=probability)
    except HTTPException:
        raise
    except Exception as e:
        error_message = f"Ошибка при предсказании: {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=True)
