import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, Literal
import random  # Added for demo prediction
import pickle
import pandas as pd
import os

app = FastAPI(
    title="Auto Subscription Prediction API",
    description="API for predicting auto subscription likelihood",
    version="1.0.0",
    docs_url="/",  # Setting the OpenAPI documentation to root path
)

# Загрузка модели при старте приложения
model_path = "lgbm_model.pkl"
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Модель успешно загружена из {model_path}")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    model = None


class SessionFeatures(BaseModel):
    # Numeric features
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


@app.post("/infer", response_model=PredictionResponse)
async def infer(features: SessionFeatures):
    if model is None:
        # Если модель не загружена, возвращаем фиктивное предсказание
        prediction = 1 if random.random() > 0.5 else 0
        probability = random.random()
    else:
        try:
            # Преобразуем входные данные в формат, подходящий для модели
            features_dict = features.model_dump()
            # Создаем DataFrame из входных данных
            df = pd.DataFrame([features_dict])

            # Предсказание вероятности
            probability = float(model.predict_proba(df)[0, 1])
            # Бинарный результат (1 - подпишется, 0 - не подпишется)
            prediction = 1 if probability > 0.5 else 0
        except Exception as e:
            print(f"Ошибка при предсказании: {e}")
            # В случае ошибки возвращаем фиктивное предсказание
            prediction = 0
            probability = 0.0

    return PredictionResponse(prediction=prediction, probability=probability)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
