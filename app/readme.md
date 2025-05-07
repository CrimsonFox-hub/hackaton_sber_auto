# SberAuto API

Сервис предсказания вероятности подписки на автосервис.

## Запуск

```bash
# Локально
poetry install
uvicorn main:app --host 0.0.0.0 --port 8000

# Docker
docker build -t sberauto-api .
docker run -p 8000:8000 sberauto-api
```

## API
POST `/infer` - Предсказание вероятности подписки
