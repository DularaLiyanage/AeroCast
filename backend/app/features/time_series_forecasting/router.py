from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

@router.get("/")
def test_component1():
    return {"message": "Component 1 working"}

class ForecastRequest(BaseModel):
    temperature: float
    humidity: float

@router.get("/ping")
def ping():
    return {"message": "Component 1 API working"}

@router.post("/predict")
def predict(data: ForecastRequest):
    # Fake prediction logic for demo
    result = data.temperature * 0.5 + data.humidity * 0.2
    return {
        "prediction": round(result, 2)
    }