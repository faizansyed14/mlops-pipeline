from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
import logging
from app.model import TextClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlops")

app = FastAPI(title="DistilBERT Sentiment API")
model = TextClassifier()

# Instrument Prometheus metrics
Instrumentator().instrument(app).expose(app)

@app.post("/predict")
async def predict(payload: dict):
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Payload must include 'text'")
    logger.info(f"Received text: {text}")
    result = model.predict(text)
    logger.info(f"Prediction: {result}")
    return result

@app.get("/health")
async def health():
    return {"status": "ok"}
