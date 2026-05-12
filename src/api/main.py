from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import time
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse

app = FastAPI(
    title="Student Performance API",
    description="Prediction du succes ou echec d'un etudiant",
    version="1.0.0"
)

# Charger le modele et le preprocesseur
model        = pickle.load(open("models/model.pkl", "rb"))
preprocessor = pickle.load(open("data/processed/preprocessor.pkl", "rb"))

# Metriques Prometheus
REQUEST_COUNT  = Counter("api_requests_total", "Total requetes")
LATENCY        = Histogram("api_latency_seconds", "Latence")
FAILURE_COUNT  = Counter("api_echec_total", "Predictions echec")

class StudentInput(BaseModel):
    hours_studied:    float = Field(..., ge=0,  le=24,  example=14)
    previous_scores:  float = Field(..., ge=0,  le=100, example=75)
    extracurricular:  str   = Field(...,                example="Yes")
    sleep_hours:      float = Field(..., ge=0,  le=24,  example=7)
    papers_practiced: int   = Field(..., ge=0,  le=10,  example=3)

@app.post("/predict")
async def predict(data: StudentInput):
    start = time.time()
    REQUEST_COUNT.inc()

    df    = pd.DataFrame([data.model_dump()])
    proc  = preprocessor.transform(df)
    proba = model.predict_proba(proc)[0]

    prob_success = round(float(proba[1]), 4)
    prediction   = "Succes" if prob_success >= 0.5 else "Echec"

    if prediction == "Echec":
        FAILURE_COUNT.inc()
        recommendation = "Intervention recommandee. Augmenter les heures d'etude."
    elif prob_success < 0.75:
        recommendation = "Profil moyen. Encourager la pratique de sujets."
    else:
        recommendation = "Bon profil. Maintenir les efforts."

    LATENCY.observe(time.time() - start)

    return {
        "prediction":          prediction,
        "probability_success": prob_success,
        "risk_score":          round(1 - prob_success, 4),
        "recommendation":      recommendation,
        "latency_ms":          round((time.time() - start) * 1000, 2)
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "RandomForest", "version": "1.0.0"}

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return generate_latest()