"""
api.py
FastAPI wrapper around RiskAnalyzer.
Run:
  uvicorn api:app --reload --host 0.0.0.0 --port 8000
Then POST:
  /analyze  {"text": "..."}
"""
from __future__ import annotations

from fastapi import FastAPI
import os
from pydantic import BaseModel, Field
from risk_analyzer import (
    RiskAnalyzer,
    ZeroShotScamModel,
    SklearnSpamModel,
    OpenAIRiskModel,
    FineTunedPhishingModel,
)

app = FastAPI(title="Elder Message Risk Analyzer", version="1.0")

use_zero_shot = os.getenv("USE_ZERO_SHOT", "").strip().lower() in {"1", "true", "yes", "on"}
zero_shot = ZeroShotScamModel() if use_zero_shot else None

sklearn_spam = SklearnSpamModel()

openai_model = None

analyzer = RiskAnalyzer(
    zero_shot=zero_shot,
    sklearn_spam=None,
    openai_model=None,
    finetuned_phishing=FineTunedPhishingModel(
        "./models/hebrew-phishing-model"
    ),
)

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    r = analyzer.analyze(req.text)
    return {
        "risk_score": r.risk_score,
        "top_risk": r.top_risk,
        "risks": r.risks,
        "reasons": r.reasons,
        "consequences": r.consequences,
        "urls": r.urls,
    }
@app.get("/")
def root():
    return {"message": "api is working!"}
