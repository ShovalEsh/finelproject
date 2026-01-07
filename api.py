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
from pydantic import BaseModel, Field
from risk_analyzer import (
    RiskAnalyzer,
    ZeroShotScamModel,
    SklearnSpamModel,
    OpenAIRiskModel,
    FineTunedPhishingModel,
)

app = FastAPI(title="Elder Message Risk Analyzer", version="1.0")

zero_shot = ZeroShotScamModel()

sklearn_spam = SklearnSpamModel()

openai_model = None

analyzer = RiskAnalyzer(
    zero_shot=ZeroShotScamModel(),
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