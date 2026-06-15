"""
api.py
FastAPI wrapper around RiskAnalyzer.
Run:
  uvicorn api:app --reload --host 0.0.0.0 --port 8000
Then POST:
  /analyze  {"text": "..."}
"""
from __future__ import annotations
from fastapi import FastAPI, Request, Body, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Any
import json
import os
from pydantic import BaseModel, Field
from risk_analyzer import (
    RiskAnalyzer,
    ZeroShotScamModel,
    FineTunedPhishingModel,
    normalize_message_text,
)

app = FastAPI(title="Elder Message Risk Analyzer", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

use_zero_shot = os.getenv("USE_ZERO_SHOT", "").strip().lower() in {"1", "true", "yes", "on"}
zero_shot = ZeroShotScamModel() if use_zero_shot else None

analyzer = RiskAnalyzer(
    zero_shot=zero_shot,
    finetuned_phishing=FineTunedPhishingModel(
        "./models/hebrew-phishing-model"
    ),
)

class AnalyzeRequest(BaseModel):
    text: str = Field("", description="Message text (can include newlines)")

@app.post("/analyze")
async def analyze(req: AnalyzeRequest = Body(...), request: Request = None):
    text = normalize_message_text(req.text)

    if not text and request is not None:
        raw = await request.body()
        try:
            data = json.loads(raw.decode("utf-8"))
            text = str(data.get("text", "")).strip()
        except json.JSONDecodeError:
            text = raw.decode("utf-8", errors="ignore").strip()

    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    r = analyzer.analyze(text)
    
    return {
        "risk_score": r.risk_score,
        "alert_level": r.alert_level,
        "top_risk": r.top_risk,
        "message_category": r.message_category,
        "risks": r.risks,
        "reasons": r.reasons,
        "recommendations": r.recommendations,
        "consequences": r.consequences,
        "urls": r.urls,
        "suspicious_urls": r.suspicious_urls,
    }

@app.get("/")
def root():
    return {"message": "api is working!"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)
