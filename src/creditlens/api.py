# run: uvicorn creditlens.api:app --reload
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import policy
from .data import FEATURES, PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "scorecard.joblib"
STATIC = Path(__file__).parent / "static" / "index.html"

app = FastAPI(title="CreditLens", version="1.0")


class Applicant(BaseModel):
    revolving_utilization: float = Field(..., ge=0)  # balance / limit, 0.35 = 35%
    age: int = Field(..., ge=18, le=110)
    past_due_30_59: int = Field(0, ge=0)
    past_due_60_89: int = Field(0, ge=0)
    past_due_90_plus: int = Field(0, ge=0)
    debt_ratio: float = Field(..., ge=0)
    monthly_income: float | None = Field(None, ge=0)  # can be missing
    open_credit_lines: int = Field(..., ge=0)
    real_estate_loans: int = Field(0, ge=0)
    dependents: int | None = Field(None, ge=0)

    model_config = {"json_schema_extra": {"example": {
        "revolving_utilization": 0.85, "age": 29, "past_due_30_59": 2, "past_due_60_89": 0,
        "past_due_90_plus": 0, "debt_ratio": 0.45, "monthly_income": 4200, "open_credit_lines": 7,
        "real_estate_loans": 0, "dependents": 1}}}


class Reason(BaseModel):
    feature: str
    points_lost: int
    reason: str


class ScoreResponse(BaseModel):
    probability_of_default: float
    score: int
    grade: str
    decision: str
    reasons: list[Reason]


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise HTTPException(503, "model not trained, run python -m creditlens.train first")
    return joblib.load(MODEL_PATH)


def score_frame(card, df):
    pds = card.predict_pd(df)
    scores = card.score(df)
    reasons = card.reason_codes(df)
    out = []
    for p, s, r in zip(pds, scores, reasons):
        out.append({
            "probability_of_default": round(float(p), 4),
            "score": int(round(float(s))),
            "grade": policy.grade(float(p)),
            "decision": policy.decide(float(p)),
            "reasons": r,
        })
    return out


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return STATIC.read_text(encoding="utf-8")


@app.get("/health")
def health():
    card = get_model()
    return {"status": "ok", "model": "woe_logistic_scorecard", "features": card.features}


@app.post("/score", response_model=ScoreResponse)
def score(applicant: Applicant):
    card = get_model()
    df = pd.DataFrame([applicant.model_dump()])[FEATURES].astype(float)
    return score_frame(card, df)[0]


@app.get("/scorecard")
def scorecard():
    card = get_model()
    t = card.points_table().drop(columns="points_exact")
    return {"pdo": card.pdo, "base_score": card.base_score, "base_odds": card.base_odds,
            "rows": t.to_dict(orient="records")}
