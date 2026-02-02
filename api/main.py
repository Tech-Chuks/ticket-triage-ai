import json
import os
import re
from typing import Any, Dict

import numpy as np
import httpx
from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

TRITON_HTTP = os.environ.get("TRITON_HTTP", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "intent_router")

LABELS_PATH = os.environ.get("LABELS_PATH", "model-repo/intent_router/labels.json")
VECTORIZER_PATH = os.environ.get("VECTORIZER_PATH", "api/vectorizer.joblib")

LABELS = json.load(open(LABELS_PATH, "r"))["labels"]
VECTORIZER = load(VECTORIZER_PATH)
N_FEATURES = len(VECTORIZER.get_feature_names_out())

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
CARD_RE  = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

def detect_pii(text: str) -> bool:
    return bool(EMAIL_RE.search(text) or PHONE_RE.search(text) or CARD_RE.search(text))

def priority_rules(intent: str, text: str) -> str:
    t = text.lower()
    urgent_words = ["urgent", "asap", "immediately", "now", "fraud", "stolen", "chargeback", "lawsuit"]
    if any(w in t for w in urgent_words):
        return "high"
    if intent in ["billing", "login"] and any(w in t for w in ["charged", "refund", "locked", "can't log in", "cannot log in"]):
        return "high"
    if intent == "shipping" and any(w in t for w in ["delivered but", "late", "missing", "never arrived"]):
        return "medium"
    return "low"

REQS = Counter("api_requests_total", "Total requests", ["endpoint"])
ERRS = Counter("api_errors_total", "Total errors", ["endpoint"])
LAT  = Histogram("api_request_latency_seconds", "Latency", ["endpoint"])

app = FastAPI(title="Ticket Triage + Auto-Routing API")

class TriageRequest(BaseModel):
    ticket_id: str
    message: str

def vectorize(text: str) -> np.ndarray:
    X = VECTORIZER.transform([text])
    return X.toarray().astype(np.float32)

async def call_triton(vec: np.ndarray) -> Dict[str, Any]:
    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, N_FEATURES],
                "datatype": "FP32",
                "data": vec.reshape(-1).tolist()
            }
        ],
        "outputs": [
            {"name": "probabilities"},
            {"name": "label"}
        ]
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(f"{TRITON_HTTP}/v2/models/{MODEL_NAME}/infer", json=payload)
        r.raise_for_status()
        return r.json()

def pick_intent(triton_json: Dict[str, Any]) -> str:
    probs = None
    for out in triton_json.get("outputs", []):
        if out.get("name") == "probabilities":
            probs = out.get("data", [])
            shape = out.get("shape", [])
            if shape and len(shape) == 2 and shape[0] == 1:
                probs = probs[:shape[1]]
            break

    if not probs:
        return "other"

    best_idx = int(np.argmax(np.array(probs, dtype=np.float32)))
    if 0 <= best_idx < len(LABELS):
        return LABELS[best_idx]
    return "other"

@app.get("/health")
def health():
    return {"status": "ok", "features": N_FEATURES, "labels": LABELS}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/triage")
async def triage(req: TriageRequest):
    endpoint = "/triage"
    REQS.labels(endpoint=endpoint).inc()

    with LAT.labels(endpoint=endpoint).time():
        try:
            pii = detect_pii(req.message)
            vec = vectorize(req.message)
            triton_json = await call_triton(vec)
            intent = pick_intent(triton_json)
            priority = priority_rules(intent, req.message)

            return {
                "ticket_id": req.ticket_id,
                "intent": intent,
                "priority": priority,
                "contains_pii": pii,
                "recommended_queue": f"{intent}-queue",
                "notes": "intent from Triton probabilities, priority from rules, pii from regex"
            }
        except Exception as e:
            ERRS.labels(endpoint=endpoint).inc()
            return {"ticket_id": req.ticket_id, "error": str(e)}
