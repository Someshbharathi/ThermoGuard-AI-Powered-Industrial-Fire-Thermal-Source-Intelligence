from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RISK_FILE = PROJECT_ROOT / "results" / "risk" / "thermoguard_risk_scores.csv"
SHAP_FILE = PROJECT_ROOT / "results" / "shap" / "event_explanations.csv"
TEST_PREDICTIONS_FILE = PROJECT_ROOT / "models" / "xgboost" / "test_predictions.csv"

app = FastAPI(title="ThermoGuard Dashboard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str, default: float = 0) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def classification(value: str) -> str:
    return {
        "Industrial": "Industrial Fire",
        "Mining": "Mining Activity",
        "Natural": "Wildfire",
        "Other_Uncertain": "Other / Unknown",
    }.get(value, value or "Other / Unknown")


def risk(value: str) -> str:
    return (value or "LOW").upper()


def load_rows() -> list[dict[str, str]]:
    return read_csv(RISK_FILE)


def prediction_lookup() -> dict[str, dict[str, str]]:
    return {row.get("satellite_id", ""): row for row in read_csv(TEST_PREDICTIONS_FILE)}


def event_payload(row: dict[str, str], index: int) -> dict[str, Any]:
    prediction = prediction_lookup().get(row.get("satellite_id", ""), {})
    probability_keys = {"Natural": "prob_natural", "Industrial Fire": "prob_industrial", "Mining Activity": "prob_mining", "Other / Unknown": "prob_other_uncertain"}
    probabilities = {label: round(number(prediction, key) * 100, 2) for label, key in probability_keys.items()}
    proximity = number(row, "distance_to_industrial_area_km")
    return {
        "id": row.get("satellite_id") or f"TG-{index:05d}",
        "thermalLocationId": row.get("thermal_location_id", ""),
        "latitude": number(row, "latitude"),
        "longitude": number(row, "longitude"),
        "classification": classification(row.get("predicted_class", "")),
        "rawClassification": row.get("predicted_class", ""),
        "risk": risk(row.get("risk_level")),
        "score": round(number(row, "risk_score"), 2),
        "modelConfidence": round(number(row, "model_confidence"), 2),
        "frp": round(number(row, "frp"), 2),
        "persistence": f"{round(number(row, 'active_days')):.0f} days active",
        "activeDays": round(number(row, "active_days"), 2),
        "durationDays": round(number(row, "duration_days"), 2),
        "detectionCount": round(number(row, "detection_count"), 2),
        "location": f"{number(row, 'latitude'):.3f}, {number(row, 'longitude'):.3f}",
        "status": "Priority review" if risk(row.get("risk_level")) in {"HIGH", "CRITICAL"} else "Monitored",
        "explanation": row.get("risk_explanation", ""),
        "detectedAt": prediction.get("acq_datetime", ""),
        "satelliteDate": prediction.get("acq_datetime", "")[:10],
        "nearestFacility": "Industrial area" if proximity else "Not identified",
        "distanceKm": round(proximity, 2),
        "probabilities": probabilities,
        "riskFactors": {"FRP intensity": min(100, number(row, "thermal_score") * 10), "Persistence": min(100, number(row, "persistence_score")), "Industrial proximity": min(100, number(row, "industrial_score")), "Environmental risk": min(100, number(row, "mining_score"))},
    }


def overview_payload(rows: list[dict[str, str]]) -> dict[str, Any]:
    classifications = Counter(row.get("predicted_class", "Other_Uncertain") for row in rows)
    risks = Counter(risk(row.get("risk_level")) for row in rows)
    timeline_rows = read_csv(TEST_PREDICTIONS_FILE)
    monthly: dict[str, int] = defaultdict(int)
    for row in timeline_rows:
        date = row.get("acq_datetime", "")[:7]
        if date:
            monthly[date] += 1
    months = sorted(monthly)
    colors = {"Industrial": "#ef7d32", "Mining": "#d6a34d", "Natural": "#55b8a4", "Other_Uncertain": "#64748b"}
    kpis = [
        {"label": "Test-set thermal events", "value": f"{len(rows):,}", "detail": "Final fused risk output", "trend": "Verified output", "tone": "cyan"},
        {"label": "Industrial predictions", "value": f"{classifications['Industrial']:,}", "detail": "Fusion model classification", "trend": "Verified output", "tone": "amber"},
        {"label": "High / critical risk", "value": f"{risks['HIGH'] + risks['CRITICAL']:,}", "detail": "Risk-scored test set", "trend": "Priority review", "tone": "red"},
        {"label": "Persistent sources", "value": f"{sum(number(row, 'active_days') > 30 for row in rows):,}", "detail": "More than 30 active days", "trend": "Derived from output", "tone": "green"},
    ]
    distribution = [{"label": classification(key), "value": count, "color": colors.get(key, "#64748b")} for key, count in classifications.items()]
    risk_distribution = [{"label": key.title(), "value": count, "count": f"{count:,}", "color": {"Low": "#55b8a4", "Medium": "#d6a34d", "High": "#ef7d32", "Critical": "#df5d5d"}.get(key.title(), "#64748b")} for key, count in risks.items()]
    return {"source": str(RISK_FILE.relative_to(PROJECT_ROOT)), "rowCount": len(rows), "kpis": kpis, "timeline": {"labels": months, "values": [monthly[month] for month in months]}, "classificationDistribution": distribution, "riskDistribution": risk_distribution, "events": [event_payload(row, index) for index, row in enumerate(sorted(rows, key=lambda item: number(item, "risk_score"), reverse=True)[:8])], "mapPoints": [{"id": row.get("satellite_id", ""), "latitude": number(row, "latitude"), "longitude": number(row, "longitude"), "risk": risk(row.get("risk_level")), "classification": classification(row.get("predicted_class", "")), "activeDays": number(row, "active_days"), "frp": number(row, "frp")} for row in rows]}


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "dataAvailable": RISK_FILE.exists(), "source": str(RISK_FILE.relative_to(PROJECT_ROOT))}


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    return overview_payload(load_rows())


@app.get("/api/events")
def events(search: str = "", classification_filter: str = Query("", alias="classification"), risk_filter: str = Query("", alias="risk"), limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)) -> dict[str, Any]:
    rows = load_rows()
    records = [event_payload(row, index) for index, row in enumerate(rows)]
    search_value = search.strip().lower()
    if search_value:
        records = [item for item in records if search_value in " ".join(str(value) for value in item.values()).lower()]
    if classification_filter:
        records = [item for item in records if item["classification"] == classification_filter]
    if risk_filter:
        records = [item for item in records if item["risk"] == risk_filter.upper()]
    records.sort(key=lambda item: item["score"], reverse=True)
    return {"items": records[offset : offset + limit], "total": len(records), "source": str(RISK_FILE.relative_to(PROJECT_ROOT))}


@app.get("/api/events/{event_id}")
def event(event_id: str) -> dict[str, Any]:
    for index, row in enumerate(load_rows()):
        if row.get("satellite_id") == event_id:
            return event_payload(row, index)
    return {"detail": "Event not found"}


@app.get("/api/alerts")
def alerts() -> dict[str, Any]:
    rows = sorted(load_rows(), key=lambda item: number(item, "risk_score"), reverse=True)
    return {"items": [{"id": row.get("satellite_id", ""), "eventId": row.get("satellite_id", ""), "level": risk(row.get("risk_level")), "title": f"{classification(row.get('predicted_class', ''))} anomaly", "location": f"{number(row, 'latitude'):.3f}, {number(row, 'longitude'):.3f}", "event": event_payload(row, index)} for index, row in enumerate(rows[:8])], "source": str(RISK_FILE.relative_to(PROJECT_ROOT))}


@app.get("/api/insights")
def insights() -> dict[str, Any]:
    rows = read_csv(SHAP_FILE)
    items = []
    for row in rows[:50]:
        items.append({"id": row.get("satellite_id", ""), "classification": row.get("predicted_class", ""), "confidence": number(row, "prediction_confidence"), "positive": row.get("top_positive_features", ""), "negative": row.get("top_negative_features", "")})
    return {"items": items, "source": str(SHAP_FILE.relative_to(PROJECT_ROOT))}


@app.get("/api/shap/{event_id}")
def shap(event_id: str) -> dict[str, Any]:
    for row in read_csv(SHAP_FILE):
        if row.get("satellite_id") == event_id:
            return {"id": event_id, "classification": row.get("predicted_class", ""), "confidence": number(row, "prediction_confidence"), "positive": row.get("top_positive_features", ""), "negative": row.get("top_negative_features", ""), "source": str(SHAP_FILE.relative_to(PROJECT_ROOT))}
    return {"detail": "SHAP explanation not found"}
