"""
app.py

REST API serving the trained ETA model.

Framework choice: Flask + Pydantic, not FastAPI. The brief explicitly allows
either ("FastAPI/Flask for serving"); FastAPI could not be installed in the
sandbox this was developed in (no internet access to PyPI), while Flask and
Pydantic were already available, so this stays fully runnable and testable.
Pydantic still does the request schema validation FastAPI would normally
give you for free -- we just wire it in manually via a decorator.

Endpoints:
  GET  /health            liveness/readiness check
  POST /predict           predict trip_duration_min for one trip

Input validation & error handling:
  - Pydantic model enforces types, required fields, and value ranges
    (latitude/longitude bounds, non-negative temperature bounds, timestamp
    format). Validation failures return HTTP 422 with a field-level error
    list, not a raw stack trace.
  - Malformed / non-JSON bodies return HTTP 400.
  - Unexpected server errors return HTTP 500 with a generic message (no
    internal details leaked) and are logged server-side.

Every request/response is logged to logs/predictions.jsonl (timestamp,
input, prediction, latency_ms) -- this log is the direct input to the Week 4
monitoring & drift-simulation step.
"""
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
from flask import Flask, request, jsonify
from pydantic import BaseModel, Field, field_validator, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "features"))
from serving_features import to_model_input, VALID_WEATHER  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "best_model.joblib"
FEATURE_COLUMNS_PATH = BASE / "models" / "best_model_feature_columns.json"
LOG_PATH = BASE / "logs" / "predictions.jsonl"
LOG_PATH.parent.mkdir(exist_ok=True)

app = Flask(__name__)

_model = joblib.load(MODEL_PATH)
_feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text())
_model_version = MODEL_PATH.stat().st_mtime_ns  # simple content-change version marker


class PredictRequest(BaseModel):
    pickup_datetime: str = Field(..., description="ISO 8601 timestamp, e.g. 2025-03-15T18:30:00")
    pickup_lat: float = Field(..., ge=-90, le=90)
    pickup_lon: float = Field(..., ge=-180, le=180)
    dropoff_lat: float = Field(..., ge=-90, le=90)
    dropoff_lon: float = Field(..., ge=-180, le=180)
    weather: str = Field(default="unknown")
    temperature_c: float = Field(default=25.0, ge=-50, le=60)

    @field_validator("pickup_datetime")
    @classmethod
    def _valid_timestamp(cls, v):
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError("pickup_datetime must be a valid ISO 8601 timestamp")
        return v

    @field_validator("weather")
    @classmethod
    def _valid_weather(cls, v):
        v = (v or "unknown").lower()
        if v not in VALID_WEATHER:
            raise ValueError(f"weather must be one of {sorted(VALID_WEATHER)}")
        return v


def _log_prediction(request_id, payload, prediction_min, latency_ms, error=None):
    entry = {
        "request_id": request_id,
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "input": payload,
        "predicted_duration_min": prediction_min,
        "latency_ms": latency_ms,
        "model_version": _model_version,
        "error": error,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": _model is not None,
        "model_version": _model_version,
        "n_features": len(_feature_columns),
    })


@app.post("/predict")
def predict():
    start = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]

    raw_body = request.get_json(silent=True)
    if raw_body is None:
        return jsonify({"request_id": request_id, "error": "Request body must be valid JSON"}), 400

    try:
        payload = PredictRequest(**raw_body)
    except ValidationError as e:
        # e.errors() can embed non-JSON-serializable exception objects in "ctx"
        # for custom @field_validator errors; e.json() is pydantic's own
        # JSON-safe serialization, so round-trip through that instead.
        return jsonify({
            "request_id": request_id,
            "error": "Input validation failed",
            "details": json.loads(e.json(include_url=False)),
        }), 422

    try:
        X = to_model_input(payload.model_dump(), _feature_columns)
        prediction_min = float(_model.predict(X)[0])
        prediction_min = max(prediction_min, 0.5)  # ETA can't be non-positive
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: never leak internals to the client
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        _log_prediction(request_id, raw_body, None, latency_ms, error=str(exc))
        app.logger.exception("Prediction failed for request %s", request_id)
        return jsonify({"request_id": request_id, "error": "Internal error while predicting"}), 500

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    _log_prediction(request_id, raw_body, prediction_min, latency_ms)

    return jsonify({
        "request_id": request_id,
        "predicted_duration_min": round(prediction_min, 2),
        "model_version": _model_version,
        "latency_ms": latency_ms,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
