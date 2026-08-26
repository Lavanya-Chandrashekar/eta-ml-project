"""
Unit tests for the Flask serving API, using Flask's built-in test client
(no real network socket needed). Run with:
    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
import app as api_app  # noqa: E402

VALID_PAYLOAD = {
    "pickup_datetime": "2025-03-15T18:30:00",
    "pickup_lat": 12.95,
    "pickup_lon": 77.60,
    "dropoff_lat": 12.98,
    "dropoff_lon": 77.63,
    "weather": "rain",
    "temperature_c": 24.0,
}


class TestAPI(unittest.TestCase):
    def setUp(self):
        api_app.app.testing = True
        self.client = api_app.app.test_client()

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["status"], "ok")

    def test_valid_prediction_returns_200(self):
        resp = self.client.post("/predict", json=VALID_PAYLOAD)
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIn("predicted_duration_min", body)
        self.assertGreater(body["predicted_duration_min"], 0)
        self.assertIn("request_id", body)
        self.assertIn("latency_ms", body)

    def test_missing_required_field_returns_422(self):
        bad_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "pickup_lat"}
        resp = self.client.post("/predict", json=bad_payload)
        self.assertEqual(resp.status_code, 422)
        self.assertIn("details", resp.get_json())

    def test_invalid_weather_returns_422(self):
        bad_payload = {**VALID_PAYLOAD, "weather": "tornado"}
        resp = self.client.post("/predict", json=bad_payload)
        self.assertEqual(resp.status_code, 422)

    def test_out_of_range_latitude_returns_422(self):
        bad_payload = {**VALID_PAYLOAD, "pickup_lat": 999}
        resp = self.client.post("/predict", json=bad_payload)
        self.assertEqual(resp.status_code, 422)

    def test_invalid_timestamp_returns_422(self):
        bad_payload = {**VALID_PAYLOAD, "pickup_datetime": "not-a-timestamp"}
        resp = self.client.post("/predict", json=bad_payload)
        self.assertEqual(resp.status_code, 422)

    def test_malformed_json_returns_400(self):
        resp = self.client.post(
            "/predict", data="not json at all", content_type="application/json"
        )
        self.assertEqual(resp.status_code, 400)

    def test_missing_weather_defaults_to_unknown(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "weather"}
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 200)

    def test_response_is_deterministic_for_same_input(self):
        resp1 = self.client.post("/predict", json=VALID_PAYLOAD)
        resp2 = self.client.post("/predict", json=VALID_PAYLOAD)
        self.assertEqual(
            resp1.get_json()["predicted_duration_min"],
            resp2.get_json()["predicted_duration_min"],
        )


if __name__ == "__main__":
    unittest.main()
