"""Prometheus metrics for observability.

A small middleware records request counts + latency for every endpoint; the predict endpoints
additionally record the predicted label. Exposed at ``GET /metrics`` in the standard exposition
format so Prometheus / Grafana can scrape it.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUEST_COUNT = Counter(
    "deepfake_requests_total",
    "Total HTTP requests",
    ["endpoint", "method", "status"],
)

REQUEST_LATENCY = Histogram(
    "deepfake_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
)

PREDICTIONS = Counter(
    "deepfake_predictions_total",
    "Predictions by endpoint and label",
    ["endpoint", "label"],
)


def render_latest() -> tuple[bytes, str]:
    """Return (payload, content_type) for the /metrics response."""
    return generate_latest(), CONTENT_TYPE_LATEST
