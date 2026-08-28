"""Tests for request correlation and safe backend logging."""

import logging
import uuid

from fastapi.testclient import TestClient

from app.api.routes import health as health_routes
from main import app


def test_request_logging_generates_id_and_logs_safe_metadata(monkeypatch, caplog):
    monkeypatch.setattr(health_routes, "check_database_connection", lambda: None)
    caplog.set_level(logging.INFO, logger="codefrog.api")

    response = TestClient(app).get("/api/v1/health?token=unsafe")

    request_id = response.headers["X-Request-ID"]
    uuid.UUID(hex=request_id)
    record = next(record for record in caplog.records if record.name == "codefrog.api")
    assert "method=GET" in record.message
    assert "path=/api/v1/health" in record.message
    assert "status_code=200" in record.message
    assert "duration_ms=" in record.message
    assert "unsafe" not in record.message


def test_request_logging_preserves_client_request_id(monkeypatch):
    monkeypatch.setattr(health_routes, "check_database_connection", lambda: None)

    response = TestClient(app).get(
        "/api/v1/health", headers={"X-Request-ID": "client-request-42"}
    )

    assert response.headers["X-Request-ID"] == "client-request-42"


def test_unexpected_error_logs_without_sensitive_exception_message(caplog):
    from backend.tests.test_error_handling import create_error_test_client

    caplog.set_level(logging.ERROR, logger="app.core.exception_handlers")
    response = create_error_test_client().get("/unexpected-error")

    assert response.status_code == 500
    assert any("Unexpected server error" in record.message for record in caplog.records)
    assert "unsafe" not in caplog.text
    assert "C:/private/path" not in caplog.text
