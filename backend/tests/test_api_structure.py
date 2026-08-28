"""Tests for the versioned API router registration."""

from fastapi.testclient import TestClient

from app.api.routes import health as health_routes
from main import app


client = TestClient(app)


def test_versioned_health_endpoint_reports_database_available(monkeypatch):
    monkeypatch.setattr(health_routes, "check_database_connection", lambda: None)

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "database": "healthy"}


def test_versioned_resource_endpoints_are_registered():
    for path, resource in (
        ("/api/v1/users", "users"),
        ("/api/v1/repositories", "repositories"),
        ("/api/v1/tasks", "tasks"),
    ):
        response = client.get(path)

        assert response.status_code == 200
        assert response.json() == {"status": "available", "resource": resource}


def test_openapi_includes_versioned_routes():
    paths = client.get("/openapi.json").json()["paths"]

    assert {"/api/v1/health", "/api/v1/users", "/api/v1/repositories", "/api/v1/tasks"} <= set(paths)
