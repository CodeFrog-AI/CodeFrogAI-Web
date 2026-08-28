from fastapi.testclient import TestClient

from app.db.database import DatabaseConnectionError
from main import app


def test_health_reports_database_available(monkeypatch):
    monkeypatch.setattr("main.check_database_connection", lambda: None)

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "database": "healthy"}


def test_health_returns_safe_service_unavailable_response(monkeypatch):
    def unavailable() -> None:
        raise DatabaseConnectionError("password=unsafe")

    monkeypatch.setattr("main.check_database_connection", unavailable)

    response = TestClient(app).get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "SERVICE_UNAVAILABLE",
            "message": "Service unavailable",
            "details": None,
        }
    }
