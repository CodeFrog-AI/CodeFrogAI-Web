"""Tests for centralized and safe API error responses."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.core.exception_handlers import register_exception_handlers
from app.core.exceptions import (
    BadRequestError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    UnauthorizedError,
)


def create_error_test_client() -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/bad-request")
    def bad_request() -> None:
        raise BadRequestError("Invalid repository selection")

    @app.get("/unauthorized")
    def unauthorized() -> None:
        raise UnauthorizedError()

    @app.get("/forbidden")
    def forbidden() -> None:
        raise ForbiddenError()

    @app.get("/not-found")
    def not_found() -> None:
        raise NotFoundError("Repository not found")

    @app.get("/conflict")
    def conflict() -> None:
        raise ConflictError()

    @app.get("/validation/{item_id}")
    def validation(item_id: int) -> dict[str, int]:
        return {"item_id": item_id}

    @app.get("/database-error")
    def database_error() -> None:
        raise SQLAlchemyError("password=unsafe DATABASE_URL=unsafe")

    @app.get("/unexpected-error")
    def unexpected_error() -> None:
        raise RuntimeError("token=unsafe at C:/private/path")

    return TestClient(app, raise_server_exceptions=False)


def test_application_errors_use_the_common_envelope():
    client = create_error_test_client()

    cases = (
        ("/bad-request", 400, "BAD_REQUEST"),
        ("/unauthorized", 401, "UNAUTHORIZED"),
        ("/forbidden", 403, "FORBIDDEN"),
        ("/not-found", 404, "RESOURCE_NOT_FOUND"),
        ("/conflict", 409, "CONFLICT"),
    )
    for path, status_code, code in cases:
        response = client.get(path)

        assert response.status_code == status_code
        assert response.json()["error"]["code"] == code
        assert response.json()["error"]["details"] is None


def test_validation_errors_only_expose_safe_validation_details():
    response = create_error_test_client().get("/validation/not-an-integer")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert response.json()["error"]["details"] == [
        {"location": ["path", "item_id"], "message": "Input should be a valid integer, unable to parse string as an integer", "type": "int_parsing"}
    ]


def test_database_and_unexpected_errors_do_not_leak_sensitive_details():
    client = create_error_test_client()

    for path, code, message in (
        ("/database-error", "DATABASE_ERROR", "Database operation failed"),
        ("/unexpected-error", "INTERNAL_SERVER_ERROR", "An unexpected error occurred"),
    ):
        response = client.get(path)
        body = response.json()

        assert response.status_code == 500
        assert body == {"error": {"code": code, "message": message, "details": None}}
        assert "unsafe" not in response.text
        assert "C:/private/path" not in response.text
