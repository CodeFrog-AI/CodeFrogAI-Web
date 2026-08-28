from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import OperationalError

from app.db import database


def test_database_health_check_executes_select_one(monkeypatch):
    connection = MagicMock()
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = connection
    monkeypatch.setattr(database, "engine", engine)

    database.check_database_connection()

    connection.execute.assert_called_once()


def test_database_health_check_hides_connection_details(monkeypatch):
    engine = MagicMock()
    engine.connect.side_effect = OperationalError(
        "SELECT 1", {}, Exception("password=unsafe")
    )
    monkeypatch.setattr(database, "engine", engine)

    with pytest.raises(database.DatabaseConnectionError, match="Database connection failed"):
        database.check_database_connection()
