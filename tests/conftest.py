import pytest

from app.shared.config.settings import settings


@pytest.fixture(autouse=True)
def isolate_forecast_snapshots(monkeypatch, tmp_path):
    """Tests must never append dummy forecasts to the versioned demo snapshot."""
    monkeypatch.setattr(settings, "forecast_snapshot_path", str(tmp_path / "forecast_snapshots.json"))
