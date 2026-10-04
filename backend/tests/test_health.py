from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app

client = TestClient(app)

def test_health_check_endpoint():
    with patch("app.graph.tigergraph_client.TigerGraphClient.get_status", return_value={"configured": True, "status": "connected"}):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()

        assert data["api_status"] == "ok"

        # Ollama should be configured by default via config.py fallbacks
        assert "ollama" in data
        assert data["ollama"]["configured"] is True

        # Qdrant should be configured by default
        assert "qdrant" in data
        assert data["qdrant"]["configured"] is True

        # TigerGraph should be configured in Phase 4
        assert "tigergraph" in data
        assert data["tigergraph"]["configured"] is True
        assert data["tigergraph"]["status"] == "connected"
