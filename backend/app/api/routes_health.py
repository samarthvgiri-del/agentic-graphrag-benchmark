from fastapi import APIRouter
from app.models.schemas import HealthResponse, ServiceStatus
from app.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    # In Phase 1, we just report configuration status and assumed availability
    # based on configurations. Real checks will be added when services are implemented.

    ollama_status = ServiceStatus(
        configured=bool(settings.OLLAMA_BASE_URL),
        status="configured" if settings.OLLAMA_BASE_URL else "unavailable/not configured"
    )

    try:
        from app.rag.vector_store import build_vector_store
        v_store = build_vector_store()
        v_status = v_store.get_status()
        qdrant_status = ServiceStatus(
            configured=bool(settings.QDRANT_URL),
            status="connected" if "error" not in v_status else f"error: {v_status['error']}"
        )
    except Exception as e:
        qdrant_status = ServiceStatus(
            configured=bool(settings.QDRANT_URL),
            status=f"error: {str(e)}"
        )
    from app.graph.tigergraph_client import TigerGraphClient
    tg_client = TigerGraphClient()
    tg_status_dict = tg_client.get_status()

    tigergraph_status = ServiceStatus(
        configured=tg_status_dict["configured"],
        status=tg_status_dict["status"]
    )

    return HealthResponse(
        api_status="ok",
        ollama=ollama_status,
        qdrant=qdrant_status,
        tigergraph=tigergraph_status
    )
