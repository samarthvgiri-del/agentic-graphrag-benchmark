from fastapi import FastAPI
from app.api.routes_health import router as health_router
from app.api.routes_rag import router as rag_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_agentic import router as agentic_router
from app.api.routes_benchmark import router as benchmark_router
from app.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend API for Agentic GraphRAG Benchmarking Platform",
    version="0.6.0",
)

app.include_router(health_router, tags=["health"])
app.include_router(rag_router)
app.include_router(ingest_router)
app.include_router(agentic_router)
app.include_router(benchmark_router)


@app.get("/")
async def root():
    return {"message": f"Welcome to the {settings.PROJECT_NAME} API"}
