from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Agentic GraphRAG Benchmarking Platform"

    # Ollama settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3:4b"

    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Provider selection
    LLM_PROVIDER: str = "ollama"
    EVALUATOR_PROVIDER: str = "ollama"
    EVALUATOR_MODEL: Optional[str] = None
    ORCHESTRATOR_PROVIDER: str = "ollama"
    ORCHESTRATOR_MODEL: Optional[str] = None

    # Qdrant settings
    QDRANT_URL: str = "http://localhost:6333"

    # Timeouts
    OLLAMA_REQUEST_TIMEOUT_MS: int = 120000
    AGENT_RUNTIME_BUDGET_MS: int = 60000
    AGENT_MAX_STEPS: int = 5
    AGENT_MAX_TOOL_REPEATS: int = 2

    # TigerGraph settings
    TIGERGRAPH_HOST: Optional[str] = None
    TIGERGRAPH_SECRET: Optional[str] = None
    TIGERGRAPH_GRAPHNAME: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

settings = Settings()
