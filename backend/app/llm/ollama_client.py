"""
Ollama LLM client for local inference via the Ollama REST API.
"""

import httpx
import logging
from typing import Optional, Union

from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 300


class OllamaClient:
    """Client for interacting with a local Ollama instance."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout or (settings.OLLAMA_REQUEST_TIMEOUT_MS / 1000.0)

    # ------------------------------------------------------------------
    # Health / availability
    # ------------------------------------------------------------------
    async def is_available(self) -> bool:
        """Check whether the Ollama server is reachable."""
        import asyncio
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, asyncio.TimeoutError) as exc:
            logger.warning("Ollama unreachable: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        format: Optional[Union[str, dict]] = None,
        think: bool = True,
        num_predict: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Send a prompt to Ollama and return the generated text.

        Uses the /api/generate endpoint with stream=false so we get a
        single JSON response containing the full reply.
        """
        import asyncio

        effective_timeout = timeout if timeout is not None else self.timeout

        options = {"temperature": temperature}
        if num_predict is not None:
            options["num_predict"] = num_predict

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if format:
            payload["format"] = format
        if system_prompt:
            payload["system"] = system_prompt

        async def _make_request():
            # In newer Ollama versions, thinking models cannot be disabled via system prompts
            # without breaking JSON schema generation. We let the model think normally,
            # or pass any top-level/option configurations if supported in the future.
            # We explicitly DO NOT inject "DO NOT output <think> blocks" as it causes 100% fallback.
            if think is False:
                # Some API versions might respect it in options
                options["think"] = False

            async with httpx.AsyncClient(timeout=effective_timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                response = data.get("response", "")
                # qwen3 thinking-mode quirk: when format is set,
                # the model may put structured output in "thinking"
                # and leave "response" empty.
                if not response and data.get("thinking"):
                    response = data["thinking"]
                return response

        try:
            return await asyncio.wait_for(_make_request(), timeout=effective_timeout + 2.0)
        except (httpx.TimeoutException, asyncio.TimeoutError):
            logger.error("Ollama request timed out after %ss", effective_timeout)
            raise RuntimeError(
                f"Ollama request timed out after {effective_timeout}s. "
                "Consider increasing the timeout or using a smaller model."
            )
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama at %s", self.base_url)
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is the Ollama server running?"
            )
        except httpx.HTTPStatusError as exc:
            logger.error("Ollama HTTP error: %s", exc)
            raise RuntimeError(f"Ollama returned an error: {exc}")


# Module-level singleton for convenience
ollama_client = OllamaClient()
