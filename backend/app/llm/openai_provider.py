import json
import logging
from typing import Optional, Union, Dict, Any

import openai
from app.config import settings

logger = logging.getLogger(__name__)

class OpenAIProvider:
    """Client for interacting with OpenAI API."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model or settings.OPENAI_MODEL
        api_key = api_key or settings.OPENAI_API_KEY

        if not api_key:
            logger.warning("OPENAI_API_KEY is not set. OpenAIProvider will fail on generation.")
            api_key = "dummy-key-to-allow-init"

        self.client = openai.AsyncOpenAI(api_key=api_key)
        self._input_tokens = 0
        self._output_tokens = 0
        self._api_calls = 0

    @property
    def input_tokens(self) -> int:
        return self._input_tokens

    @property
    def output_tokens(self) -> int:
        return self._output_tokens

    @property
    def api_calls(self) -> int:
        return self._api_calls

    async def is_available(self) -> bool:
        """Check whether the OpenAI API is available by validating key presence."""
        return bool(self.client.api_key and self.client.api_key != "dummy-key-to-allow-init")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        format: Optional[Union[str, Dict[str, Any]]] = None,
        think: bool = True,  # Unused for standard OpenAI models
        num_predict: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Send a prompt to OpenAI and return the generated text."""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if timeout is not None:
            kwargs["timeout"] = timeout

        if num_predict:
            kwargs["max_tokens"] = num_predict

        # Handle structured output format
        if format:
            if isinstance(format, dict):
                # Ensure it's a valid JSON schema for OpenAI Structured Outputs
                # OpenAI requires a specific structure for response_format:
                # {"type": "json_schema", "json_schema": {"name": "schema_name", "schema": {...}}}
                def _add_additional_properties(schema_obj):
                    if isinstance(schema_obj, dict):
                        if schema_obj.get("type") == "object":
                            schema_obj["additionalProperties"] = False
                        for key, value in schema_obj.items():
                            _add_additional_properties(value)
                    elif isinstance(schema_obj, list):
                        for item in schema_obj:
                            _add_additional_properties(item)

                _add_additional_properties(format)

                schema_name = format.get("title", "ResponseSchema")
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": format,
                        "strict": True
                    }
                }
            elif format == "json":
                kwargs["response_format"] = {"type": "json_object"}
                # Add instruction to output JSON if missing
                if not system_prompt or "json" not in system_prompt.lower():
                    messages.insert(0, {"role": "system", "content": "You must output valid JSON."})
                    kwargs["messages"] = messages

        try:
            response = await self.client.chat.completions.create(**kwargs)

            if response.usage:
                self._input_tokens += response.usage.prompt_tokens
                self._output_tokens += response.usage.completion_tokens
            self._api_calls += 1

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise RuntimeError(f"OpenAI generation failed: {e}")
