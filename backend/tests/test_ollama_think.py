import pytest
import httpx
from app.llm.ollama_client import OllamaClient

@pytest.mark.asyncio
async def test_ollama_think_parameter():
    client = OllamaClient()

    schema = {
        "type": "object",
        "properties": {
            "result": {"type": "string"},
        },
        "required": ["result"]
    }

    # Generate with think=False and a schema
    # Without the harmful prompt injection, this should now return valid JSON
    response_text = await client.generate(
        prompt="Output a JSON object with key 'result' and value 'success'.",
        format=schema,
        think=False,
        num_predict=100
    )

    import json
    try:
        data = json.loads(response_text)
        assert "result" in data
        assert data["result"].lower() == "success"
    except json.JSONDecodeError:
        pytest.fail(f"Failed to parse JSON response: {response_text}")
