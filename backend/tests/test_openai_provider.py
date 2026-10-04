import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.llm.openai_provider import OpenAIProvider

@pytest.mark.asyncio
async def test_openai_provider_generate_json():
    provider = OpenAIProvider(api_key="test_key", model="gpt-4o-mini")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"action": "stop", "reason": "test"}'

    with patch("openai.AsyncOpenAI") as mock_openai:
        mock_client_instance = MagicMock()
        mock_openai.return_value = mock_client_instance
        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)

        provider.client = mock_client_instance

        schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "reason": {"type": "string"}
            }
        }

        result = await provider.generate(prompt="Test", format=schema)

        assert result == '{"action": "stop", "reason": "test"}'
        mock_client_instance.chat.completions.create.assert_called_once()
        kwargs = mock_client_instance.chat.completions.create.call_args.kwargs
        assert "response_format" in kwargs
        assert kwargs["response_format"]["type"] == "json_schema"
        assert kwargs["response_format"]["json_schema"]["schema"] == schema
        assert kwargs["response_format"]["json_schema"]["strict"] is True

@pytest.mark.asyncio
async def test_openai_provider_generate_error():
    provider = OpenAIProvider(api_key="test_key", model="gpt-4o-mini")

    with patch("openai.AsyncOpenAI") as mock_openai:
        mock_client_instance = MagicMock()
        mock_openai.return_value = mock_client_instance
        mock_client_instance.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

        provider.client = mock_client_instance

        with pytest.raises(RuntimeError, match="OpenAI generation failed: API Error"):
            await provider.generate(prompt="Test")
