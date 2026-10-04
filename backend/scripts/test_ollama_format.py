import asyncio
import sys

from app.llm.ollama_client import ollama_client
from app.models.schemas import OrchestratorAction

async def run():
    prompt = '''You are the Orchestrator. Output valid JSON:
{
  "action": "vector_search",
  "reason": "test",
  "target": "test",
  "expected_information": "test"
}
'''
    print('Testing format="json"...')
    try:
        res1 = await ollama_client.generate(prompt=prompt, format="json")
        print('RES1:', repr(res1))
    except Exception as e:
        print('Error 1:', e)

    print('Testing format=schema...')
    try:
        res2 = await ollama_client.generate(prompt=prompt, format=OrchestratorAction.model_json_schema())
        print('RES2:', repr(res2))
    except Exception as e:
        print('Error 2:', e)

if __name__ == "__main__":
    asyncio.run(run())
