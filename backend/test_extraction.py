import asyncio
import json
from app.llm.ollama_client import ollama_client
from app.models.schemas import EntityExtraction

EXTRACT_PROMPT = """Extract the key entities from this question to query a knowledge graph.
Allowed types: Athlete, Country, Event, Games, GenericEntity

Example Question: Who won the men's 100m sprint in the 2012 London Olympics?
Output:
<think>I need to extract the entities.</think>
{{
  "entities": [
    {{"name": "men's 100m sprint", "type": "Event"}},
    {{"name": "2012 London Olympics", "type": "Games"}}
  ]
}}

Question: {question}
Output:
<think>I will extract the entities now.</think>
"""

async def test_extraction():
    question = "Who won the gold medal in the event held at Olympic Weightlifting Gymnasium on 20 September 1988?"
    resp = await ollama_client.generate(
        prompt=EXTRACT_PROMPT.format(question=question),
        temperature=0.0,
        think=False,
        num_predict=300
    )
    print("RAW RESP:")
    print(resp)

if __name__ == "__main__":
    asyncio.run(test_extraction())
