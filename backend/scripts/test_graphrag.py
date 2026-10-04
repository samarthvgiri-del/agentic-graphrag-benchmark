import asyncio
from app.api.routes_rag import RAGQueryRequest, graphrag_query

async def test_graphrag():
    request = RAGQueryRequest(question="According to the provided corpus, how many shooting events at the 2004 Summer Olympics had more than 37 competitors?")
    try:
        response = await graphrag_query(request)
        print("--- RESPONSE ---")
        print(response.answer)
        print("\n--- CITATIONS ---")
        print(response.citations)
        print("\n--- ENTITIES ---")
        print(response.entities)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_graphrag())
