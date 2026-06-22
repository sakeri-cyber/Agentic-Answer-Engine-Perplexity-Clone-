import os
import json
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import AsyncGroq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from elasticsearch import AsyncElasticsearch
from qdrant_client import AsyncQdrantClient

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = AsyncGroq(api_key=GROQ_API_KEY)

es_client = AsyncElasticsearch("http://localhost:9200")
qdrant_client = AsyncQdrantClient(url="http://localhost:6333")
ColLECTION_NAME = "arxiv_papers"

print("Loading query embedder...")
embedder = SentenceTransformer("perplexity-ai/pplx-embed-context-v1-0.6B")

app = FastAPI(title = "Answer Engine")

class QueryRequest(BaseModel):
    query: str

async def understand_query(user_query: str) -> dict:
    """
    Acts as the Traffic Cop. Takes a messy user query and structures it.
    We use Groq because it runs Llama-3 at ~800 tokens per second (virtually zero latency).
    """

    system_prompt = """
    You are a search query analyzer. Analyze the user's input and output ONLY a valid JSON object.
    Do not include markdown blocks or any other text.

    Format:
    {
        "is_safe": boolean, // False if asking for illegal acts or diagnostics
        "es_keywords": "string", // 2-4 core keywords for BM25 exact match
        "qdrant_hyde": "string", // Rewrite the query as a hypothetical academic abstract sentence
        "requires_recent": boolean // True if they ask for 'latest', 'new', '2024', etc.
    }

    """

    try: 
        response = await groq_client.chat.completions.create(
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            model="llama-3.1-8b-instant",
            temperature = 0.1
            max_tokens = 150
            response_format = {"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)
    
    except Exception as e:
        print(f"Groq Routing Failed: {e}")

        return {
            "is_safe": True,
            "es_keywords": user_query,
            "qdrant_hyde": user_query,
            "requires_recent": False
        }

def reciprocal_rank_fusion(qdrant_results, es_results, k=60):
    """
    Fuses the results from Dense and Sparse retrievers using RRF.
    Returns a sorted list of unique documents with their metadata.
    """
    rrf_socres = {}
    document_store = {}

    for rank, point in enumerate(qdrant_results, start=1):
        doc_id = point.id
        payload = point.payload

        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0.0
            document_store[doc_id] = payload

            # Store the raw dense scores for XGBoost later
            document_store[doc_id]['dense_score'] = point.score
            document_store[doc_id]['sparse_score'] = 0.0

        rrf_scores[doc_id] += 1 / (rank + k)
    
    for rank, hit in enumerate(es_results, start=1):
        doc_id = hit['_id']
        source = hit['_source']

        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0.0
            document_store[doc_id] = source

            document_store[doc_id]['dense_score'] = 0.0

        document_store[doc_id]['sparse_score'] = hit['_score']

        rrf_scores[doc_id] += 1 / (rank + k)
    
    sorted_docs = sorted(
        rrf_scores.keys(),
        key=lambda x: rrf_scores[x],
        reverse=True
    )

    final_candidates = []
    for doc_id in sorted_docs:
        doc_data = document_store[doc_id]
        doc_data['chunk_id'] = doc_id
        doc_data['rrf_score'] = rrf_scores[doc_id]
        final_candidates.append(doc_data)
    
    return final_candidates




@app.post("/search")
async def execute_search(request: QueryRequest):
    print(f"Received query: {request.query}")

    # Step 1: Route and Understand the Query
    routing_data = await understand_query(request.query)

    if not routing_data.get("is_safe"):
        raise HTTPException(status_code=400, detail="Query violates safety guidelines. Please rephrase.")

    es_query = routing_data.get("es_keywords", "")
    qdrant_query = routing_data.get("qdrant_hyde", "")
    
    query_vector = embedder.encode(qdrant_query).tolist()

    async def search_qdrant():
        """Searches the dense vector space"""
        return await qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=100
        )
    
    async def search_elasticsearch():
        """Searches the sparse lexical space"""
        response = await es_client.search(
            index=COLLECTION_NAME,
            query={
                "match": {
                    "text": es_query
                }
            },
            size=100
        )
        return response['hits']['hits']
    
    print("Executing parallel database search ...")
    qdrant_results, es_results = await asyncio.gather(
        search_qdrant(),
        search_elasticsearch()
    )

    print(f"Qdrant found {len(qdrant_results)} hits.")
    print(f"Elasticsearch found {len(es_results)} hits.")

    print("Fusing results via RRF...")
    fused_candidates = reciprocal_rank_fusion(qdrant_results, es_results)
    
    print(f"Total unique candidates after fusion: {len(fused_candidates)}")
    
    # Just returning the top 5 just to verify the math worked before we add XGBoost
    return {
        "status": "success",
        "routing": routing_data,
        "top_qdrant_hit": qdrant_results[0].payload['title'] if qdrant_results else None,
        "top_es_hit": es_results[0]['_source']['title'] if es_results else None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=" 0.0.0.0", port=8000)       