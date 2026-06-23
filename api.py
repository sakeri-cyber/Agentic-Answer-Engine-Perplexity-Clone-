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
import re

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = AsyncGroq(api_key=GROQ_API_KEY)

# FIXED: IPv4 Localhost routing
es_client = AsyncElasticsearch(hosts=["http://127.0.0.1:9200"])
qdrant_client = AsyncQdrantClient(url="http://127.0.0.1:6333")
COLLECTION_NAME = "arxiv_papers"

print("Loading query embedder...")
# We use trust_remote_code=True based on our app.py findings
embedder = SentenceTransformer("perplexity-ai/pplx-embed-context-v1-0.6B", trust_remote_code=True)

app = FastAPI(title="ArXiv Answer Engine API")

class QueryRequest(BaseModel):
    query: str

async def understand_query(user_query: str) -> dict:
    """Acts as the Traffic Cop. Takes a messy user query and structures it."""
    
    system_prompt = """
    You are a search query analyzer. 
    Analyze the user query and output a JSON object with these exact keys:
    "is_safe" (boolean), "es_keywords" (string), "qdrant_hyde" (string), "requires_recent" (boolean).
    """
    
    try: 
        response = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=200
            # ABSOLUTELY NO response_format PARAMETER HERE
        )
        
        raw_output = response.choices[0].message.content
        
        # ---------------------------------------------------------
        # BULLETPROOF PARSING: Use Regex to find the JSON brackets {}
        # This completely ignores "Here is your JSON:" or ```markdown
        # ---------------------------------------------------------
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if match:
            clean_json_string = match.group(0)
            return json.loads(clean_json_string)
        else:
            raise ValueError("No JSON brackets found in the LLM response.")
    
    except Exception as e:
        print(f"❌ Groq Routing Failed: {e}")
        # Safe Fallback
        return {
            "is_safe": True,
            "es_keywords": user_query,
            "qdrant_hyde": user_query,
            "requires_recent": False
        }
    
    except Exception as e:
        print(f"❌ Groq Routing Failed: {e}")
        return {
            "is_safe": True,
            "es_keywords": user_query,
            "qdrant_hyde": user_query,
            "requires_recent": False
        }

def reciprocal_rank_fusion(qdrant_results, es_results, k=60):
    """Fuses the results from Dense and Sparse retrievers using RRF."""
    rrf_scores = {}
    document_store = {}

    # Qdrant Results Processing
    for rank, point in enumerate(qdrant_results, start=1):
        doc_id = point.id
        payload = point.payload

        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0.0
            document_store[doc_id] = payload
            document_store[doc_id]['dense_score'] = point.score
            document_store[doc_id]['sparse_score'] = 0.0

        rrf_scores[doc_id] += 1 / (rank + k)
    
    # Elasticsearch Results Processing
    for rank, hit in enumerate(es_results, start=1):
        doc_id = hit['_id']
        source = hit['_source']

        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0.0
            document_store[doc_id] = source
            document_store[doc_id]['dense_score'] = 0.0

        document_store[doc_id]['sparse_score'] = hit['_score']
        rrf_scores[doc_id] += 1 / (rank + k)
    
    sorted_docs = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    final_candidates = []
    for doc_id in sorted_docs:
        doc_data = document_store[doc_id]
        doc_data['chunk_id'] = doc_id
        doc_data['rrf_score'] = rrf_scores[doc_id]
        final_candidates.append(doc_data)
    
    return final_candidates

@app.post("/search")
async def execute_search(request: QueryRequest):
    print(f"\nReceived query: {request.query}")

    # Step 1: Route and Understand the Query
    routing_data = await understand_query(request.query)

    if not routing_data.get("is_safe"):
        raise HTTPException(status_code=400, detail="Query violates safety guidelines.")

    es_query = routing_data.get("es_keywords", request.query)
    qdrant_query = routing_data.get("qdrant_hyde", request.query)
    
    query_vector = embedder.encode(qdrant_query).tolist()

    async def search_qdrant():
        """Searches the dense vector space"""
        # FIXED: Updated deprecated search method
        response = await qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=100
        )
        return response.points
    
    async def search_elasticsearch():
        """Searches the sparse lexical space"""
        # FIXED: Updated ES syntax
        response = await es_client.search(
            index=COLLECTION_NAME,
            query={"match": {"text": es_query}},
            size=100
        )
        return response['hits']['hits']
    
    print("Executing parallel database search...")
    qdrant_results, es_results = await asyncio.gather(
        search_qdrant(),
        search_elasticsearch()
    )

    print(f"Qdrant found {len(qdrant_results)} hits.")
    print(f"Elasticsearch found {len(es_results)} hits.")

    print("Fusing results via RRF...")
    fused_candidates = reciprocal_rank_fusion(qdrant_results, es_results)
    
    print(f"Total unique candidates after fusion: {len(fused_candidates)}")
    
    return {
        "status": "success",
        "routing": routing_data,
        "top_fused_hit": fused_candidates[0]['title'] if fused_candidates else None,
        "total_results": len(fused_candidates)
    }

if __name__ == "__main__":
    import uvicorn
    # Silence HF Symlink warning
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    uvicorn.run(app, host="127.0.0.1", port=8001)  