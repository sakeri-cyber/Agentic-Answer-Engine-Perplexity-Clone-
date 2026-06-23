import asyncio
import os
from sentence_transformers import SentenceTransformer
from elasticsearch import AsyncElasticsearch
from qdrant_client import AsyncQdrantClient

COLLECTION_NAME = "arxiv_papers"

async def retrieve_context(user_query: str):
    embedder = SentenceTransformer("perplexity-ai/pplx-embed-context-v1-0.6B", trust_remote_code=True)
    query_vector = embedder.encode(user_query).tolist()

    es_client = AsyncElasticsearch(hosts=["http://127.0.0.1:9200"])
    qdrant_client = AsyncQdrantClient(url="http://127.0.0.1:6333")

    fused_context = {}

    try:
        # 1. Fetch from Qdrant
        qdrant_response = await qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3
        )
        for hit in qdrant_response.points:
            chunk_id = hit.id
            fused_context[chunk_id] = {
                "title": hit.payload.get('title'),
                "text": hit.payload.get('text'),
                "source": "Qdrant (Semantic)"
            }

        # 2. Fetch from Elasticsearch
        es_results = await es_client.search(
            index=COLLECTION_NAME, 
            query={"match": {"text": user_query}}, 
            size=3
        )
        for hit in es_results['hits']['hits']:
            chunk_id = hit['_id']
            # If already found by Qdrant, we just note that it was a hybrid match
            if chunk_id in fused_context:
                fused_context[chunk_id]["source"] = "Hybrid (Both)"
            else:
                fused_context[chunk_id] = {
                    "title": hit['_source'].get('title'),
                    "text": hit['_source'].get('text'),
                    "source": "Elasticsearch (Keyword)"
                }

    finally:
        await es_client.close()

    return list(fused_context.values())

def generate_llm_prompt(query: str, contexts: list):
    prompt = "You are an expert AI Research Assistant. Answer the user's question using ONLY the provided ArXiv paper excerpts below.\n"
    prompt += "If the answer cannot be derived from the context, say 'I cannot find the answer in the ingested papers.'\n"
    prompt += "Always cite the paper title when referencing information.\n\n"
    prompt += "=== CONTEXT STRATA ===\n"
    
    for i, ctx in enumerate(contexts, 1):
        prompt += f"\n[{i}] Paper: {ctx['title']} (Retrieved via: {ctx['source']})\n"
        prompt += f"Excerpt: {ctx['text']}\n"
        prompt += "-" * 40 + "\n"
        
    prompt += f"\nUser Question: {query}\n"
    prompt += "Detailed Answer:"
    return prompt

async def main():
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    query = input("Ask your Answer Engine a question: ")
    print("\nRetrieving and fusing context from Hybrid storage layer...")
    
    contexts = await retrieve_context(query)
    
    if not contexts:
        print("No matching papers found in the database.")
        return

    compiled_prompt = generate_llm_prompt(query, contexts)
    
    print("\n" + "="*60)
    print("🚀 GENERATED LLM PROMPT (Ready for Synthesis)")
    print("="*60)
    print(compiled_prompt)

if __name__ == "__main__":
    asyncio.run(main())