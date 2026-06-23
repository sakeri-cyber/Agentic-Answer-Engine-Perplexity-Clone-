import asyncio
from sentence_transformers import SentenceTransformer
from elasticsearch import AsyncElasticsearch
from qdrant_client import AsyncQdrantClient

COLLECTION_NAME = "arxiv_papers"

async def hybrid_search(user_query: str):
    print("\nLoading Perplexity embedding model (cached)...")
    embedder = SentenceTransformer("perplexity-ai/pplx-embed-context-v1-0.6B", trust_remote_code=True)
    
    # 1. Convert the user's question into a math vector
    print(f"\nEmbedding query: '{user_query}'...")
    query_vector = embedder.encode(user_query).tolist()

    # 2. Open DB connections
    es_client = AsyncElasticsearch(hosts=["http://127.0.0.1:9200"])
    qdrant_client = AsyncQdrantClient(url="http://127.0.0.1:6333")

    try:
        # ==========================================
        # SEARCH 1: QDRANT (Semantic Vector Search)
        # Finds text that *means* the same thing as the query
        # ==========================================
        print("\n" + "="*50)
        print("🧠 QDRANT RESULTS (Semantic Match)")
        print("="*50)
        
        qdrant_response = await qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3  # Bring back the top 3 chunks
        )
        
        for i, hit in enumerate(qdrant_response.points, 1):
            title = hit.payload.get('title', 'Unknown')
            score = round(hit.score, 4)
            print(f"{i}. [Score: {score}] {title}")

        # ==========================================
        # SEARCH 2: ELASTICSEARCH (Sparse Keyword Search)
        # Finds text that uses the *exact words* in the query
        # ==========================================
        print("\n" + "="*50)
        print("🔍 ELASTICSEARCH RESULTS (Keyword Match)")
        print("="*50)

        # es_query = {
        #     "query": {
        #         "match": {
        #             "text": user_query
        #         }
        #     }
        # }
        
        es_results = await es_client.search(
            index=COLLECTION_NAME, 
            query = {
                "match": {
                    "text": user_query
                }
            },
            size=3
        )
        
        for i, hit in enumerate(es_results['hits']['hits'], 1):
            title = hit['_source'].get('title', 'Unknown')
            score = round(hit['_score'], 4)
            print(f"{i}. [Score: {score}] {title}")

    except Exception as e:
        print(f"\n❌ Search failed: {e}")
        
    finally:
        await es_client.close()

async def main():
    print("Welcome to the ArXiv Hybrid Search Engine!")
    while True:
        query = input("\nAsk a question about your papers (or type 'quit'): ")
        if query.lower() in ['quit', 'q', 'exit']:
            break
            
        await hybrid_search(query)

if __name__ == "__main__":
    # Silence HuggingFace token warnings for cleaner output
    import os
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    asyncio.run(main())