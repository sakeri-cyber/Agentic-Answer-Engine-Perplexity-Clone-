import uuid
from sentence_transformers import SentenceTransformer
from elasticsearch import AsyncElasticsearch
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

# Initialising Client and Models 

# es_client = AsyncElasticsearch("http://localhost:9200")
# qdrant_client = AsyncQdrantClient(url = "http://localhost:6333")
COLLECTION_NAME = "arxiv_papers"

print("Loading pplx-embed-context-v1-0.6B model...")
embedder = SentenceTransformer("perplexity-ai/pplx-embed-context-v1-0.6B" , trust_remote_code=True)

def chunk_text(text , chunk_size=250, overlap=50):
    """Splits text into chunks with specified size and overlap"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

async def process_and_ingest(paper):
    """Takes a single paper dictionary, chunks it, embeds it and upserts top DBs"""

    

    # Chunk the document
    chunks = chunk_text(paper['abstract'])

    points_to_upsert = []
    es_operations = []

    for idx, chunk in enumerate(chunks):
        # --- CONTEXT INJECTION ---
        # To simulate the context model's strength, we prepend the paper title 
        # to the chunk so the vector space knows what this chunk belongs to.

        contextualized_chunk = f"Title: {paper['title']} \n\n Content: {chunk}"

        # Dense Embedding
        vector = embedder.encode(contextualized_chunk).tolist()

        # IDs for chunks 
        chunk_id = f"{paper['id']}_chunk_{idx}"
        chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

        # Prepare Qdrant payload
        points_to_upsert.append(
            PointStruct(
                id=chunk_uuid,
                vector=vector,
                payload={
                    "paper_id": paper['id'],
                    "title": paper['title'],
                    "text": chunk,
                    "published_date": paper['published_date'],
                    "citation_count": paper['citation_count']
                }
            )
        )   

        # Prepare Elasticsearch payload
        es_operations.append({
            "index": {
                "_index": COLLECTION_NAME,
                "_id" : chunk_uuid
            }
        })
        es_operations.append({
            "arxiv": paper['id'],
            "title": paper['title'],
            "text": chunk,
            "published_date": paper['published_date']
        })
    
    es_client = AsyncElasticsearch("http://127.0.0.1:9200", request_timeout=60)
    qdrant_client = AsyncQdrantClient(url="http://127.0.0.1:6333", timeout=60)
    
    try: 
        # Upsert to Qdrant
        await qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_to_upsert
        )

        # Bulk insert to Elasticsearch
        await es_client.bulk(operations=es_operations)

        print(f"Successfully ingested paper: {paper['title']} with {len(chunks)} chunks.")
    except Exception as e:
        print(f"Error ingesting paper {paper['title']}: {str(e)}")
    
    finally:
        # Clean up the connections so we don't leak memory
        await es_client.close()

