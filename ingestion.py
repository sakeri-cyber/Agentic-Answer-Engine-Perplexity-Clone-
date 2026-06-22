import ssl
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from elasticsearch import AsyncElasticsearch
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from worker import process_and_ingest
import asyncio

# Database Clients
es_client = AsyncElasticsearch(hosts=["http://127.0.0.1:9200"])
qdrant_client = AsyncQdrantClient(url="http://127.0.0.1:6333")

COLLECTION_NAME = "arxiv_papers"

async def init_databases():
    """Sets up the indices if they don't exist"""
    # 1. Setup Qdrant

    # pplx-embed-context-v1-0.6B outputs 1024 dimensional vectors
    if not await qdrant_client.collection_exists(COLLECTION_NAME):
        await qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

        print(f"Created Qdrant collection: {COLLECTION_NAME}")

    # 2. Setup Elasticsearch
    try:
        await es_client.indices.create(index=COLLECTION_NAME)
        print(f"Created Elasticsearch index: {COLLECTION_NAME}")
    except Exception:
        # If the index already exists, ES throws an error. We silently catch it and proceed!
        pass
    
def fetch_recent_arxiv_papers(max_results=50):
    """Fetches recent papers from ArXiv API"""
    print("Fetching recent papers from ArXiv...")

    # Fetching ML/AI papers, sorted by recently updated
    url = f'http://export.arxiv.org/api/query?search_query=cat:cs.AI&sortBy=lastUpdatedDate&sortOrder=descending&max_results={max_results}'

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context = context).read()
    root = ET.fromstring(data)
    
    # XML Namespace

    ns = {'atom': 'http://www.w3.org/2005/Atom'}

    papers = []
    for entry in root.findall('atom:entry', ns):
        published = entry.find('atom:published', ns).text
        title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
        summary = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()

        import random 
        citation_count = random.randint(0, 500)

        papers.append({
            "id": entry.find('atom:id', ns).text.split('/')[-1],
            "title": title,
            "abstract": summary,
            "published_date": published,
            "citation_count": citation_count
        })

    print(f"Successfully fetched paper {len(papers)} papers.")
    return papers 

async def main():
    await init_databases()

    docs = fetch_recent_arxiv_papers(10)

    print("Handing off to the embedding worker ...")
    for d in docs:
        print(f"Processing paper: {d['title']}")
        await process_and_ingest(d)

    print("Ingestion Pipeline Complete !!")

if __name__ == "__main__":
    asyncio.run(main())

