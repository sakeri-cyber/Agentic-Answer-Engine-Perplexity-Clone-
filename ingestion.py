import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from elasticsearch import AsyncElasticsearch
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

# Database Clients
es_client = AsyncElasticsearch(hosts=["http://localhost:9200"])
qdrant_client = AsyncQdrantClient(url="http://localhost:6333")

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
    if not await es_client.indices.exists(index=COLLECTION_NAME):
        await es_client.indices.create(index=COLLECTION_NAME)
        print(f"Created Elasticsearch index: {COLLECTION_NAME}")
    
def fetch_recent_arxiv_papers(max_results=50):
    """Fetches recent papers from ArXiv API"""
    print("Fetching recent papers from ArXiv...")

    # Fetching ML/AI papers, sorted by recently updated
    url = f'http://export.arxiv.org/api/query?search_query=cat:cs.AI&sortBy=lastUpdatedDate&sortOrder=descending&max_results={max_results}'

    data = urllib.request.urlopen(url).read()
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

if __name__ == "__main__":
    import asyncio 
    asyncio.run(init_databases())
    docs = fetch_recent_arxiv_papers(10)

    for d in docs:
        print(f"Title: {d['title']}")
        print(f"Published: {d['published_date']}")
        print("-" * 80)

