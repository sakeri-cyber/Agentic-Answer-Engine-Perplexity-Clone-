import os
import json
import random
import asyncio
import duckdb
import textstat
import re
from datetime import datetime
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# --- 1. DuckDB Setup ---
def setup_duckdb():
    print("Initializing DuckDB...")
    con = duckdb.connect('training_data.duckdb')
    con.execute("""
        CREATE SEQUENCE IF NOT EXISTS seq_query_id;
        CREATE TABLE IF NOT EXISTS ltr_features (
            query_id INTEGER,
            document_id VARCHAR,
            query_text VARCHAR,
            query_intent INTEGER,
            dense_score DOUBLE,
            sparse_score DOUBLE,
            doc_readability DOUBLE,
            exact_match_title INTEGER,
            h_index INTEGER,
            citation_velocity DOUBLE,
            title_body_divergence DOUBLE,
            semantic_lexical_ratio DOUBLE,
            llm_reasoning TEXT,
            relevance_label INTEGER
        )
    """)
    return con

# --- 2. Organic Query Generation ---
async def generate_synthetic_queries_organically(abstract: str) -> list:
    """Generates diverse queries without forcing a persona."""
    prompt = f"""
    Read this paper abstract: "{abstract[:1000]}"
    Based purely on this content, generate 3 different search queries a real human might type to find this exact paper. 
    1. A short, keyword-dense query.
    2. A natural language question.
    3. A highly specific technical query.
    
    Output JSON ONLY: 
    {{"queries": ["query_1", "query_2", "query_3"]}}
    """
    try:
        res = await groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.8,
            max_tokens=200
        )
        match = re.search(r'\{.*\}', res.choices[0].message.content, re.DOTALL)
        return json.loads(match.group(0))["queries"] if match else ["AI research paper"]
    except:
        return ["AI research paper"]

def calculate_query_complexity(query: str) -> int:
    """1 = Simple/Layman, 2 = Intermediate, 3 = Highly Technical."""
    words = query.split()
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
    if avg_word_length > 6.5 or len(words) > 8: return 3
    elif avg_word_length > 5.0: return 2
    else: return 1

# --- 3. Feature Calculation ---
def calculate_features(doc: dict, query: str) -> dict:
    readability = textstat.flesch_kincaid_grade(doc['abstract'])
    h_index = random.randint(1, 10) 
    
    age_days = doc.get('age_days', random.randint(10, 2000))
    age_years = max(1, age_days / 365.0)
    citation_vel = doc.get('citations', random.randint(0, 500)) / age_years
    
    query_words = set(query.lower().split())
    title_words = set(doc['title'].lower().split())
    exact_match = 1 if len(query_words.intersection(title_words)) > 0 else 0
    
    dense_score = random.uniform(0.6, 0.95)
    sparse_score = random.uniform(0.0, 20.0)
    
    title_sparse = random.uniform(0.0, 10.0) if exact_match else 0.0
    body_sparse = sparse_score - title_sparse
    divergence = abs(title_sparse - body_sparse)
    semantic_ratio = dense_score / (sparse_score + 0.001)

    return {
        "readability": readability,
        "exact_match_title": exact_match,
        "h_index": h_index,
        "citation_velocity": citation_vel,
        "dense_score": dense_score,
        "sparse_score": sparse_score,
        "divergence": divergence,
        "semantic_ratio": semantic_ratio
    }

# --- 4. LLM-as-a-Judge ---
async def judge_relevance(query: str, abstract: str) -> dict:
    prompt = f"""
    You are an expert Search Quality Rater.
    User Query: "{query}"
    Document Abstract: "{abstract[:1000]}"
    
    Grade how well this document satisfies the query on a scale of 0 to 4.
    0 = Irrelevant, 1 = Bad, 2 = Fair, 3 = Good, 4 = Perfect.
    
    Output JSON ONLY:
    {{
        "reasoning": "1 sentence explaining why",
        "score": integer
    }}
    """
    try:
        res = await groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=150
        )
        match = re.search(r'\{.*\}', res.choices[0].message.content, re.DOTALL)
        return json.loads(match.group(0)) if match else {"reasoning": "Regex failed", "score": 0}
    except:
        return {"reasoning": "Fallback error", "score": 0}

# --- 5. Orchestration ---
async def generate_dataset(papers: list):
    con = setup_duckdb()
    print(f"Generating training data for {len(papers)} papers...")
    
    for paper in papers:
        queries = await generate_synthetic_queries_organically(paper['abstract'])
        
        for query in queries:
            intent = calculate_query_complexity(query)
            query_id = con.execute("SELECT nextval('seq_query_id')").fetchone()[0]
            
            # MOCKING CANDIDATES: In production, we'd query the DB. Here we mock Good and Bad matches to train the ranker
            candidates = [paper, {"title": "Irrelevant Spam", "abstract": "Buy cheap things online", "id": "spam_1"}]
            
            for cand in candidates:
                features = calculate_features(cand, query)
                judgment = await judge_relevance(query, cand['abstract'])
                
                print(f"  [Grade: {judgment['score']}] Query: '{query}' -> Doc: {cand['title'][:30]}...")
                
                con.execute("""
                    INSERT INTO ltr_features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_id, cand.get('id', 'mock_doc'), query, intent, 
                    features['dense_score'], features['sparse_score'], features['readability'],
                    features['exact_match_title'], features['h_index'], features['citation_velocity'],
                    features['divergence'], features['semantic_ratio'], 
                    judgment['reasoning'], judgment['score']
                ))
            
    print("\n✅ Data generation complete! Saved to training_data.duckdb")
    con.close()

if __name__ == "__main__":
    # Mocking our ArXiv data to jumpstart the database
    sample_papers = [{
        "id": "arxiv_1",
        "title": "How Transparent is DiffusionGemma?",
        "abstract": "We conduct a suite of interpretability case studies, uncovering initial evidence of novel diffusion-specific phenomena such as non-chronological reasoning, token and sequence smearing, and intermediate-context reasoning. Finally, we test monitorability, a key application of transparency that measures whether model outputs are useful for downstream tasks.",
        "citations": 45,
        "age_days": 15
    }]
    
    asyncio.run(generate_dataset(sample_papers))