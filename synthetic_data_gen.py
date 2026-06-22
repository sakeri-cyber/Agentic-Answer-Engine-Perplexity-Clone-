import os
import json
import random
import asyncio
import duckdb
import textstat
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

# --- 2. Synthetic Query Generation ---
PERSONAS = [
    ("Layman", "You are a curious person with no science background Googling a concept simply."),
    ("Student", "You are an undergrad student looking for a clear explanation for a project."),
    ("Expert", "You are a PhD researcher looking for highly technical methodologies.")
]

async def generate_synthetic_query(abstract: str) -> dict:
    """
    Generates diverse queries without forcing a persona, mimicking real-world variance.
    """
    prompt = f"""
    Read this paper abstract: "{abstract[:1000]}"
    Based purely on this content, generate 3 different search queries a real human might type into a search engine to find this exact paper. 
    Ensure organic diversity in how people search:
    1. A short, keyword-dense query (like someone rushing).
    2. A natural language question (like someone asking a conversational assistant).
    3. A highly specific, technical query targeting a specific methodology or finding in the text.
    
    Output JSON ONLY: 
    {{"queries": ["query_1", "query_2", "query_3"]}}
    """
    try:
        res = await groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.8, # Higher temp for more creative, organic variance
            response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)["queries"]
    except:
        return ["AI research paper"]

def calculate_query_complexity(query: str) -> int:
    """
    Extracts the feature AFTER the organic generation.
    1 = Simple/Layman, 2 = Intermediate, 3 = Highly Technical.
    """
    # A simple heuristic: longer words and specific jargon increase complexity
    words = query.split()
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
    
    if avg_word_length > 6.5 or len(words) > 8:
        return 3 # Technical / Complex
    elif avg_word_length > 5.0:
        return 2 # Intermediate
    else:
        return 1 # Simple / Keyword heavy

# --- 3. The LLM Judge (Chain of Thought) ---
async def judge_relevance(query: str, abstract: str) -> dict:
    prompt = f"""
    You are an expert Search Quality Rater.
    User Query: "{query}"
    Document Abstract: "{abstract[:1000]}"
    
    Grade how well this document satisfies the query on a scale of 0 to 4.
    0 = Irrelevant, 1 = Bad, 2 = Fair, 3 = Good, 4 = Perfect.
    
    Output JSON ONLY:
    {{
        "reasoning": "2 sentences explaining why it deserves this score",
        "score": integer (0-4)
    }}
    """
    try:
        res = await groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192", # In prod, use a heavier model like 70b or GPT-4o for judging
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)
    except:
        return {"reasoning": "Fallback error", "score": 0}

# --- 4. Feature Calculation & Orchestration ---
def calculate_features(doc: dict, query: str) -> dict:
    # 1. Readability
    readability = textstat.flesch_kincaid_grade(doc['abstract'])
    
    # 2. Simulated Authority
    h_index = random.randint(1, 10) 
    
    # 3. Citation Velocity
    age_days = doc.get('age_days', random.randint(10, 2000))
    age_years = max(1, age_days / 365.0)
    citation_vel = doc.get('citations', random.randint(0, 500)) / age_years
    
    # 4. Keyword Matches (Simulated Sparse/Dense for now)
    query_words = set(query.lower().split())
    title_words = set(doc['title'].lower().split())
    exact_match = 1 if len(query_words.intersection(title_words)) > 0 else 0
    
    dense_score = random.uniform(0.6, 0.95)
    sparse_score = random.uniform(0.0, 20.0)
    
    # 5. Ratios & Divergence
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

async def generate_dataset(papers: list):
    con = setup_duckdb()
    
    print(f"Generating training data for {len(papers)} queries...")
    for paper in papers:
        # 1. Generate Query
        q_data = await generate_synthetic_query(paper['abstract'])
        query = q_data['query']
        intent = q_data['intent']
        
        # 2. Fetch Candidates (Mocking 3 candidates: Good, Average, Bad)
        # In reality, you'd call your execute_search() function here!
        candidates = [paper, paper, paper] # Duplicated just to show the loop structure
        
        # We need a shared ID for this query group
        query_id = con.execute("SELECT nextval('seq_query_id')").fetchone()[0]
        
        for cand in candidates:
            # 3. Calculate Advanced Features
            features = calculate_features(cand, query)
            
            # 4. Get LLM Judgment
            judgment = await judge_relevance(query, cand['abstract'])
            
            # 5. Insert into DuckDB
            con.execute("""
                INSERT INTO ltr_features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query_id, cand.get('id', 'doc_1'), query, intent, 
                features['dense_score'], features['sparse_score'], features['readability'],
                features['exact_match_title'], features['h_index'], features['citation_velocity'],
                features['divergence'], features['semantic_ratio'], 
                judgment['reasoning'], judgment['score']
            ))
            
    print("Data generation complete! Saved to training_data.duckdb")
    con.close()

# Execute
if __name__ == "__main__":
    # Mocking a fetched paper to test the pipeline
    sample_papers = [{
        "id": "1234.5678",
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        "citations": 105000,
        "age_days": 2500
    }]
    
    asyncio.run(generate_dataset(sample_papers))