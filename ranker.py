import xgboost as xgb
import pandas as pd
import textstat
from datetime import datetime

# --- 1. Load the REAL Trained Model ---
def load_xgboost_model():
    print("Loading Trained XGBoost Re-ranker...")
    model = xgb.Booster()
    try:
        model.load_model("perplexity_ranker_v1.json")
    except Exception as e:
        print(f"Warning: Could not load model ({e}). Ensure you ran train_ranker.py!")
    return model

xgb_model = load_xgboost_model()

# --- 2. Live Feature Engineering ---
def calculate_age_in_days(published_date_str: str) -> int:
    try:
        pub_date = datetime.strptime(published_date_str[:10], "%Y-%m-%d")
        return max(0, (datetime.now() - pub_date).days)
    except:
        return 1825 # Default 5 years

def calculate_query_complexity(query: str) -> int:
    """Classifies query intent on the fly: 1=Layman, 2=Student, 3=Expert"""
    words = query.split()
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
    if avg_word_length > 6.5 or len(words) > 8: return 3
    elif avg_word_length > 5.0: return 2
    return 1

def prepare_live_feature_matrix(candidates: list, query: str) -> pd.DataFrame:
    """
    Transforms live RRF candidates into the EXACT 10-feature schema used in training.
    """
    features = []
    query_intent = calculate_query_complexity(query)
    query_words = set(query.lower().split())
    
    for doc in candidates:
        # Extract base values
        dense_score = doc.get('dense_score', 0.0)
        sparse_score = doc.get('sparse_score', 0.0)
        age_days = calculate_age_in_days(doc.get('published_date', '2000-01-01'))
        citations = int(doc.get('citation_count', 0))
        title = doc.get('title', '')
        abstract = doc.get('text', '') # Our chunks are stored under 'text'
        
        # Calculate derived features
        age_years = max(1, age_days / 365.0)
        citation_velocity = citations / age_years
        readability = textstat.flesch_kincaid_grade(abstract) if abstract else 10.0
        
        title_words = set(title.lower().split())
        exact_match = 1 if len(query_words.intersection(title_words)) > 0 else 0
        
        # Simulated metrics (in prod, these would be pulled from Qdrant payload)
        h_index = doc.get('h_index', 5) 
        title_sparse = sparse_score * 0.4 if exact_match else 0.0
        divergence = abs(title_sparse - (sparse_score - title_sparse))
        semantic_ratio = dense_score / (sparse_score + 0.001)

        row = {
            'query_intent': query_intent,
            'dense_score': dense_score,
            'sparse_score': sparse_score,
            'doc_readability': readability,
            'exact_match_title': exact_match,
            'h_index': h_index,
            'citation_velocity': citation_velocity,
            'title_body_divergence': divergence,
            'semantic_lexical_ratio': semantic_ratio
        }
        features.append(row)
        
    # CRITICAL: Feature order must exactly match the DuckDB training schema!
    return pd.DataFrame(features)

# --- 3. The Core Inference Function ---
def rerank_candidates(candidates: list, query: str, top_k: int = 5) -> list:
    if not candidates:
        return []

    # 1. Feature Extraction
    df_features = prepare_live_feature_matrix(candidates, query)
    
    # 2. Convert to XGBoost DMatrix
    dmatrix = xgb.DMatrix(df_features)
    
    # 3. Predict new ranking scores
    xgb_scores = xgb_model.predict(dmatrix)
    
    # 4. Attach and re-sort
    for i, doc in enumerate(candidates):
        doc['xgb_score'] = float(xgb_scores[i])
        
    reranked_docs = sorted(candidates, key=lambda x: x['xgb_score'], reverse=True)
    
    return reranked_docs[:top_k]