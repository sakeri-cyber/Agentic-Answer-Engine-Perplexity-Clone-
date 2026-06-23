import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Prevents OpenMP errors on some systems
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads to avoid conflicts with asyncio

import xgboost as xgb
import pandas as pd
import textstat
from datetime import datetime

# --- 1. Load the Trained Model ---
print("Loading Trained XGBoost Re-ranker...")
xgb_model = xgb.Booster()
try:
    xgb_model.load_model("perplexity_ranker_v1.json")
    xgb_model.set_param({"nthread": 1})  # Ensure single-threaded operation for async compatibility
except Exception as e:
    print(f"Warning: Could not load model ({e}).")

# --- 2. Live Feature Engineering ---
def calculate_query_complexity(query: str) -> int:
    words = query.split()
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
    if avg_word_length > 6.5 or len(words) > 8: return 3
    elif avg_word_length > 5.0: return 2
    return 1

def prepare_live_feature_matrix(candidates: list, query: str) -> pd.DataFrame:
    """Transforms live RRF candidates into the EXACT 9-feature schema used in training."""
    features = []
    query_intent = calculate_query_complexity(query)
    query_words = set(query.lower().split())
    
    for doc in candidates:
        dense_score = doc.get('dense_score', 0.0)
        sparse_score = doc.get('sparse_score', 0.0)
        abstract = doc.get('text', '') 
        title = doc.get('title', '')
        
        # Mocking missing metadata for the live pipeline
        age_days = 15 
        citations = 50
        h_index = 5
        
        age_years = max(1, age_days / 365.0)
        citation_velocity = citations / age_years
        readability = textstat.flesch_kincaid_grade(abstract) if abstract else 10.0
        
        title_words = set(title.lower().split())
        exact_match = 1 if len(query_words.intersection(title_words)) > 0 else 0
        
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
        
    return pd.DataFrame(features)

# --- 3. Core Inference Function ---
def rerank_candidates(candidates: list, query: str, top_k: int = 5) -> list:
    if not candidates:
        return []

    df_features = prepare_live_feature_matrix(candidates, query)
    dmatrix = xgb.DMatrix(df_features)
    xgb_scores = xgb_model.predict(dmatrix)
    
    for i, doc in enumerate(candidates):
        doc['xgb_score'] = float(xgb_scores[i])
        
    reranked_docs = sorted(candidates, key=lambda x: x['xgb_score'], reverse=True)
    return reranked_docs[:top_k]