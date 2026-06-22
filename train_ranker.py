import duckdb
import xgboost as xgb
import pandas as pd

def train_xgboost_model():
    print("Connecting to DuckDB...")
    con = duckdb.connect('training_data.duckdb')
    
    # 1. Extract the Feature Matrix and Target Labels
    # We explicitly exclude query_id, document_id, query_text, and llm_reasoning 
    # because XGBoost only understands raw numbers, not strings or IDs.
    print("Extracting feature matrix...")
    df = con.execute("""
        SELECT 
            query_intent,
            dense_score,
            sparse_score,
            doc_readability,
            exact_match_title,
            h_index,
            citation_velocity,
            title_body_divergence,
            semantic_lexical_ratio,
            relevance_label
        FROM ltr_features
        WHERE relevance_label IS NOT NULL
    """).df()
    
    con.close()

    if df.empty:
        print("Error: No training data found in DuckDB.")
        return

    # 2. Prepare X (Features) and y (Target)
    X = df.drop('relevance_label', axis=1)
    y = df['relevance_label']
    
    # 3. Create XGBoost DMatrix
    # Note: For true LTR, XGBoost expects a 'group' array telling it how many 
    # documents belong to each query, but for this simplified portfolio script, 
    # we will use pairwise ranking natively handled by the DMatrix.
    print(f"Training on {len(X)} query-document pairs...")
    dtrain = xgb.DMatrix(X, label=y)
    
    # 4. Set LambdaMART / NDCG Parameters
    params = {
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg',
        'learning_rate': 0.1,
        'max_depth': 4, # Keep trees shallow to prevent overfitting on synthetic data
        'tree_method': 'hist' # Highly efficient histogram-based algorithm
    }
    
    # 5. Train the Model
    print("Running gradient boosting...")
    model = xgb.train(params, dtrain, num_boost_round=50)
    
    # 6. Save the Model
    model_path = "perplexity_ranker_v1.json"
    model.save_model(model_path)
    print(f"Model successfully saved to {model_path}!")
    
    # Optional: Print Feature Importance so you can explain it in your interview
    importance = model.get_score(importance_type='gain')
    print("\nFeature Importance (Gain):")
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f" - {feature}: {score:.2f}")

if __name__ == "__main__":
    train_xgboost_model()
    xgb.plot_importance(model)
    plt.savefig("feature_importance.png")