import duckdb
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

def train_xgboost_model():
    print("Connecting to DuckDB...")
    con = duckdb.connect('training_data.duckdb')
    
    # 1. Extract the Feature Matrix and Target Labels
    # We strictly exclude query_id, document_id, query_text, and llm_reasoning
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
    
    print(f"Training on {len(X)} query-document pairs...")
    
    # 3. Create XGBoost DMatrix
    dtrain = xgb.DMatrix(X, label=y)
    
    # 4. Set LambdaMART / NDCG Parameters
    params = {
        'objective': 'rank:ndcg',    # Learning to Rank objective
        'eval_metric': 'ndcg',       # Normalized Discounted Cumulative Gain
        'learning_rate': 0.1,
        'max_depth': 4,              # Keep shallow to prevent overfitting on small data
        'tree_method': 'hist'
    }
    
    # 5. Train the Model
    print("Running gradient boosting...")
    model = xgb.train(params, dtrain, num_boost_round=50)
    
    # 6. Save the Model File
    model_path = "perplexity_ranker_v1.json"
    model.save_model(model_path)
    print(f"✅ Model successfully saved to {model_path}!")
    
    # 7. Generate Feature Importance Dashboard
    print("\nGenerating Feature Importance Graph...")
    importance = model.get_score(importance_type='gain')
    
    # Print to console
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f" - {feature}: {score:.2f}")
        
    # Save to image
    xgb.plot_importance(model, importance_type='gain', title='XGBoost Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("📊 Dashboard saved as 'feature_importance.png'")

if __name__ == "__main__":
    train_xgboost_model()