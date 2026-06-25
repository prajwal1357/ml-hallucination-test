import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Try importing sentence-transformers, fall back to TF-IDF if unavailable
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not found. Falling back to TF-IDF cosine similarity.")


def compute_semantic_similarity(texts_a, texts_b):
    """Computes similarity between list of original contexts and perturbed contexts."""
    if not texts_a or not texts_b:
        return [0.0] * len(texts_a)

    if HAS_SENTENCE_TRANSFORMERS:
        try:
            # Using a very light and fast embedding model
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings_a = model.encode(texts_a, convert_to_tensor=True)
            embeddings_b = model.encode(texts_b, convert_to_tensor=True)
            cosine_scores = util.cos_sim(embeddings_a, embeddings_b)
            # Extract diagonal scores (matching pairs)
            return [float(cosine_scores[i][i]) for i in range(len(texts_a))]
        except Exception as e:
            print(f"Error in SentenceTransformer embedding: {e}. Falling back to TF-IDF.")
            
    # Fallback: TF-IDF similarity for each pair
    similarities = []
    for ta, tb in zip(texts_a, texts_b):
        try:
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
            tfidf = vectorizer.fit_transform([ta, tb])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            similarities.append(float(sim))
        except Exception:
            similarities.append(0.0)
    return similarities


def run_predictive_analysis(evaluation_results):
    """
    Given a list of evaluated samples, trains scikit-learn models to predict
    adherence (1) vs. reversion (0).
    """
    df_rows = []
    
    # 1. Gather raw inputs
    for domain, samples in evaluation_results.items():
        for sample in samples:
            # Exclude inconclusive 'Other' samples from predictive model
            if sample.get("score") == -1:
                continue
                
            df_rows.append({
                "domain": domain,
                "question": sample["question"],
                "baseline_context": sample["baseline_context"] or "",
                "perturbed_context": sample["perturbed_context"] or "",
                "baseline_answer": sample["baseline_answer"] or "",
                "perturbed_answer": sample["perturbed_answer"] or "",
                "target": sample["score"]  # 1 = Adherence, 0 = Reversion
            })
            
    if len(df_rows) < 10:
        print(f"Not enough samples ({len(df_rows)}) to train ML model. Need at least 10 samples.")
        return None

    df = pd.DataFrame(df_rows)

    # 2. Extract Features
    print("Computing semantic similarity features...")
    df["semantic_similarity"] = compute_semantic_similarity(
        df["baseline_context"].tolist(), df["perturbed_context"].tolist()
    )
    
    df["context_word_count"] = df["perturbed_context"].apply(lambda x: len(str(x).split()))
    
    # Word count difference
    df["len_diff"] = df.apply(
        lambda r: abs(len(str(r["baseline_context"]).split()) - len(str(r["perturbed_context"]).split())),
        axis=1
    )

    # One-hot encode domain
    domain_dummies = pd.get_dummies(df["domain"], prefix="domain")
    
    # Ensure all four domains are present in features
    for d in ["general", "medical", "legal", "finance"]:
        col = f"domain_{d}"
        if col not in domain_dummies.columns:
            domain_dummies[col] = 0
            
    df = pd.concat([df, domain_dummies], axis=1)

    # Define Feature matrix X and Target vector y
    feature_cols = [
        "semantic_similarity",
        "context_word_count",
        "len_diff",
        "domain_general",
        "domain_medical",
        "domain_legal",
        "domain_finance"
    ]
    
    X = df[feature_cols].astype(float)
    y = df["target"].astype(int)

    # 3. Train-Test Split (with stratify if classes are balanced enough, fallback to simple split)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    print(f"Dataset Size: {len(df)} samples (Train: {len(X_train)}, Test: {len(X_test)})")
    
    # 4. Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred_rf)
    
    # Extract Feature Importances
    rf_importances = dict(zip(feature_cols, rf.feature_importances_))

    # 5. Train Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, y_pred_lr)
    
    # Extract Coefficients
    lr_coefs = dict(zip(feature_cols, lr.coef_[0]))

    # Print baseline classification reports
    print("\n--- ML Model Performance ---")
    print(f"Random Forest Test Accuracy: {rf_acc:.2%}")
    print(f"Logistic Regression Test Accuracy: {lr_acc:.2%}")

    # Return summary dict
    return {
        "dataset_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "random_forest": {
            "accuracy": float(rf_acc),
            "feature_importances": {k: float(v) for k, v in rf_importances.items()},
            "report": classification_report(y_test, y_pred_rf, output_dict=True, zero_division=0)
        },
        "logistic_regression": {
            "accuracy": float(lr_acc),
            "coefficients": {k: float(v) for k, v in lr_coefs.items()},
            "report": classification_report(y_test, y_pred_lr, output_dict=True, zero_division=0)
        }
    }


if __name__ == "__main__":
    # Test analysis script
    mock_results = {
        "medical": [
            {"question": "Q1", "baseline_context": "Effective drug A.", "perturbed_context": "Ineffective drug A.", "baseline_answer": "yes", "perturbed_answer": "no", "score": 1},
            {"question": "Q2", "baseline_context": "No change.", "perturbed_context": "Flipped positive.", "baseline_answer": "no", "perturbed_answer": "yes", "score": 0},
            {"question": "Q3", "baseline_context": "Effective.", "perturbed_context": "Ineffective.", "baseline_answer": "yes", "perturbed_answer": "no", "score": 1},
            {"question": "Q4", "baseline_context": "None.", "perturbed_context": "Flipped.", "baseline_answer": "no", "perturbed_answer": "yes", "score": 1},
            {"question": "Q5", "baseline_context": "Test.", "perturbed_context": "Test.", "baseline_answer": "yes", "perturbed_answer": "no", "score": -1} # Should be ignored
        ],
        "legal": [
            {"question": "L1", "baseline_context": "Limits.", "perturbed_context": "Unlimited.", "baseline_answer": "Yes", "perturbed_answer": "No", "score": 0},
            {"question": "L2", "baseline_context": "Limits.", "perturbed_context": "Unlimited.", "baseline_answer": "Yes", "perturbed_answer": "No", "score": 0},
            {"question": "L3", "baseline_context": "Limits.", "perturbed_context": "Unlimited.", "baseline_answer": "Yes", "perturbed_answer": "No", "score": 1},
            {"question": "L4", "baseline_context": "Limits.", "perturbed_context": "Unlimited.", "baseline_answer": "Yes", "perturbed_answer": "No", "score": 0},
            {"question": "L5", "baseline_context": "Limits.", "perturbed_context": "Unlimited.", "baseline_answer": "Yes", "perturbed_answer": "No", "score": 0},
            {"question": "L6", "baseline_context": "Limits.", "perturbed_context": "Unlimited.", "baseline_answer": "Yes", "perturbed_answer": "No", "score": 1}
        ]
    }
    stats = run_predictive_analysis(mock_results)
    print("\nPredictive Analysis Output:")
    print(json.dumps(stats, indent=4))
