"""
Enhanced ML Analysis Script
============================
Reads existing results.json files from deepseek/, llama3/, and mistral/ directories.
Applies improved feature engineering, scaling, and hyperparameter-tuned classifiers
to produce higher-quality predictive analysis WITHOUT re-running any LLM queries.

Usage:
    python enhanced_analysis.py
"""

import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

# Try importing sentence-transformers
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not found. Falling back to TF-IDF.")


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def compute_semantic_similarity(texts_a, texts_b):
    """Computes pairwise cosine similarity using sentence-transformers or TF-IDF."""
    if not texts_a or not texts_b:
        return [0.0] * len(texts_a)

    if HAS_SENTENCE_TRANSFORMERS:
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings_a = model.encode(texts_a, convert_to_tensor=True)
            embeddings_b = model.encode(texts_b, convert_to_tensor=True)
            cosine_scores = util.cos_sim(embeddings_a, embeddings_b)
            return [float(cosine_scores[i][i]) for i in range(len(texts_a))]
        except Exception as e:
            print(f"SentenceTransformer error: {e}. Falling back to TF-IDF.")

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


def count_numeric_tokens(text):
    """Counts the number of numeric tokens (integers, decimals, percentages) in the text."""
    return len(re.findall(r'\b\d+[\.,]?\d*%?\b', str(text)))


def compute_flesch_kincaid(text):
    """Computes a simplified Flesch-Kincaid Grade Level for a text passage."""
    text = str(text)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = max(len(sentences), 1)

    words = text.split()
    num_words = max(len(words), 1)

    # Count syllables (simplified: count vowel groups per word)
    num_syllables = 0
    for word in words:
        word = word.lower()
        vowel_groups = re.findall(r'[aeiouy]+', word)
        num_syllables += max(len(vowel_groups), 1)

    grade = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
    return max(grade, 0.0)


def compute_answer_length_ratio(baseline_ans, perturbed_ans):
    """Ratio of the length of the perturbed answer to the baseline answer."""
    base_len = max(len(str(baseline_ans)), 1)
    pert_len = max(len(str(perturbed_ans)), 1)
    return pert_len / base_len


def compute_perturbation_position(baseline_ctx, perturbed_ctx):
    """
    Estimates where in the text the perturbation was introduced.
    Returns a value between 0.0 (beginning) and 1.0 (end).
    """
    base_words = str(baseline_ctx).split()
    pert_words = str(perturbed_ctx).split()
    if not base_words or not pert_words:
        return 0.5

    # Find first differing word position
    min_len = min(len(base_words), len(pert_words))
    for i in range(min_len):
        if base_words[i] != pert_words[i]:
            return i / max(min_len, 1)
    return 1.0


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def build_feature_dataframe(evaluation_results, model_name="unknown"):
    """
    Converts raw evaluation results into a feature-engineered DataFrame.
    Returns df with features and target column ready for ML.
    """
    df_rows = []

    for domain, samples in evaluation_results.items():
        for sample in samples:
            # Exclude inconclusive 'Other' samples
            if sample.get("score") == -1:
                continue

            baseline_ctx = sample.get("baseline_context") or ""
            perturbed_ctx = sample.get("perturbed_context") or ""
            baseline_ans = sample.get("baseline_answer") or ""
            perturbed_ans = sample.get("perturbed_answer") or ""

            df_rows.append({
                "model": model_name,
                "domain": domain,
                "baseline_context": baseline_ctx,
                "perturbed_context": perturbed_ctx,
                "baseline_answer": baseline_ans,
                "perturbed_answer": perturbed_ans,
                "target": sample["score"]
            })

    if not df_rows:
        return None

    df = pd.DataFrame(df_rows)

    # --- Core Features (Original) ---
    print(f"  Computing semantic similarity for {model_name}...")
    df["semantic_similarity"] = compute_semantic_similarity(
        df["baseline_context"].tolist(), df["perturbed_context"].tolist()
    )
    df["context_word_count"] = df["perturbed_context"].apply(lambda x: len(str(x).split()))
    df["len_diff"] = df.apply(
        lambda r: abs(len(str(r["baseline_context"]).split()) - len(str(r["perturbed_context"]).split())),
        axis=1
    )

    # --- New Engineered Features ---
    # 1. Readability / Text Complexity
    df["flesch_kincaid_grade"] = df["perturbed_context"].apply(compute_flesch_kincaid)

    # 2. Numeric Density (count of numbers in the perturbed context)
    df["numeric_density"] = df["perturbed_context"].apply(count_numeric_tokens)

    # 3. Answer Length Ratio (how much longer/shorter is the perturbed answer)
    df["answer_length_ratio"] = df.apply(
        lambda r: compute_answer_length_ratio(r["baseline_answer"], r["perturbed_answer"]),
        axis=1
    )

    # 4. Perturbation Position (where in the text the change was introduced)
    df["perturbation_position"] = df.apply(
        lambda r: compute_perturbation_position(r["baseline_context"], r["perturbed_context"]),
        axis=1
    )

    # 5. Question length
    df["question_length"] = df.apply(
        lambda r: len(str(r.get("baseline_context", "")).split()) if r["domain"] != "general" else 0,
        axis=1
    )

    # One-hot encode domain
    domain_dummies = pd.get_dummies(df["domain"], prefix="domain")
    for d in ["general", "medical", "legal", "finance"]:
        col = f"domain_{d}"
        if col not in domain_dummies.columns:
            domain_dummies[col] = 0

    df = pd.concat([df, domain_dummies], axis=1)

    return df


def get_feature_cols():
    """Returns the ordered list of feature column names."""
    return [
        "semantic_similarity",
        "context_word_count",
        "len_diff",
        "flesch_kincaid_grade",
        "numeric_density",
        "answer_length_ratio",
        "perturbation_position",
        "domain_general",
        "domain_medical",
        "domain_legal",
        "domain_finance"
    ]


def run_enhanced_analysis(df, model_name):
    """
    Trains improved ML classifiers with feature scaling and hyperparameter tuning.
    Returns a results dictionary with all metrics.
    """
    feature_cols = get_feature_cols()

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].astype(float)
    y = df["target"].astype(int)

    print(f"\n{'='*60}")
    print(f"  Enhanced ML Analysis for: {model_name}")
    print(f"  Dataset Size: {len(df)} usable samples (excluded ODR)")
    print(f"  Class Distribution: Adherence={int(y.sum())}, Reversion={int(len(y)-y.sum())}")
    print(f"{'='*60}")

    # Train-Test Split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {
        "dataset_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    # ---- 1. Random Forest with GridSearchCV ----
    print("\n  Training Random Forest (with GridSearchCV)...")
    rf_param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
    cv = StratifiedKFold(n_splits=min(5, min(int(y_train.sum()), int(len(y_train) - y_train.sum()))), shuffle=True, random_state=42)
    
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=0
    )
    rf_grid.fit(X_train_scaled, y_train)
    rf_best = rf_grid.best_estimator_
    y_pred_rf = rf_best.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf, zero_division=0)

    rf_importances = dict(zip(feature_cols, rf_best.feature_importances_))
    print(f"  Random Forest — Accuracy: {rf_acc:.2%}, F1: {rf_f1:.4f}")
    print(f"  Best Params: {rf_grid.best_params_}")

    results["random_forest"] = {
        "accuracy": float(rf_acc),
        "f1_score": float(rf_f1),
        "best_params": rf_grid.best_params_,
        "feature_importances": {k: round(float(v), 4) for k, v in rf_importances.items()},
        "report": classification_report(y_test, y_pred_rf, output_dict=True, zero_division=0)
    }

    # ---- 2. Logistic Regression (Scaled) ----
    print("\n  Training Logistic Regression (scaled)...")
    lr = LogisticRegression(max_iter=2000, random_state=42, C=1.0, solver="lbfgs")
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, y_pred_lr)
    lr_f1 = f1_score(y_test, y_pred_lr, zero_division=0)

    lr_coefs = dict(zip(feature_cols, lr.coef_[0]))
    print(f"  Logistic Regression — Accuracy: {lr_acc:.2%}, F1: {lr_f1:.4f}")

    results["logistic_regression"] = {
        "accuracy": float(lr_acc),
        "f1_score": float(lr_f1),
        "coefficients": {k: round(float(v), 4) for k, v in lr_coefs.items()},
        "report": classification_report(y_test, y_pred_lr, output_dict=True, zero_division=0)
    }

    # ---- 3. Gradient Boosting ----
    print("\n  Training Gradient Boosting Classifier...")
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    )
    gb.fit(X_train_scaled, y_train)
    y_pred_gb = gb.predict(X_test_scaled)
    gb_acc = accuracy_score(y_test, y_pred_gb)
    gb_f1 = f1_score(y_test, y_pred_gb, zero_division=0)

    gb_importances = dict(zip(feature_cols, gb.feature_importances_))
    print(f"  Gradient Boosting — Accuracy: {gb_acc:.2%}, F1: {gb_f1:.4f}")

    results["gradient_boosting"] = {
        "accuracy": float(gb_acc),
        "f1_score": float(gb_f1),
        "feature_importances": {k: round(float(v), 4) for k, v in gb_importances.items()},
        "report": classification_report(y_test, y_pred_gb, output_dict=True, zero_division=0)
    }

    # ---- 4. Support Vector Machine ----
    print("\n  Training SVM (RBF kernel)...")
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, y_pred_svm)
    svm_f1 = f1_score(y_test, y_pred_svm, zero_division=0)
    print(f"  SVM (RBF) — Accuracy: {svm_acc:.2%}, F1: {svm_f1:.4f}")

    results["svm"] = {
        "accuracy": float(svm_acc),
        "f1_score": float(svm_f1),
        "report": classification_report(y_test, y_pred_svm, output_dict=True, zero_division=0)
    }

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    model_dirs = {
        "DeepSeek-R1 (8B)": "deepseek",
        "Llama-3 (8B)": "llama3",
        "Mistral (7B)": "mistral"
    }

    all_results = {}

    for model_name, dir_name in model_dirs.items():
        results_path = os.path.join(dir_name, "results.json")
        if not os.path.exists(results_path):
            print(f"\nSkipping {model_name}: {results_path} not found.")
            continue

        print(f"\n{'#'*60}")
        print(f"# Loading results for: {model_name}")
        print(f"{'#'*60}")

        with open(results_path, "r", encoding="utf-8") as f:
            evaluation_results = json.load(f)

        df = build_feature_dataframe(evaluation_results, model_name)
        if df is None or len(df) < 10:
            print(f"  Not enough usable samples for {model_name}. Skipping.")
            continue

        model_results = run_enhanced_analysis(df, model_name)
        all_results[model_name] = model_results

    # Save consolidated results
    output_path = os.path.join("enhanced_ml_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n\nAll enhanced ML results saved to: {output_path}")

    # Print summary comparison table
    print(f"\n{'='*80}")
    print("ENHANCED ML MODEL COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<22} {'Classifier':<25} {'Accuracy':>10} {'F1 Score':>10}")
    print(f"{'-'*22} {'-'*25} {'-'*10} {'-'*10}")
    for model_name, res in all_results.items():
        for clf_name in ["random_forest", "logistic_regression", "gradient_boosting", "svm"]:
            if clf_name in res:
                clf_display = clf_name.replace("_", " ").title()
                acc = res[clf_name]["accuracy"]
                f1 = res[clf_name]["f1_score"]
                print(f"{model_name:<22} {clf_display:<25} {acc:>9.2%} {f1:>9.4f}")
        print()
