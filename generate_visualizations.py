"""
Enhanced Visualization Script
==============================
Generates publication-quality charts from existing results.json and enhanced_ml_results.json.
Saves all plots to enhanced_plots/ directory.

Usage:
    python generate_visualizations.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use("Agg")

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

OUTPUT_DIR = "enhanced_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = ["DeepSeek-R1 (8B)", "Llama-3 (8B)", "Mistral (7B)"]
MODEL_DIRS = {"DeepSeek-R1 (8B)": "deepseek", "Llama-3 (8B)": "llama3", "Mistral (7B)": "mistral"}
DOMAINS = ["General", "Medical", "Legal", "Finance"]
MODEL_COLORS = {"DeepSeek-R1 (8B)": "#6366f1", "Llama-3 (8B)": "#f59e0b", "Mistral (7B)": "#10b981"}
BEHAVIOR_COLORS = {"CAR": "#ef4444", "PRR": "#3b82f6", "ODR": "#9ca3af"}


# ============================================================================
# LOAD DATA
# ============================================================================
def load_behavior_data():
    """Load raw behavior metrics from results.json files."""
    behavior = {}
    for model_name, dir_name in MODEL_DIRS.items():
        path = os.path.join(dir_name, "results.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        behavior[model_name] = {}
        for domain, samples in data.items():
            total = len(samples)
            car = sum(1 for s in samples if s.get("score") == 1)
            prr = sum(1 for s in samples if s.get("score") == 0)
            odr = sum(1 for s in samples if s.get("score") == -1)
            behavior[model_name][domain.capitalize()] = {
                "CAR": car / total * 100 if total else 0,
                "PRR": prr / total * 100 if total else 0,
                "ODR": odr / total * 100 if total else 0,
                "total": total
            }
    return behavior


def load_ml_results():
    """Load enhanced ML results."""
    with open("enhanced_ml_results.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# PLOT 1: Cross-Model Domain Behavior Comparison (Grouped Stacked Bars)
# ============================================================================
def plot_cross_model_behavior(behavior):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    fig.suptitle("Cross-Model Behavior Distribution by Domain", fontsize=16, fontweight="bold", y=1.02)

    bar_width = 0.55
    models = list(behavior.keys())

    for i, domain in enumerate(DOMAINS):
        ax = axes[i]
        car_vals = [behavior[m].get(domain, {}).get("CAR", 0) for m in models]
        prr_vals = [behavior[m].get(domain, {}).get("PRR", 0) for m in models]
        odr_vals = [behavior[m].get(domain, {}).get("ODR", 0) for m in models]

        x = np.arange(len(models))
        bars1 = ax.bar(x, car_vals, bar_width, label="Context Adherence", color=BEHAVIOR_COLORS["CAR"], edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x, prr_vals, bar_width, bottom=car_vals, label="Parametric Reversion", color=BEHAVIOR_COLORS["PRR"], edgecolor="white", linewidth=0.5)
        bars3 = ax.bar(x, odr_vals, bar_width, bottom=[c+p for c,p in zip(car_vals, prr_vals)], label="Other/Indeterminate", color=BEHAVIOR_COLORS["ODR"], edgecolor="white", linewidth=0.5)

        # Add percentage labels on bars
        for j, (c, p, o) in enumerate(zip(car_vals, prr_vals, odr_vals)):
            if c > 8:
                ax.text(j, c/2, f"{c:.0f}%", ha="center", va="center", fontsize=8, fontweight="bold", color="white")
            if p > 8:
                ax.text(j, c + p/2, f"{p:.0f}%", ha="center", va="center", fontsize=8, fontweight="bold", color="white")
            if o > 8:
                ax.text(j, c + p + o/2, f"{o:.0f}%", ha="center", va="center", fontsize=8, fontweight="bold", color="white")

        ax.set_title(domain, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["DeepSeek", "Llama-3", "Mistral"], rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, 105)
        if i == 0:
            ax.set_ylabel("Percentage (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.98), frameon=True, fancybox=True, shadow=True)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(OUTPUT_DIR, "cross_model_behavior.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# PLOT 2: Classifier Accuracy Comparison (Grouped Bar Chart)
# ============================================================================
def plot_classifier_comparison(ml_results):
    classifiers = ["random_forest", "logistic_regression", "gradient_boosting", "svm"]
    clf_labels = ["Random\nForest", "Logistic\nRegression", "Gradient\nBoosting", "SVM\n(RBF)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Enhanced Classifier Performance Comparison", fontsize=16, fontweight="bold")

    x = np.arange(len(classifiers))
    width = 0.22

    # Accuracy subplot
    for i, model in enumerate(MODELS):
        if model not in ml_results:
            continue
        accs = [ml_results[model].get(clf, {}).get("accuracy", 0) * 100 for clf in classifiers]
        bars = ax1.bar(x + i * width - width, accs, width, label=model, color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, accs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax1.set_title("Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(clf_labels, fontsize=9)
    ax1.set_ylim(50, 90)
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend(fontsize=9, loc="lower right")

    # F1 Score subplot
    for i, model in enumerate(MODELS):
        if model not in ml_results:
            continue
        f1s = [ml_results[model].get(clf, {}).get("f1_score", 0) for clf in classifiers]
        bars = ax2.bar(x + i * width - width, f1s, width, label=model, color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, f1s):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax2.set_title("F1 Score")
    ax2.set_xticks(x)
    ax2.set_xticklabels(clf_labels, fontsize=9)
    ax2.set_ylim(0.65, 0.92)
    ax2.set_ylabel("F1 Score")
    ax2.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "classifier_comparison.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# PLOT 3: Feature Importance Heatmap (Random Forest)
# ============================================================================
def plot_rf_feature_heatmap(ml_results):
    feature_order = [
        "context_word_count", "flesch_kincaid_grade", "perturbation_position",
        "semantic_similarity", "len_diff", "numeric_density",
        "answer_length_ratio", "domain_general", "domain_legal",
        "domain_finance", "domain_medical"
    ]
    feature_labels = [
        "Context Word Count", "Flesch-Kincaid Grade", "Perturbation Position",
        "Semantic Similarity", "Length Difference", "Numeric Density",
        "Answer Length Ratio", "Domain: General", "Domain: Legal",
        "Domain: Finance", "Domain: Medical"
    ]

    data = []
    model_labels = []
    for model in MODELS:
        if model not in ml_results:
            continue
        imp = ml_results[model].get("random_forest", {}).get("feature_importances", {})
        row = [imp.get(f, 0) for f in feature_order]
        data.append(row)
        model_labels.append(model)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        data, annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=feature_labels, yticklabels=model_labels,
        linewidths=1, linecolor="white", cbar_kws={"label": "Importance Score"},
        ax=ax
    )
    ax.set_title("Random Forest Feature Importance Heatmap (Tuned)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticklabels(feature_labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(model_labels, rotation=0, fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "rf_feature_heatmap.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# PLOT 4: Gradient Boosting Feature Importance Heatmap
# ============================================================================
def plot_gb_feature_heatmap(ml_results):
    feature_order = [
        "context_word_count", "len_diff", "perturbation_position",
        "flesch_kincaid_grade", "numeric_density", "semantic_similarity",
        "answer_length_ratio", "domain_legal", "domain_medical",
        "domain_finance", "domain_general"
    ]
    feature_labels = [
        "Context Word Count", "Length Difference", "Perturbation Position",
        "Flesch-Kincaid Grade", "Numeric Density", "Semantic Similarity",
        "Answer Length Ratio", "Domain: Legal", "Domain: Medical",
        "Domain: Finance", "Domain: General"
    ]

    data = []
    model_labels = []
    for model in MODELS:
        if model not in ml_results:
            continue
        imp = ml_results[model].get("gradient_boosting", {}).get("feature_importances", {})
        row = [imp.get(f, 0) for f in feature_order]
        data.append(row)
        model_labels.append(model)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        data, annot=True, fmt=".3f", cmap="PuBuGn",
        xticklabels=feature_labels, yticklabels=model_labels,
        linewidths=1, linecolor="white", cbar_kws={"label": "Importance Score"},
        ax=ax
    )
    ax.set_title("Gradient Boosting Feature Importance Heatmap", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticklabels(feature_labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(model_labels, rotation=0, fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "gb_feature_heatmap.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# PLOT 5: Logistic Regression Coefficient Diverging Heatmap
# ============================================================================
def plot_lr_coefficients_heatmap(ml_results):
    feature_order = [
        "domain_general", "domain_finance", "numeric_density",
        "answer_length_ratio", "perturbation_position", "context_word_count",
        "len_diff", "flesch_kincaid_grade", "domain_medical",
        "semantic_similarity", "domain_legal"
    ]
    feature_labels = [
        "Domain: General", "Domain: Finance", "Numeric Density",
        "Answer Length Ratio", "Perturbation Position", "Context Word Count",
        "Length Difference", "Flesch-Kincaid Grade", "Domain: Medical",
        "Semantic Similarity", "Domain: Legal"
    ]

    data = []
    model_labels = []
    for model in MODELS:
        if model not in ml_results:
            continue
        coefs = ml_results[model].get("logistic_regression", {}).get("coefficients", {})
        row = [coefs.get(f, 0) for f in feature_order]
        data.append(row)
        model_labels.append(model)

    data = np.array(data)
    vmax = max(abs(data.min()), abs(data.max()))

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        data, annot=True, fmt="+.3f", cmap="RdBu_r", center=0,
        vmin=-vmax, vmax=vmax,
        xticklabels=feature_labels, yticklabels=model_labels,
        linewidths=1, linecolor="white",
        cbar_kws={"label": "Coefficient (+ = Adherence, − = Reversion)"},
        ax=ax
    )
    ax.set_title("Logistic Regression Coefficients (Scaled Features)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticklabels(feature_labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(model_labels, rotation=0, fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "lr_coefficients_heatmap.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# PLOT 6: Radar / Spider Chart - Feature Profile per Model
# ============================================================================
def plot_radar_chart(ml_results):
    features = [
        "context_word_count", "semantic_similarity", "flesch_kincaid_grade",
        "perturbation_position", "len_diff", "numeric_density", "answer_length_ratio"
    ]
    labels = [
        "Context\nWord Count", "Semantic\nSimilarity", "Flesch-Kincaid\nGrade",
        "Perturbation\nPosition", "Length\nDifference", "Numeric\nDensity", "Answer Length\nRatio"
    ]

    N = len(features)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)

    for model in MODELS:
        if model not in ml_results:
            continue
        imp = ml_results[model].get("random_forest", {}).get("feature_importances", {})
        values = [imp.get(f, 0) for f in features]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model, color=MODEL_COLORS[model])
        ax.fill(angles, values, alpha=0.1, color=MODEL_COLORS[model])

    ax.set_ylim(0, 0.25)
    ax.set_title("Feature Importance Radar — Random Forest (Tuned)", fontsize=14, fontweight="bold", y=1.12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=10, frameon=True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_radar.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# PLOT 7: CAR Rate Comparison Across Models (Line Chart)
# ============================================================================
def plot_car_line_comparison(behavior):
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(DOMAINS))
    for model in MODELS:
        if model not in behavior:
            continue
        car_rates = [behavior[model].get(d, {}).get("CAR", 0) for d in DOMAINS]
        ax.plot(x, car_rates, "o-", linewidth=2.5, markersize=10, label=model, color=MODEL_COLORS[model])
        for xi, val in zip(x, car_rates):
            ax.annotate(f"{val:.0f}%", (xi, val), textcoords="offset points",
                       xytext=(0, 12), ha="center", fontsize=9, fontweight="bold", color=MODEL_COLORS[model])

    ax.set_xticks(x)
    ax.set_xticklabels(DOMAINS, fontsize=12)
    ax.set_ylabel("Context Adherence Rate (%)", fontsize=12)
    ax.set_title("Context Adherence Rate (CAR) Across Domains", fontsize=14, fontweight="bold")
    ax.set_ylim(-5, 110)
    ax.legend(fontsize=11, loc="lower left", frameon=True, fancybox=True)
    ax.axhline(y=50, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "car_line_comparison.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# PLOT 8: Feature Importance Bar Chart (Averaged Across Models)
# ============================================================================
def plot_avg_feature_importance(ml_results):
    features = [
        "context_word_count", "flesch_kincaid_grade", "perturbation_position",
        "len_diff", "semantic_similarity", "numeric_density",
        "answer_length_ratio", "domain_legal", "domain_general",
        "domain_finance", "domain_medical"
    ]
    labels = [
        "Context Word Count", "Flesch-Kincaid Grade", "Perturbation Position",
        "Length Difference", "Semantic Similarity", "Numeric Density",
        "Answer Length Ratio", "Domain: Legal", "Domain: General",
        "Domain: Finance", "Domain: Medical"
    ]

    avg_vals = []
    for f in features:
        vals = []
        for model in MODELS:
            if model in ml_results:
                imp = ml_results[model].get("random_forest", {}).get("feature_importances", {})
                vals.append(imp.get(f, 0))
        avg_vals.append(np.mean(vals) if vals else 0)

    # Sort by importance
    sorted_pairs = sorted(zip(labels, avg_vals), key=lambda x: x[1], reverse=True)
    sorted_labels, sorted_vals = zip(*sorted_pairs)

    # Color: new features in orange, original in blue, domain in gray
    new_features = {"Flesch-Kincaid Grade", "Numeric Density", "Answer Length Ratio", "Perturbation Position"}
    domain_features = {"Domain: Legal", "Domain: General", "Domain: Finance", "Domain: Medical"}
    colors = []
    for l in sorted_labels:
        if l in new_features:
            colors.append("#f59e0b")
        elif l in domain_features:
            colors.append("#9ca3af")
        else:
            colors.append("#3b82f6")

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(sorted_labels)), sorted_vals, color=colors, edgecolor="white", linewidth=0.5, height=0.65)

    for bar, val in zip(bars, sorted_vals):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2, f"{val:.4f}",
                va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Average Importance Score", fontsize=12)
    ax.set_title("Average Feature Importance Across All Models (Random Forest)", fontsize=14, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3b82f6", label="Original Features"),
        Patch(facecolor="#f59e0b", label="New Engineered Features"),
        Patch(facecolor="#9ca3af", label="Domain Indicators")
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10, frameon=True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "avg_feature_importance.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("Loading data...")
    behavior = load_behavior_data()
    ml_results = load_ml_results()

    print("\nGenerating visualizations...")

    print("\n1. Cross-Model Domain Behavior Comparison")
    plot_cross_model_behavior(behavior)

    print("2. Classifier Accuracy & F1 Comparison")
    plot_classifier_comparison(ml_results)

    print("3. Random Forest Feature Importance Heatmap")
    plot_rf_feature_heatmap(ml_results)

    print("4. Gradient Boosting Feature Importance Heatmap")
    plot_gb_feature_heatmap(ml_results)

    print("5. Logistic Regression Coefficients Diverging Heatmap")
    plot_lr_coefficients_heatmap(ml_results)

    print("6. Feature Importance Radar Chart")
    plot_radar_chart(ml_results)

    print("7. CAR Line Comparison Across Domains")
    plot_car_line_comparison(behavior)

    print("8. Average Feature Importance Bar Chart")
    plot_avg_feature_importance(ml_results)

    print(f"\nAll visualizations saved to: {OUTPUT_DIR}/")
    print(f"Total charts generated: 8")
