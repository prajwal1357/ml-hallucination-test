import os
import matplotlib
matplotlib.use("Agg")  # Run headlessly to prevent GUI popup errors
import matplotlib.pyplot as plt
import numpy as np


def generate_domain_plots(evaluation_results, output_dir="results"):
    """
    Generates a stacked bar chart showing the distribution of
    Context Adherence, Parametric Reversion, and Other behaviors across domains.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    domains = list(evaluation_results.keys())
    car_list = []
    prr_list = []
    odr_list = []

    for domain in domains:
        samples = evaluation_results[domain]
        total = len(samples)
        if total == 0:
            car_list.append(0)
            prr_list.append(0)
            odr_list.append(0)
            continue

        car = sum(1 for s in samples if s.get("score") == 1)
        prr = sum(1 for s in samples if s.get("score") == 0)
        odr = sum(1 for s in samples if s.get("score") == -1)

        car_list.append((car / total) * 100)
        prr_list.append((prr / total) * 100)
        odr_list.append((odr / total) * 100)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Harmonious colors
    color_car = "#2E86AB"  # Blueish
    color_prr = "#A23B72"  # Magenta/Purpleish
    color_odr = "#D9D9D9"  # Grayish

    bar_width = 0.5
    indices = np.arange(len(domains))

    # Stacked Bars
    ax.bar(indices, car_list, bar_width, label="Context Adherence (CAR)", color=color_car)
    ax.bar(indices, prr_list, bar_width, bottom=car_list, label="Parametric Reversion (PRR)", color=color_prr)
    
    bottom_odr = [c + p for c, p in zip(car_list, prr_list)]
    ax.bar(indices, odr_list, bar_width, bottom=bottom_odr, label="Other / Indeterminate (ODR)", color=color_odr)

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Cross-Domain LLM Behavior under Factual Conflict", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(indices)
    ax.set_xticklabels([d.capitalize() for d in domains], fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", framealpha=0.9)
    
    # Add grid lines
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Tight layout and save
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "behavior_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved domain behavior plot to: {plot_path}")


def generate_feature_importance_plot(ml_results, output_dir="results"):
    """
    Generates a horizontal bar chart showing the predictive feature importances
    extracted from the Random Forest classifier.
    """
    if not ml_results or "random_forest" not in ml_results:
        return

    importances = ml_results["random_forest"]["feature_importances"]
    
    # Sort features
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [f[0].replace("_", " ").capitalize() for f in sorted_features]
    values = [f[1] for f in sorted_features]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    color_bar = "#F26419"  # Vibrant orange
    
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, align="center", color=color_bar, alpha=0.9, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()  # Top-down feature list
    ax.set_xlabel("Relative Importance Score", fontsize=12)
    ax.set_title("Prompt Features Driving LLM Susceptibility to False Context", fontsize=13, fontweight="bold", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved feature importance plot to: {plot_path}")


if __name__ == "__main__":
    # Test plotting
    mock_results = {
        "medical": [{"score": 1}, {"score": 1}, {"score": 0}, {"score": -1}],
        "legal": [{"score": 1}, {"score": 0}, {"score": 0}, {"score": 0}],
        "general": [{"score": 1}, {"score": 1}, {"score": 1}, {"score": 0}],
        "finance": [{"score": 1}, {"score": 0}, {"score": -1}, {"score": -1}]
    }
    mock_ml = {
        "random_forest": {
            "feature_importances": {
                "semantic_similarity": 0.45,
                "context_word_count": 0.25,
                "len_diff": 0.10,
                "domain_legal": 0.08,
                "domain_medical": 0.07,
                "domain_general": 0.03,
                "domain_finance": 0.02
            }
        }
    }
    generate_domain_plots(mock_results)
    generate_feature_importance_plot(mock_ml)
