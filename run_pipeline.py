import os
import json
import argparse
from datetime import datetime
from src.loaders import load_all_domains
from src.generator import evaluate_sample
from src.evaluator import evaluate_sample_result
from src.analysis import run_predictive_analysis
from src.plots import generate_domain_plots, generate_feature_importance_plot


def create_markdown_report(model_name, evaluation_results, ml_results, output_dir):
    """Generates a detailed summary Markdown report of the experiment findings."""
    report_path = os.path.join(output_dir, "report.md")
    
    # Calculate metrics per domain
    domain_rows = []
    for domain, samples in evaluation_results.items():
        total = len(samples)
        if total == 0:
            continue
        car = sum(1 for s in samples if s.get("score") == 1)
        prr = sum(1 for s in samples if s.get("score") == 0)
        odr = sum(1 for s in samples if s.get("score") == -1)
        
        domain_rows.append(
            f"| {domain.capitalize()} | {total} | {car} ({car/total:.1%}) | {prr} ({prr/total:.1%}) | {odr} ({odr/total:.1%}) |"
        )
        
    domain_table = "\n".join(domain_rows)
    
    # Extract ML findings
    ml_section = ""
    if ml_results:
        rf_acc = ml_results["random_forest"]["accuracy"]
        lr_acc = ml_results["logistic_regression"]["accuracy"]
        
        importances = ml_results["random_forest"]["feature_importances"]
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        imp_rows = [f"| {feat.replace('_', ' ').capitalize()} | {score:.4f} |" for feat, score in sorted_imp]
        imp_table = "\n".join(imp_rows)
        
        coefs = ml_results["logistic_regression"]["coefficients"]
        sorted_coef = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
        coef_rows = [f"| {feat.replace('_', ' ').capitalize()} | {val:.4f} |" for feat, val in sorted_coef]
        coef_table = "\n".join(coef_rows)
        
        ml_section = f"""
## 2. Predictive Machine Learning Analysis

We trained classical ML classifiers (`scikit-learn`) on the prompt characteristics to predict whether the model would succumb to false contexts (`1`) or stick to memory (`0`).

*   **Random Forest Classifier Accuracy**: `{rf_acc:.2%}`
*   **Logistic Regression Classifier Accuracy**: `{lr_acc:.2%}`

### Feature Importances (Random Forest)
This table shows which factors were most influential in predicting whether the model believed the false context:

| Feature | Importance Score |
| :--- | :---: |
{imp_table}

### Feature Coefficients (Logistic Regression)
Positive values indicate features that drive the model towards **Context Adherence** (believing the false context), whereas negative values drive it towards **Parametric Reversion** (sticking to real-world truth):

| Feature | Regression Coefficient |
| :--- | :---: |
{coef_table}
"""

    report_content = f"""# Cross-Domain Factual Conflict Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model Tested: `{model_name}`

---

## 1. Domain Performance Summary

This table compares model behaviors under factual conflicts.
- **Context Adherence Rate (CAR)**: Model believed the false/perturbed context.
- **Parametric Reversion Rate (PRR)**: Model ignored the false context and stuck to its memory.
- **Other/Indeterminate (ODR)**: Model outputted nonsense, got confused, or refused to answer.

| Domain | Total Samples | Context Adherence (CAR) | Parametric Reversion (PRR) | Other/Indeterminate (ODR) |
| :--- | :---: | :---: | :---: | :---: |
{domain_table}

### Key Visualization
![Cross-Domain Behavior Chart](behavior_distribution.png)

---
{ml_section}

### Key Visualization
![ML Feature Importance Chart](feature_importance.png)

---

## 3. Executive Summary & Findings
1. **Domain Vulnerability**: Look at which domains have the highest CAR (Context Adherence Rate). Typically, models show higher adherence in domains like Legal and Finance due to a lack of strong pre-trained parametric safety alignment on those specific, dense text passages.
2. **Parametric Resistance**: Observe PRR (Parametric Reversion Rate). In General Knowledge (TruthfulQA) and Medical domains, the model's pre-trained weights often provide strong "cognitive resistance" to falsehoods.
3. **ML Classifier Insights**: Look at the top features in Section 2. If **Semantic Similarity** has the highest importance, it indicates that the model is highly sensitive to the plausibility of the lie (more likely to believe lies that sound close to the truth).
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content.strip())
    print(f"\nGenerated research report at: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Cross-Domain Factual Conflict Analysis Pipeline")
    parser.add_argument("--model", type=str, default="mistral", help="Ollama model to evaluate (e.g. mistral, llama3)")
    parser.add_argument("--evaluator", type=str, default="mistral", help="Ollama model to use as LLM-as-a-judge")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to evaluate per domain")
    parser.add_argument("--force-mutate", action="store_true", help="Force regenerate mutated datasets")
    parser.add_argument("--output", type=str, default="results", help="Directory to save output files")
    args = parser.parse_args()

    print(f"==================================================")
    print(f"Starting Cross-Domain Conflict Analysis Pipeline")
    print(f"Target Model: {args.model}")
    print(f"Evaluator Model: {args.evaluator}")
    print(f"Sample Limit per Domain: {args.limit}")
    print(f"==================================================")

    # 1. Load data
    data_dict = load_all_domains(limit=args.limit, force_mutate=args.force_mutate, mutator_model=args.model)

    # 2. Run Generation and Evaluation loop
    # Create results folder first
    os.makedirs(args.output, exist_ok=True)
    raw_results_path = os.path.join(args.output, "results.json")

    # 2. Load existing checkpoints if they exist
    evaluation_results = {}
    if os.path.exists(raw_results_path):
        try:
            with open(raw_results_path, "r", encoding="utf-8") as f:
                evaluation_results = json.load(f)
            print(f"Loaded existing checkpoint from {raw_results_path}.")
        except Exception as e:
            print(f"Warning: Could not parse existing checkpoint {raw_results_path}: {e}")
            evaluation_results = {}

    import sys
    
    try:
        for domain, samples in data_dict.items():
            print(f"\nRunning baseline vs. conflict query pipeline for domain: {domain.upper()}")
            if domain not in evaluation_results:
                evaluation_results[domain] = []
            
            for idx, sample in enumerate(samples):
                # Check if this sample has already been successfully evaluated in this run
                already_evaluated = None
                for existing in evaluation_results[domain]:
                    if (existing.get("question") == sample["question"] and 
                        existing.get("baseline_context") == sample["baseline_context"]):
                        # Retain only if it was successfully queried (no ERROR in outputs)
                        if "ERROR" not in str(existing.get("model_baseline_output", "")) and "ERROR" not in str(existing.get("model_conflict_output", "")):
                            already_evaluated = existing
                            break
                
                if already_evaluated is not None:
                    print(f"  [{idx+1}/{len(samples)}] Skipping (already evaluated: {already_evaluated.get('classification', 'Unknown')})")
                    continue

                print(f"  [{idx+1}/{len(samples)}] Querying Ollama...")
                # Query the target model on baseline vs. conflict prompt configurations
                sample_result = evaluate_sample(args.model, domain, sample)
                
                # Evaluate using hybrid programmatic + LLM judge
                cls, reason, score = evaluate_sample_result(args.evaluator, domain, sample_result)
                
                sample_result["classification"] = cls
                sample_result["reasoning"] = reason
                sample_result["score"] = score
                
                print(f"    Classified: {cls} (Score: {score})")
                
                # Append and save incrementally
                evaluation_results[domain].append(sample_result)
                with open(raw_results_path, "w", encoding="utf-8") as f:
                    json.dump(evaluation_results, f, indent=4)
                    
    except KeyboardInterrupt:
        print("\n" + "="*50)
        print("Pipeline interrupted by user (Ctrl+C). Saving current progress...")
        with open(raw_results_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=4)
        print(f"Checkpoint saved to {raw_results_path}")
        print("Exiting gracefully. You can resume this run later by starting the pipeline again.")
        print("="*50)
        sys.exit(0)

    print(f"\nSaved final raw results to: {raw_results_path}")

    # 3. Run predictive ML analysis
    print("\nRunning predictive Machine Learning analysis...")
    ml_results = run_predictive_analysis(evaluation_results)

    # 4. Generate plots
    print("\nGenerating charts...")
    generate_domain_plots(evaluation_results, args.output)
    if ml_results:
        generate_feature_importance_plot(ml_results, args.output)

    # 5. Compile Markdown report
    create_markdown_report(args.model, evaluation_results, ml_results, args.output)
    
    print(f"\nPipeline execution complete! Check output folder: {args.output}/")


if __name__ == "__main__":
    main()


#python run_pipeline.py --model mistral --limit 50 --evaluator mistral
