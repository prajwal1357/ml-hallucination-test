# SECTION 3: METHODOLOGY

## 3.1 Experimental Framework & System Overview

This study proposes a systematic, reproducible evaluation framework to investigate the "cognitive tension" between large language models' (LLMs) internal parametric memory (pre-trained weights) and external prompt contexts. We call this the Parametric vs. Contextual Conflict.

The pipeline operates across four distinct textual domains (General Knowledge, Medical, Legal, and Finance) to capture differing text structures, vocabularies, and logical densities.

The architecture consists of four distinct phases:

1. **Perturbation and Factual Mutation**: Algorithmic and LLM-guided context generation.
2. **Context-Driven Model Ingestion**: Dual baseline vs. conflict query execution.
3. **Response Classification**: Hybrid exact-string and LLM-as-a-judge behavior mapping.
4. **Statistical Feature Extraction and Predictive ML Modeling**: Feature extraction and scikit-learn classifier training.

```
       [Raw Corpora (TruthfulQA, PubMedQA, LegalBench, FinQA)]
                                  │
                                  ▼
                    [Perturbation Generator (mutator.py)]
                                  │
       ┌──────────────────────────┴──────────────────────────┐
       ▼                                                     ▼
[Baseline Prompts (generator.py)]                     [Conflict Prompts (generator.py)]
       │                                                     │
       ▼                                                     ▼
[LLM Parametric Inference]                             [LLM Conflict Inference]
       │                                                     │
       └──────────────────────────┬──────────────────────────┘
                                  ▼
                [Hybrid Classifier (evaluator.py)]
                   ├── Programmatic Containment Check
                   └── Optimized LLM-as-a-judge
                                  │
       ┌──────────────────────────┼──────────────────────────┐
       ▼                          ▼                          ▼
[Context Adherence (CAR)]  [Parametric Reversion (PRR)]  [Other/Indeterminate (ODR)]
       │
       ▼
[Feature Extraction & Predictive Classifier Training (analysis.py)]
       ├── Semantic Similarity (all-MiniLM-L6-v2)
       ├── Linguistic Complexity (word count, length differences)
       └── Domain Encoding
```

---

## 3.2 Dataset Selection and Factual Mutation

To evaluate model stability, we mutated factual truths across four domains into logical falsehoods, formatting them into unified schemas containing: `question`, `baseline_context`, `perturbed_context`, `baseline_answer`, and `perturbed_answer`.

1. **General Knowledge (TruthfulQA)**: Uses common misconceptions. The baseline context is marked as `None`. The perturbed context asserts the misconception as an absolute fact (e.g., "According to a widely cited educational report: Swallowing gum takes 7 years to digest."). The perturbed answer matches the misconception, while the baseline answer contains the scientific truth.
2. **Medical (PubMedQA)**: Alters the abstracts of peer-reviewed clinical studies. The conclusion is flipped programmatically or via LLM-rewrite (e.g., changing "significantly reduced mortality" to "resulted in no significant change in mortality rates"). The answer is mutated from "yes" to "no" (or vice versa).
3. **Legal (LegalBench - Limitation of Liability)**: Mutates contractual clauses. A clause restricting a party's liability is rewritten to state unlimited liability, or a clause with unlimited liability is modified to introduce explicit dollar caps. The target classification is flipped from "Yes" to "No".
4. **Finance (FinQA)**: Perturbs earnings statements. Key financial numbers in both tables and paragraphs are inverted (e.g., changing "revenue grew 15%" to "revenue shrank 15%" or mutating specific table indices). The target mathematical calculation and answer are updated to reflect the false metrics.

---

## 3.3 Prompt Engineering and Conflict Instructions

Models are queried under two experimental configurations with temperature set to $0.0$ to ensure deterministic, reproducible outputs:

### A. Baseline Configuration

Queries the model with the original question and original context to establish baseline accuracy:

> **Prompt Template**:
> `Context: {baseline_context}`
> `Question: {question}`
> `Instruction: Based on the provided context, answer the question. Keep your answer brief and direct.`

### B. Conflict Configuration

Queries the model with the mutated context and explicit system constraints ordering the model to prioritize context over outside memory:

> **Prompt Template**:
> `Context: {perturbed_context}`
> `Question: {question}`
> `Instruction: Based on the provided context, answer the question. You MUST answer based ONLY on the provided context. If the context contradicts your outside knowledge, prioritize the context and ignore your outside knowledge. Keep your answer brief and direct.`

---

## 3.4 Response Classification Methodology

Outputs are classified into three behaviors:

* **Context Adherence (CAR)** (Score: 1): The model follows the perturbed context and outputs the contextual falsehood.
* **Parametric Reversion (PRR)** (Score: 0): The model ignores the false context and outputs the real-world parametric truth.
* **Other/Indeterminate (ODR)** (Score: -1): The model output is nonsense, ambiguous, or refuses to answer.

To handle reasoning models (e.g., `deepseek-r1`) and speed up execution, we employ a **hybrid classification pipeline**:

1. **Reasoning Tag Stripping**: The system strips `<think>...</think>` tags to isolate the final answer block.
2. **Programmatic Containment Bypass**: Normalizes text and runs substring containment checks. If the output contains only the perturbed answer, it is labeled CAR. If it contains only the baseline answer, it is labeled PRR. This bypasses the LLM judge for ~80% of samples.
3. **Optimized LLM Judge Fallback**: If programmatic check is ambiguous, the prompt is evaluated by a secondary evaluator model. Crucially, the evaluator prompt excludes the raw 1,000+ token context (as the evaluator only needs to compare the baseline/perturbed answers with the model response), cutting VRAM and prompt pre-fill overhead by 90%.

---

## 3.5 Feature Extraction and Classifier Training

To diagnose the drivers of model failure, we extract linguistic, semantic, and structural features from the prompts:

### A. Core Semantic Features

* **Semantic Similarity**: The cosine similarity between the embeddings of the baseline context and the perturbed context, encoded using the `all-MiniLM-L6-v2` Sentence-Transformer model:
  $$
  \text{Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}
  $$
* **Context Word Count**: The total number of words in the perturbed context.
* **Length Difference**: The absolute difference in word count between the baseline and perturbed contexts.

### B. Engineered Linguistic Features

* **Flesch-Kincaid Grade Level**: A readability metric estimating the U.S. school grade level required to comprehend the perturbed context. Calculated as:

  $$
  \text{FK} = 0.39 \times \frac{\text{words}}{\text{sentences}} + 11.8 \times \frac{\text{syllables}}{\text{words}} - 15.59
  $$

  Higher grades indicate more complex, technical language (e.g., financial reports grade 14+ vs. general knowledge grade 8).
* **Numeric Density**: The count of numeric tokens (integers, decimals, percentages) in the perturbed context. Financial and tabular data produces high numeric density, which may confuse or anchor the model.
* **Answer Length Ratio**: The ratio of the perturbed answer length to the baseline answer length. A high ratio (e.g., 5.0) indicates the perturbed answer is significantly more verbose than the baseline, which may signal implausibility to the model.
* **Perturbation Position**: A normalized value between 0.0 (beginning) and 1.0 (end) indicating where in the text the first divergence between the baseline and perturbed contexts occurs. This captures the "lost in the middle" phenomenon, where LLMs are known to lose track of information embedded deep within long passages.

### C. Domain Encoding

* **One-Hot Categorical Indicators**: Binary features for each of the four domains (General, Medical, Legal, Finance).

### D. Feature Normalization

All continuous features are standardized using `StandardScaler` (zero mean, unit variance) before training. This prevents high-magnitude features (e.g., context word count ~500) from dominating low-magnitude features (e.g., semantic similarity ~0.8) in distance-based and gradient-based classifiers.

### E. Classifier Training

Using the 11-dimensional feature vector as input ($X$) and the classification score ($y \in \{0, 1\}$) as the target, we train four classical ML classifiers:

1. **Random Forest** (with `GridSearchCV` hyperparameter tuning over `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`)
2. **Logistic Regression** (L2-regularized, LBFGS solver)
3. **Gradient Boosting Classifier** (200 estimators, learning rate 0.1)
4. **Support Vector Machine** (RBF kernel, `gamma=scale`)

Feature importances (RF, GB) and regression coefficients (LR) are extracted to measure which prompt characteristics most strongly predict model vulnerability.

---

---

# SECTION 4: RESULTS AND COMPARATIVE ANALYSIS

We conducted evaluations on three diverse local models: **DeepSeek-R1 (8B)** (a reasoning-centric model), **Llama-3 (8B)** (meta's general-purpose model), and **Mistral (7B)**. Each domain was evaluated with a limit of 50 samples.

## 4.1 Cross-Domain Behavior Comparison

The following table summarizes the behavior rates (Context Adherence, Parametric Reversion, and Other/Indeterminate) across the models:

| Model                      | Domain  | Total Samples | Context Adherence (CAR) | Parametric Reversion (PRR) | Other/Indeterminate (ODR) |
| :------------------------- | :------ | :-----------: | :---------------------: | :------------------------: | :-----------------------: |
| **DeepSeek-R1 (8B)** | General |      50      |       50 (100.0%)       |          0 (0.0%)          |         0 (0.0%)         |
|                            | Medical |      53      |       26 (49.1%)       |         21 (39.6%)         |         6 (11.3%)         |
|                            | Legal   |      50      |       29 (58.0%)       |         21 (42.0%)         |         0 (0.0%)         |
|                            | Finance |      50      |        4 (8.0%)        |         25 (50.0%)         |        21 (42.0%)        |
| **Llama-3 (8B)**     | General |      50      |       48 (96.0%)       |          2 (4.0%)          |         0 (0.0%)         |
|                            | Medical |      50      |       30 (60.0%)       |         20 (40.0%)         |         0 (0.0%)         |
|                            | Legal   |      50      |       25 (50.0%)       |         25 (50.0%)         |         0 (0.0%)         |
|                            | Finance |      50      |       48 (96.0%)       |          1 (2.0%)          |         1 (2.0%)         |
| **Mistral (7B)**     | General |      50      |       48 (96.0%)       |          2 (4.0%)          |         0 (0.0%)         |
|                            | Medical |      50      |       35 (70.0%)       |         15 (30.0%)         |         0 (0.0%)         |
|                            | Legal   |      50      |       20 (40.0%)       |         30 (60.0%)         |         0 (0.0%)         |
|                            | Finance |      50      |       47 (94.0%)       |          0 (0.0%)          |         3 (6.0%)         |

### Key Behavioral Analysis

1. **General Knowledge**: All three models adhered strongly to the false context (96% to 100% CAR). When presented with simple, misconception-based text assertions under strict context instructions, the models readily outputted the falsehoods.
2. **Medical Domain**: The medical domain showed the highest cognitive resistance. Across all models, the Parametric Reversion Rate (PRR) ranged between 30% and 40%, indicating that pre-trained weights containing strong medical correlations (e.g., drug efficacy or anatomical truths) push back against mutated contextual falsehoods.
3. **Legal Domain**: Llama-3 and DeepSeek-R1 showed a near 50-50 split between CAR and PRR, while Mistral favored parametric memory (60.0% PRR).
4. **Financial Domain (The Divergence)**: This domain showed the most significant contrast:
   * **Llama-3 and Mistral** both exhibited near-total adherence (96.0% and 94.0% CAR respectively). They accepted the mutated financial metrics and calculation tables blindly.
   * **DeepSeek-R1** had a low CAR of 8.0% and a high PRR (50.0%) and ODR (42.0%). Because DeepSeek-R1 is a reasoning model that evaluates problems step-by-step, it caught mathematical and textual contradictions introduced by the mutations (e.g., text asserting a revenue growth of -15% alongside numbers that didn't add up). This caused it to either revert to its parametric understanding or output confused, indeterminate responses (resulting in 42.0% ODR).

**Figure 1: Cross-Model Behavior Distribution by Domain**
![Cross-Model Behavior Distribution by Domain](P:/project/research/dataset/enhanced_plots/cross_model_behavior.png)

**Figure 2: Context Adherence Rate (CAR) Trend Across Domains**
![Context Adherence Rate (CAR) Across Domains](P:/project/research/dataset/enhanced_plots/car_line_comparison.png)

---

## 4.2 Enhanced Machine Learning Feature Importance Analysis

We trained four classifiers (Random Forest with GridSearchCV, Logistic Regression with StandardScaler, Gradient Boosting, and SVM) on an 11-dimensional engineered feature vector to predict model vulnerability to context adherence. ODR (Other/Indeterminate) samples were excluded from training, yielding 176 (DeepSeek), 199 (Llama-3), and 197 (Mistral) usable samples.

### 4.2.1 Classifier Performance Comparison

The following table compares classifier accuracy and F1 scores across all models:

| Model                      | Classifier                   |     Accuracy     |     F1 Score     |
| :------------------------- | :--------------------------- | :--------------: | :--------------: |
| **DeepSeek-R1 (8B)** | Random Forest (Tuned)        |      75.00%      |      0.8000      |
|                            | Logistic Regression (Scaled) |      69.44%      |      0.7843      |
|                            | Gradient Boosting            | **77.78%** | **0.8182** |
|                            | SVM (RBF)                    |      69.44%      |      0.7843      |
| **Llama-3 (8B)**     | Random Forest (Tuned)        |      67.50%      |      0.7937      |
|                            | Logistic Regression (Scaled) |      67.50%      |      0.7869      |
|                            | Gradient Boosting            | **70.00%** | **0.7931** |
|                            | SVM (RBF)                    |      67.50%      |      0.7869      |
| **Mistral (7B)**     | Random Forest (Tuned)        | **80.00%** | **0.8621** |
|                            | Logistic Regression (Scaled) |      77.50%      |      0.8421      |
|                            | Gradient Boosting            |      77.50%      |      0.8475      |
|                            | SVM (RBF)                    |      77.50%      |      0.8421      |

**Key Observations**:

* **Mistral (7B)** had the most predictable behavior, with a tuned Random Forest achieving **80.00% accuracy** and an F1 of **0.8621**. This suggests that Mistral's failure patterns are highly systematic and driven by a small number of surface-level features.
* **DeepSeek-R1 (8B)** was best predicted by Gradient Boosting (77.78%), consistent with a non-linear interaction between reasoning depth and context structure.
* **Llama-3 (8B)** was the hardest to predict (peak 70.00%), suggesting that its failure patterns are more stochastic and less feature-dependent.

**Figure 3: Classifier Performance Comparison (Accuracy & F1 Score)**
![Classifier Performance Comparison](P:/project/research/dataset/enhanced_plots/classifier_comparison.png)

---

### 4.2.2 Random Forest Feature Importance Comparison (Enhanced)

The following table compares the predictive feature importance scores using the tuned Random Forest across all models. Features are ordered by average importance:

| Feature                         | DeepSeek-R1 (8B) | Llama-3 (8B) | Mistral (7B) |  Avg.  |
| :------------------------------ | :--------------: | :----------: | :----------: | :----: |
| **Context word count**    |      0.1801      |    0.2046    |    0.1307    | 0.1718 |
| **Semantic similarity**   |      0.1628      |    0.0906    |    0.0922    | 0.1152 |
| **Flesch-Kincaid grade**  |      0.1194      |    0.1653    |    0.1485    | 0.1444 |
| **Perturbation position** |      0.1034      |    0.1310    |    0.1562    | 0.1302 |
| **Length difference**     |      0.1485      |    0.1233    |    0.0991    | 0.1236 |
| **Numeric density**       |      0.1151      |    0.0842    |    0.0712    | 0.0902 |
| **Answer length ratio**   |      0.0346      |    0.0449    |    0.1481    | 0.0759 |
| **Domain general**        |      0.0761      |    0.0313    |    0.0129    | 0.0401 |
| **Domain legal**          |      0.0097      |    0.0753    |    0.0986    | 0.0612 |
| **Domain finance**        |      0.0346      |    0.0186    |    0.0316    | 0.0283 |
| **Domain medical**        |      0.0155      |    0.0310    |    0.0108    | 0.0191 |

**Insights from Enhanced Feature Engineering**:

1. The features (**Flesch-Kincaid grade**, **Perturbation position**, **Numeric density**) collectively contribute **~35% of total importance** across all models, demonstrating that they capture meaningful signal that the original 3-feature model could not.
2. **Perturbation Position** is the 4th most important feature overall (avg. 0.1302). Mutations injected early in a passage are more likely to be followed by the model, consistent with the "primacy bias" in autoregressive transformers.
3. **Flesch-Kincaid Grade** is the 3rd most important feature (avg. 0.1444). More complex, technical passages increase model susceptibility to false contexts — the model appears to "defer" to complex-sounding authority.
4. For Mistral specifically, **Answer Length Ratio** (0.1481) is the 3rd most important feature, meaning Mistral is particularly susceptible when the perturbed answer is much longer/shorter than expected.

**Figure 4: Random Forest Feature Importance Heatmap**
![Random Forest Feature Importance Heatmap](P:/project/research/dataset/enhanced_plots/rf_feature_heatmap.png)

**Figure 5: Feature Importance Radar Chart**
![Feature Importance Radar](P:/project/research/dataset/enhanced_plots/feature_radar.png)

**Figure 6: Average Feature Importance Across All Models**
![Average Feature Importance](P:/project/research/dataset/enhanced_plots/avg_feature_importance.png)

---

### 4.2.3 Gradient Boosting Feature Importance Comparison

Gradient Boosting captures non-linear feature interactions. The following table shows its feature importances:

| Feature                         | DeepSeek-R1 (8B) | Llama-3 (8B) | Mistral (7B) |
| :------------------------------ | :--------------: | :----------: | :----------: |
| **Context word count**    |      0.4006      |    0.2080    |    0.1571    |
| **Length difference**     |      0.1579      |    0.2032    |    0.0401    |
| **Perturbation position** |      0.1016      |    0.1392    |    0.1776    |
| **Flesch-Kincaid grade**  |      0.0761      |    0.1131    |    0.1566    |
| **Numeric density**       |      0.1121      |    0.1166    |    0.0610    |
| **Semantic similarity**   |      0.1015      |    0.0966    |    0.0620    |
| **Answer length ratio**   |      0.0338      |    0.0020    |    0.3348    |
| **Domain legal**          |      0.0040      |    0.1071    |    0.0049    |
| **Domain medical**        |      0.0116      |    0.0040    |    0.0056    |
| **Domain finance**        |      0.0009      |    0.0102    |    0.0001    |
| **Domain general**        |      0.0000      |    0.0000    |    0.0000    |

**Notable**: For Mistral, the Gradient Boosting model places **33.48%** of importance on **Answer Length Ratio** — the single largest importance for any feature-model pair. This strongly indicates that Mistral uses the verbosity/brevity of an expected answer as a heuristic for plausibility.

**Figure 7: Gradient Boosting Feature Importance Heatmap**
![Gradient Boosting Feature Importance Heatmap](P:/project/research/dataset/enhanced_plots/gb_feature_heatmap.png)

---

### 4.2.4 Logistic Regression Coefficients (Scaled)

Positive coefficients indicate features that drive the model towards **Context Adherence** (believing the lie), whereas negative coefficients drive it towards **Parametric Reversion** (sticking to real-world truth). All features are standardized (zero mean, unit variance), making coefficient magnitudes directly comparable:

| Feature                         | DeepSeek-R1 (8B) | Llama-3 (8B) | Mistral (7B) |
| :------------------------------ | :--------------: | :----------: | :----------: |
| **Domain general**        |     +0.9911     |   +0.0935   |   +0.2285   |
| **Domain finance**        |     -0.7887     |   +1.2071   |   +0.9103   |
| **Numeric density**       |     +0.0946     |   +0.6926   |   +0.3931   |
| **Answer length ratio**   |     +0.4171     |   +0.1254   |   +0.1976   |
| **Perturbation position** |     -0.1111     |   +0.1691   |   +0.0216   |
| **Context word count**    |     -0.3790     |   -0.3236   |   +0.1983   |
| **Length difference**     |     -0.2187     |   -0.3826   |   -0.0084   |
| **Flesch-Kincaid grade**  |     -0.0850     |   -0.2567   |   -0.2130   |
| **Domain medical**        |     -0.1670     |   -0.5643   |   -0.3651   |
| **Semantic similarity**   |     -0.6760     |   -0.4966   |   -0.4095   |
| **Domain legal**          |     -0.1891     |   -0.7481   |   -0.7944   |

#### Key Insights from Scaled Coefficients:

1. **Domain General**: General Knowledge prompts push DeepSeek-R1 strongly toward adherence (+0.99), while the effect is weaker for Llama-3 (+0.09) and Mistral (+0.23). This is because simple misconception-based prompts are universally followed, making the domain label less discriminative for non-reasoning models.
2. **Semantic Similarity**: Consistently negative across all models (-0.68, -0.50, -0.41). Paradoxically, as the perturbed context becomes more semantically similar to the baseline, models are more likely to revert — because highly similar perturbations may be too subtle to override strong parametric memory.
3. **Domain Finance**: For Llama-3 (+1.21) and Mistral (+0.91), the Finance domain is the single strongest driver of context adherence. For DeepSeek-R1 (-0.79), it drives reversion — the reasoning model's chain-of-thought catches contradictions.
4. **Numeric Density**: Positive across all models (+0.09, +0.69, +0.39). Prompts containing many numbers increase context adherence — the models appear to anchor on numerical authority without verification.
5. **Domain Legal**: The strongest negative driver for both Llama-3 (-0.75) and Mistral (-0.79). Legal language triggers strong parametric resistance, suggesting that pre-trained legal reasoning patterns (e.g., distinguishing liability clauses) are deeply embedded.

**Figure 8: Logistic Regression Coefficients Heatmap (Diverging)**
![Logistic Regression Coefficients Heatmap](P:/project/research/dataset/enhanced_plots/lr_coefficients_heatmap.png)

---

## 4.3 Diagnostic Visualizations

All visualization charts are generated by [generate_visualizations.py](file:///P:/project/research/dataset/generate_visualizations.py)

### Enhanced Analysis Charts (New)

| Figure   | Description                                       | File Path                                                                                                    |
| :------- | :------------------------------------------------ | :----------------------------------------------------------------------------------------------------------- |
| Figure 1 | Cross-Model Behavior Distribution by Domain       | [cross_model_behavior.png](file:///P:/project/research/dataset/enhanced_plots/cross_model_behavior.png)       |
| Figure 2 | Context Adherence Rate (CAR) Trend Across Domains | [car_line_comparison.png](file:///P:/project/research/dataset/enhanced_plots/car_line_comparison.png)         |
| Figure 3 | Classifier Performance Comparison (Accuracy & F1) | [classifier_comparison.png](file:///P:/project/research/dataset/enhanced_plots/classifier_comparison.png)     |
| Figure 4 | Random Forest Feature Importance Heatmap          | [rf_feature_heatmap.png](file:///P:/project/research/dataset/enhanced_plots/rf_feature_heatmap.png)           |
| Figure 5 | Feature Importance Radar Chart                    | [feature_radar.png](file:///P:/project/research/dataset/enhanced_plots/feature_radar.png)                     |
| Figure 6 | Average Feature Importance Bar Chart              | [avg_feature_importance.png](file:///P:/project/research/dataset/enhanced_plots/avg_feature_importance.png)   |
| Figure 7 | Gradient Boosting Feature Importance Heatmap      | [gb_feature_heatmap.png](file:///P:/project/research/dataset/enhanced_plots/gb_feature_heatmap.png)           |
| Figure 8 | Logistic Regression Coefficients Heatmap          | [lr_coefficients_heatmap.png](file:///P:/project/research/dataset/enhanced_plots/lr_coefficients_heatmap.png) |

### Per-Model Behavior Charts (Original Pipeline)

| Model            | Behavior Distribution                                                                              | Feature Importance                                                                           |
| :--------------- | :------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------- |
| DeepSeek-R1 (8B) | [behavior_distribution.png](file:///P:/project/research/dataset/deepseek/behavior_distribution.png) | [feature_importance.png](file:///P:/project/research/dataset/deepseek/feature_importance.png) |
| Llama-3 (8B)     | [behavior_distribution.png](file:///P:/project/research/dataset/llama3/behavior_distribution.png)   | [feature_importance.png](file:///P:/project/research/dataset/llama3/feature_importance.png)   |
| Mistral (7B)     | [behavior_distribution.png](file:///P:/project/research/dataset/mistral/behavior_distribution.png)  | [feature_importance.png](file:///P:/project/research/dataset/mistral/feature_importance.png)  |
