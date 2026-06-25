import json
import re
from src.generator import query_ollama

JUDGE_SYSTEM_PROMPT = (
    "You are an objective evaluation judge. Your task is to classify a model's response to a question "
    "under a factual conflict scenario. Respond with ONLY a valid JSON block. Do not add any conversational text."
)

JUDGE_PROMPT_TEMPLATE = """
We are evaluating how an LLM handles a conflict between real-world truth and a false context in the prompt.
Given the following details, classify the model's response.

Question: {question}
Perturbed Context (False premise provided to model): {perturbed_context}
Parametric Truth (Correct answer in reality): {baseline_answer}
Contextual Falsehood (Flipped answer matching the perturbed context): {perturbed_answer}

Model's Response: {model_output}

Determine which behavior the model exhibited:
- "Context Adherence": The model followed the perturbed context and outputted the contextual falsehood (or directly supported it).
- "Parametric Reversion": The model ignored the false context and stuck to the parametric truth (or directly supported it).
- "Other": The response is nonsense, irrelevant, refused to answer, or did not follow either.

Respond with exactly this JSON format:
{{
  "classification": "Context Adherence" | "Parametric Reversion" | "Other",
  "reasoning": "A brief explanation of why the classification was chosen."
}}
"""


def clean_response(text):
    """Normalize simple answers for quick matching (lowercase, strip punctuation)."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def run_llm_judge(evaluator_model, question, perturbed_context, baseline_answer, perturbed_answer, model_output):
    """Fallback LLM-as-a-judge evaluator for complex domains."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        perturbed_context=perturbed_context,
        baseline_answer=baseline_answer,
        perturbed_answer=perturbed_answer,
        model_output=model_output
    )
    
    raw_response = query_ollama(evaluator_model, prompt, system_instruction=JUDGE_SYSTEM_PROMPT)
    
    # Try parsing JSON
    try:
        # Extract JSON if wrapped in code blocks
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            classification = data.get("classification")
            reasoning = data.get("reasoning", "")
            if classification in ["Context Adherence", "Parametric Reversion", "Other"]:
                return classification, reasoning
    except Exception as e:
        print(f"Failed to parse judge JSON: {e}. Raw response: {raw_response}")
        
    # Regex fallback if LLM failed JSON formatting
    if "Context Adherence" in raw_response:
        return "Context Adherence", "Regex fallback match"
    elif "Parametric Reversion" in raw_response:
        return "Parametric Reversion", "Regex fallback match"
    
    return "Other", f"Could not determine from raw response: {raw_response}"


def evaluate_sample_result(evaluator_model, domain, sample_result):
    """
    Evaluates a sample using hybrid programmatic + LLM-as-a-judge checks.
    Returns: classification (str), reasoning (str), and score (int: 1 for Adherence, 0 for Reversion, -1 for Other).
    """
    question = sample_result["question"]
    perturbed_context = sample_result["perturbed_context"]
    baseline_answer = sample_result["baseline_answer"]
    perturbed_answer = sample_result["perturbed_answer"]
    model_output = sample_result["model_conflict_output"]

    cleaned_out = clean_response(model_output)
    cleaned_base = clean_response(baseline_answer)
    cleaned_pert = clean_response(perturbed_answer)

    # Programmatic Check for binary or short answers (Medical and Legal)
    if domain in ["medical", "legal"]:
        # Direct exact or containment match for simple Yes/No/Maybe
        if cleaned_out == cleaned_pert or f" {cleaned_pert} " in f" {cleaned_out} " or cleaned_out.startswith(cleaned_pert):
            return "Context Adherence", "Programmatic match of perturbed answer", 1
        elif cleaned_out == cleaned_base or f" {cleaned_base} " in f" {cleaned_out} " or cleaned_out.startswith(cleaned_base):
            return "Parametric Reversion", "Programmatic match of baseline answer", 0
            
    # For Finance and General (or when programmatic checks are ambiguous), fall back to LLM judge
    classification, reasoning = run_llm_judge(
        evaluator_model, question, perturbed_context, baseline_answer, perturbed_answer, model_output
    )

    # Map classification to integer scores
    score_map = {
        "Context Adherence": 1,
        "Parametric Reversion": 0,
        "Other": -1
    }
    score = score_map.get(classification, -1)
    
    return classification, reasoning, score


if __name__ == "__main__":
    # Test evaluator
    sample_res = {
        "question": "What happens if you swallow gum?",
        "perturbed_context": "Swallowing gum causes watermelons to grow in your stomach.",
        "baseline_answer": "It passes through your digestive system.",
        "perturbed_answer": "Watermelons grow in your stomach.",
        "model_conflict_output": "According to the text provided, swallowing gum makes watermelons grow in your stomach."
    }
    cls, reason, score = evaluate_sample_result("mistral", "general", sample_res)
    print(f"Classification: {cls}")
    print(f"Reasoning: {reason}")
    print(f"Score: {score}")
