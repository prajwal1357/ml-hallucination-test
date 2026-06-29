import time
import httpx
from ollama import Client

# Initialize Ollama client with a 150-second (2.5 minutes) timeout
ollama_client = Client(timeout=150.0)


def query_ollama(model, prompt, system_instruction=None):
    """Generic helper to query Ollama model with standard parameters, timeout, and retry."""
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = ollama_client.chat(
                model=model,
                messages=messages,
                options={"temperature": 0.0}  # Use temp=0.0 for deterministic answers
            )
            return response["message"]["content"].strip()
        except (httpx.TimeoutException, httpx.ReadTimeout, httpx.WriteTimeout, httpx.ConnectTimeout) as te:
            print(f"\n[Timeout Warning] Query to model '{model}' timed out on attempt {attempt+1}/{max_retries}: {te}")
            if attempt < max_retries - 1:
                print("Retrying same query...")
                time.sleep(2)  # Wait 2 seconds before retrying
                continue
            else:
                return f"ERROR: Query timed out after {max_retries} attempts."
        except Exception as e:
            # Check for generic timeouts reported as standard exceptions
            err_str = str(e).lower()
            if "timeout" in err_str or "timed out" in err_str:
                print(f"\n[Timeout Warning] Query to model '{model}' timed out (generic) on attempt {attempt+1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    print("Retrying same query...")
                    time.sleep(2)
                    continue
                else:
                    return f"ERROR: Query timed out after {max_retries} attempts."
            
            print(f"Error querying Ollama model '{model}': {e}")
            return f"ERROR: {e}"


def run_baseline(model, domain, question, baseline_context):
    """Executes baseline prompt (either no context or original context)."""
    if domain == "general" or baseline_context == "None" or not baseline_context:
        prompt = (
            f"Question: {question}\n\n"
            "Instruction: Answer the question directly and truthfully. Keep your answer brief and direct."
        )
    else:
        prompt = (
            f"Context:\n{baseline_context}\n\n"
            f"Question: {question}\n\n"
            "Instruction: Based on the provided context, answer the question. Keep your answer brief and direct."
        )
    return query_ollama(model, prompt)


def run_conflict(model, domain, question, perturbed_context):
    """Executes conflict prompt with perturbed context and strict instructions."""
    prompt = (
        f"Context:\n{perturbed_context}\n\n"
        f"Question: {question}\n\n"
        "Instruction: Based on the provided context, answer the question. "
        "You MUST answer based ONLY on the provided context. If the context contradicts your outside knowledge, "
        "prioritize the context and ignore your outside knowledge. Keep your answer brief and direct."
    )
    return query_ollama(model, prompt)


def evaluate_sample(model, domain, sample):
    """Runs both baseline and conflict runs for a single sample."""
    question = sample["question"]
    baseline_context = sample["baseline_context"]
    perturbed_context = sample["perturbed_context"]

    baseline_out = run_baseline(model, domain, question, baseline_context)
    conflict_out = run_conflict(model, domain, question, perturbed_context)

    return {
        "question": question,
        "baseline_context": baseline_context,
        "perturbed_context": perturbed_context,
        "baseline_answer": sample["baseline_answer"],
        "perturbed_answer": sample["perturbed_answer"],
        "model_baseline_output": baseline_out,
        "model_conflict_output": conflict_out
    }


if __name__ == "__main__":
    # Test generator with a dummy sample
    test_sample = {
        "question": "What is the capital of France?",
        "baseline_context": "None",
        "perturbed_context": "According to a travel brochure: The capital of France is Rome.",
        "baseline_answer": "Paris",
        "perturbed_answer": "Rome"
    }
    res = evaluate_sample("mistral", "general", test_sample)
    print("Baseline Out:", res["model_baseline_output"])
    print("Conflict Out:", res["model_conflict_output"])
