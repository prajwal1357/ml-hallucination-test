import os
import json
import time
import httpx
from ollama import Client

# Initialize Ollama client with a 150-second (2.5 minutes) timeout
ollama_client = Client(timeout=150.0)

# Prompts for LLM-based mutations
MEDICAL_PROMPT = (
    "Rewrite the following medical abstract to invert its clinical findings and final conclusion. "
    "If the abstract shows a positive outcome or correlation (e.g. effective drug, increased survival), "
    "change it to a negative or neutral outcome (e.g. ineffective drug, no survival change). "
    "If it was negative/neutral, change it to positive. "
    "Keep all other scientific terms, measurements, and general background identical. "
    "Do not explain the changes, simply return the rewritten text."
)

LEGAL_PROMPT = (
    "Rewrite the following contract clause to flip whether it contains a limitation of liability. "
    "If the clause limits liability, rewrite it to state unlimited liability or remove the limitation. "
    "If it has no limitation of liability, rewrite it to include a clear limitation of liability. "
    "Keep the general legal context and terminology identical. "
    "Do not explain the changes, simply return the rewritten text."
)

FINANCE_PROMPT = (
    "Rewrite the following financial report text and table to invert the primary financial metric or outcome. "
    "For example, if net income or revenue increased, change the numbers and description to show a decrease. "
    "If it showed a loss, change it to show a gain. Keep the company names and text formatting identical. "
    "Do not explain the changes, simply return the rewritten text."
)


def query_ollama_mutation(system_instruction, text, model="mistral"):
    """Queries local Ollama to mutate a given text according to instructions with timeout and retry."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = ollama_client.chat(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise text-rewriting engine. Output ONLY the rewritten text. Do not add any conversational introductions, notes, markdown formatting, or explanations."
                    },
                    {
                        "role": "user",
                        "content": f"{system_instruction}\n\nOriginal Text:\n{text}"
                    }
                ]
            )
            return response["message"]["content"].strip()
        except (httpx.TimeoutException, httpx.ReadTimeout, httpx.WriteTimeout, httpx.ConnectTimeout) as te:
            print(f"\n[Mutation Timeout Warning] Attempt {attempt+1}/{max_retries} timed out: {te}")
            if attempt < max_retries - 1:
                print("Retrying mutation...")
                time.sleep(2)
                continue
            else:
                print("Mutation failed due to persistent timeouts. Returning original text.")
                return text
        except Exception as e:
            err_str = str(e).lower()
            if "timeout" in err_str or "timed out" in err_str:
                print(f"\n[Mutation Timeout Warning] Attempt {attempt+1}/{max_retries} timed out (generic): {e}")
                if attempt < max_retries - 1:
                    print("Retrying mutation...")
                    time.sleep(2)
                    continue
                else:
                    print("Mutation failed due to persistent timeouts. Returning original text.")
                    return text
            print(f"Error querying Ollama for mutation: {e}")
            return text


def format_finance_context(sample):
    """Combines pre_text, table, and post_text of FinQA into a single string."""
    context = ""
    if sample.get("pre_text"):
        context += " ".join(sample["pre_text"]) + "\n"
    if sample.get("table"):
        table_str = "\n".join([" | ".join(row) for row in sample["table"]])
        context += "Table:\n" + table_str + "\n"
    if sample.get("post_text"):
        context += " ".join(sample["post_text"])
    return context.strip()


def mutate_general(sample):
    """
    Programmatic mutation for TruthfulQA (general.json).
    Pick the first incorrect answer (misconception) and present it as a fact in the context.
    """
    question = sample["Question"]
    incorrect_answers = sample["Incorrect Answers"].split("; ")
    # Pick the first misconception
    misconception = incorrect_answers[0]
    
    # Formulate false context
    perturbed_context = f"According to a widely cited educational report: {misconception}."
    
    return {
        "question": question,
        "baseline_context": "None",
        "perturbed_context": perturbed_context,
        "baseline_answer": sample["Best Answer"],
        "perturbed_answer": misconception
    }


def mutate_medical(key, sample, model="mistral"):
    """Mutate PubMedQA sample using Ollama."""
    question = sample["QUESTION"]
    original_context = "\n".join(sample["CONTEXTS"])
    
    print(f"Mutating Medical sample {key}...")
    perturbed_context = query_ollama_mutation(MEDICAL_PROMPT, original_context, model)
    
    # Flip the answer
    original_ans = sample["final_decision"]
    if original_ans == "yes":
        perturbed_ans = "no"
    elif original_ans == "no":
        perturbed_ans = "yes"
    else:
        perturbed_ans = "maybe"
        
    return {
        "question": question,
        "baseline_context": original_context,
        "perturbed_context": perturbed_context,
        "baseline_answer": original_ans,
        "perturbed_answer": perturbed_ans
    }


def mutate_legal(sample, model="mistral"):
    """Mutate LegalBench sample using Ollama."""
    original_context = sample["text"]
    original_ans = sample["answer"]
    
    # Question is fixed for contract_qa
    question = "Does this clause contain a limitation of liability?"
    
    print(f"Mutating Legal sample {sample['index']}...")
    perturbed_context = query_ollama_mutation(LEGAL_PROMPT, original_context, model)
    
    perturbed_ans = "No" if original_ans == "Yes" else "Yes"
    
    return {
        "question": question,
        "baseline_context": original_context,
        "perturbed_context": perturbed_context,
        "baseline_answer": original_ans,
        "perturbed_answer": perturbed_ans
    }


def mutate_finance(sample, model="mistral"):
    """Mutate FinQA sample using Ollama."""
    original_context = format_finance_context(sample)
    question = sample["qa"]["question"]
    original_ans = sample["qa"]["answer"]
    
    print(f"Mutating Financial sample...")
    perturbed_context = query_ollama_mutation(FINANCE_PROMPT, original_context, model)
    
    # For finance, since the answer is numerical or specific, we ask LLM to provide a flipped answer too
    ans_prompt = f"Given the original financial answer '{original_ans}' and the original context: \n\n{original_context}\n\nState a plausible opposite numerical or final answer that would match the mutated context. Respond with ONLY the answer value."
    perturbed_ans = query_ollama_mutation(ans_prompt, "", model)
    
    return {
        "question": question,
        "baseline_context": original_context,
        "perturbed_context": perturbed_context,
        "baseline_answer": original_ans,
        "perturbed_answer": perturbed_ans
    }


def run_mutation_pipeline(model="mistral", limit=100, force=False):
    """Runs the mutation process on data/ folder and saves perturbed versions."""
    os.makedirs(os.path.join("data", "perturbed"), exist_ok=True)
    
    # 1. General
    general_path = os.path.join("data", "general.json")
    perturbed_general_path = os.path.join("data", "perturbed", "general.json")
    if force or not os.path.exists(perturbed_general_path):
        print("Processing General Domain (TruthfulQA)...")
        with open(general_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        perturbed_data = [mutate_general(s) for s in data[:limit]]
        with open(perturbed_general_path, "w", encoding="utf-8") as f:
            json.dump(perturbed_data, f, indent=4)
        print(f"Saved {len(perturbed_data)} mutated General samples.")
    else:
        print("General Domain mutations already cached.")

    # 2. Medical
    medical_path = os.path.join("data", "medical.json")
    perturbed_medical_path = os.path.join("data", "perturbed", "medical.json")
    if force or not os.path.exists(perturbed_medical_path):
        print("Processing Medical Domain (PubMedQA)...")
        with open(medical_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        perturbed_data = []
        keys = list(data.keys())[:limit]
        for k in keys:
            perturbed_data.append(mutate_medical(k, data[k], model))
        with open(perturbed_medical_path, "w", encoding="utf-8") as f:
            json.dump(perturbed_data, f, indent=4)
        print(f"Saved {len(perturbed_data)} mutated Medical samples.")
    else:
        print("Medical Domain mutations already cached.")

    # 3. Legal
    legal_path = os.path.join("data", "legal.json")
    perturbed_legal_path = os.path.join("data", "perturbed", "legal.json")
    if force or not os.path.exists(perturbed_legal_path):
        print("Processing Legal Domain (LegalBench)...")
        with open(legal_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        perturbed_data = []
        for s in data[:limit]:
            perturbed_data.append(mutate_legal(s, model))
        with open(perturbed_legal_path, "w", encoding="utf-8") as f:
            json.dump(perturbed_data, f, indent=4)
        print(f"Saved {len(perturbed_data)} mutated Legal samples.")
    else:
        print("Legal Domain mutations already cached.")

    # 4. Finance
    finance_path = os.path.join("data", "finance.json")
    perturbed_finance_path = os.path.join("data", "perturbed", "finance.json")
    if force or not os.path.exists(perturbed_finance_path):
        print("Processing Financial Domain (FinQA)...")
        with open(finance_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        perturbed_data = []
        for s in data[:limit]:
            perturbed_data.append(mutate_finance(s, model))
        with open(perturbed_finance_path, "w", encoding="utf-8") as f:
            json.dump(perturbed_data, f, indent=4)
        print(f"Saved {len(perturbed_data)} mutated Financial samples.")
    else:
        print("Financial Domain mutations already cached.")


if __name__ == "__main__":
    # For testing the mutator independently
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_mutation_pipeline(model="mistral", limit=limit, force=True)
