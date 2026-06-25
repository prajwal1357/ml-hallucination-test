import os
import json
from src.mutator import run_mutation_pipeline

DOMAINS = ["general", "medical", "legal", "finance"]


def load_domain_data(domain, limit=100, force_mutate=False, mutator_model="mistral"):
    """
    Loads unified baseline & perturbed samples for a given domain.
    If the perturbed cache file is missing or force_mutate=True, it triggers the mutator first.
    """
    if domain not in DOMAINS:
        raise ValueError(f"Invalid domain: {domain}. Must be one of {DOMAINS}")

    perturbed_file = os.path.join("data", "perturbed", f"{domain}.json")

    # If cache doesn't exist or we force mutation, generate it
    if force_mutate or not os.path.exists(perturbed_file):
        print(f"Perturbed cache for '{domain}' not found. Generating...")
        # Mutate a slightly larger limit to ensure we have enough samples
        run_mutation_pipeline(model=mutator_model, limit=limit, force=True)

    with open(perturbed_file, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    # Self-heal: If cached data has fewer samples than requested, regenerate it
    if len(data) < limit:
        print(f"Perturbed cache for '{domain}' has only {len(data)} samples. Need {limit}. Regenerating...")
        run_mutation_pipeline(model=mutator_model, limit=limit, force=True)
        with open(perturbed_file, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

    # Return up to the requested limit
    return data[:limit]


def load_all_domains(limit=50, force_mutate=False, mutator_model="mistral"):
    """
    Loads unified datasets for all domains.
    Returns a dictionary: { domain_name: list_of_samples }
    """
    all_data = {}
    for domain in DOMAINS:
        print(f"\nLoading data for domain: {domain}")
        all_data[domain] = load_domain_data(
            domain, limit=limit, force_mutate=force_mutate, mutator_model=mutator_model
        )
    return all_data


if __name__ == "__main__":
    # Test loader
    data = load_all_domains(limit=3)
    for domain, samples in data.items():
        print(f"\nDomain: {domain} ({len(samples)} samples)")
        if samples:
            print(f"Sample 1 Question: {samples[0]['question']}")
            print(f"Sample 1 Perturbed Answer: {samples[0]['perturbed_answer']}")
