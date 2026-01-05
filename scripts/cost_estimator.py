"""
Cost Estimator - Estimate API costs for dataset evaluation.

Usage:
    python scripts/cost_estimator.py
    python scripts/cost_estimator.py --samples 1000 --model gpt-4
"""

import argparse

# Token estimates (average per sample)
TOKENS_PER_SAMPLE = {
    "prompt": 150,
    "response": 50
}

# Pricing (USD per 1M tokens) - as of 2024
PRICING = {
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1-preview": {"input": 15.0, "output": 60.0},
    "o1-mini": {"input": 3.0, "output": 12.0},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
}

# Default sample counts
DEFAULT_SAMPLES = {
    "cladder": 600,
    "p3sa": 500,
    "total": 1100
}


def estimate_cost(
    n_samples: int,
    model: str,
    prompt_tokens: int = TOKENS_PER_SAMPLE["prompt"],
    response_tokens: int = TOKENS_PER_SAMPLE["response"]
) -> dict:
    """
    Estimate API cost for evaluation.

    Args:
        n_samples: Number of samples to evaluate
        model: Model name
        prompt_tokens: Average prompt tokens per sample
        response_tokens: Average response tokens per sample

    Returns:
        Dict with cost breakdown
    """
    if model not in PRICING:
        raise ValueError(f"Unknown model: {model}. Available: {list(PRICING.keys())}")

    prices = PRICING[model]

    total_input = n_samples * prompt_tokens
    total_output = n_samples * response_tokens

    input_cost = (total_input / 1_000_000) * prices["input"]
    output_cost = (total_output / 1_000_000) * prices["output"]
    total_cost = input_cost + output_cost

    return {
        "model": model,
        "samples": n_samples,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate API evaluation costs")
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES["total"],
        help=f"Number of samples (default: {DEFAULT_SAMPLES['total']})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model (default: show all)"
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=TOKENS_PER_SAMPLE["prompt"],
        help=f"Average prompt tokens (default: {TOKENS_PER_SAMPLE['prompt']})"
    )
    parser.add_argument(
        "--response-tokens",
        type=int,
        default=TOKENS_PER_SAMPLE["response"],
        help=f"Average response tokens (default: {TOKENS_PER_SAMPLE['response']})"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("API COST ESTIMATION")
    print("=" * 60)
    print(f"\nSamples: {args.samples}")
    print(f"Prompt tokens/sample: {args.prompt_tokens}")
    print(f"Response tokens/sample: {args.response_tokens}")
    print()

    if args.model:
        models = [args.model]
    else:
        models = list(PRICING.keys())

    print(f"{'Model':<20} {'Input $':<12} {'Output $':<12} {'Total $':<12}")
    print("-" * 56)

    for model in models:
        try:
            result = estimate_cost(
                args.samples,
                model,
                args.prompt_tokens,
                args.response_tokens
            )
            print(
                f"{model:<20} "
                f"${result['input_cost']:<11.2f} "
                f"${result['output_cost']:<11.2f} "
                f"${result['total_cost']:<11.2f}"
            )
        except ValueError:
            continue

    print()
    print("=" * 60)

    # Show dataset-specific estimates
    print("\nPer-Dataset Estimates (gpt-4o-mini):")
    print("-" * 40)
    for dataset, count in DEFAULT_SAMPLES.items():
        if dataset == "total":
            continue
        result = estimate_cost(count, "gpt-4o-mini")
        print(f"  {dataset.upper()}: {count} samples → ${result['total_cost']:.2f}")


if __name__ == "__main__":
    main()
