"""
Evaluation pipeline for the Product Review Summarizer.

Computes ROUGE-L, BERTScore, and LLM-as-Judge metrics comparing
the fine-tuned model against the base model on the test set.

Usage:
    python src/evaluate.py \
        --adapter-path ./adapters_run4-2ep-lr1.5e4-r16 \
        --test-data ./data/processed/test_data.json \
        --output ./evaluation/eval_results.json \
        --openai-key sk-...  # Optional, for LLM-as-judge
"""

import argparse
import json
import os
import random

import torch
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from unsloth import FastLanguageModel

load_dotenv()


def load_finetuned_model(adapter_path: str):
    """Load base model with fine-tuned LoRA adapters."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42,
    )
    model.load_adapter(adapter_path, adapter_name="default")
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_base_model():
    """Load base model without any fine-tuning."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_summary(model, tokenizer, instruction: str, input_text: str) -> str:
    """Generate a structured summary for a single review."""
    prompt = f"<s>[INST] {instruction}\n\n{input_text} [/INST]\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return response.strip()


def generate_all_predictions(model, tokenizer, test_data: list, label: str) -> list:
    """Generate predictions for the entire test set."""
    predictions = []
    for i, example in enumerate(test_data):
        if i % 25 == 0:
            print(f"  {label}: {i}/{len(test_data)}...")
        pred = generate_summary(
            model, tokenizer, example["instruction"], example["input"]
        )
        predictions.append(pred)
    return predictions


def compute_rouge(predictions: list, references: list) -> float:
    """Compute average ROUGE-L F1."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score["rougeL"].fmeasure)
    return sum(scores) / len(scores)


def compute_bertscore(predictions: list, references: list) -> float:
    """Compute average BERTScore F1."""
    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn(
        predictions, references,
        lang="en",
        model_type="roberta-large",
        verbose=True,
        batch_size=8,
    )
    return F1.mean().item()


def compute_llm_judge(
    test_data: list,
    predictions: list,
    references: list,
    api_key: str,
    sample_size: int = 50,
) -> dict:
    """Score predictions using GPT-4o-mini as judge."""
    import openai

    client = openai.OpenAI(api_key=api_key)

    JUDGE_PROMPT = """You are evaluating a product review summarizer.

ORIGINAL REVIEW: {review}
REFERENCE SUMMARY: {reference}
MODEL OUTPUT: {prediction}

Score the model output on these criteria (1-5 each):
1. STRUCTURE: Does it have clear Pros, Cons, and Verdict sections?
2. ACCURACY: Are the pros/cons actually mentioned in the review?
3. COMPLETENESS: Does it capture the key points?
4. CONCISENESS: Is it appropriately brief without losing meaning?

Respond in JSON only:
{{"structure": N, "accuracy": N, "completeness": N, "conciseness": N, "overall": N}}"""

    random.seed(42)
    indices = random.sample(range(len(predictions)), min(sample_size, len(predictions)))

    scores = []
    for i, idx in enumerate(indices):
        if i % 10 == 0:
            print(f"  Judging: {i}/{len(indices)}...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        review=test_data[idx]["input"],
                        reference=references[idx],
                        prediction=predictions[idx],
                    ),
                }],
                temperature=0,
                max_tokens=100,
                response_format={"type": "json_object"},
            )
            score = json.loads(response.choices[0].message.content)
            scores.append(score)
        except Exception as e:
            print(f"  Error: {e}")

    # Average scores
    result = {}
    for key in ["structure", "accuracy", "completeness", "conciseness", "overall"]:
        values = [s[key] for s in scores if key in s]
        result[key] = sum(values) / len(values) if values else 0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--output", type=str, default="./evaluation/eval_results.json")
    parser.add_argument("--openai-key", type=str, default=None)
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()

    # Load test data
    with open(args.test_data) as f:
        test_data = json.load(f)
    references = [ex["output"] for ex in test_data]
    print(f"Test examples: {len(test_data)}")

    # Generate fine-tuned predictions
    print("\nLoading fine-tuned model...")
    ft_model, ft_tokenizer = load_finetuned_model(args.adapter_path)
    ft_predictions = generate_all_predictions(
        ft_model, ft_tokenizer, test_data, "Fine-tuned"
    )
    del ft_model, ft_tokenizer
    torch.cuda.empty_cache()

    # Generate base model predictions
    print("\nLoading base model...")
    base_model, base_tokenizer = load_base_model()
    base_predictions = generate_all_predictions(
        base_model, base_tokenizer, test_data, "Base"
    )
    del base_model, base_tokenizer
    torch.cuda.empty_cache()

    # ROUGE-L
    print("\nComputing ROUGE-L...")
    ft_rouge = compute_rouge(ft_predictions, references)
    base_rouge = compute_rouge(base_predictions, references)
    rouge_improvement = ((ft_rouge - base_rouge) / base_rouge) * 100

    print(f"  Base: {base_rouge:.4f}")
    print(f"  Fine-tuned: {ft_rouge:.4f}")
    print(f"  Improvement: {rouge_improvement:.1f}%")

    results = {
        "rouge_l": {
            "base": base_rouge,
            "finetuned": ft_rouge,
            "improvement_pct": rouge_improvement,
        }
    }

    # BERTScore
    if not args.skip_bertscore:
        print("\nComputing BERTScore...")
        ft_bert = compute_bertscore(ft_predictions, references)
        base_bert = compute_bertscore(base_predictions, references)
        bert_improvement = ((ft_bert - base_bert) / base_bert) * 100

        print(f"  Base: {base_bert:.4f}")
        print(f"  Fine-tuned: {ft_bert:.4f}")
        print(f"  Improvement: {bert_improvement:.1f}%")

        results["bertscore"] = {
            "base": base_bert,
            "finetuned": ft_bert,
            "improvement_pct": bert_improvement,
        }

    # LLM-as-Judge
    api_key = args.openai_key or os.getenv("OPENAI_API_KEY")
    if not args.skip_judge and api_key:
        print("\nRunning LLM-as-Judge...")
        ft_judge = compute_llm_judge(
            test_data, ft_predictions, references, api_key
        )
        base_judge = compute_llm_judge(
            test_data, base_predictions, references, api_key
        )
        results["llm_judge"] = {"base": base_judge, "finetuned": ft_judge}

        print(f"\n  {'Metric':<16} {'Base':>8} {'Fine-tuned':>12}")
        for key in ["structure", "accuracy", "completeness", "conciseness", "overall"]:
            print(f"  {key:<16} {base_judge[key]:>8.2f} {ft_judge[key]:>12.2f}")

    # Save results
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()