"""
Generate structured labels for product reviews using GPT-4o-mini.

Uses Pydantic schema validation to ensure consistent, clean output.
Each review gets: pros, cons, verdict, and a 1-5 rating.

Usage:
    python src/label_generation.py \
        --input ./data/processed/balanced_reviews.parquet \
        --output ./data/processed/labeled_reviews.parquet \
        --api-key sk-...
"""

import argparse
import json
import os
import time

import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # Reads from .env file automatically
from pydantic import BaseModel, field_validator
from typing import List


# ============================================
# Pydantic Schema — validates at the source
# ============================================

class ReviewLabel(BaseModel):
    """Schema for a structured product review summary.

    Handles all edge cases from GPT-4o-mini output:
    - Float ratings (3.5) → rounded to int
    - Out-of-range ratings (7) → clamped to 1-5
    - Sets instead of lists → converted
    - Empty pros/cons → filled with defaults
    - None verdict → filled with default
    """

    pros: List[str]
    cons: List[str]
    verdict: str
    rating: int

    @field_validator("rating", mode="before")
    @classmethod
    def clamp_rating_1_to_5(cls, v):
        return max(1, min(5, int(round(float(v)))))

    @field_validator("pros", "cons", mode="before")
    @classmethod
    def ensure_non_empty_string_list(cls, v):
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            v = list(v)
        v = [str(item) for item in v if item is not None]
        if len(v) == 0:
            v = ["No notable points mentioned"]
        return v

    @field_validator("verdict", mode="before")
    @classmethod
    def ensure_verdict_is_string(cls, v):
        if v is None or v == "":
            return "Mixed experience overall."
        return str(v)


# ============================================
# Labeling Prompt
# ============================================

LABELING_PROMPT = """Analyze this product review and extract a structured summary.

Review: {review_text}
Sentiment: {sentiment}

Respond in EXACTLY this JSON format and nothing else:
{{
  "pros": ["pro1", "pro2"],
  "cons": ["con1", "con2"],
  "verdict": "One sentence overall verdict",
  "rating": N
}}

Rules:
- Extract 1-4 pros and 1-4 cons from the ACTUAL review content
- Each pro/con should be a concise phrase (5-15 words)
- The verdict should be one clear sentence
- Assign a rating that is exactly one of: 1, 2, 3, 4, or 5 (integers only)
- If the review has NO positives, use: ["No notable strengths mentioned"]
- If the review has NO negatives, use: ["No significant drawbacks mentioned"]
- Be SPECIFIC — pull actual details from the review, not generic phrases"""


def generate_label(
    client: openai.OpenAI,
    review_text: str,
    sentiment: int,
    max_chars: int = 800,
    max_retries: int = 3,
) -> dict | None:
    """Generate a structured label for a single review.

    Returns a validated dictionary or None if all retries fail.
    """
    sentiment_str = "positive" if sentiment == 1 else "negative"

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": LABELING_PROMPT.format(
                            review_text=review_text[:max_chars],
                            sentiment=sentiment_str,
                        ),
                    }
                ],
                temperature=0.3,
                max_tokens=300,
                response_format={"type": "json_object"},
            )

            raw_result = json.loads(response.choices[0].message.content)

            # Validate + clean with Pydantic
            label = ReviewLabel(**raw_result)
            return label.model_dump()

        except json.JSONDecodeError:
            print(f"  Attempt {attempt + 1}: Invalid JSON, retrying...")
            time.sleep(2)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt + 1}: {type(e).__name__}, retrying...")
                time.sleep(2)
            else:
                print(f"  FAILED after {max_retries} attempts: {e}")

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--api-key", type=str, default=None, help="Overrides OPENAI_API_KEY from .env")
    parser.add_argument("--max-chars", type=int, default=800)
    args = parser.parse_args()

    # Priority: CLI arg > .env file > environment variable
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No API key found. Set OPENAI_API_KEY in .env or pass --api-key")

    client = openai.OpenAI(api_key=api_key)
    df = pd.read_parquet(args.input)

    labels = []
    failed = 0

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 100 == 0:
            print(f"Labeling... {i}/{len(df)} ({failed} failed)")

        label = generate_label(
            client, row["text"], row["sentiment"], max_chars=args.max_chars
        )
        if label:
            labels.append(label)
        else:
            labels.append(None)
            failed += 1

    df["label"] = labels
    df = df.dropna(subset=["label"])
    df["rating"] = df["label"].apply(lambda x: x["rating"])

    df.to_parquet(args.output)
    print(f"\nDone! {len(df)} labeled, {failed} failed")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()