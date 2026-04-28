#!/usr/bin/env python3
"""
Example client for the Product Review Summarizer API.

Usage:
    python serving/client.py --review "Great laptop but heavy" --rating 4
    python serving/client.py --batch reviews.json
    python serving/client.py  # Demo mode
"""

import argparse
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

SYSTEM_PROMPT = (
    "You are a product review analyzer. Given a customer review, "
    "generate a structured summary with pros, cons, and an overall verdict."
)


def summarize_review(review_text: str, rating: int) -> str:
    response = client.chat.completions.create(
        model="review-summarizer",
        messages=[
            {
                "role": "user",
                "content": f"{SYSTEM_PROMPT}\n\nReview: {review_text}\nRating: {rating}/5",
            }
        ],
        temperature=0.7,
        max_tokens=256,
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=str, help="Single review text")
    parser.add_argument("--rating", type=int, default=3)
    parser.add_argument("--batch", type=str, help="Path to JSON file with reviews")
    args = parser.parse_args()

    if args.review:
        print(summarize_review(args.review, args.rating))
    elif args.batch:
        with open(args.batch) as f:
            reviews = json.load(f)
        for i, r in enumerate(reviews):
            print(f"\n--- Review {i + 1} ---")
            print(summarize_review(r["text"], r["rating"]))
    else:
        demo = (
            "This tablet is decent for the price. The screen is bright "
            "and colorful, and the battery easily lasts 8 hours. However, "
            "the speakers are really tinny, and it gets noticeably slow "
            "when you have more than 3 apps open."
        )
        print("--- Demo ---\n")
        print(summarize_review(demo, 3))


if __name__ == "__main__":
    main()
