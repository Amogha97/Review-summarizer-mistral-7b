"""
Data preparation pipeline for the Product Review Summarizer.

Loads Amazon Polarity reviews, cleans them, filters to English,
balances across sentiments, and prepares for label generation.

Usage:
    python src/data_prep.py --output-dir ./data/processed
"""

import argparse
import pandas as pd
from datasets import load_dataset
from langdetect import detect, LangDetectException
from sklearn.model_selection import train_test_split


def is_english(text: str) -> bool:
    """Detect if text is English."""
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def load_and_clean(sample_size: int = 10000) -> pd.DataFrame:
    """Load Amazon Polarity and apply cleaning filters."""
    print("Loading Amazon Polarity dataset...")
    dataset = load_dataset("amazon_polarity", split="train")
    df = pd.DataFrame(dataset)
    df = df.rename(columns={"content": "text", "label": "sentiment"})

    print(f"Raw reviews: {len(df):,}")

    # Filter by length
    df = df[df["text"].str.len().between(100, 1000)].copy()
    df = df.drop_duplicates(subset=["text"])
    print(f"After cleaning: {len(df):,}")

    # Sample for language detection (don't process all 3.6M)
    pos_sample = df[df["sentiment"] == 1].sample(n=sample_size, random_state=42)
    neg_sample = df[df["sentiment"] == 0].sample(n=sample_size, random_state=42)
    sample_df = pd.concat([pos_sample, neg_sample]).reset_index(drop=True)

    print(f"Sampled {len(sample_df)} for language detection...")
    sample_df["lang"] = sample_df["text"].apply(
        lambda t: detect(t) if t else "unknown"
    )

    english_df = sample_df[sample_df["lang"] == "en"].copy()
    english_df = english_df.drop(columns=["lang"])
    print(f"English reviews: {len(english_df):,}")

    return english_df


def balance_dataset(
    df: pd.DataFrame, per_sentiment: int = 1500
) -> pd.DataFrame:
    """Balance positive and negative reviews."""
    pos = df[df["sentiment"] == 1].sample(
        n=min(per_sentiment, len(df[df["sentiment"] == 1])), random_state=42
    )
    neg = df[df["sentiment"] == 0].sample(
        n=min(per_sentiment, len(df[df["sentiment"] == 0])), random_state=42
    )
    balanced = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(
        drop=True
    )
    print(f"Balanced dataset: {len(balanced)} reviews")
    print(f"  Positive: {(balanced['sentiment'] == 1).sum()}")
    print(f"  Negative: {(balanced['sentiment'] == 0).sum()}")
    return balanced


def stratified_split(
    data: list, ratings: list, test_size: float = 0.2, random_state: int = 42
) -> tuple:
    """Split data with stratification on ratings."""
    train_data, temp_data, train_ratings, temp_ratings = train_test_split(
        data, ratings, test_size=test_size, random_state=random_state, stratify=ratings
    )
    val_data, test_data, _, _ = train_test_split(
        temp_data, temp_ratings, test_size=0.5, random_state=random_state,
        stratify=temp_ratings
    )
    return train_data, val_data, test_data


def oversample_minority(
    train_examples: list, train_ratings: list, target_per_class: int = None
) -> list:
    """Duplicate minority class examples to balance training set."""
    from collections import Counter
    import random

    counts = Counter(train_ratings)
    if target_per_class is None:
        target_per_class = max(counts.values())

    oversampled = list(train_examples)

    for rating, count in counts.items():
        if count < target_per_class:
            deficit = target_per_class - count
            class_examples = [
                ex for ex, r in zip(train_examples, train_ratings) if r == rating
            ]
            random.seed(42)
            extras = random.choices(class_examples, k=deficit)
            oversampled.extend(extras)
            print(
                f"  Rating {rating}: {count} -> {count + deficit} (+{deficit} duplicates)"
            )

    random.shuffle(oversampled)
    return oversampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./data/processed")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--per-sentiment", type=int, default=1500)
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and clean
    df = load_and_clean(sample_size=args.sample_size)

    # Balance
    balanced_df = balance_dataset(df, per_sentiment=args.per_sentiment)

    # Save
    output_path = os.path.join(args.output_dir, "balanced_reviews.parquet")
    balanced_df.to_parquet(output_path)
    print(f"\nSaved to {output_path}")
    print("Next step: Run label_generation.py to create training labels")


if __name__ == "__main__":
    main()
