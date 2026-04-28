"""
QLoRA fine-tuning script for the Product Review Summarizer.

Loads a pre-trained Mistral-7B, applies LoRA adapters, trains on
labeled product reviews, and saves adapters + MLflow logs.

Usage:
    python src/train.py \
        --config configs/training_config.yaml \
        --train-data ./data/processed/train_dataset_sft.jsonl \
        --val-data ./data/processed/val_dataset_sft.jsonl \
        --output-dir ./outputs
"""

import argparse
import os
import time

import mlflow
import torch
import yaml
from datasets import Dataset
from dotenv import load_dotenv
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
from unsloth import FastLanguageModel

load_dotenv()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_experiment(
    config: dict,
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: str,
    run_name: str,
):
    """Run a single training experiment with MLflow logging."""
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {run_name}")
    print(f"{'=' * 60}\n")

    os.environ["MLFLOW_RUN_NAME"] = run_name

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"]["name"],
        max_seq_length=config["model"]["max_seq_length"],
        dtype=None,
        load_in_4bit=config["model"]["load_in_4bit"],
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["rank"],
        target_modules=config["lora"]["target_modules"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias=config["lora"]["bias"],
        use_gradient_checkpointing=config["lora"]["gradient_checkpointing"],
        random_state=config["training"]["seed"],
    )
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, f"results_{run_name}"),
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["per_device_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        lr_scheduler_type=config["training"]["lr_scheduler"],
        warmup_steps=config["training"]["warmup_steps"],
        weight_decay=config["training"]["weight_decay"],
        fp16=config["training"]["fp16"],
        bf16=not config["training"]["fp16"],
        optim=config["training"]["optimizer"],
        logging_steps=10,
        eval_strategy=config["evaluation"]["strategy"],
        eval_steps=config["evaluation"]["eval_steps"],
        save_strategy="steps",
        save_steps=config["evaluation"]["save_steps"],
        save_total_limit=config["evaluation"]["save_total_limit"],
        load_best_model_at_end=config["evaluation"]["load_best_model_at_end"],
        metric_for_best_model=config["evaluation"]["metric_for_best_model"],
        seed=config["training"]["seed"],
        report_to="mlflow",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config["model"]["max_seq_length"],
        packing=False,
        args=training_args,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config["early_stopping"]["patience"]
            )
        ],
    )

    # Train
    start_time = time.time()
    stats = trainer.train()
    train_time = time.time() - start_time

    # Save adapters IMMEDIATELY to output dir
    adapter_dir = os.path.join(output_dir, f"adapters_{run_name}")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Adapters saved to: {adapter_dir}")

    # Log extra params to MLflow
    runs = mlflow.search_runs(order_by=["start_time DESC"])
    if len(runs) > 0:
        with mlflow.start_run(run_id=runs.iloc[0]["run_id"]):
            mlflow.log_params({
                "lora_rank": config["lora"]["rank"],
                "lora_alpha": config["lora"]["alpha"],
                "quantization": config["model"]["quantization"],
            })
            mlflow.log_metric("training_time_minutes", train_time / 60)

    print(f"\nTrain loss: {stats.training_loss:.4f}")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Time: {train_time / 60:.1f} min")

    # Cleanup
    del model, tokenizer, trainer
    torch.cuda.empty_cache()

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--run-name", type=str, default="qlora-run")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)

    # Set up MLflow — reads from .env or falls back to defaults
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "product-review-summarizer"))

    # Load datasets
    train_dataset = Dataset.from_json(args.train_data)
    val_dataset = Dataset.from_json(args.val_data)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Run experiment
    run_experiment(config, train_dataset, val_dataset, args.output_dir, args.run_name)


if __name__ == "__main__":
    main()