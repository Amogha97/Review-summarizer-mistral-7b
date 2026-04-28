#!/usr/bin/env python3
"""
Product Review Summarizer — vLLM Serving Script.

Starts an OpenAI-compatible API server with the fine-tuned model.

Usage:
    python serving/serve.py --adapter-path ./adapters_run4-2ep-lr1.5e4-r16
"""

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Serve the review summarizer")
    parser.add_argument("--adapter-path", type=str, help="Path to LoRA adapters")
    parser.add_argument(
        "--model-path",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Path to base or merged model",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--max-model-len", type=int, default=2048)
    args = parser.parse_args()

    cmd = [
        "vllm", "serve", args.model_path,
        "--host", args.host,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--enable-prefix-caching",
        "--gpu-memory-utilization", "0.9",
    ]

    if args.adapter_path:
        cmd.extend([
            "--enable-lora",
            "--lora-modules", f"review-summarizer={args.adapter_path}",
            "--max-lora-rank", "16",
        ])

    print(f"Starting vLLM server...")
    print(f"API: http://{args.host}:{args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
