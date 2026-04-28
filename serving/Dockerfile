FROM vllm/vllm-openai:latest

COPY ./adapters_run4-2ep-lr1.5e4-r16 /models/adapters

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["--model", "mistralai/Mistral-7B-Instruct-v0.3", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--max-model-len", "2048", \
     "--enable-prefix-caching", \
     "--gpu-memory-utilization", "0.9", \
     "--enable-lora", \
     "--lora-modules", "review-summarizer=/models/adapters", \
     "--max-lora-rank", "16"]
