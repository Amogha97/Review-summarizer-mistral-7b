# Serving the Product Review Summarizer

## Quick Start

### Option 1: Direct vLLM (development)
```bash
pip install vllm
python serve.py --adapter-path ../adapters_run4-2ep-lr1.5e4-r16
```

### Option 2: Docker (production)
```bash
docker compose up --build
```

### Test the API
```bash
# Using curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "review-summarizer",
    "messages": [{"role": "user", "content": "You are a product review analyzer...\n\nReview: Great laptop but heavy\nRating: 4/5"}],
    "temperature": 0.7,
    "max_tokens": 256
  }'

# Using the Python client
python client.py --review "Great laptop but heavy" --rating 4
```

## Requirements
- NVIDIA GPU with 16+ GB VRAM (T4, RTX 3090, A100)
- CUDA 12.1+
- For Docker: nvidia-container-toolkit

## Cost Estimates for Cloud Deployment

| Provider | Instance | GPU | Cost/hr | Monthly (24/7) |
|----------|----------|-----|---------|-----------------|
| RunPod | Community | RTX 3090 | $0.22 | ~$160 |
| AWS | g4dn.xlarge | T4 | $0.53 | ~$380 |
| GCP | n1 + T4 | T4 | $0.35 | ~$250 |
| Lambda Labs | On-demand | A10 | $0.75 | ~$540 |
