# streaming-qwen

RAM-streamed MoE inference for Qwen3-Coder-30B-A3B on consumer hardware (24GB VRAM + 64GB RAM).

## Key Insight

Layers 6-35 have near-zero KL divergence on expert substitution. Cache misses are essentially free - just use whatever expert is already hot.

## Memory Layout

| Location | Content | Size (FP8) |
|----------|---------|------------|
| VRAM | Attention + embeddings | ~1.5 GB |
| VRAM | Pinned experts (layers 0-5, 36-47) | ~5.1 GB |
| VRAM | KV cache (128K context) | ~12.5 GB |
| VRAM | Hot cache for streaming layers | ~2 GB |
| **VRAM Total** | | **~21 GB** |
| RAM | Fungible experts (layers 6-35) | ~12.5 GB |

## Usage

```bash
# Install
pip install -e .

# CLI
streaming-qwen                      # Interactive REPL
streaming-qwen --prompt "Hello"     # Single prompt
streaming-qwen --server             # OpenAI-compatible API
streaming-qwen --benchmark          # Run benchmarks

# Docker
docker compose build
docker compose up
```

## API

OpenAI-compatible endpoints at `http://localhost:8000`:

- `POST /v1/chat/completions` - Chat with tool calling
- `POST /v1/completions` - Legacy completions
- `GET /v1/models` - List models
- `GET /stats` - Cache hit rates, memory usage
- `GET /health` - Health check

## Configuration

```bash
# Custom layer pinning
streaming-qwen --vram-layers "0-7,40-47" --cache-size 32

# Connect opencode
OPENAI_API_BASE=http://localhost:8000/v1 opencode
```
