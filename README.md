# llm-serving-go

A from-scratch LLM inference engine written in Go. Built to understand how transformer models actually work under the hood — no ML frameworks, no abstractions, just raw math on weights.

## What this implements

- BPE tokenizer (byte-level, GPT-2 style)
- Safetensors weight loading
- RMSNorm
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU MLP
- Greedy token sampling
- Text generation loop

Model: SmolLM2-135M (HuggingFaceTB/SmolLM2-135M)

## Running

**1. Download model weights**

```bash
uvx huggingface-cli download HuggingFaceTB/SmolLM2-135M \
  --include "*.safetensors" "*.json" \
  --local-dir ./SmolLM2-135M
```

**2. Run**

```bash
go run .
```