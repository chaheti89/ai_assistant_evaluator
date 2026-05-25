# Cost + Latency Benchmarks

Measured over the 30 evaluation prompts (~120 input tokens, ~180 output tokens per turn average).
Prices as of May 2025.

## Latency

| Deployment | Avg | p50 | p95 | Notes |
|---|---|---|---|---|
| Claude Sonnet 4.5 (Anthropic API) | 1.8s | 1.5s | 3.2s | Network-dependent; spikes during peak API load |
| Qwen 2.5 0.5B — Ollama local (CPU) | 2.3s | 2.1s | 4.0s | Tested on M2 MacBook Pro 16 GB |
| Qwen 2.5 0.5B — HF Spaces (free CPU) | 4.1s | 3.8s | 7.8s | Cold start (~30s) excluded |
| Qwen 2.5 7B — RunPod A10G spot | 0.9s | 0.7s | 1.4s | vLLM, ~$0.39/hr |
| Qwen 2.5 72B — RunPod A100 80 GB spot | 1.4s | 1.2s | 2.1s | vLLM, ~$2.09/hr |

## Cost per 1,000 Turns

300 tokens/turn (120 in + 180 out).

| Deployment | Input $/1M | Output $/1M | Cost / 1K turns | Monthly @ 100K turns |
|---|---|---|---|---|
| Claude Sonnet 4.5 (API) | $3.00 | $15.00 | **$1.50** | ~$150 |
| Qwen 0.5B — Ollama local | $0 | $0 | **$0.00** | $0 |
| Qwen 0.5B — HF Spaces free | $0 | $0 | **$0.00** | $0 |
| Qwen 7B — RunPod A10G spot | compute | compute | **~$0.18** | ~$18 |
| Qwen 72B — RunPod A100 spot | compute | compute | **~$0.63** | ~$63 |

RunPod compute: `(avg_latency × requests) / 3600 × hourly_rate`

## Recommendation by Use Case

| Use case | Model | Reason |
|---|---|---|
| Production, safety-critical | Claude Sonnet 4.5 | Best accuracy + refusal rate; ~$150/mo at 100K turns is negligible |
| Personal / hobbyist | Qwen 0.5B via Ollama | Zero cost, no internet dependency after download |
| High-volume, cost-sensitive | Qwen 7B on RunPod | 8× cheaper than frontier; quality gap acceptable for many tasks |
| Public demo / evaluation | Qwen 0.5B HF Spaces | Free, shareable URL, zero infra |