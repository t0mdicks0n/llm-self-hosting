# Self-Hosted vs. API Economics

Date: 2026-02-09

## The Question

Is self-hosting Kimi K2.5 on cloud GPUs cheaper than using a frontier model API?

**Short answer: No, not for light R&D usage.**

## The Math

### GPU costs

| Setup | Hourly Cost | 2h Session |
|---|---|---|
| 8x A100-80GB (GCP spot) | ~$20/hr | ~$40 |
| 8x H100 (GCP spot) | ~$26/hr | ~$52 |

### API equivalent

**Gemini 3.0 Pro API pricing:** $2/1M input tokens, $12/1M output tokens.

| GPU Setup | Hourly Cost | Equivalent API Output Tokens | Equivalent API Input Tokens |
|---|---|---|---|
| 8x A100-80GB spot | $20/hr | 1.68M output tokens | 10M input tokens |
| 8x H100 spot | $26/hr | 2.13M output tokens | 12.8M input tokens |

For a typical mixed workload (4:1 input:output ratio), $20/hr buys the equivalent of ~4M input + 1M output tokens via the Gemini API.

### What we actually measured

Kimi K2.5 full precision on 8xH100 with vLLM 0.15.0:

| Workload | Throughput | Tokens/hr |
|---|---|---|
| Single request (short prompts) | 112 tok/s | ~400K |
| Single request (longer reasoning) | 62-80 tok/s | ~225K-290K |
| 2 concurrent requests | 108 tok/s per request | ~780K total |
| Higher concurrency | Untested, likely KV cache bottlenecked | — |

At typical R&D usage (1-2 concurrent requests, mix of short and long prompts), you'd generate **~225K-400K output tokens/hr**.

### Break-even analysis

The break-even point on 8xA100 spot ($20/hr) is **~1.68M output tokens/hr**. Our measured throughput is 4-7x below that.

To break even, you'd need to sustain **~15 concurrent requests** at 112 tok/s — which requires far more KV cache than the 0.64 GiB available in the full precision config. INT4 quantization would help here by freeing memory for larger KV cache.

**Bottom line:** At the usage levels where self-hosting makes economic sense, you're running a production serving cluster, not doing interactive R&D.

## When self-hosting makes sense anyway

Cost isn't the only variable. Self-hosting is worth it when you need:

- **Full model control** — quantization experiments, custom sampling strategies, weight surgery, fine-tuning
- **No rate limits** — APIs throttle you during peak hours; your own cluster doesn't
- **No content filtering** — APIs have safety filters that may block legitimate research queries
- **Privacy** — data never leaves your VPC, no third-party logging
- **Access to models not available via API** — Kimi K2.5 has no hosted API; open-weight MoE models are only available to self-host
- **Batch workloads at high concurrency** — if you can saturate 16+ concurrent requests, the economics flip

## Hidden costs not in the math

- **Debugging time:** Getting Kimi K2.5 running took a full day and ~$100-130 in GPU time burned on troubleshooting. This is a one-time cost amortized over future sessions, but it's real.
- **Cold start:** Each session burns ~15-20 min (and ~$6-8) just starting up before you can run a single query.
- **Opportunity cost:** Time spent fighting CUDA/NCCL/driver issues is time not spent on actual research.
- **Knowledge perishability:** The working configuration (vLLM 0.15.0 + driver 580 + NCCL 2.29.3) may break when any component updates. Maintenance is ongoing.
