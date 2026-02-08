# When Does Self-Hosting an LLM Start to Make Sense?

A practical breakdown of the economics, based on real experience spinning up ephemeral GPU clusters on GCP to serve large open-weight models (700B+ params) via vLLM.

## The Core Question

If you're spending $1,000-5,000/month on LLM API calls (Gemini, Claude, OpenAI, etc.), could you save money by renting GPUs and running an open-weight model yourself?

**Short answer:** It depends entirely on your workload pattern. Interactive usage? Probably not. Predictable daily batch jobs? Quite possibly.

## The Two Pricing Models

**API pricing** scales with tokens. You pay per million input/output tokens regardless of how long the request takes. Double the tokens, double the cost.

**Self-hosted GPU pricing** scales with time. You rent hardware by the hour. Whether you push 100K tokens or 2.5M tokens through the GPUs in that hour, the cost is the same. The tokens themselves are free -- you're paying for the clock.

This distinction is everything.

## Real GPU Costs (GCP Spot, Feb 2026)

| Setup | Hourly Cost |
|---|---|
| 8x A100-80GB (spot) | ~$20/hr |
| 8x H100 (spot) | ~$26/hr |

These are the actual prices SkyPilot reports for running a 1T parameter MoE model (Kimi K2.5) on GCP spot instances in us-central1. On-demand is roughly 2-3x more.

## API Costs for Comparison (Gemini 3.0 Pro, Feb 2026)

| | Per 1M Tokens |
|---|---|
| Input | $2.00 |
| Output | $12.00 |

Other frontier APIs (Claude, GPT) are in a similar range, some higher.

## The Break-Even Math

At $20/hr GPU cost, your hourly spend is equivalent to:
- **1.68M output tokens** at Gemini API rates ($12/1M)
- **10M input tokens** at Gemini API rates ($2/1M)
- For a mixed workload (10:1 input:output): roughly 16M input + 1.6M output tokens

The question becomes: **can your self-hosted setup process that many tokens in an hour?**

## Throughput: The Key Variable

A large MoE model (1T params, 32B active) on 8x A100-80GB with vLLM:

| Concurrency | Est. Output Tokens/sec | Output Tokens/hr |
|---|---|---|
| 1 request (interactive) | ~60-90 | ~250K |
| 4 concurrent | ~250-400 | ~1.1M |
| 16 concurrent | ~500-670 | ~2.0M |
| 32+ concurrent (saturated) | ~600-800 | ~2.5M |

Input tokens (prefill) process much faster -- ~5,000-15,000 tokens/sec -- because they're parallelized. Output tokens are the bottleneck since they're generated autoregressively.

**At low concurrency (1-2 requests, interactive R&D), you produce ~250-500K output tokens/hr. The API is cheaper.**

**At high concurrency (16+ requests, batch processing), you produce ~2M+ output tokens/hr. Self-hosting starts winning.**

## The Workload Pattern Matters More Than the Spend

### Interactive / Ad-hoc Usage
You fire requests one at a time throughout the day. The GPUs sit mostly idle between requests. You're paying $20/hr for a machine that's working 5% of the time.

**Verdict: Use the API.** You'd need to be spending $10k+/month on API calls for the idle GPU time to make sense, and even then you'd want a reserved instance, not ephemeral.

### Predictable Daily Batch
You have a pipeline that runs once a day, processing a known set of documents/prompts. All jobs can be queued and run concurrently. You spin up the cluster, saturate it for 1-2 hours, tear it down.

**This is where self-hosting shines.** The math for a ~$1,500/month API spend:

| Daily GPU Time | Monthly GPU Cost | Monthly API Equivalent | Savings |
|---|---|---|---|
| 1 hour/day | $600/mo | $1,500/mo | **60%** |
| 1.5 hours/day | $900/mo | $1,500/mo | **40%** |
| 2 hours/day | $1,200/mo | $1,500/mo | **20%** |
| 2.5 hours/day | $1,500/mo | $1,500/mo | Break-even |

And the savings scale linearly with spend. At $5,000/month API:

| Daily GPU Time | Monthly GPU Cost | Savings |
|---|---|---|
| 2 hours/day | $1,200/mo | **76%** |
| 4 hours/day | $2,400/mo | **52%** |

### Input-Heavy Workloads Favor Self-Hosting

If your workload is mostly input tokens (large context, document processing) with limited output (classifications, short summaries), self-hosting benefits doubly:

1. **Input prefill is fast on GPUs** (~10,000 tok/s), so your batch completes quicker
2. **You avoid paying the 6x output premium** ($12/1M vs $2/1M) on the API -- GPUs don't distinguish

A workload processing 16M input + 1.5M output tokens daily might take ~63 minutes of GPU time (~$21/day) vs ~$48/day on the API.

## The Complexity Tax

Self-hosting isn't free operationally. Here's what you're signing up for:

### Infrastructure Orchestration
You need to automate: spin up cluster, wait for ready, submit batch, wait for completion, tear down. Tools like SkyPilot make the infra part easy, but you still need to build the job submission pipeline.

### Batch Coordination
To maximize GPU utilization, all your LLM jobs need to happen in the same time window. If you have other pipeline stages (document processing, OCR, etc.) that produce LLM inputs throughout the day, you either:
- Buffer jobs and run them all at once (adds latency)
- Keep the cluster running longer (adds cost)

### Spot Instance Preemption
Spot instances save 60-80% but can be reclaimed mid-batch. You need retry logic or checkpointing. For a 1-hour batch this is usually fine (low probability of preemption), but for multi-hour jobs it becomes a real concern.

### Model Quality Mismatch
Open-weight models are good but may not match the frontier API model for your specific task. You need to validate that Kimi K2.5 / Llama / DeepSeek / etc. actually produces acceptable outputs for your use case before committing to the infrastructure.

### Maintenance Burden
vLLM updates, CUDA driver updates, model weight updates, quota management, cloud billing monitoring. None of this is hard, but it's ongoing work that "just calling an API" doesn't require.

## Decision Framework

| Monthly API Spend | Workload Pattern | Recommendation |
|---|---|---|
| < $500 | Any | **Use the API.** Not worth the complexity. |
| $500 - $2,000 | Interactive | **Use the API.** GPUs will sit idle. |
| $500 - $2,000 | Daily batch | **Consider self-hosting.** 30-60% savings possible if batch completes in 1-2 hrs. Run a benchmark first. |
| $2,000 - $10,000 | Daily batch | **Strong case for self-hosting.** Savings of $1,000-6,000/month. Worth the engineering investment. |
| $10,000+ | Any pattern | **Self-host or negotiate enterprise API pricing.** At this spend, even interactive usage can justify always-on GPU clusters. |

## How to Validate Before Committing

1. **Measure your actual token volumes.** Check your API dashboard for daily input/output token counts.
2. **Spin up a test cluster.** Use SkyPilot + vLLM to launch an ephemeral GPU cluster (~$20 for a 1-hour test).
3. **Run your actual workload.** Not synthetic benchmarks -- your real prompts at your real concurrency.
4. **Measure wall-clock time.** How long does your daily batch take on self-hosted hardware?
5. **Calculate:** `daily_gpu_hours * $20 * 30` vs your current monthly API bill.

If the self-hosted monthly cost is less than 60% of your API bill, it's probably worth it. The remaining 40% margin covers your engineering time, spot preemption retries, and the occasional day where things break.

## What We're Using

- **[SkyPilot](https://github.com/skypilot-org/skypilot)** - Ephemeral GPU cluster management (launch, auto-shutdown, multi-cloud)
- **[vLLM](https://github.com/vllm-project/vllm)** - High-throughput LLM serving with continuous batching
- **GCP Spot Instances** - 60-80% cheaper than on-demand, good enough for batch workloads
- See the [main README](../README.md) for setup instructions
