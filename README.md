# Self-Hosting Kimi K2.5 on Ephemeral Cloud GPUs

Run large open-weight LLMs (1T+ parameters) on cloud GPUs for short 1-2 hour R&D sessions — testing quantization, parallelization, and inference techniques — then tear everything down automatically to avoid ongoing costs.

Uses [SkyPilot](https://github.com/skypilot-org/skypilot) for infrastructure management and [vLLM](https://github.com/vllm-project/vllm) for serving.

```bash
sky launch -c kimi kimi-k2.5.yaml --idle-minutes-to-autostop 20 --down --detach-run
```

One command. ~15-20 minutes later, **Kimi K2.5** (1 trillion parameter MoE, 32B active params, ~595GB weights) is serving on `localhost:8000` with an OpenAI-compatible API. Disconnect SSH and it auto-deletes after 20 minutes of idle.

### Results (8xH100 80GB, GCP)

| Metric | Value |
|---|---|
| Sequential throughput | 112 tok/s |
| Concurrent throughput (2x) | 108 tok/s per request (~4% drop) |
| Cold start (GCS cached) | ~15-20 min |
| Cost per 2h session (H100 spot) | ~$52 |
| Context length (full precision) | 8K tokens |
| Context length (INT4) | 32K tokens |

### What we learned

Getting here was significantly harder than expected. The YAML files encode solutions to **12 distinct issues** discovered through a full day of debugging on a live 8xH100 cluster. The major ones:

1. **GCP's GIB libraries break single-node NCCL** — Pre-installed networking middleware for multi-node InfiniBand silently intercepts NCCL and breaks P2P/NVSwitch on single-node setups. Required complete removal. Took ~15 failed attempts to diagnose.

2. **NVIDIA driver 570 too old for Marlin kernels** — Kimi K2.5 uses compressed tensor weights that need Marlin MoE CUDA kernels, which require PTX features only available in driver 580+. The model card doesn't mention this.

3. **vLLM defaults don't work for 1T models** — Default `gpu_memory_utilization=0.9` leaves negative KV cache space. Default `max_num_seqs=1024` causes OOM during warmup. Default context length (256K) is impossible with the remaining memory. All three had to be manually tuned.

4. **No local iteration loop** — Every debugging attempt required 5+ minutes to reload 595GB of weights into GPU memory. Total GPU time burned on debugging: ~4-5 hours (~$100-130).

The deeper issue is that the ML infrastructure stack (GPU driver, CUDA, PyTorch, vLLM, NCCL, model format) has no cross-layer dependency resolution. Each tool documents its own happy path and assumes the rest of the stack is someone else's problem.

### Is self-hosting worth it?

For pure cost, no. At typical R&D usage (1-2 concurrent requests), self-hosting generates ~225K-400K output tokens/hr — well below the ~1.68M token break-even point vs. Gemini 3.0 Pro API pricing. Self-hosting makes sense when you need full model control, no rate limits, privacy, access to models not available via API, or heavy batch workloads. See [full cost analysis](#self-hosted-vs-gemini-30-pro-api) below.

### Detailed notes

- [notes/kimi-k2.5-setup-log.md](notes/kimi-k2.5-setup-log.md) — Every issue we hit and how we fixed it, plus full VM bash history
- [notes/kimi-k2.5-benchmark-results.md](notes/kimi-k2.5-benchmark-results.md) — Performance measurements on 8xH100
- [notes/why-this-is-so-painful.md](notes/why-this-is-so-painful.md) — Honest analysis of all 12 issues and why MLOps is so flaky
- [Self-hosted vs. API economics](#self-hosted-vs-gemini-30-pro-api) — Cost comparison with Gemini 3.0 Pro API

---

## Prerequisites

1. **GCP account** with billing enabled and `gcloud` CLI configured
2. **GPU quota** for A100-80GB or H100 (at least 8 GPUs in your target region)
3. **Python 3.9+**

## Quick Start

### 1. Install SkyPilot

```bash
pip install "skypilot[gcp]"
sky check   # Verify GCP credentials are working
```

### 2. Seed Model Weights (First Time Only)

The model is ~595GB. SkyPilot's default GCS FUSE mount is unusably slow for files this large (1+ hours for both reads and writes). Instead, we pre-seed a GCS bucket using a cheap CPU-only VM (~$0.30/hr), then GPU launches pull via `gsutil` which gets ~1.2 GiB/s:

```bash
sky launch -c seed seed-weights.yaml --idle-minutes-to-autostop 60 --down --detach-run
```

This downloads the weights from HuggingFace to local disk, then uploads to GCS via `gsutil`. Takes ~30 min total. You can disconnect — the job runs detached on the VM.

Monitor progress: `sky logs seed`

Once complete, the GCS bucket is seeded and all future GPU launches pull from there instead of HuggingFace.

### 3. Launch Kimi K2.5

```bash
# Full precision (595GB model, 8x A100-80GB or H100, spot instances)
sky launch -c kimi kimi-k2.5.yaml --idle-minutes-to-autostop 20 --down --detach-run

# OR INT4 quantized (smaller, ~2x faster inference)
sky launch -c kimi kimi-k2.5-int4.yaml --idle-minutes-to-autostop 20 --down --detach-run
```

Flags:
- `-c kimi` names the cluster "kimi" (used for SSH, status, teardown)
- `--idle-minutes-to-autostop 20` auto-shuts down 20 min after SSH disconnects
- `--down` deletes the instance (not just stops it) to avoid any lingering costs
- `--detach-run` runs the vLLM server in the background so you can disconnect from `sky launch` output

Setup takes ~15-20 min total (GCS weight copy ~8 min, weight loading ~5.5 min, CUDA graph capture ~23s). The setup phase also upgrades the NVIDIA driver from 570 to 580 and removes GCP's GIB networking libraries — see `notes/kimi-k2.5-setup-log.md` for why.

### 4. Connect

**Terminal:**
```bash
sky ssh kimi
```

**VS Code:**
1. Install the "Remote - SSH" extension
2. `Cmd+Shift+P` > "Remote-SSH: Connect to Host"
3. Select "kimi" from the list (SkyPilot auto-configures your SSH config)
4. Open the terminal and run commands on the GPU instance

### 5. Copy Scripts & Test the Endpoint

First, copy the test scripts to the VM (from your local machine):
```bash
scp -O -r scripts/ kimi:~/scripts/
```

Then SSH in and test:
```bash
sky ssh kimi

# Check GPUs are available
nvidia-smi

# Check vLLM is serving
curl localhost:8000/v1/models

# Run test suite
python scripts/test_inference.py

# Run benchmark
python scripts/benchmark.py --rounds 3 --concurrent 2
```

Note: the `-O` flag is needed for scp to work with GCP VMs (forces legacy SCP protocol).

### 6. Tear Down

```bash
sky down kimi                  # Immediate teardown
# OR just disconnect SSH - auto-deletes after 20 min idle
```

## Available Profiles

| File | Model | GPUs | Context | Notes |
|---|---|---|---|---|
| `kimi-k2.5.yaml` | Kimi K2.5 (1T MoE) | 8x A100-80GB / H100 | 8K tokens | Full precision, 112 tok/s, spot |
| `kimi-k2.5-int4.yaml` | Kimi K2.5 NVFP4 | 8x A100-80GB / H100 | 32K tokens | INT4 quantized, spot |
| `seed-weights.yaml` | — | CPU only | — | Seeds GCS bucket with model weights (~$0.30/hr) |

Context length is limited by available KV cache memory after model weights are loaded. Full precision leaves only 0.64 GiB for KV cache per GPU. INT4 quantization frees significantly more memory, allowing 4x the context length.

## Useful SkyPilot Commands

```bash
sky status              # List running clusters
sky ssh kimi            # SSH into cluster
sky exec kimi -- CMD    # Run a command on the cluster
sky logs kimi           # View cluster logs
sky down kimi           # Tear down cluster
sky down --all          # Tear down ALL clusters (safety net)
sky cost-report         # View cost history
```

## Cost Estimates (per session)

| Setup | Hourly Cost | 2h Session |
|---|---|---|
| 8x A100-80GB (GCP spot) | ~$20/hr | ~$40 |
| 8x H100 (GCP spot) | ~$26/hr | ~$52 |

Cold start is ~15-20 min: driver upgrade (~2 min), GCS weight copy (~8 min), weight loading into GPU memory (~5.5 min), CUDA graph capture (~23s). Seed the GCS bucket first (step 2) to avoid downloading from HuggingFace on GPU time.

### Self-hosted vs. Gemini 3.0 Pro API

Is self-hosting worth it purely on cost? Probably not for light usage. Here's the math:

**Gemini 3.0 Pro API pricing:** $2/1M input tokens, $12/1M output tokens.

| GPU Setup | Hourly Cost | Equivalent API Output Tokens | Equivalent API Input Tokens |
|---|---|---|---|
| 8x A100-80GB spot | $20/hr | 1.68M output tokens | 10M input tokens |
| 8x H100 spot | $26/hr | 2.13M output tokens | 12.8M input tokens |

For a typical mixed workload (4:1 input:output ratio), $20/hr buys the equivalent of ~4M input + 1M output tokens via the Gemini API.

**Measured self-hosted throughput** (Kimi K2.5 full precision on 8xH100 with vLLM 0.15.0):
- Single request: 112 tok/s (short prompts), 62-80 tok/s (longer reasoning)
- 2 concurrent requests: 108 tok/s per request (~4% drop)
- Higher concurrency untested but likely bottlenecked by KV cache (0.64 GiB)

At typical R&D usage (1-2 concurrent requests), you'd generate ~225K-400K output tokens/hr — well below the ~1.68M break-even point. **The API is cheaper for light, interactive use.**

**Self-hosting makes sense when you need:**
- Full model control (quantization experiments, custom sampling, weight surgery)
- No rate limits or content filtering
- Privacy (data never leaves your VPC)
- Access to models not available via API (Kimi K2.5, open-weight MoE research)
- Heavy batch workloads at high concurrency where you saturate the GPUs

## GCP GPU Quota Setup (Required First Time)

GCP does not give you GPU quota by default. Modern GPUs (A100, H100, etc.) show as "System limit" with value 0 and "Not adjustable" in the GCP Console quota UI. The console UI is essentially useless for requesting these - you have to use the `gcloud` CLI.

### Step 1: Find available quota IDs

```bash
gcloud beta quotas info list \
  --service=compute.googleapis.com \
  --project=YOUR_PROJECT_ID \
  --filter="quotaId:GPU" \
  --format="table(quotaId)"
```

### Step 2: Request quota increases

You need three quotas for spot GPU instances. Run each command separately:

```bash
# GPU family quota (general gate that blocks all GPU provisioning)
gcloud beta quotas preferences create \
  --preferred-value=8 \
  --quota-id=GPUS-PER-GPU-FAMILY-per-project-region \
  --service=compute.googleapis.com \
  --project=YOUR_PROJECT_ID \
  --dimensions=region=us-central1,gpu_family=NVIDIA_H100 \
  --email=YOUR_EMAIL \
  --justification="ML research - large language model inference with 8x H100 GPUs"

# H100 spot instances
gcloud beta quotas preferences create \
  --preferred-value=8 \
  --quota-id=PREEMPTIBLE-NVIDIA-H100-GPUS-per-project-region \
  --service=compute.googleapis.com \
  --project=YOUR_PROJECT_ID \
  --dimensions=region=us-central1 \
  --email=YOUR_EMAIL \
  --justification="ML research - large language model inference with 8x H100 GPUs spot"

# A100-80GB spot instances (fallback)
gcloud beta quotas preferences create \
  --preferred-value=8 \
  --quota-id=PREEMPTIBLE-NVIDIA-A100-80GB-GPUS-per-project-region \
  --service=compute.googleapis.com \
  --project=YOUR_PROJECT_ID \
  --dimensions=region=us-central1 \
  --email=YOUR_EMAIL \
  --justification="ML research - large language model inference with 8x A100 80GB GPUs spot"
```

### Step 3: Check approval status

```bash
gcloud beta quotas preferences list \
  --project=YOUR_PROJECT_ID \
  --format="table(quotaId,quotaConfig.preferredValue,quotaConfig.grantedValue,reconciling)"
```

When `GRANTED_VALUE` equals your `PREFERRED_VALUE` and `RECONCILING` is empty, you're approved. This can take anywhere from instant to a few business days.

### Why is this so painful?

GCP's quota UI shows GPU quotas as non-adjustable "System limits". The only way to request increases for modern GPUs (A100/H100/H200) is via the `gcloud beta quotas` CLI. The GCP Console quota page will not let you edit or request these directly. This is a known pain point.

## Troubleshooting

**Quota errors (`quotaExceeded`, `GPUS_PER_GPU_FAMILY` limit 0):**
Follow the GPU Quota Setup section above. SkyPilot will try every region and fail if you have 0 quota everywhere.

**Auth errors (`Reauthentication failed`):**
```bash
gcloud auth login
gcloud auth application-default login
```

**No spot capacity:**
Remove `use_spot: true` from the YAML to use on-demand (more expensive but always available).

**Cold start too slow (~20 min):**
This is the time to copy ~595GB from GCS to local disk + load into GPU memory. It's unavoidable for this model size. Make sure you've seeded the GCS bucket first (step 2) — without it, the GPU VM downloads from HuggingFace directly which is even slower and wastes GPU time.

**vLLM crashes with `PTX JIT compilation failed`:**
The NVIDIA driver is too old for the model's Marlin CUDA kernels. The YAML already handles this by upgrading from driver 570 to 580. If this fails, SSH in and run `nvidia-smi | head -3` to check the driver version — it should be 580.x.

**NCCL errors or vLLM hangs during startup:**
GCP's GPU images ship with GIB (GPU Interconnect Bridge) libraries that break single-node NCCL. The YAML removes these automatically. If you still see NCCL issues, SSH in and verify: `ls /usr/local/gib` should return "No such file or directory". See `notes/kimi-k2.5-setup-log.md` for the full debugging story.

**`scp` fails with "realpath: No such file":**
Use `scp -O` (capital O) to force the legacy SCP protocol. GCP VM SSH configs don't work with the newer SFTP-based scp.

**Fallback to RunPod/Vast.ai:**
```bash
pip install "skypilot[runpod]"
# Edit YAML: remove `cloud: gcp` to let SkyPilot pick cheapest provider
```
