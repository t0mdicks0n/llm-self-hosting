# gpu_cli - Ephemeral GPU Instances for Large LLM R&D

Spin up multi-GPU instances, serve large LLMs, connect via VS Code SSH, auto-shutdown when done.

Uses [SkyPilot](https://github.com/skypilot-org/skypilot) for infrastructure management.

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

### 5. Test the Endpoint

Once connected via SSH:
```bash
# Check GPUs are available
nvidia-smi

# Check vLLM is serving
curl localhost:8000/v1/models

# Run test suite
python scripts/test_inference.py

# Run benchmark
python scripts/benchmark.py --rounds 3 --concurrent 2
```

### 6. Tear Down

```bash
sky down kimi                  # Immediate teardown
# OR just disconnect SSH - auto-deletes after 20 min idle
```

## Available Profiles

| File | Model | GPUs | Notes |
|---|---|---|---|
| `kimi-k2.5.yaml` | Kimi K2.5 (1T MoE) | 8x A100-80GB / H100 | Full precision, spot |
| `kimi-k2.5-int4.yaml` | Kimi K2.5 NVFP4 | 8x A100-80GB / H100 | INT4 quantized, ~2x faster |
| `seed-weights.yaml` | — | CPU only | Seeds GCS bucket with model weights (~$0.30/hr) |

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

Cold start is ~20 min (GCS copy + weight loading). Seed the GCS bucket first (step 2) to avoid downloading from HuggingFace on GPU time.

### Self-hosted vs. Gemini 3.0 Pro API

Is self-hosting worth it purely on cost? Probably not for light usage. Here's the math:

**Gemini 3.0 Pro API pricing:** $2/1M input tokens, $12/1M output tokens.

| GPU Setup | Hourly Cost | Equivalent API Output Tokens | Equivalent API Input Tokens |
|---|---|---|---|
| 8x A100-80GB spot | $20/hr | 1.68M output tokens | 10M input tokens |
| 8x H100 spot | $26/hr | 2.13M output tokens | 12.8M input tokens |

For a typical mixed workload (4:1 input:output ratio), $20/hr buys the equivalent of ~4M input + 1M output tokens via the Gemini API.

**Expected self-hosted throughput** (Kimi K2.5 on 8xA100/H100 with vLLM, estimated):
- Single request: ~50-100 output tokens/sec
- 4 concurrent requests: ~200-400 output tokens/sec
- High concurrency (16+): ~500-1000+ output tokens/sec

At typical R&D usage (1-2 concurrent requests), you'd generate ~180K-720K output tokens/hr -- well below the ~1.68M break-even point. **The API is cheaper for light, interactive use.**

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

**Fallback to RunPod/Vast.ai:**
```bash
pip install "skypilot[runpod]"
# Edit YAML: remove `cloud: gcp` to let SkyPilot pick cheapest provider
```
