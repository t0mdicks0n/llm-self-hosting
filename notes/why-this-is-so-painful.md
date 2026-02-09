# Why Self-Hosting Large LLMs Is So Painful

Date: 2026-02-09

An honest post-mortem on getting Kimi K2.5 (1T MoE, 595GB) running on 8xH100 via SkyPilot + vLLM. Despite using mature abstractions (vLLM for serving, SkyPilot for infra), this took an entire day of debugging. Here's every issue we hit, why it happened, and what it says about the state of MLOps.

---

## Every Issue We Hit

### 1. CUDA wheel version mismatch
- **What happened:** `pip install vllm` pulled CUDA 12.9 wheels. The GCP VM image had driver 570, which only supports up to CUDA 12.8.
- **Error:** Cryptic CUDA initialization failure.
- **Fix:** `uv pip install --system vllm --torch-backend=cu128`
- **Root cause:** pip has no way to detect the installed GPU driver version and select compatible wheels. You have to know to specify the CUDA backend manually.

### 2. Driver too old for Marlin CUDA kernels
- **What happened:** Kimi K2.5 uses compressed tensor weights (CompressedTensorsWNA16MarlinMoEMethod). The Marlin kernels require PTX features not available in driver 570.
- **Error:** `PTX JIT compilation failed` — the error message does not tell you which driver version you need.
- **Fix:** Upgrade from nvidia-driver-570 to nvidia-driver-575-server (which installs version 580.126.09).
- **Root cause:** The model card says "use vLLM." It doesn't mention driver requirements. The vLLM docs don't list per-model driver requirements either. You find out at runtime.

### 3. Driver package naming makes no sense
- **What happened:** We needed driver version 580. The apt package is called `nvidia-driver-575-server`. 575 is the "branch," 580 is the version it installs.
- **Root cause:** NVIDIA's package naming uses branch numbers, not version numbers. There's no `nvidia-driver-580` package. You have to know that branch 575 gives you version 580.

### 4. Can't reboot during SkyPilot setup
- **What happened:** After installing a new driver, you normally reboot. But SkyPilot runs the setup phase via SSH. A reboot kills the SSH session and SkyPilot treats it as a setup failure.
- **Fix:** Instead of rebooting, we rmmod the old kernel modules and modprobe the new ones. This is fragile and may not work in all cases (e.g., if a process holds a GPU handle).
- **Root cause:** SkyPilot's execution model (SSH → run setup script → run serve script) doesn't support reboots. This is a fundamental mismatch between "cloud infra provisioning" and "bare metal system administration."

### 5. GCP's GIB libraries silently breaking NCCL
- **What happened:** GCP's a3-highgpu-8g images ship with GIB (GPU Interconnect Bridge) — a set of NCCL plugins for multi-node InfiniBand networking. On our single-node setup, these plugins intercepted NCCL initialization and broke P2P/NVSwitch communication.
- **Error:** NCCL errors about failing to open plugins, or hanging during initialization.
- **Fix attempts that didn't work:**
  - Setting `NCCL_NVLS_ENABLE=0`, `NCCL_CUMEM_ENABLE=0`
  - Stripping GIB paths from `LD_LIBRARY_PATH`
  - Pointing `VLLM_NCCL_SO_PATH` to system NCCL
  - Sourcing GIB's own env script
  - Renaming GIB .so files to .disabled
  - Copying system NCCL over pip-installed NCCL
- **Fix that actually worked:** Complete removal of GIB: `sudo rm -rf /usr/local/gib && sudo rm -f /etc/ld.so.conf.d/*gib* && sudo ldconfig`. The reason nothing else worked is that NCCL loads plugins via ldconfig, not just LD_LIBRARY_PATH. As long as the .so files existed anywhere on the system and were registered in ldconfig, NCCL would find and load them.
- **Root cause:** GCP pre-installs middleware for their multi-node GPU networking. It's not documented that this breaks single-node setups. There's no "I'm only using one node" flag. This was by far the hardest issue to diagnose — we tried ~15 different approaches before finding the fix.

### 6. NCCL version incompatibility
- **What happened:** After fixing GIB, the pip-installed NCCL version still didn't work cleanly with this driver/vLLM combination.
- **Fix:** Pin `nvidia-nccl-cu12==2.29.3`.
- **Root cause:** NCCL, the NVIDIA driver, and vLLM all version independently. There's no compatibility matrix published anywhere. You find working combinations through trial and error.

### 7. OOM during sampler warmup (max_num_seqs)
- **What happened:** vLLM defaults to `max_num_seqs=1024` (max concurrent sequences). Pre-allocating memory for 1024 sequences on top of a 72GB-per-GPU model caused OOM.
- **Error:** CUDA out of memory during sampler warmup.
- **Fix:** `--max-num-seqs 64`
- **Root cause:** vLLM's defaults are tuned for smaller models. For a 1T parameter model that consumes 90%+ of GPU memory, you need to manually tune concurrency limits. The error message doesn't suggest which parameter to change.

### 8. Negative KV cache memory (gpu_memory_utilization)
- **What happened:** The model weights alone take 72.02 GiB per GPU. vLLM's default `gpu_memory_utilization=0.9` reserves 90% of 80GB = 71.26 GiB, which is less than the model needs. This left negative memory for KV cache.
- **Error:** `Available KV cache memory: -3.32 GiB`
- **Fix:** `--gpu-memory-utilization 0.95`
- **Root cause:** vLLM doesn't check whether the model even fits at the requested utilization level before starting. It loads the model, then discovers there's negative cache space.

### 9. Context length too long for available KV cache
- **What happened:** With 0.95 utilization, only 0.64 GiB was left for KV cache. The model's native 256K context length requires far more.
- **Error:** Need 1.07 GiB for 16384 tokens, only 0.64 GiB available.
- **Fix:** `--max-model-len 8192` (full precision) or `--max-model-len 32768` (INT4).
- **Root cause:** vLLM tries to use the model's configured max context length by default, without checking if there's enough memory. For a model that barely fits in GPU memory, you have to manually cap the context length.

### 10. Hidden reasoning tokens crashing test scripts
- **What happened:** Kimi K2.5 uses internal chain-of-thought reasoning. It generates reasoning tokens that are counted in `usage.completion_tokens` but don't appear in the response `content` field. With a 256-token budget, the model spent all tokens on reasoning and returned `content: null`.
- **Error:** `'NoneType' object is not subscriptable` when trying to slice the response string.
- **Fix:** Increase `max_tokens` to 1024 and add None handling.
- **Root cause:** This is actually well-designed behavior (reasoning models need token budget for thinking), but it's not obvious from the API response. The OpenAI-compatible API doesn't distinguish "no content because still thinking" from "empty response."

### 11. SCP protocol mismatch
- **What happened:** `scp -r scripts/ kimi:~/scripts/` failed with "realpath scripts/: No such file or directory."
- **Fix:** `scp -O` flag to use the legacy SCP protocol instead of SFTP.
- **Root cause:** Modern OpenSSH defaults to SFTP protocol for scp transfers. GCP VM SSH configuration doesn't support this cleanly. This is not specific to ML but adds to the friction.

### 12. GCS FUSE mount unusably slow for large models
- **What happened:** SkyPilot's default file mount uses GCS FUSE. For a 595GB model, reads through FUSE took over an hour.
- **Fix:** Built a separate GCS seeding workflow (CPU VM downloads from HuggingFace, uploads to GCS bucket) and use `gsutil -m cp` which gets ~1.2 GiB/s.
- **Root cause:** FUSE filesystem overhead on random reads of many large files is catastrophic. This is well-known but SkyPilot doesn't warn about it or offer alternatives by default.

---

## The Systemic Problems

These 12 issues aren't random bad luck. They point to structural problems in the ML infrastructure stack:

### 1. Six-layer compatibility matrix with no contract enforcement

```
GPU Driver ↔ CUDA Toolkit ↔ PyTorch ↔ vLLM ↔ NCCL ↔ Model Format
  570/580      12.8/12.9     2.9.1    0.15.0   2.29.3   Marlin MoE
```

Every layer must be compatible with every adjacent layer. Unlike normal software where you have stable ABIs and semver, here the "interface" between layers is "does this compiled PTX binary happen to execute on this driver version." There's no dependency resolver that spans all six layers. pip handles Python packages. apt handles system packages. Neither talks to the other, and neither understands CUDA compatibility.

### 2. Cloud providers inject invisible middleware

GCP's GIB is the perfect example. It's installed by default on GPU images, it's not documented in the image release notes, and it silently intercepts a critical system library (NCCL). You can't opt out of it during image selection. You find out about it when things break, and the error messages give no hint that a vendor-injected library is the cause.

AWS has similar issues with EFA (Elastic Fabric Adapter). It's the same pattern: the cloud provider optimizes for their multi-node networking product and inadvertently breaks single-node use cases.

### 3. Error messages are written for NVIDIA engineers, not users

- `PTX JIT compilation failed` — doesn't say which driver version you need
- `NCCL WARN Failed to open ...plugin...` — doesn't say which plugin or why
- `Available KV cache memory: -3.32 GiB` — doesn't suggest gpu_memory_utilization
- CUDA OOM during "sampler warmup" — doesn't suggest max_num_seqs

Every error required us to understand the internals of the system to diagnose. "PTX compilation failed" means "your driver is too old for this kernel" but you need to already know what PTX is and how driver/CUDA versioning works to make that connection.

### 4. No local iteration loop

In normal software development, you run the build locally, see the error, fix it, re-run in seconds. Here:

- Each iteration requires an 8xH100 cluster ($26/hr)
- Cold start is ~20 minutes (GCS copy + weight loading)
- You can't test driver upgrades or NCCL fixes locally
- You can't test "does this 595GB model load" without 640GB of VRAM

This means every debugging cycle costs real money and 20+ minutes of wall time. The GIB issue alone took ~15 attempts to fix — each requiring a vLLM restart (5+ minutes to reload weights). That's over an hour of GPU time (~$26) on a single issue.

### 5. Configuration through negative space

Most vLLM parameters have defaults tuned for smaller models (7B-70B). For a 1T model, you need to override `gpu_memory_utilization`, `max_num_seqs`, and `max_model_len`. But there's no "large model" preset or warning like "this model uses 90% of VRAM, you probably want to adjust these settings." You discover each parameter by hitting a different error.

This is "configuration through negative space" — the system doesn't tell you what to set, it only tells you (via crashes) what the defaults can't handle.

### 6. Documentation assumes happy path only

The Kimi K2.5 model card has a single vLLM serve command. It works on the exact hardware/driver/CUDA combination the authors tested. It doesn't mention:
- Minimum driver version (580+)
- Memory parameters needed for 80GB GPUs
- NCCL requirements
- That reasoning tokens are hidden from response content

vLLM's docs list all the flags but don't explain when you need them. SkyPilot's docs cover provisioning but not GPU driver management. Each tool documents its own happy path and assumes the rest of the stack is someone else's problem.

### 7. Version churn makes knowledge perishable

The fix that works today (vLLM 0.15.0 + driver 580 + NCCL 2.29.3) may break next month when any of these releases a new version. There's no LTS or stability guarantee across the stack. Community fixes on GitHub issues become outdated within weeks. This is why so many ML deployment guides start with "pin everything."

---

## What Would Make This Better

This isn't just complaining. Here's what the ecosystem actually needs:

1. **Cross-layer dependency resolution.** Something that understands "this model needs Marlin kernels, which need driver ≥580, which needs NCCL ≥2.29, which needs vLLM ≥0.15." pip can't do this. It needs to be a new tool or a major extension.

2. **Model cards with deployment requirements.** Not just "use vLLM" but minimum driver version, minimum VRAM per GPU, recommended vLLM flags for different GPU configs.

3. **Better error messages.** `PTX JIT compilation failed` should say `"Your driver (570) is too old for this model's Marlin kernels. Minimum required: 580. Install with: sudo apt-get install nvidia-driver-575-server"`.

4. **Cloud provider transparency.** GCP should document that their GPU images include GIB and how to disable it for single-node use cases. Or better: don't install it on single-node instance types.

5. **Verified deployment configurations.** Docker images or environment lockfiles that capture the entire working stack (driver + CUDA + PyTorch + vLLM + NCCL + model-specific flags) as a tested unit.

6. **Dry-run mode.** Let me check "will this model fit on 8xH100 with these settings" without actually provisioning the cluster. vLLM could compute memory requirements offline.

---

## The Bottom Line

We have good abstractions at each layer (vLLM, SkyPilot, PyTorch). What's missing is the glue between them. The failure modes are all at boundaries: driver↔CUDA, cloud image↔NCCL, model↔serving engine, serving engine↔available memory.

Normal software doesn't have this problem because the boundaries are well-defined (syscall ABI, HTTP APIs, language package managers). In ML infra, the boundaries are defined by hardware errata, compiled binary compatibility, and undocumented cloud vendor middleware.

This will get better — it has to, because the economic pressure to deploy large models is enormous. But right now (early 2026), self-hosting large LLMs feels like Linux desktop in 2005: powerful if you get it working, but the getting-it-working part is a full day of archaeology.

---

## Time and Cost

- **Wall time to first successful inference:** ~8-10 hours (including wait times)
- **GPU hours burned on debugging:** ~4-5 hours at $26/hr ≈ $100-130 in compute
- **Issues that could have been avoided with better docs/tooling:** 9 out of 12
- **Issues that are fundamentally hard:** 3 (model size → cold start, memory pressure, reasoning token behavior)
