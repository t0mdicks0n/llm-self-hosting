# Kimi K2.5 on GCP 8xH100: Setup Log

What it took to get Kimi K2.5 (1T param MoE, ~595GB) serving via vLLM on GCP a3-highgpu-8g (8x H100 80GB). Documenting every issue and fix so future launches don't repeat the debugging.

Date: 2026-02-09

## Final Working Configuration

- **VM image:** `pytorch-2-7-cu128-ubuntu-2204-nvidia-570-v20260129`
- **Driver:** 580.126.09 (installed via `nvidia-driver-575-server` package)
- **CUDA:** 13.0 (driver-level), cu128 wheels (pip-level)
- **torch:** 2.9.1+cu128
- **vLLM:** 0.15.0
- **NCCL:** 2.29.3 (`nvidia-nccl-cu12==2.29.3`)
- **GIB:** Completely removed
- **Model weights:** `$HOME/Kimi-K2.5` on local NVMe (copied from GCS)

### Working vLLM command

```bash
vllm serve $HOME/Kimi-K2.5 \
  -tp 8 \
  --served-model-name Kimi-K2.5 \
  --mm-encoder-tp-mode data \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 64 \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 8000
```

### Performance observed

- Model loading: ~5.5 min (64 shards, 72.02 GiB per GPU)
- torch.compile: ~15s
- CUDA graph capture: ~8s
- Total cold start (after weights on disk): ~7 min
- Generation throughput: 62-80 tokens/s (single request)
- Test 1 ("What is 2+2?"): 0.87s, 96 completion tokens
- Test 2 (prime function): 5.65s, 638 completion tokens
- Test 3 (TP vs PP explanation): 6.07s, 687 completion tokens

Note: high token counts relative to output length are because Kimi K2.5 uses chain-of-thought reasoning internally. The reasoning tokens are counted but hidden from the response content.

---

## Issue 1: Marlin PTX Kernel Failure (Driver 570)

**Symptom:** `RuntimeError: PTX compilation failed` during model loading, specifically in `CompressedTensorsWNA16MarlinMoEMethod` (Marlin quantized MoE weight repacking).

**Root cause:** Driver 570 lacks PTX support for the Marlin CUDA kernels that vLLM 0.15.0 uses for MoE weight handling. The PTX ISA version emitted by the CUDA compiler is newer than what driver 570's JIT compiler supports.

**Fix:** Upgrade to driver 575+ (package `nvidia-driver-575-server` installs version 580.126.09).

```bash
sudo apt-get update -qq
sudo apt-get install -y nvidia-driver-575-server --no-install-recommends
sudo reboot  # Required for new kernel module to load
```

**YAML note:** A reboot is needed for the kernel module swap. SkyPilot's setup can't do `sudo reboot` (kills SSH). The YAML uses rmmod/modprobe to reload modules without rebooting. This is untested on a fresh VM — if it fails, fall back to a two-step launch.

---

## Issue 2: NCCL "Failed to initialize any NET plugin" (GIB)

**Symptom:** After model weights loaded (64/64 shards, ~6 min), vLLM crashed during the profiling phase with:
```
ncclInternalError: Internal check failed.
Failed to initialize any NET plugin
```
This happened during an `all_gather` operation in the vision tower encoder profiling.

**Root cause:** GCP's a3-highgpu-8g VMs ship with GIB (GPU Interconnect Bridge) at `/usr/local/gib/`. GIB provides NCCL plugins (`libnccl-net.so`, `libnccl-tuner.so`) designed for multi-node InfiniBand communication. On single-node setups, these plugins intercept NCCL initialization and fail because there's no InfiniBand fabric.

**What did NOT fix it:**
1. Stripping GIB from `LD_LIBRARY_PATH` — NCCL loads plugins via ldconfig, not just LD_LIBRARY_PATH
2. `NCCL_NET_PLUGIN=""` and `NCCL_TUNER_PLUGIN=""` env vars — still failed
3. Renaming `.so` files to `.so.disabled` — NCCL still found them somehow
4. `NCCL_IB_DISABLE=1` — didn't help
5. Setting `VLLM_NCCL_SO_PATH` to system NCCL — still loaded GIB plugins
6. Sourcing GIB's own `set_nccl_env.sh` — designed for multi-node, made things worse

**What DID fix it:** Complete removal of GIB + ldconfig rebuild:
```bash
sudo rm -rf /usr/local/gib
sudo rm -f /etc/ld.so.conf.d/*gib*
sudo ldconfig
```

After this, NCCL correctly auto-detected P2P/CUMEM transport over NVLink/NVSwitch. Verified with an 8-GPU `all_gather` test via `torchrun`.

**Key insight:** The ldconfig cache (`/etc/ld.so.cache`) is what NCCL uses to discover plugins at runtime. Even if the `.so` files are renamed or LD_LIBRARY_PATH is cleaned, if the ldconfig cache still points to them, NCCL will try to load them. The only reliable fix is removing the files AND rebuilding ldconfig.

---

## Issue 3: NCCL Version Mismatch

**Symptom:** Part of the NCCL debugging — the system NCCL was 2.26.2 (installed with the VM image), while vLLM/PyTorch expected a newer version.

**Fix:** Pin NCCL 2.29.3 which is confirmed compatible:
```bash
pip install nvidia-nccl-cu12==2.29.3
```

This was done during troubleshooting. It may or may not be strictly required after GIB removal, but pinning the known-working version avoids any risk.

---

## Issue 4: CUDA OOM During Sampler Warmup

**Symptom:** After model loaded and torch.compile succeeded:
```
CUDA out of memory occurred when warming up sampler with 1024 dummy requests.
Please try lowering `max_num_seqs` or `gpu_memory_utilization`
```

**Root cause:** Model takes 72.02 GiB per GPU out of 79.18 GiB available. Default `max_num_seqs=1024` tried to allocate memory for 1024 dummy sequences during warmup, which exceeded the remaining ~7 GiB.

**Fix:** `--max-num-seqs 64`

---

## Issue 5: No Available KV Cache Memory

**Symptom:** After fixing sampler warmup:
```
Available KV cache memory: -3.32 GiB
ValueError: No available memory for the cache blocks.
```

**Root cause:** vLLM's default `gpu_memory_utilization=0.9` caps total usage at 79.18 * 0.9 = 71.26 GiB. But the model alone takes 72.02 GiB — exceeding the budget entirely.

**Fix:** `--gpu-memory-utilization 0.95` (allows 75.2 GiB, leaving ~3.2 GiB for KV cache).

---

## Issue 6: KV Cache Too Small for Default Context Length

**Symptom:** With 0.95 utilization:
```
Available KV cache memory: 0.64 GiB
To serve at least one request with max seq len (16384), 1.07 GiB KV cache is needed.
Estimated maximum model length is 9664.
```

**Fix:** `--max-model-len 8192` (fits within 0.64 GiB KV cache). 8K context is sufficient for R&D testing. The INT4 quantized variant has more headroom and can use `--max-model-len 32768`.

---

## Issue 7: CUDA Version Mismatch (cu128 vs cu129)

**Symptom:** Early in setup, default `pip install vllm` pulled cu129 wheels which were incompatible with the driver 570 image.

**Fix:** Use uv with explicit torch backend:
```bash
pip install uv
uv pip install --system vllm --torch-backend=cu128
```

---

## Issue 8: GCS FUSE Mount Too Slow for 595GB Model

**Symptom:** SkyPilot's default `file_mounts` uses GCS FUSE which has extremely poor performance for large sequential reads (1+ hours for 595GB).

**Fix:** Two-part solution:
1. Pre-seed a GCS bucket with weights using a cheap CPU VM (`seed-weights.yaml`)
2. GPU VM copies from GCS to local NVMe via `gsutil -m cp` (~8 min at 1.2 GiB/s)

---

## Order of Operations (Setup)

This is the exact order that matters:

1. Upgrade NVIDIA driver (570 → 580 via `nvidia-driver-575-server`)
2. Reload kernel modules (or reboot)
3. Remove GIB completely + rebuild ldconfig
4. Install vLLM with cu128 wheels
5. Install flash-attn
6. Pin NCCL 2.29.3
7. Copy model weights from GCS to local disk

## Things That Were Red Herrings

- `NCCL_NVLS_ENABLE=0` / `NCCL_CUMEM_ENABLE=0` — not needed after GIB removal
- `VLLM_NCCL_SO_PATH` — not needed after GIB removal
- `nvidia-fabricmanager` — auto-starts on boot, manual restart only needed if driver is swapped without reboot
- `--quantization None` — the model uses compressed tensors natively, not a separate quantization config
- Trying cu130 wheels — unnecessary and potentially problematic
- Copying system NCCL .so over pip-installed one — brittle, fixed by proper NCCL version pin

---

## Appendix: Full VM Bash History

Raw bash history from the VM session for reference. Includes all failed attempts and successful commands.

```bash
    1  #!/bin/bash
    2  source ~/.bashrc
    3  set -a
    4  . $(conda info --base 2> /dev/null)/etc/profile.d/conda.sh > /dev/null 2>&1 || true
    5  set +a
    6  export PATH=$(echo $PATH | sed "s|$(echo ~)/skypilot-runtime/bin:||") && unset VIRTUAL_ENV && unset VIRTUAL_ENV_PROMPT
    7  export PYTHONUNBUFFERED=1
    8  cd ~/sky_workdir
    9  export SKYPILOT_CLUSTER_INFO='{"cluster_name": "kimi", "cloud": "GCP", "region": "us-central1", "zone": "us-central1-a"}'
   10  export SKYPILOT_SETUP_NODE_IPS=10.128.0.11
   11  export SKYPILOT_SETUP_NODE_RANK=0
   12  export SKYPILOT_SETUP_NUM_GPUS_PER_NODE=8
   13  set -e
   14  # Install vLLM with CUDA 12.8 wheels to match the VM image (driver 570).
   15  # Default pip install pulls cu129 which fails on driver 570.
   16  pip install uv
   17  uv pip install vllm --torch-backend=cu128
   18  #!/bin/bash
   19  source ~/.bashrc
   20  set -a
   21  . $(conda info --base 2> /dev/null)/etc/profile.d/conda.sh > /dev/null 2>&1 || true
   22  set +a
   23  export PATH=$(echo $PATH | sed "s|$(echo ~)/skypilot-runtime/bin:||") && unset VIRTUAL_ENV && unset VIRTUAL_ENV_PROMPT
   24  export PYTHONUNBUFFERED=1
   25  cd ~/sky_workdir
   26  export SKYPILOT_CLUSTER_INFO='{"cluster_name": "kimi", "cloud": "GCP", "region": "us-central1", "zone": "us-central1-a"}'
   27  export SKYPILOT_SETUP_NODE_IPS=10.128.0.11
   28  export SKYPILOT_SETUP_NODE_RANK=0
   29  export SKYPILOT_SETUP_NUM_GPUS_PER_NODE=8
   30  set -e
   31  # Install vLLM with CUDA 12.8 wheels to match the VM image (driver 570).
   32  # Default pip install pulls cu129 which fails on driver 570.
   33  pip install uv
   34  uv pip install --system vllm --torch-backend=cu128
   35  pip install flash-attn --no-build-isolation
   36  # Download model weights to local disk (fast), not GCS FUSE mount (slow)
   37  # Check GCS cache first for subsequent launches
   38  LOCAL_MODEL=$HOME/Kimi-K2.5
   39  BUCKET=kimi-k2-5-weights
   40  if gsutil -q stat gs://$BUCKET/.download_complete 2>/dev/null; then   echo "Model weights cached in GCS. Downloading to local disk via gsutil...";   mkdir -p $LOCAL_MODEL;   gsutil -m cp -r gs://$BUCKET/Kimi-K2.5/* $LOCAL_MODEL/;   echo "Copy from GCS complete."; else   echo "Downloading Kimi K2.5 model weights to local disk (~595GB)...";   python -c "
   41  from huggingface_hub import snapshot_download
   42  snapshot_download('moonshotai/Kimi-K2.5', local_dir='$LOCAL_MODEL')
   43  ";   echo "Download complete. Uploading to GCS cache for future launches...";   gsutil -m cp -r $LOCAL_MODEL/* gs://$BUCKET/Kimi-K2.5/ && gsutil cp /dev/null gs://$BUCKET/.download_complete &   echo "GCS sync started in background."; fi
   44  # SkyPilot run phase — first vLLM attempt (failed: driver 570 PTX error)
   45  echo "Starting vLLM server for Kimi K2.5..."
   46  echo "GPUs available: $SKYPILOT_NUM_GPUS_PER_NODE"
   47  nvidia-smi
   48  LOCAL_MODEL=$HOME/Kimi-K2.5
   49  vllm serve $LOCAL_MODEL   -tp $SKYPILOT_NUM_GPUS_PER_NODE   --served-model-name Kimi-K2.5   --mm-encoder-tp-mode data   --tool-call-parser kimi_k2   --reasoning-parser kimi_k2   --trust-remote-code   --host 0.0.0.0   --port 8000
   50  # Troubleshooting: fabricmanager
   51  sudo systemctl start nvidia-fabricmanager
   52  sudo systemctl status nvidia-fabricmanager.service
   53  sudo journalctl -xeu nvidia-fabricmanager.service --no-pager | tail -30
   54  dpkg -l | grep fabric
   55  nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
   56  # Attempt: NCCL env vars
   57  export NCCL_NVLS_ENABLE=0
   58  export NCCL_CUMEM_ENABLE=0
   59  export NCCL_DEBUG=WARN
   60  vllm serve ~/Kimi-K2.5 -tp 8 --served-model-name Kimi-K2.5 --mm-encoder-tp-mode data --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --trust-remote-code --host 0.0.0.0 --port 8000
   61  # Attempt: system NCCL via VLLM_NCCL_SO_PATH
   62  unset NCCL_NVLS_ENABLE
   63  unset NCCL_CUMEM_ENABLE
   64  export NCCL_DEBUG=WARN
   65  export VLLM_NCCL_SO_PATH=$(find /usr -name "libnccl.so.2" 2>/dev/null | head -1)
   66  echo "Using system NCCL at: $VLLM_NCCL_SO_PATH"
   67  vllm serve ~/Kimi-K2.5 -tp 8 --served-model-name Kimi-K2.5 --mm-encoder-tp-mode data --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --trust-remote-code --host 0.0.0.0 --port 8000
   68  # Investigating GIB
   69  curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/machine-type
   70  ls /usr/local/gib/
   71  ls /usr/local/gib/scripts/ 2>/dev/null
   72  ls /usr/local/gib/*.sh 2>/dev/null
   73  systemctl list-units | grep -i gib
   74  nvidia-smi topo -m
   75  # Attempt: source GIB env script
   76  unset NCCL_NVLS_ENABLE NCCL_CUMEM_ENABLE VLLM_NCCL_SO_PATH NCCL_DEBUG
   77  source /usr/local/gib/scripts/set_nccl_env.sh
   78  env | grep NCCL
   79  vllm serve ~/Kimi-K2.5 -tp 8 --served-model-name Kimi-K2.5 --mm-encoder-tp-mode data --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --trust-remote-code --host 0.0.0.0 --port 8000
   80  # Investigating GIB libraries
   81  cat /usr/local/gib/scripts/set_nccl_env.sh
   82  ls -la /usr/local/gib/lib64/
   83  echo $LD_LIBRARY_PATH
   84  find /usr/local/gib/ -name "*.so*" -type f 2>/dev/null
   85  # Attempt: GIB env + GIB NCCL + debug
   86  source /usr/local/gib/scripts/set_nccl_env.sh
   87  export VLLM_NCCL_SO_PATH=/usr/local/gib/lib64/libnccl.so.2
   88  export NCCL_DEBUG=INFO
   89  vllm serve ~/Kimi-K2.5 -tp 8 --served-model-name Kimi-K2.5 --mm-encoder-tp-mode data --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --trust-remote-code --host 0.0.0.0 --port 8000
   90  # Attempt: strip GIB from LD_LIBRARY_PATH + unset all NCCL vars
   91  export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v gib | paste -sd ':')
   92  unset NCCL_NET NCCL_CROSS_NIC NCCL_NET_GDR_LEVEL NCCL_P2P_NET_CHUNKSIZE NCCL_NVLS_CHUNKSIZE
   93  unset NCCL_IB_ADAPTIVE_ROUTING NCCL_IB_QPS_PER_CONNECTION NCCL_IB_TC NCCL_IB_FIFO_TC
   94  unset NCCL_TUNER_CONFIG_PATH VLLM_NCCL_SO_PATH
   95  export NCCL_DEBUG=INFO
   96  vllm serve ~/Kimi-K2.5 -tp 8 --served-model-name Kimi-K2.5 --mm-encoder-tp-mode data --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --trust-remote-code --host 0.0.0.0 --port 8000
   97  # Checking versions
   98  python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"
   99  pip show vllm | grep Version
  100  nvcc --version 2>/dev/null || echo "no nvcc"
  101  python -c "import json; c=json.load(open('$HOME/Kimi-K2.5/config.json')); print('quant:', c.get('quantization_config', 'None'))"
  102  python -c "
  103  import safetensors.torch
  104  import json
  105  with open('$HOME/Kimi-K2.5/model.safetensors.index.json') as f:
  106      idx = json.load(f)
  107  print('metadata keys:', list(idx.get('metadata', {}).keys())[:20])
  108  "
  109  # Attempt: downgrade vLLM to 0.14.3
  110  uv pip install --system "vllm==0.14.3" --torch-backend=cu128
  111  pip index versions vllm 2>/dev/null || uv pip install --system "vllm==0.0.0" --torch-backend=cu128 2>&1 | tail -5
  112  # Attempt: explicit --quantization None
  113  export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v gib | paste -sd ':')
  114  vllm serve $HOME/Kimi-K2.5   -tp 8   --served-model-name Kimi-K2.5   --mm-encoder-tp-mode data   --tool-call-parser kimi_k2   --reasoning-parser kimi_k2   --trust-remote-code   --host 0.0.0.0   --port 8000   --quantization None
  115  # Upgrade to vLLM 0.15.0 (fixes Marlin support)
  116  uv pip install --system "vllm==0.15.0" --torch-backend=cu128
  117  export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v gib | paste -sd ':')
  118  vllm serve $HOME/Kimi-K2.5   -tp 8   --served-model-name Kimi-K2.5   --mm-encoder-tp-mode data   --tool-call-parser kimi_k2   --reasoning-parser kimi_k2   --trust-remote-code   --host 0.0.0.0   --port 8000
  119  # Driver upgrade: 570 → 575-server (installs 580.126.09)
  120  apt list --installed 2>/dev/null | grep nvidia-driver
  121  sudo apt-get update && apt-cache search nvidia-driver | grep -E 'nvidia-driver-5[7-9]'
  122  sudo apt-get install -y nvidia-driver-575-server
  123  nvidia-smi | head -3
  124  sudo reboot
  125  # After reboot — driver 580.126.09 now active
  126  nvidia-smi | head -3
  127  export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v gib | paste -sd ':')
  128  vllm serve $HOME/Kimi-K2.5   -tp 8   --served-model-name Kimi-K2.5   --mm-encoder-tp-mode data   --tool-call-parser kimi_k2   --reasoning-parser kimi_k2   --trust-remote-code   --host 0.0.0.0   --port 8000
  129  echo $LD_LIBRARY_PATH | tr ':' '\n' | grep gib
  130  NCCL_DEBUG=WARN vllm serve $HOME/Kimi-K2.5   -tp 8   --served-model-name Kimi-K2.5   --mm-encoder-tp-mode data   --tool-call-parser kimi_k2   --reasoning-parser kimi_k2   --trust-remote-code   --host 0.0.0.0   --port 8000 2>&1 | head -200
  131  pkill -9 -f "vllm serve"
  132  vllm serve $HOME/Kimi-K2.5   -tp 8   --served-model-name Kimi-K2.5   --mm-encoder-tp-mode data   --tool-call-parser kimi_k2   --reasoning-parser kimi_k2   --trust-remote-code   --host 0.0.0.0   --port 8000
  133  NCCL_CUMEM_ENABLE=0 vllm serve $HOME/Kimi-K2.5   -tp 8   --served-model-name Kimi-K2.5   --mm-encoder-tp-mode data   --tool-call-parser kimi_k2   --reasoning-parser kimi_k2   --trust-remote-code   --host 0.0.0.0   --port 8000
  134  # Attempt: cu130 wheels
  135  uv pip install --system vllm --torch-backend=cu130
  136  find / -name "libnccl*.so*" 2>/dev/null
  137  # Attempt: VLLM_NCCL_SO_PATH to system NCCL
  138  VLLM_NCCL_SO_PATH=/usr/lib/x86_64-linux-gnu/libnccl.so.2 vllm serve $HOME/Kimi-K2.5   -tp 8   --served-model-name Kimi-K2.5   --mm-encoder-tp-mode data   --tool-call-parser kimi_k2   --reasoning-parser kimi_k2   --trust-remote-code   --host 0.0.0.0   --port 8000
  139  # Attempt: copy system NCCL over pip NCCL
  140  cp /usr/lib/x86_64-linux-gnu/libnccl.so.2.26.2 /home/gcpuser/miniconda3/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
  141  VLLM_NCCL_SO_PATH=/usr/lib/x86_64-linux-gnu/libnccl.so.2 vllm serve $HOME/Kimi-K2.5   -tp 8   --served-model-name Kimi-K2.5   --mm-encoder-tp-mode data   --tool-call-parser kimi_k2   --reasoning-parser kimi_k2   --trust-remote-code   --host 0.0.0.0   --port 8000
  142  cp /home/gcpuser/.cache/uv/archive-v0/mA2oSdDqbI6HrKQWIDbDj/nvidia/nccl/lib/libnccl.so.2 /home/gcpuser/miniconda3/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
  143  # Attempt: rename GIB plugins (not enough — need full removal)
  144  sudo mv /usr/local/gib/lib64/libnccl-net.so /usr/local/gib/lib64/libnccl-net.so.disabled
  145  sudo mv /usr/local/gib/lib64/libnccl-tuner.so /usr/local/gib/lib64/libnccl-tuner.so.disabled
  146  export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v gib | paste -sd ':')
  147  vllm serve $HOME/Kimi-K2.5   -tp 8   --served-model-name Kimi-K2.5   --mm-encoder-tp-mode data   --tool-call-parser kimi_k2   --reasoning-parser kimi_k2   --trust-remote-code   --host 0.0.0.0   --port 8000
  148  # Upgrade NCCL
  149  pip install --force-reinstall nvidia-nccl-cu12
  150  vllm serve $HOME/Kimi-K2.5   -tp 8   --served-model-name Kimi-K2.5   --mm-encoder-tp-mode data   --tool-call-parser kimi_k2   --reasoning-parser kimi_k2   --trust-remote-code   --host 0.0.0.0   --port 8000
  151  VLLM_NCCL_SO_PATH=/usr/lib/x86_64-linux-gnu/libnccl.so.2 vllm serve $HOME/Kimi-K2.5   -tp 8   --served-model-name Kimi-K2.5   --mm-encoder-tp-mode data   --tool-call-parser kimi_k2   --reasoning-parser kimi_k2   --trust-remote-code   --host 0.0.0.0   --port 8000
  152  # Attempt: NCCL_NET_PLUGIN="" and NCCL_TUNER_PLUGIN=""
  153  NCCL_NET_PLUGIN="" NCCL_TUNER_PLUGIN="" VLLM_NCCL_SO_PATH=/usr/lib/x86_64-linux-gnu/libnccl.so.2 vllm serve $HOME/Kimi-K2.5   -tp 8   --served-model-name Kimi-K2.5   --mm-encoder-tp-mode data   --tool-call-parser kimi_k2   --reasoning-parser kimi_k2   --trust-remote-code   --host 0.0.0.0   --port 8000
  154  # THE FIX: Complete GIB removal
  155  sudo rm -rf /usr/local/gib
  156  sudo rm -f /etc/ld.so.conf.d/*gib*
  157  sudo ldconfig
  158  # NCCL test — verified 8-GPU all_gather works
  159  cat > /tmp/test_nccl.py << 'EOF'
  160  import torch
  161  import torch.distributed as dist
  162  dist.init_process_group("nccl")
  163  rank = dist.get_rank()
  164  t = torch.ones(10, device=f"cuda:{rank}")
  165  output = torch.empty(80, device=f"cuda:{rank}")
  166  dist.all_gather_into_tensor(output, t)
  167  if rank == 0:
  168      print(f"NCCL 8-GPU all_gather succeeded! Sum={output.sum().item()}")
  169  dist.destroy_process_group()
  170  EOF
  171  torchrun --nproc_per_node=8 /tmp/test_nccl.py
  172  NCCL_IB_DISABLE=1 torchrun --nproc_per_node=8 /tmp/test_nccl.py
  173  NCCL_DEBUG=INFO torchrun --nproc_per_node=8 /tmp/test_nccl.py &> /tmp/nccl_debug.log
  174  head -100 /tmp/nccl_debug.log
  175  tail -200 /tmp/nccl_debug.log
  176  # vLLM attempts after NCCL fix (OOM issues, then memory tuning)
  177  vllm serve $HOME/Kimi-K2.5   -tp 8   --served-model-name Kimi-K2.5   --mm-encoder-tp-mode data   --tool-call-parser kimi_k2   --reasoning-parser kimi_k2   --trust-remote-code   --host 0.0.0.0   --port 8000
  178  vllm serve $HOME/Kimi-K2.5   -tp 8   --served-model-name Kimi-K2.5   --mm-encoder-tp-mode data   --tool-call-parser kimi_k2   --reasoning-parser kimi_k2   --trust-remote-code   --max-num-seqs 64   --host 0.0.0.0   --port 8000
  179  # Final working command (with gpu-memory-utilization and max-model-len added)
  180  # vllm serve $HOME/Kimi-K2.5 -tp 8 --served-model-name Kimi-K2.5 --mm-encoder-tp-mode data --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --trust-remote-code --gpu-memory-utilization 0.95 --max-num-seqs 64 --max-model-len 8192 --host 0.0.0.0 --port 8000
  181  # Testing
  182  curl localhost:8000/v1/models
  183  python scripts/test_inference.py
  184  python scripts/benchmark.py --rounds 3 --concurrent 2
  185  history
```
