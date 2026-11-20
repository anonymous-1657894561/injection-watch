#!/usr/bin/env bash
# E112 pod side: Qwen3.5-27B rollouts under vLLM, then its residual states along them. Invoked only by rpjob's timed,
# self-terminating script supervisor, from the repo root with the repo-relative pod config. Weights and venvs on container disk.
set -euo pipefail
repo=$(pwd)
job_config="$repo/${1:?repo-relative pod configuration required}"
cd "$repo/prototypes/violation-forecast"
cfg="$job_config"   # absolute: both scripts resolve HERE / cfg to it
logs=results/e112-cloud/stage-logs; mkdir -p "$logs"
python3 - <<'PY' 2>&1 | tee "$logs/setup.log"
import shutil
for location, minimum in [('/root', 120), ('.', 10)]:
    free = shutil.disk_usage(location).free / 2**30
    print(f'{location}: {free:.1f} GiB free (need {minimum})', flush=True)
    if free < minimum:
        raise SystemExit(f'{location}: {free:.1f} GiB free; need {minimum} GiB before setup')
PY
export HF_HOME=/root/hf HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.total --format=csv > "$logs/gpu.log"

# 1. rollouts (vLLM pins its own torch, so its own venv); skipped once they are on dataset_repo (skip_rollouts: true)
if grep -q '^skip_rollouts: true' "$cfg"; then echo 'skip_rollouts: states stage only' | tee "$logs/rollouts.log"; else
python3 -m venv /root/vllm-venv
/root/vllm-venv/bin/pip install --no-cache-dir -r requirements-e112-vllm.txt >"$logs/install-vllm.log" 2>&1
# The venv is not activated, so its bin/ (ninja et al.) must be put on PATH by hand. The pod has no nvcc, so FlashInfer's
# top-k/top-p sampler cannot JIT-build (first launch, 2026-11-10: FileNotFoundError 'ninja'); vLLM's PyTorch sampler applies
# the same top-k / top-p with no build step.
PATH=/root/vllm-venv/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 \
  /root/vllm-venv/bin/python -u e112_rollouts.py "$cfg" run 2>&1 | tee "$logs/rollouts.log"
fi

# 2. states (E79 v2's venv), read back from the rollout repo the first stage wrote
python3 -m venv /root/e79v2-venv
/root/e79v2-venv/bin/pip install --no-cache-dir -r requirements-e24-pod.txt >"$logs/install-states.log" 2>&1
/root/e79v2-venv/bin/python -u injecforks_states.py "$cfg" run 2>&1 | tee "$logs/states.log"
