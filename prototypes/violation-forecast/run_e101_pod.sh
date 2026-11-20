#!/usr/bin/env bash
# E101 distill observer states (docs/plan-2026-10-30-distill-observer.md), pod side. Invoked only by rpjob's timed,
# self-terminating script supervisor, from the repo root with the repo-relative pod config. Weights and venv on container disk.
# Mirrors run_e79v2_pod.sh (same venv pins, same disk check). Two steps, one process each, so the GPU holds one model at a time:
#   1. the observer extraction (injecforks_observer_states.py run; checks E101's config_sha256, uploads states to the Hub)
#   2. E79's test cache check: Qwen3.5-9B on 3 realistic test eval cases (injecforks_states.py cachecheck), results only
set -euo pipefail   # pipefail: a failed disk check still fails the job after tee records it
repo=$(pwd)
job_config="$repo/${1:?repo-relative pod configuration required}"
cd "$repo/prototypes/violation-forecast"
logs=results/e101-cloud/stage-logs
mkdir -p "$logs" results/e79v2-cloud/testcache
python3 - <<'PY' 2>&1 | tee "$logs/setup.log"
import shutil
for location, minimum in [('/root', 55), ('.', 10)]:   # 16.1 GB observer + 19.3 GB 9B weights + venv
    free = shutil.disk_usage(location).free / 2**30
    print(f'{location}: {free:.1f} GiB free (need {minimum})', flush=True)
    if free < minimum:
        raise SystemExit(f'{location}: {free:.1f} GiB free; need {minimum} GiB before setup')
PY
python3 -m venv /root/e101-venv
/root/e101-venv/bin/pip install --no-cache-dir -r requirements-e24-pod.txt >"$logs/install.log" 2>&1
export HF_HOME=/root/hf HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.total --format=csv > "$logs/gpu.log"
/root/e101-venv/bin/python -u injecforks_observer_states.py "$job_config" run 2>&1 | tee "$logs/run.log"
/root/e101-venv/bin/python -u injecforks_states.py configs/e101-9b-testcache.yaml cachecheck 2>&1 | tee "$logs/e79-testcache.log"
test -s results/e79v2-cloud/testcache/cache_check.json
