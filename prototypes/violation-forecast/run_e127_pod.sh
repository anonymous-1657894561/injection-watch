#!/usr/bin/env bash
# E127 pod side: steered Qwen3.5-9B rollouts (e127_steer.py). Invoked only by rpjob's timed, self-terminating script supervisor,
# from the repo root with the repo-relative pod config. Weights and venv live on container disk.
set -euo pipefail
repo=$(pwd)
job_config="$repo/${1:?repo-relative pod configuration required}"
cd "$repo/prototypes/violation-forecast"
logs=results/e127-cloud/stage-logs; mkdir -p "$logs"
python3 - <<'PY' 2>&1 | tee "$logs/setup.log"
import shutil
for location, minimum in [('/root', 45), ('.', 5)]:
    free = shutil.disk_usage(location).free / 2**30
    print(f'{location}: {free:.1f} GiB free (need {minimum})', flush=True)
    if free < minimum:
        raise SystemExit(f'{location}: {free:.1f} GiB free; need {minimum} GiB before setup')
PY
python3 -m venv /root/e127-venv
/root/e127-venv/bin/pip install --no-cache-dir -r requirements-e24-pod.txt >"$logs/install.log" 2>&1
export HF_HOME=/root/hf HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.total --format=csv > "$logs/gpu.log"
/root/e127-venv/bin/python -u e127_steer.py "$job_config" run 2>&1 | tee "$logs/run.log"
