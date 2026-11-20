#!/usr/bin/env bash
# E103 variant states, pod side (experiments/E103-classifier-brittleness.md, agent side). Invoked only by rpjob's timed,
# self-terminating script supervisor, from the repo root with the repo-relative pod configuration. As run_e102_pod.sh's first
# stage: own venv, weights on container disk; Qwen3.5-9B on the T2/T4/T5 rows (injecforks_states.py run) -> e79v2-states
set -euo pipefail
repo=$(pwd)
job_config="$repo/${1:?repo-relative pod configuration required}"
cd "$repo/prototypes/violation-forecast"
logs=results/e103-cloud/stage-logs
mkdir -p "$logs"
python3 - <<'PY' 2>&1 | tee "$logs/setup.log"
import shutil
for location, minimum in [('/root', 30), ('.', 10)]:   # 19.3 GB 9B weights + venv
    free = shutil.disk_usage(location).free / 2**30
    print(f'{location}: {free:.1f} GiB free (need {minimum})', flush=True)
    if free < minimum:
        raise SystemExit(f'{location}: {free:.1f} GiB free; need {minimum} GiB before setup')
PY
python3 -m venv /root/e103-venv
/root/e103-venv/bin/pip install --no-cache-dir -r requirements-e24-pod.txt >"$logs/install.log" 2>&1
export HF_HOME=/root/hf HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.total --format=csv > "$logs/gpu.log"
/root/e103-venv/bin/python -u injecforks_states.py "$job_config" run 2>&1 | tee "$logs/run-9b.log"
