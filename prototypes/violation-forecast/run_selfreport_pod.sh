#!/usr/bin/env bash
# Self-report run, any config (injecforks_selfreport.py), pod side. Invoked only by rpjob's timed, self-terminating script supervisor, from
# the repo root with the repo-relative pod config. Weights and venv live on container disk.
set -euo pipefail   # pipefail: a failed disk check still fails the job after tee records it
repo=$(pwd)
job_config="$repo/${1:?repo-relative pod configuration required}"
out=$(sed -n "s/^results_dir: *//p" "$job_config"); [ -n "$out" ] || { echo "results_dir missing from $job_config" >&2; exit 1; }
cd "$repo/prototypes/violation-forecast"
mkdir -p "$out"/stage-logs
python3 - <<'PY' 2>&1 | tee "$out"/stage-logs/setup.log
import shutil
for location, minimum in [('/root', 45), ('.', 10)]:
    free = shutil.disk_usage(location).free / 2**30
    print(f'{location}: {free:.1f} GiB free (need {minimum})', flush=True)
    if free < minimum:
        raise SystemExit(f'{location}: {free:.1f} GiB free; need {minimum} GiB before setup')
PY
python3 -m venv /root/sr-venv
/root/sr-venv/bin/pip install --no-cache-dir -r requirements-e24-pod.txt \
  >"$out"/stage-logs/install.log 2>&1
export HF_HOME=/root/hf HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.total --format=csv > "$out"/stage-logs/gpu.log
/root/sr-venv/bin/python -u injecforks_selfreport.py "$job_config" run 2>&1 | tee "$out"/stage-logs/run.log
