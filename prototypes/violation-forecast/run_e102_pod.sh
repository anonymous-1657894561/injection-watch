#!/usr/bin/env bash
# E102 look-alike states, pod side (experiments/E102-lookalike-negatives.md). Invoked only by rpjob's timed, self-terminating
# script supervisor, from the repo root with the repo-relative pod configuration. As run_e101_pod.sh: own venv, weights on
# container disk, one model per process:
#   1. Qwen3.5-9B on the look-alike rows (injecforks_states.py run; E79 v2's model and reads) -> e79v2-states
#   2. DeepSeek-R1-Distill-Llama-8B on the same rows (injecforks_observer_states.py run; E101's) -> distill-observer-states
set -euo pipefail   # pipefail: a failed disk check still fails the job after tee records it
repo=$(pwd)
job_config="$repo/${1:?repo-relative pod configuration required}"
cd "$repo/prototypes/violation-forecast"
logs=results/e102-cloud/stage-logs
mkdir -p "$logs"
python3 - <<'PY' 2>&1 | tee "$logs/setup.log"
import shutil
for location, minimum in [('/root', 55), ('.', 10)]:   # 19.3 GB 9B + 16.1 GB observer weights + venv
    free = shutil.disk_usage(location).free / 2**30
    print(f'{location}: {free:.1f} GiB free (need {minimum})', flush=True)
    if free < minimum:
        raise SystemExit(f'{location}: {free:.1f} GiB free; need {minimum} GiB before setup')
PY
python3 -m venv /root/e102-venv
/root/e102-venv/bin/pip install --no-cache-dir -r requirements-e24-pod.txt >"$logs/install.log" 2>&1
export HF_HOME=/root/hf HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.total --format=csv > "$logs/gpu.log"
/root/e102-venv/bin/python -u injecforks_states.py "$job_config" run 2>&1 | tee "$logs/run-9b.log"
/root/e102-venv/bin/python -u injecforks_observer_states.py configs/e102-distill-lookalike.yaml run 2>&1 | tee "$logs/run-distill.log"
