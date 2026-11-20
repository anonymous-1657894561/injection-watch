#!/usr/bin/env bash
# E107 N = 16 reminder (always on and gated) on realistic look-alikes and twins, pod side. Invoked only by rpjob's timed, self-terminating script supervisor, from the repo root
# with the repo-relative pod config. Weights and venv live on container disk. The split comes from the config; test is read once (--final), and the runner refuses
# if the artifact repo already holds a completed E107 test run.
set -euo pipefail   # pipefail: a failed disk check still fails the job after tee records it
repo=$(pwd)
job_config="$repo/${1:?repo-relative pod configuration required}"
cd "$repo/prototypes/violation-forecast"
mkdir -p results/e107-cloud/stage-logs
python3 - <<'PY' 2>&1 | tee results/e107-cloud/stage-logs/setup.log
import shutil
for location, minimum in [('/root', 45), ('.', 10)]:
    free = shutil.disk_usage(location).free / 2**30
    print(f'{location}: {free:.1f} GiB free (need {minimum})', flush=True)
    if free < minimum:
        raise SystemExit(f'{location}: {free:.1f} GiB free; need {minimum} GiB before setup')
PY
python3 -m venv /root/e107-venv
/root/e107-venv/bin/pip install --no-cache-dir -r requirements-e24-pod.txt \
  >results/e107-cloud/stage-logs/install.log 2>&1
export HF_HOME=/root/hf HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.total --format=csv > results/e107-cloud/stage-logs/gpu.log
split=$(/root/e107-venv/bin/python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['split'])" "$job_config")
final=$([ "$split" = test ] && echo --final || true)
/root/e107-venv/bin/python -u injecforks_e107.py "$job_config" run --split "$split" $final 2>&1 | tee results/e107-cloud/stage-logs/run.log
