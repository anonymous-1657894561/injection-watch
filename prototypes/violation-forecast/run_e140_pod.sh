#!/usr/bin/env bash
# E140, pod side (experiments/E140-injecforks-gptoss.md): gpt-oss-20b on InjecForks with E115's design. Invoked only by rpjob's timed,
# self-terminating script supervisor, from the repo root with the repo-relative pod config. E133's two venvs (vLLM's own; a client venv for
# openai-harmony). Order: inputs; unit checks; HF states smoke before any generation; vLLM serve; smoke; train U; eligibility (frozen rule:
# test is generated only if it passes); test U, test P; HF states. Outputs go to the private artifact repo after every stage; test outcomes
# are never printed here.
set -euo pipefail
repo=$(pwd)
job_config="$repo/${1:?repo-relative pod configuration required}"
cd "$repo/prototypes/violation-forecast"
L=results/e140-cloud/stage-logs; mkdir -p $L data/e140
python3 - <<'PY' 2>&1 | tee $L/setup.log
import shutil
for location, minimum in [('/root', 60), ('.', 5)]:
    free = shutil.disk_usage(location).free / 2**30
    print(f'{location}: {free:.1f} GiB free (need {minimum})', flush=True)
    if free < minimum:
        raise SystemExit(f'{location}: {free:.1f} GiB free; need {minimum} GiB before setup')
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > $L/gpu.log
export VLLM_EXTRA="accelerate openai-harmony==0.0.8 huggingface_hub"
source pod_setup_vllm.sh
/root/vllm-venv/bin/pip install -q ninja >>$L/vllm-install.log 2>&1
export PATH=/root/vllm-venv/bin:$PATH
python3 -m venv /root/c-venv
/root/c-venv/bin/pip install -q openai-harmony==0.0.8 requests==2.32.5 pyyaml==6.0.3 huggingface_hub >$L/client-install.log 2>&1
export HF_HOME=/root/hf HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false
RUN=${RPJOB_RUN_ID:-$(date +%Y%m%dT%H%M%S)}
stage() {  # name, command...: log to $L/name.log, time it, stop the job on failure
  local name=$1; shift; local t0=$(date +%s)
  "$@" >$L/$name.log 2>&1 || { echo "stage $name FAILED (see $L/$name.log)" | tee -a $L/stages.log; tail -30 $L/$name.log; exit 1; }
  echo "stage $name ok $(( ($(date +%s) - t0) / 60 )) min" | tee -a $L/stages.log
}
G="/root/c-venv/bin/python -u e140_injecforks.py"
upload() { $G upload --run "$RUN" >>$L/upload.log 2>&1 || echo "upload after $1 failed" | tee -a $L/stages.log; }
stage inputs $G fetch-inputs
stage unit /root/c-venv/bin/python test_e140.py
stage states-smoke /root/vllm-venv/bin/python e125_states.py smoke
cfg() { /root/c-venv/bin/python -c "import yaml; print(yaml.safe_load(open('configs/e117.yaml'))['models']['gpt-oss-20b']['$1'])"; }
/root/vllm-venv/bin/vllm serve "$(cfg hf_id)" --revision "$(cfg revision)" --served-model-name "$(cfg served_name)" \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --enable-prefix-caching --seed 0 --port 8000 >$L/serve.log 2>&1 &
SERVER=$!
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:8000/health >/dev/null && { echo "server up after $((i * 10)) s" | tee -a $L/stages.log; break; }
  kill -0 $SERVER 2>/dev/null || { echo "server died" | tee -a $L/stages.log; tail -40 $L/serve.log; exit 1; }
  sleep 10
done
curl -sf http://127.0.0.1:8000/health >/dev/null || { echo "server not up after 40 min" | tee -a $L/stages.log; exit 1; }
stage smoke-train-U $G gen --split train --arm U --limit 1
stage gen-train-U $G gen --split train --arm U; upload gen-train-U
ELIG=1
$G eligibility >$L/eligibility.log 2>&1 || { rc=$?; [ $rc -eq 3 ] && ELIG=0 || { echo "eligibility FAILED" | tee -a $L/stages.log; exit 1; }; }
echo "eligible=$ELIG" | tee -a $L/stages.log; upload eligibility
if [ $ELIG -eq 1 ]; then
  stage gen-test-U $G gen --split test --arm U; upload gen-test-U
  stage gen-test-P $G gen --split test --arm P; upload gen-test-P
fi
kill $SERVER; wait $SERVER 2>/dev/null || true; sleep 10
stage states /root/vllm-venv/bin/python -u e140_injecforks.py states; upload states
cp data/e140/states/receipt.json results/e140-cloud/states_receipt.json
cp data/e140/eligibility.json results/e140-cloud/eligibility.json
