#!/usr/bin/env bash
# E125, pod side (experiments/E125-agentdojo-gptoss-heldout.md). Invoked only by rpjob's timed, self-terminating script supervisor, from the
# repo root with the repo-relative pod config. E117's two venvs (vLLM's own, a client venv for AgentDojo + openai-harmony); the HF state
# extraction runs in vLLM's venv after the server stops. Order: unit checks; HF states smoke (load, forward, next-token check) before any
# generation; vLLM serve; generation smoke; prefixes (dev, test); continuations (U dev, U test, P test, T test); states; dev summary.
# Episodes and states go to the private artifact repo after every stage (test outcomes are never printed or committed here).
set -euo pipefail
repo=$(pwd)
job_config="$repo/${1:?repo-relative pod configuration required}"
cd "$repo/prototypes/violation-forecast"
L=results/e125-cloud/stage-logs; D=data/e125
mkdir -p $L $D
python3 - <<'PY' 2>&1 | tee $L/setup.log
import shutil
for location, minimum in [('/root', 60), ('.', 5)]:
    free = shutil.disk_usage(location).free / 2**30
    print(f'{location}: {free:.1f} GiB free (need {minimum})', flush=True)
    if free < minimum:
        raise SystemExit(f'{location}: {free:.1f} GiB free; need {minimum} GiB before setup')
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > $L/gpu.log
export VLLM_EXTRA="accelerate openai-harmony==0.0.8"
source pod_setup_vllm.sh
/root/vllm-venv/bin/pip install -q ninja >>$L/vllm-install.log 2>&1   # FlashInfer's sampler JIT-builds with ninja (E117)
export PATH=/root/vllm-venv/bin:$PATH
python3 -m venv /root/ad-venv
/root/ad-venv/bin/pip install -q agentdojo==0.1.35 openai-harmony==0.0.8 transformers==5.16.1 jinja2==3.1.6 requests==2.32.5 pyyaml==6.0.3 \
  >$L/client-install.log 2>&1
export HF_HOME=/root/hf HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false
RUN=${RPJOB_RUN_ID:-$(date +%Y%m%dT%H%M%S)}
stage() {  # name, command...: log to $L/name.log, time it, stop the job on failure
  local name=$1; shift; local t0=$(date +%s)
  "$@" >$L/$name.log 2>&1 || { echo "stage $name FAILED (see $L/$name.log)" | tee -a $L/stages.log; tail -30 $L/$name.log; exit 1; }
  echo "stage $name ok $(( ($(date +%s) - t0) / 60 )) min" | tee -a $L/stages.log
}
upload() { /root/vllm-venv/bin/python e125_pod_util.py upload --data $D --run "$RUN" >>$L/upload.log 2>&1 || echo "upload after $1 failed" | tee -a $L/stages.log; }
stage repo-check /root/vllm-venv/bin/python e125_pod_util.py check-repo
stage unit /root/ad-venv/bin/python test_e125.py
stage states-smoke /root/vllm-venv/bin/python e125_states.py smoke
cfg() { /root/ad-venv/bin/python -c "import yaml; print(yaml.safe_load(open('configs/e117.yaml'))['models']['gpt-oss-20b']['$1'])"; }
/root/vllm-venv/bin/vllm serve "$(cfg hf_id)" --revision "$(cfg revision)" --served-model-name "$(cfg served_name)" \
  --max-model-len 32768 --gpu-memory-utilization 0.90 --enable-prefix-caching --seed 0 --port 8000 >$L/serve.log 2>&1 &
SERVER=$!
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:8000/health >/dev/null && { echo "server up after $((i * 10)) s" | tee -a $L/stages.log; break; }
  kill -0 $SERVER 2>/dev/null || { echo "server died" | tee -a $L/stages.log; tail -40 $L/serve.log; exit 1; }
  sleep 10
done
curl -sf http://127.0.0.1:8000/health >/dev/null || { echo "server not up after 40 min" | tee -a $L/stages.log; exit 1; }
G="/root/ad-venv/bin/python -u e125_agentdojo.py"
stage smoke-prefix $G prefix --split dev --limit 1 --out data/e125-smoke
for arm in U P T; do stage smoke-cont-$arm $G cont --split dev --arm $arm --limit 1 --out data/e125-smoke; done
stage smoke-check /root/ad-venv/bin/python e125_pod_util.py smoke-check --data data/e125-smoke
stage prefix-dev $G prefix --split dev; upload prefix-dev
stage prefix-test $G prefix --split test; upload prefix-test
stage cont-dev-U $G cont --split dev --arm U; upload cont-dev-U
stage cont-test-U $G cont --split test --arm U; upload cont-test-U
stage cont-test-P $G cont --split test --arm P; upload cont-test-P
stage cont-test-T $G cont --split test --arm T; upload cont-test-T
kill $SERVER; wait $SERVER 2>/dev/null || true; sleep 10
stage states /root/vllm-venv/bin/python -u e125_states.py run --data $D; upload states
stage summary /root/ad-venv/bin/python e125_pod_util.py summary --data $D --out results/e125-cloud/pod_summary.json
cp $D/states/receipt.json results/e125-cloud/states_receipt.json
