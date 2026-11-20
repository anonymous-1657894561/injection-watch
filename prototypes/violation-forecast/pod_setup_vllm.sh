# Sourced by run_e50_pod.sh and run_e59_pod.sh: vLLM's own venv at /root/vllm-venv (it pins its own torch), installed as E43's
# recipe does. Needs $L (the stage-log directory). Extra pip packages for that venv go in $VLLM_EXTRA (space-separated).
python3 -m venv /root/vllm-venv
/root/vllm-venv/bin/pip install -q uv==0.12.22 >$L/vllm-install.log 2>&1
# vLLM brings its own transformers and numpy; e50_labels.py asserts the HF venv rebuilds every prompt's token ids exactly.
/root/vllm-venv/bin/uv pip install -q --python /root/vllm-venv/bin/python --torch-backend=auto vllm==0.30.0 pyyaml==6.0.3 \
  ${VLLM_EXTRA:-} >>$L/vllm-install.log 2>&1
/root/vllm-venv/bin/python -c 'import torch, vllm; x = torch.ones(4, device="cuda"); print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, "gpu ok", float(x.sum()))' \
  | tee -a $L/vllm-install.log
