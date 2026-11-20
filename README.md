# Forecasting prompt-injection compliance from an agent's activations

Code and data for the paper: a linear probe on Qwen3.5-9B's residual stream, read after the first N reasoning tokens that
follow a tool result, predicts whether the agent will carry out an injected instruction, and gates a cheap mitigation
(a one-sentence reminder, steering, or resampling). The dataset, InjecForks, ships in this repository.

## Install

Python 3.12 (the analysis ran on 3.12.13; the GPU runs used the same pins).

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

The AgentDojo, gpt-oss-20b and Qwen3.5-27B rollout runs need a second environment with vLLM; see the end of
`requirements.txt`. Diagrams also need `rsvg-convert` and the TeX Gyre Termes fonts.

Credentials, only for the steps that use them: `OPENROUTER_API_KEY` (LLM monitor, LLM judge, API rollouts),
`OPENAI_API_KEY` (AgentDojo monitor), a Hugging Face token that has accepted the Llama 4 Community licence (PromptGuard 2).

## Dataset

`injecforks.tar.xz` (sha256 `1ef9745387fb7f2386d7853581c324c125b88311f03ae9e370848ee420d62c3f`) holds InjecForks: 2,417 cases (867 train / 186 dev / 1,364 test) per arm,
two arms, 8 / 8 / 16 Qwen3.5-9B replies per case, labels, reasoning-token cut offsets, benign twins and look-alikes.

```bash
sha256sum injecforks.tar.xz                    # must print the hash above
mkdir -p data && tar -xJf injecforks.tar.xz -C data
(cd data/injecforks && sha256sum -c SHA256SUMS*)
# optional: injecforks_load.py unpacks the archive into data/ on first use; INJECFORKS_ROOT points it elsewhere
```

This creates the `injecforks` folder under `data/`. The data card inside it gives the schema, splits and the rules for
each split (fit on train, select on dev, measure on test once). Code loads it through `injecforks_load.py`; a few
build and rewording scripts read the same files from `data/e78/injecforks_v1/` under the code folder, so unpack a second
copy there (or symlink it) before running those.

## How to read the tables

- Run every command from `prototypes/violation-forecast/` (figure scripts: from the repository root).
- **Expected file** paths are relative to `prototypes/violation-forecast/results/`. Only summaries ship; the per-row score
  files, activation caches and generated replies they were computed from do not (too large, or they carry run records), so most CPU
  steps need the GPU or API step above them first. The **Needs** column says what is missing.
- Times and costs are what the original runs billed. GPU prices are the cloud rates paid at the time; "local" means a
  6 GB laptop GPU (RTX 2060). API costs are OpenRouter's billed amounts.
- Most scripts that read `test` require `--final`.

## GPU and API runs (generation)

The original runs were launched through a private cloud launcher that rented one GPU, built a venv from these pins,
downloaded the model and ran one script. The commands below are the equivalent on any machine with a GPU of at least the
listed VRAM and the model weights available to `transformers` (or vLLM). This changes how runs start, not what they compute.

Every `run` stage writes its outputs to a private Hugging Face dataset repo named by `artifact_repo:` in its config and
refuses a public one. Set `artifact_repo:` (and `dataset_repo:` where present) to a private repo you own before running.

| # | Produces (paper) | Command | Hardware (VRAM) | Time, cost (original) | Needs (not in this release) |
|---|---|---|---|---|---|
| G1 | Qwen3.5-9B states, train/dev/test: every probe result | `python injecforks_states.py configs/e79v2-9b-pod.yaml run` | A40 (40 GiB) | 2.61 h, $1.54 | none |
| G2 | States of the 8 extra test replies per eval case (Fig. forecast, N = 8, 16) | `python injecforks_states.py <evalextra config> run` | H200 | 0.38 h, $1.99 | the `evalextra` test config |
| G3 | Self-report scores (Fig. forecast, Table cost) | `python injecforks_selfreport.py configs/selfreport-9b-pod.yaml run` (dev config shipped) | A100 80 GB (40 GiB) | test 7.08 h, $12.67; extra replies 1.55 h, $2.78 | the fp32 test and `evalextra` configs |
| G4 | LLM monitor scores (Fig. forecast) | `python injecforks_monitor.py run --model openai/gpt-6.1-sol --effort low --max-tokens 8000 --prompt v1 --split test --configs smoke realistic smoke-benign realistic-benign smoke-lookalike realistic-lookalike --tag test-gpt61sol-low-v1 --workers 16 --max-usd 60`, then `... export --tag test-gpt61sol-low-v1`; extra replies: `... run --split test --subset evalextra --configs realistic --ns 8 16` | API | 74 min, $46.39; extra replies $4.27 | none |
| G5 | Distill-observer states and test cache check (Distill and prefix-only comparison) | `python injecforks_observer_states.py configs/e101-distill-observer-pod.yaml run`, then `python injecforks_states.py configs/e101-9b-testcache.yaml cachecheck` | A40 (40 GiB) | 1.18 h, $0.70 | the prefix-only re-encode run (`injecforks_resample.py`) config |
| G6 | Look-alike states, 9B and Distill (Sec. classifiers, probe vs look-alikes) | `python injecforks_states.py configs/e102-9b-lookalike-pod.yaml run`, then `python injecforks_observer_states.py configs/e102-distill-lookalike.yaml run` | L40S (40 GiB) | 0.35 h, $0.38 | none |
| G7 | States on reworded test injections (App. rewording, probe) | `python injecforks_states.py configs/e103-9b-variants-test-pod.yaml run` | RTX PRO 6000 (40 GiB) | 0.95 h, $2.36 (plus $6.02 API rollouts, already in the dataset) | none |
| G8 | Timings (Table cost) | `python e104_timing.py configs/e104-timing-pod.yaml run`; text readers: `python e134_timing.py` | A100 80 GB SXM (40 GiB); local | 0.29 h, $0.52 (plus $0.70 monitor sample); local $0 | monitor latency records |
| G9 | Reminder on harmless inputs (Sec. mitigation, "what the reminder costs") | `python injecforks_e107.py configs/e107-9b-test-pod.yaml run --split test --final`, then `... score --prefix RUN_ID` | A40 (40 GiB) | 2.66 h, $1.57 (judges est. ~$1.5) | none |
| G10 | Gated reminder (Table mitigation, Fig. budget) | `python injecforks_e110.py configs/e110-9b-pod.yaml run --split test --final` | A40 (40 GiB) | 4.45 h, $2.63; API generation $0.44, judges $0.43 | fresh look-alike inputs (read from the artifact repo) |
| G11 | Gated resampling (Sec. mitigation) | `python injecforks_e120.py configs/e120-9b-pod.yaml run --split test --final` | RTX 6000 Ada (40 GiB) | 0.72 h, $0.71 | G10's outputs |
| G12 | Steering (Sec. mitigation) | `python e127_steer.py configs/e127-9b-steer.yaml run` | A40 (40 GiB) | 3.12 h, $1.84 | `steer.npz` (read from the artifact repo; `configs/e127_directions.npz` holds the frozen directions) |
| G13 | Qwen3.5-27B rollouts and states (Sec. forecast, 27B) | vLLM env: `VLLM_USE_FLASHINFER_SAMPLER=0 python e112_rollouts.py configs/e112-27b-pod.yaml run`; then `python injecforks_states.py configs/e112-27b-pod.yaml run` | rollouts RTX PRO 6000; states A100 80 GB (72 GiB) | rollouts ~1.4 h, ~$4.10; states 3.31 h, $5.93; judge $0.09; monitor $5.59 | none |
| G14 | Prompt probe on six new tools, Qwen3.5-9B (App. further) | `python injecforks_e115.py configs/e115-9b-pod-s{0,1,2,3}.yaml run --split test --final` (four shards) | 4 x A40 (40 GiB) | 9.96 h total, ~$5.88; API generation $3.19, monitor $5.71, judges $3.84 | the E115 test inputs (read from the artifact repo) |
| G15 | Casual-style transfer (App. further) | `python injecforks_e122.py configs/e122-9b-pod.yaml run --split test --final` | RTX 6000 Ada (40 GiB) | 0.59 h, $0.59; API rollouts $5.12, writer $3.87 | rendered inputs (read from the artifact repo; slot texts in `configs/e122_inputs.json`) |
| G16 | gpt-oss-20b on InjecForks (App. further) | vLLM env: `python e140_injecforks.py build`; serve `openai/gpt-oss-20b` on :8000 (`--max-model-len 16384`); `python e140_injecforks.py gen --split train --arm U`, `eligibility`, `gen --split test --arm U`, `gen --split test --arm P`; stop the server; `python e140_injecforks.py states` | A100 80 GB SXM (60 GiB) | 0.91 h, $2.00 | none |
| G17 | gpt-oss-20b on AgentDojo (App. further) | vLLM env: serve `openai/gpt-oss-20b` on :8000 (`--max-model-len 32768`); `python e125_agentdojo.py prefix --split {dev,test}`; `python e125_agentdojo.py cont --split test --arm {U,P,T}` (and `--split dev --arm U`); stop the server; `python e125_states.py run --data data/e125` | A100 80 GB SXM (60 GiB) | 1.12 h, $2.00; monitor $0.59 + ~$1.9 | none |
| G18 | AgentDojo multi-attack, gpt-oss-20b (App. further) | as G17 with `e129_agentdojo.py gpt-oss-20b ...` and `e129_states.py gpt-oss-20b run --data data/e129/gpt-oss-20b --split test` | A100 80 GB (60 GiB) | test 2.38 h, $4.98; dev 0.89 h, $1.85 | the dev config |
| G19 | Text readers: embedding, fine-tuned DeBERTa, 27B transfer (Fig. forecast, Table mitigation) | `python e134_dev.py`, `python e134_test.py`; `python e136_grid.py`, `python e136_final.py`, `python e136_test.py`; `python e137.py dev`, `python e137.py test`; `python e135_scores.py`; `python e136_gate_scores.py`; `python e139_dev.py`, `python e139_scores.py` | local RTX 2060 (6 GB) | $0; E137 ~7 min; others not recorded | probe and monitor score files (G1, G4) |

The pod-side shell scripts (`run_*_pod.sh`) are kept for reference: they show the exact environment variables and stage
order of each original run. The dataset itself was generated through OpenRouter (Qwen3.5-9B on DeepInfra, bf16;
`e77_variants.py`, `e78_freeze.py`; about $22 and 7 h), and labelled with an LLM judge (`injecforks_judge.py`, cost not
recorded) and `injecforks_labels_v12.py`. Rebuilding it needs InjecAgent cloned at commit `f19c9f2c` into
`data/e77/injecagent/raw/repo/`. `python3 -I check_e78.py` is the structural checker for the hand-written user tools.

## CPU analysis (from scores and states)

Rows marked **runs** reproduce from this repo alone: the per-row scores, units and thresholds they read are shipped under
`results/` and `data/`. Run them from `prototypes/violation-forecast/` with `INJECFORKS_ROOT` set (figures from the repo
root). Their outputs match the shipped files byte for byte, apart from the `git` commit field some summaries record.
The other rows refit probes from activation caches, which are not released (tens of GB); rerun the listed GPU step to
regenerate them, or check the shipped expected file.

| Paper result | Command | Needs (not in this release) | Expected file |
|---|---|---|---|
| Fig. forecast, N = 8 / 16: probe 0.687 / 0.744 vs monitor and self-report, deltas with two-way CI | `python e113_analyze.py` | **runs** | `injecforks/analysis/e113.json`, `e113.md` |
| Fig. forecast, N = 32-128 (probe 0.808 / 0.846 / 0.831); Distill 0.712 and prefix-only 0.717 vs probe 0.770 | `python e79_final.py --split test --states STATES --monitor 'GLOB' --selfreport 'GLOB' --cache-check FILE --observer 'GLOB' --prefix-only 'GLOB' --final` | G1, G3, G4, G5 outputs | `injecforks/analysis/e79-final-test-verdict.json`, `.md` |
| Fig. forecast, text readers (embedding 0.700, DeBERTa 0.709 at N = 16) | G19, then `python injecforks_score.py --split test --method ... --out NAME` | G19's score files | `injecforks/analysis/e134-test/summary.md`, `e136-test/summary.md`, `e136-test-vs-emb/summary.md` |
| Fig. forecast (the plot) | `python docs/paper/figs-candidates/fig_forecast_text.py` (repo root) | **runs** | `docs/paper/fig_forecast.pdf` |
| Sec. forecast, pooled AUROC 0.78 vs 0.69 | not scripted in this release | - | - |
| Probe geometry: comply vs detection cosine, projection | `python e128_geometry.py --states STATES --split test --final` | G1 states | `e128/test.json` |
| Fig. train size (112 replies -> 0.69; all 867 cases 0.753) | `python e144/extract.py`, then `cd e144 && python3 trainsize.py` | G1 train states | `e144/trainsize/curve.json` |
| Fig. train size (the plot) | `python3 docs/paper/figs-candidates/fig_trainsize.py` (repo root) | **runs** | `docs/paper/fig_trainsize.pdf` |
| Table cost (PIGuard 17.7 ms, self-report 0.223 s, monitor 1.99 s, probe 0.05 ms) | `python e104_analyze.py` | **runs** | `e104-timing/summary.md` |
| Table mitigation: comply probe 7.9 %, detection probe 9.5 %, PIGuard 10.5 %, PromptGuard 2 14.4 % | `python e110_analyze.py evaluate RUN_ID --judge` | G10 outputs | `e110/summary.json` (cut `calibrated\|0.1`) |
| Table mitigation: embedding 11.4 %, DeBERTa 9.9 % | `python3 e135_analyze.py`; `python3 e136_gate_analyze.py` | **runs** | `e135/summary.json`; `e136/gate/summary.json` |
| Embedding gate with look-alike negatives (11.0 %) | `python3 e139_analyze.py` | **runs** | `e139/summary.json` |
| Fig. budget | `python docs/paper/figs-candidates/paper_figs.py` (repo root) | **runs** | `docs/paper/fig_budget.pdf` |
| Reminder cost on harmless inputs (-6.2 / -1.8 points always on) | `python e107_analyze.py --split test RUN_ID --judge` | G9 outputs | `e107/test-table.md` |
| Steering: 41.5 % -> 38.1 / 34.4 / 29.2 % | `python e127_analyze.py --rows data/e127-paper/rows.jsonl --tag paper` | **runs** | `e127/summary-paper.md` |
| Gated resampling: 10.4 % vs 15.4 % random | `python e120_analyze.py user-e120-9b-pod-20261111T121249Z` | **runs** | `e120/summary.json` |
| Input classifiers on look-alikes | `python e102_classifiers.py --split test --final` | none (dataset + classifier weights) | `injecforks/analysis/e102-classifiers-test.json` |
| Probe vs look-alikes 0.80 / 0.86 | `python injecforks_probe.py --fixed --lookalike ...`, then `python e102_probes.py --split test --final` | G6 outputs, probe scores | `injecforks/analysis/e102-probes-test.json` |
| Recall at 10 % look-alike FPR: probe 53 %, PIGuard 26 %, PromptGuard 2 3 % | `python e110_recall_at_fpr.py` | **runs** | `e110/recall_at_fpr.json`, `.md` |
| App. rewording, classifiers (Tables rewording) | `python e103_brittleness.py --split test --final` | none (dataset + classifier weights) | `e103/test.json` |
| App. rewording, agent: T4 23.9 vs 21.5 %, T2 32.3 % | `python e103_agent.py --split test --variants T2 T4` | the variant rollouts in the layout the script reads (`data/e78/variants/`) | `e103/agent-test.json` |
| App. rewording, probe: T4 0.732, T2 0.640 vs 0.747 | `python e103_probe_test.py` | G7 probe scores | `e103/probe-test.json` |
| Qwen3.5-27B: probe 0.730 / 0.776 / 0.834 vs monitor | `python e112_final.py --split test --states STATES --monitor FILE --final` | G13 outputs | `e112/e112-test-verdict.json` |
| Qwen3.5-27B detection probe, held out by tool | `python e112_detect_cv.py --states STATES` | G13 states | `e112/detect-cv.json` |
| Qwen3.5-27B text reader 0.741 vs probe 0.776 | `python e137.py test` | G13 states, probe scores | `e137/test.json` |
| Qwen3.5-27B prompt-read vs N = 16 read | `python e119_analyze.py --rollouts ROOT --states DIR --tag 27b` (9B check: `--tag 9b-pipeline`) | G13 / G1 outputs | `e119/27b/summary.json`, `e119/9b-pipeline/summary.json` |
| 9B compliance and prompt-risk probes flag about as many | `python e114_analyze.py` | G10 outputs | `e114/summary.json` |
| Prompt probe on new tools: 0.4 % vs 2.0 % | `python e115_freeze.py`; `python e115_analyze.py judge RUN_IDS`; `python e115_analyze.py evaluate RUN_IDS` | G14 outputs (states); the audit `python e124_audit.py` **runs** from the shipped E115 units | `e115/summary.json`, `e115/table.md`, `e124/table.md` |
| Casual-style transfer: 0 of 91 flagged | `python e122_analyze.py RUN_ID` | G15 outputs | `e122/table.md` |
| gpt-oss-20b InjecForks: 364 -> 47 vs 196 | `python e140_analyze.py fit`, then `python e140_analyze.py test` | G16 outputs | `e140/summary.json`, `e140/table.md` |
| AgentDojo gpt-oss-20b: comply 9.1 % vs random 9.9 % | `python e125_fit.py --data DIR`, then `python e125_analyze.py --data DIR` | G17 outputs | `e125/summary.json`, `e125/table.md` |
| AgentDojo multi-attack: 71 of 450 checkpoints mixed | `python e129_fit.py gpt-oss-20b --data DIR`, then `python e129_analyze.py gpt-oss-20b --data DIR` | G18 outputs | `e129/gpt-oss-20b/summary.json`, `table.md` |
| Authorized instructions: 61 % vs 23 % | `python e111_analyze.py RUN_ID`; `python e116_analyze.py RUN_IDS` | their GPU runs (not in this release) | not shipped |
| Fig. method, Fig. walkthrough (no numbers) | `python3 docs/paper/figs-candidates/diagrams.py` (repo root) | none | `docs/paper/fig_method.pdf`, `fig_walkthrough.pdf` |
| App. dataset (counts, labels, judge audit) | data card in the archive | none | the data card |

CPU steps cost $0 except where the command says `--judge` (OpenRouter LLM judge; costs are listed with the GPU run they
judge). Their run times were not recorded.

## Licence

Code and dataset: MIT (`LICENSE`). InjecForks builds on InjecAgent and AgentDojo; see `THIRD_PARTY_NOTICES.md`.
