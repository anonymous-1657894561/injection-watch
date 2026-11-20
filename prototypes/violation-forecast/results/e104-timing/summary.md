# E104 timing summary

GPU: NVIDIA A100-SXM4-80GB, bf16, $1.79/h, 10 realistic dev cases (prompt 4416–6691 tokens), median of 3 repeats. Fast linear-attention kernels: {'fla': False, 'causal_conv1d': False}.
Unit: one generated agent token = 48.6 ms; prompt prefill (the agent's own) = 1.25 s.

| method | N | s, cached | s, from scratch | $, cached | $, from scratch | from scratch, in agent tokens |
|---|---|---|---|---|---|---|
| 9B probe (comply / detect) | 16 | 0.05 ms | 1.234 s | 0.000000 | 0.000613 | 25 |
| 9B prefix-only probe | 16 | — | 0.216 s | — | 0.000107 | 4 |
| 9B self-report | 16 | 0.223 s | 1.259 s | 0.000111 | 0.000626 | 26 |
| Distill probe (8B) | 16 | 36.21 ms | 0.455 s | 0.000018 | 0.000226 | 9 |
| 9B probe (comply / detect) | 32 | 0.05 ms | 1.245 s | 0.000000 | 0.000619 | 26 |
| 9B prefix-only probe | 32 | — | 0.216 s | — | 0.000107 | 4 |
| 9B self-report | 32 | 0.223 s | 1.256 s | 0.000111 | 0.000624 | 26 |
| Distill probe (8B) | 32 | 37.71 ms | 0.457 s | 0.000019 | 0.000227 | 9 |
| pg2-86m | any | — | 17.78 ms | — | 0.000009 | 0 |
| pg2-22m | any | — | 18.11 ms | — | 0.000009 | 0 |
| piguard | any | — | 17.70 ms | — | 0.000009 | 0 |
| monitor: GPT-6.1 Sol low | 16, 32 | 1.986 s | 1.788 s | 0.000520 | 0.006264 | API round trip |
| monitor: Sonnet 5.5 low | 16, 32 | 2.788 s | 1.900 s | 0.001252 | 0.007938 | API round trip |
