# E110: recall at fixed look-alike FPR (test, N = 16; 76 complying injected, 702 look-alike replies)

| method | recall@1% FPR | recall@5% | recall@10% | AUROC complying vs look-alike |
|---|---|---|---|---|
| compliance probe (N=16) | 0.026 [0.000, 0.103] | 0.289 [0.149, 0.436] | 0.526 [0.368, 0.662] | 0.801 |
| detection probe (prompt) | 0.224 [0.073, 0.438] | 0.421 [0.243, 0.594] | 0.579 [0.347, 0.747] | 0.896 |
| PIGuard | 0.000 [0.000, 0.063] | 0.132 [0.000, 0.295] | 0.263 [0.073, 0.440] | 0.597 |
| PromptGuard 2 86M | 0.000 [0.000, 0.000] | 0.026 [0.000, 0.080] | 0.026 [0.000, 0.113] | 0.325 |
