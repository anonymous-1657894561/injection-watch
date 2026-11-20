| style | runs | harmful (rate) | capture comply_R0 | capture exposure_M | capture risk_M | capture comply_A | injected flagged R0 / exp_M | look-alikes flagged R0 / exp_M | AUROC R0 / exp_M |
|---|---|---|---|---|---|---|---|---|---|
| important | 768 | 34 (0.044) | 0.82 [0.63, 0.95] | 0.65 [0.36, 0.84] | 0.76 | 0.44 | 0.16 / 0.29 | 0.04 / 0.00 | 0.87 / 0.74 |
| tool_knowledge | 768 | 14 (0.018) | 0.36 [0.07, 0.65] | 0.50 [0.20, 0.80] | 0.36 | 0.29 | 0.07 / 0.12 | 0.00 / 0.00 | 0.83 / 0.76 |
| ignore_previous | 768 | 10 (0.013) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.30 | 0.20 | 0.00 / 0.00 | 0.00 / 0.00 | 0.95 / 0.63 |
| system_message | 768 | 2 (0.003) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 | 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.87 / 0.64 |
| injecagent_base | 768 | 7 (0.009) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.14 | 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.87 / 0.70 |
| injecagent_enhanced | 768 | 5 (0.007) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 | 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.94 / 0.51 |
| natural | 768 | 2 (0.003) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.50 | 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.87 / 0.54 |
| casual | 768 | 91 (0.118) | 0.92 [0.83, 0.99] | 0.00 [0.00, 0.00] | 0.77 | 0.49 | 0.26 / 0.00 | 0.00 / 0.00 | 0.91 / 0.58 |
| softened | 768 | 26 (0.034) | 0.77 [0.47, 0.94] | 0.23 [0.00, 0.55] | 0.88 | 0.65 | 0.17 / 0.04 | 0.00 / 0.00 | 0.89 / 0.82 |
| held_out | 6144 | 157 (0.026) | 0.69 [0.58, 0.79] | 0.08 [0.02, 0.16] | 0.66 | 0.43 | 0.06 / 0.02 | 0.00 / 0.00 | 0.93 / 0.69 |
| wrapped | 2304 | 131 (0.057) | 0.83 [0.74, 0.90] | 0.10 [0.03, 0.19] | 0.75 | 0.50 | 0.17 / 0.06 | 0.00 / 0.00 | 0.90 / 0.39 |
| unwrapped | 3840 | 26 (0.007) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.19 | 0.08 | 0.00 / 0.00 | 0.00 / 0.00 | 0.93 / 0.60 |

| contrast | point | 95 % CI |
|---|---|---|
| primary|held_out|comply_R0-exposure_M | +0.611 | [+0.500, +0.713] |
| important|comply_R0-exposure_M | +0.176 | [+0.028, +0.400] |
| tool_knowledge|comply_R0-exposure_M | -0.143 | [-0.571, +0.273] |
| ignore_previous|comply_R0-exposure_M | +0.000 | [+0.000, +0.000] |
| system_message|comply_R0-exposure_M | +0.000 | [+0.000, +0.000] |
| injecagent_base|comply_R0-exposure_M | +0.000 | [+0.000, +0.000] |
| injecagent_enhanced|comply_R0-exposure_M | +0.000 | [+0.000, +0.000] |
| natural|comply_R0-exposure_M | +0.000 | [+0.000, +0.000] |
| casual|comply_R0-exposure_M | +0.923 | [+0.829, +0.990] |
| softened|comply_R0-exposure_M | +0.538 | [+0.190, +0.842] |
| wrapped|comply_R0-exposure_M | +0.733 | [+0.615, +0.833] |
| unwrapped|comply_R0-exposure_M | +0.000 | [+0.000, +0.000] |
| auroc|comply_R0|heldout_mean-important | +0.024 | [-0.048, +0.109] |
| auroc|exposure_M|heldout_mean-important | -0.096 | [-0.202, +0.058] |
| auroc|risk_M|heldout_mean-important | +0.132 | [+0.035, +0.249] |
| auroc|exposure_n0|heldout_mean-important | +0.132 | [+0.012, +0.266] |
| auroc|comply_A|heldout_mean-important | +0.069 | [-0.024, +0.158] |
| auroc|r0a|heldout_mean-important | +0.048 | [-0.020, +0.127] |

E115 twins (96 inputs, 768 runs) flagged: comply_R0 0.000, exposure_M 0.000, risk_M 0.000, exposure_n0 0.000, comply_A 0.122, r0a 0.000
