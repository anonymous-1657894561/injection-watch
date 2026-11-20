| Policy (exact 10 % pooled) | reminder | harmful calls / attack run [95 % CI] | prevented / induced per attack run | net avoided / 100 reminders | completion (weighted) | fires attack / twin / look-alike |
|---|---|---|---|---|---|---|
| no reminder |  | 0.036 | 0.000 / 0.000 | — | 0.769 | 0.000 / 0.000 / 0.000 |
| comply_R0 | reminder_n0 | 0.004 [0.002, 0.008] | 0.032 / 0.001 | 7.815 | 0.784 | 0.400 / 0.000 / 0.000 |
| exposure_n0 | reminder_n0 | 0.010 [0.006, 0.015] | 0.026 / 0.001 | 6.303 | 0.779 | 0.350 / 0.000 / 0.050 |
| exposure_M | reminder_n0 | 0.010 [0.006, 0.017] | 0.026 / 0.000 | 6.471 | 0.783 | 0.400 / 0.000 / 0.000 |
| risk_M | reminder_n0 | 0.008 [0.004, 0.015] | 0.029 / 0.001 | 6.975 | 0.784 | 0.379 / 0.000 / 0.021 |
| comply_A | reminder | 0.020 [0.014, 0.027] | 0.016 / 0.001 | 3.866 | 0.771 | 0.152 / 0.075 / 0.098 |
| r0a | reminder | 0.007 [0.004, 0.011] | 0.033 / 0.004 | 7.143 | 0.782 | 0.369 / 0.000 / 0.031 |
| monitor | reminder | 0.021 [0.015, 0.029] | 0.016 / 0.002 | 3.537 | 0.777 | 0.283 / 0.000 / 0.117 |
| comply_R0_n16 | reminder | 0.010 [0.006, 0.014] | 0.031 / 0.005 | 6.387 | 0.785 | 0.400 / 0.000 / 0.000 |
| comply_A_random_k | reminder | 0.028 [0.021, 0.035] | 0.009 / 0.001 | 2.017 | 0.770 | 0.152 / 0.075 / 0.098 |
| always (N = 16) |  | 0.007 | 0.034 / 0.006 | 0.706 | 0.810 | 1.000 / 1.000 / 1.000 |
| always (N = 0) |  | 0.003 | 0.035 / 0.002 | 0.824 | 0.784 | 1.000 / 1.000 / 1.000 |

| Contrast | point | 95 % CI |
|---|---|---|
| primary|call_rate (comply_R0 − comply_A) | -0.016 | [-0.022, -0.009] |
| primary|completion (comply_R0 − comply_A) | +0.013 | [+0.008, +0.017] |
| target|call_rate (risk_M − exposure_M) | -0.002 | [-0.007, +0.004] |
| target|completion (risk_M − exposure_M) | +0.001 | [-0.003, +0.005] |
| vs_exposure|call_rate (comply_R0 − exposure_n0) | -0.006 | [-0.010, -0.002] |
| vs_exposure|completion (comply_R0 − exposure_n0) | +0.005 | [+0.001, +0.008] |
| n16_adds|call_rate (r0a − comply_R0_n16) | -0.003 | [-0.006, +0.000] |
| n16_adds|completion (r0a − comply_R0_n16) | -0.003 | [-0.005, +0.000] |
| monitor|call_rate (monitor − comply_R0_n16) | +0.011 | [+0.005, +0.018] |
| monitor|completion (monitor − comply_R0_n16) | -0.008 | [-0.013, -0.003] |
| within|call_rate (comply_A − comply_A_random_k) | -0.007 | [-0.011, -0.004] |
| within|completion (comply_A − comply_A_random_k) | +0.001 | [-0.002, +0.003] |

Decision: **choose comply-R0**. Untreated harmful-call rate 0.036.
