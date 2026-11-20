# InjecForks v1.2 · test · e136-test-vs-emb

Methods: ft, emb · baseline emb

## Case-mean within-case AUROC (95 % case-bootstrap CI)

| config / cut | N | ft | emb | rows | mixed cases |
|---|---|---|---|---|---|
| realistic/all | 0 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 2392 | 299 |
| realistic/all | 8 | 0.647 [0.614, 0.678] | 0.641 [0.608, 0.672] | 2392 | 299 |
| realistic/all | 16 | 0.709 [0.678, 0.740] ← | 0.700 [0.669, 0.729] ← | 2392 | 299 |
| realistic/all | 32 | 0.810 [0.785, 0.836] | 0.804 [0.777, 0.828] | 2327 | 299 |
| realistic/all | 64 | 0.875 [0.852, 0.897] | 0.846 [0.820, 0.871] | 2060 | 293 |
| realistic/all | 128 | 0.899 [0.872, 0.923] | 0.873 [0.847, 0.899] | 1477 | 260 |
| realistic/reach64 | 0 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 2060 | 293 |
| realistic/reach64 | 8 | 0.646 [0.611, 0.680] | 0.634 [0.599, 0.671] | 2060 | 293 |
| realistic/reach64 | 16 | 0.700 [0.666, 0.733] | 0.700 [0.669, 0.730] | 2060 | 293 |
| realistic/reach64 | 32 | 0.805 [0.778, 0.829] | 0.793 [0.765, 0.819] | 2060 | 293 |
| realistic/reach64 | 64 | 0.875 [0.851, 0.896] | 0.846 [0.820, 0.870] | 2060 | 293 |
| realistic/reach64 | 128 | 0.899 [0.875, 0.923] | 0.873 [0.848, 0.899] | 1477 | 260 |
| smoke/all | 0 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 2392 | 299 |
| smoke/all | 8 | 0.629 [0.598, 0.659] | 0.589 [0.558, 0.620] | 2392 | 299 |
| smoke/all | 16 | 0.669 [0.639, 0.696] ← | 0.657 [0.629, 0.686] ← | 2392 | 299 |
| smoke/all | 32 | 0.762 [0.737, 0.787] | 0.696 [0.668, 0.721] | 2381 | 299 |
| smoke/all | 64 | 0.831 [0.807, 0.854] | 0.795 [0.768, 0.820] | 2315 | 297 |
| smoke/all | 128 | 0.879 [0.855, 0.902] | 0.851 [0.828, 0.874] | 1929 | 290 |
| smoke/reach64 | 0 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 2315 | 297 |
| smoke/reach64 | 8 | 0.622 [0.590, 0.652] | 0.593 [0.563, 0.624] | 2315 | 297 |
| smoke/reach64 | 16 | 0.664 [0.635, 0.692] | 0.653 [0.623, 0.682] | 2315 | 297 |
| smoke/reach64 | 32 | 0.756 [0.730, 0.781] | 0.694 [0.667, 0.721] | 2315 | 297 |
| smoke/reach64 | 64 | 0.831 [0.807, 0.854] | 0.795 [0.768, 0.820] | 2315 | 297 |
| smoke/reach64 | 128 | 0.879 [0.857, 0.900] | 0.851 [0.829, 0.873] | 1929 | 290 |

## Paired difference vs emb (case-mean within-case AUROC)

| config / cut | N | method | Δ | 95 % CI | CI excludes 0 | Δ ≥ 0.05 |
|---|---|---|---|---|---|---|
| realistic/all | 0 | ft | +0.000 | [+0.000, +0.000] | False | False |
| realistic/all | 8 | ft | +0.006 | [-0.021, +0.034] | False | False |
| realistic/all | 16 | ft | +0.010 | [-0.011, +0.031] | False | False |
| realistic/all | 32 | ft | +0.006 | [-0.012, +0.025] | False | False |
| realistic/all | 64 | ft | +0.029 | [+0.014, +0.044] | True | False |
| realistic/all | 128 | ft | +0.026 | [+0.010, +0.042] | True | False |
| realistic/reach64 | 0 | ft | +0.000 | [+0.000, +0.000] | False | False |
| realistic/reach64 | 8 | ft | +0.011 | [-0.017, +0.041] | False | False |
| realistic/reach64 | 16 | ft | -0.000 | [-0.024, +0.024] | False | False |
| realistic/reach64 | 32 | ft | +0.011 | [-0.009, +0.030] | False | False |
| realistic/reach64 | 64 | ft | +0.029 | [+0.013, +0.043] | True | False |
| realistic/reach64 | 128 | ft | +0.026 | [+0.010, +0.042] | True | False |
| smoke/all | 0 | ft | +0.000 | [+0.000, +0.000] | False | False |
| smoke/all | 8 | ft | +0.040 | [+0.010, +0.068] | True | False |
| smoke/all | 16 | ft | +0.011 | [-0.015, +0.037] | False | False |
| smoke/all | 32 | ft | +0.066 | [+0.039, +0.094] | True | True |
| smoke/all | 64 | ft | +0.036 | [+0.014, +0.058] | True | False |
| smoke/all | 128 | ft | +0.028 | [+0.012, +0.044] | True | False |
| smoke/reach64 | 0 | ft | +0.000 | [+0.000, +0.000] | False | False |
| smoke/reach64 | 8 | ft | +0.029 | [-0.000, +0.059] | False | False |
| smoke/reach64 | 16 | ft | +0.011 | [-0.013, +0.035] | False | False |
| smoke/reach64 | 32 | ft | +0.061 | [+0.036, +0.089] | True | True |
| smoke/reach64 | 64 | ft | +0.036 | [+0.013, +0.058] | True | False |
| smoke/reach64 | 128 | ft | +0.028 | [+0.012, +0.045] | True | False |

## Average precision (95 % case-bootstrap CI; chance level in brackets)

| config / cut | N | ft: within | ft: pooled | emb: within | emb: pooled | chance within | chance pooled |
|---|---|---|---|---|---|---|---|
| realistic/all | 0 | 0.297 [0.278, 0.316] | 0.297 [0.278, 0.316] | 0.297 [0.278, 0.316] | 0.297 [0.278, 0.316] | 0.297 | 0.297 |
| realistic/all | 8 | 0.613 [0.578, 0.646] | 0.486 [0.449, 0.528] | 0.618 [0.583, 0.652] | 0.469 [0.433, 0.506] | 0.297 | 0.297 |
| realistic/all | 16 | 0.665 [0.631, 0.697] | 0.594 [0.554, 0.632] | 0.658 [0.625, 0.690] | 0.553 [0.516, 0.592] | 0.297 | 0.297 |
| realistic/all | 32 | 0.775 [0.746, 0.803] | 0.722 [0.687, 0.753] | 0.774 [0.744, 0.803] | 0.692 [0.658, 0.725] | 0.305 | 0.305 |
| realistic/all | 64 | 0.861 [0.837, 0.885] | 0.817 [0.791, 0.842] | 0.837 [0.811, 0.864] | 0.788 [0.755, 0.818] | 0.339 | 0.337 |
| realistic/all | 128 | 0.909 [0.886, 0.931] | 0.893 [0.870, 0.914] | 0.880 [0.855, 0.904] | 0.843 [0.814, 0.869] | 0.411 | 0.410 |
| realistic/reach64 | 0 | 0.339 [0.319, 0.359] | 0.337 [0.316, 0.357] | 0.339 [0.319, 0.359] | 0.337 [0.316, 0.357] | 0.339 | 0.337 |
| realistic/reach64 | 8 | 0.646 [0.613, 0.678] | 0.506 [0.468, 0.546] | 0.649 [0.614, 0.683] | 0.500 [0.462, 0.537] | 0.339 | 0.337 |
| realistic/reach64 | 16 | 0.693 [0.661, 0.725] | 0.610 [0.571, 0.648] | 0.695 [0.662, 0.725] | 0.569 [0.532, 0.606] | 0.339 | 0.337 |
| realistic/reach64 | 32 | 0.793 [0.766, 0.820] | 0.727 [0.692, 0.760] | 0.788 [0.759, 0.815] | 0.699 [0.663, 0.732] | 0.339 | 0.337 |
| realistic/reach64 | 64 | 0.861 [0.834, 0.885] | 0.817 [0.790, 0.841] | 0.837 [0.810, 0.864] | 0.788 [0.756, 0.818] | 0.339 | 0.337 |
| realistic/reach64 | 128 | 0.909 [0.887, 0.930] | 0.893 [0.870, 0.913] | 0.880 [0.854, 0.904] | 0.843 [0.815, 0.869] | 0.411 | 0.410 |
| smoke/all | 0 | 0.340 [0.321, 0.360] | 0.340 [0.321, 0.360] | 0.340 [0.321, 0.360] | 0.340 [0.321, 0.360] | 0.340 | 0.340 |
| smoke/all | 8 | 0.612 [0.580, 0.643] | 0.492 [0.459, 0.529] | 0.574 [0.542, 0.604] | 0.427 [0.395, 0.462] | 0.340 | 0.340 |
| smoke/all | 16 | 0.647 [0.615, 0.679] | 0.569 [0.535, 0.602] | 0.644 [0.614, 0.675] | 0.534 [0.500, 0.569] | 0.340 | 0.340 |
| smoke/all | 32 | 0.722 [0.691, 0.751] | 0.661 [0.627, 0.696] | 0.670 [0.641, 0.701] | 0.580 [0.548, 0.617] | 0.342 | 0.342 |
| smoke/all | 64 | 0.807 [0.780, 0.835] | 0.782 [0.754, 0.808] | 0.766 [0.737, 0.795] | 0.705 [0.672, 0.740] | 0.349 | 0.348 |
| smoke/all | 128 | 0.877 [0.852, 0.901] | 0.854 [0.828, 0.877] | 0.844 [0.818, 0.869] | 0.793 [0.764, 0.822] | 0.391 | 0.385 |
| smoke/reach64 | 0 | 0.349 [0.329, 0.370] | 0.348 [0.328, 0.369] | 0.349 [0.329, 0.370] | 0.348 [0.328, 0.369] | 0.349 | 0.348 |
| smoke/reach64 | 8 | 0.617 [0.585, 0.648] | 0.493 [0.458, 0.529] | 0.586 [0.556, 0.619] | 0.435 [0.404, 0.472] | 0.349 | 0.348 |
| smoke/reach64 | 16 | 0.651 [0.621, 0.682] | 0.568 [0.535, 0.603] | 0.649 [0.617, 0.681] | 0.536 [0.502, 0.571] | 0.349 | 0.348 |
| smoke/reach64 | 32 | 0.723 [0.694, 0.752] | 0.659 [0.624, 0.694] | 0.674 [0.645, 0.704] | 0.583 [0.550, 0.618] | 0.349 | 0.348 |
| smoke/reach64 | 64 | 0.807 [0.780, 0.834] | 0.782 [0.756, 0.808] | 0.766 [0.735, 0.795] | 0.705 [0.673, 0.738] | 0.349 | 0.348 |
| smoke/reach64 | 128 | 0.877 [0.852, 0.900] | 0.854 [0.829, 0.878] | 0.844 [0.818, 0.868] | 0.793 [0.766, 0.820] | 0.391 | 0.385 |

## Paired difference vs emb (average precision)

| config / cut | N | method | metric | Δ | 95 % CI | CI excludes 0 |
|---|---|---|---|---|---|---|
| realistic/all | 0 | ft | ap_cont | +0.000 | [+0.000, +0.000] | False |
| realistic/all | 0 | ft | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| realistic/all | 8 | ft | ap_cont | +0.017 | [-0.008, +0.045] | False |
| realistic/all | 8 | ft | ap_within_case_mean | -0.006 | [-0.032, +0.021] | False |
| realistic/all | 16 | ft | ap_cont | +0.041 | [+0.019, +0.061] | True |
| realistic/all | 16 | ft | ap_within_case_mean | +0.007 | [-0.013, +0.027] | False |
| realistic/all | 32 | ft | ap_cont | +0.029 | [+0.011, +0.048] | True |
| realistic/all | 32 | ft | ap_within_case_mean | +0.001 | [-0.023, +0.024] | False |
| realistic/all | 64 | ft | ap_cont | +0.029 | [+0.012, +0.048] | True |
| realistic/all | 64 | ft | ap_within_case_mean | +0.024 | [+0.006, +0.042] | True |
| realistic/all | 128 | ft | ap_cont | +0.050 | [+0.034, +0.065] | True |
| realistic/all | 128 | ft | ap_within_case_mean | +0.029 | [+0.014, +0.046] | True |
| realistic/reach64 | 0 | ft | ap_cont | +0.000 | [+0.000, +0.000] | False |
| realistic/reach64 | 0 | ft | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| realistic/reach64 | 8 | ft | ap_cont | +0.006 | [-0.019, +0.032] | False |
| realistic/reach64 | 8 | ft | ap_within_case_mean | -0.003 | [-0.030, +0.024] | False |
| realistic/reach64 | 16 | ft | ap_cont | +0.042 | [+0.020, +0.062] | True |
| realistic/reach64 | 16 | ft | ap_within_case_mean | -0.002 | [-0.022, +0.020] | False |
| realistic/reach64 | 32 | ft | ap_cont | +0.028 | [+0.009, +0.046] | True |
| realistic/reach64 | 32 | ft | ap_within_case_mean | +0.005 | [-0.017, +0.028] | False |
| realistic/reach64 | 64 | ft | ap_cont | +0.029 | [+0.011, +0.048] | True |
| realistic/reach64 | 64 | ft | ap_within_case_mean | +0.024 | [+0.006, +0.041] | True |
| realistic/reach64 | 128 | ft | ap_cont | +0.050 | [+0.034, +0.065] | True |
| realistic/reach64 | 128 | ft | ap_within_case_mean | +0.029 | [+0.013, +0.045] | True |
| smoke/all | 0 | ft | ap_cont | +0.000 | [+0.000, +0.000] | False |
| smoke/all | 0 | ft | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| smoke/all | 8 | ft | ap_cont | +0.065 | [+0.041, +0.091] | True |
| smoke/all | 8 | ft | ap_within_case_mean | +0.038 | [+0.015, +0.062] | True |
| smoke/all | 16 | ft | ap_cont | +0.035 | [+0.008, +0.061] | True |
| smoke/all | 16 | ft | ap_within_case_mean | +0.002 | [-0.022, +0.028] | False |
| smoke/all | 32 | ft | ap_cont | +0.080 | [+0.049, +0.110] | True |
| smoke/all | 32 | ft | ap_within_case_mean | +0.051 | [+0.029, +0.076] | True |
| smoke/all | 64 | ft | ap_cont | +0.077 | [+0.054, +0.101] | True |
| smoke/all | 64 | ft | ap_within_case_mean | +0.042 | [+0.016, +0.065] | True |
| smoke/all | 128 | ft | ap_cont | +0.060 | [+0.042, +0.079] | True |
| smoke/all | 128 | ft | ap_within_case_mean | +0.034 | [+0.015, +0.054] | True |
| smoke/reach64 | 0 | ft | ap_cont | +0.000 | [+0.000, +0.000] | False |
| smoke/reach64 | 0 | ft | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| smoke/reach64 | 8 | ft | ap_cont | +0.058 | [+0.033, +0.083] | True |
| smoke/reach64 | 8 | ft | ap_within_case_mean | +0.031 | [+0.007, +0.056] | True |
| smoke/reach64 | 16 | ft | ap_cont | +0.032 | [+0.005, +0.059] | True |
| smoke/reach64 | 16 | ft | ap_within_case_mean | +0.002 | [-0.021, +0.025] | False |
| smoke/reach64 | 32 | ft | ap_cont | +0.077 | [+0.046, +0.105] | True |
| smoke/reach64 | 32 | ft | ap_within_case_mean | +0.049 | [+0.027, +0.071] | True |
| smoke/reach64 | 64 | ft | ap_cont | +0.077 | [+0.053, +0.100] | True |
| smoke/reach64 | 64 | ft | ap_within_case_mean | +0.042 | [+0.017, +0.065] | True |
| smoke/reach64 | 128 | ft | ap_cont | +0.060 | [+0.042, +0.079] | True |
| smoke/reach64 | 128 | ft | ap_within_case_mean | +0.034 | [+0.015, +0.053] | True |

Pair-weighted `auroc_within` and across-row `auroc_cont` are in metrics.csv (secondary). Scores of emb, ft were rank-mapped into (0, 1): their Brier columns mean nothing.

## Coverage

| config | N | rows | violating_share | mixed_cases | scored_ft | scored_emb |
|---|---|---|---|---|---|---|
| smoke | 0 | 2392 | 0.340 | 299 | 2392 | 2392 |
| smoke | 8 | 2392 | 0.340 | 299 | 2392 | 2392 |
| smoke | 16 | 2392 | 0.340 | 299 | 2392 | 2392 |
| smoke | 32 | 2381 | 0.342 | 299 | 2381 | 2381 |
| smoke | 64 | 2315 | 0.348 | 297 | 2315 | 2315 |
| smoke | 128 | 1929 | 0.385 | 290 | 1929 | 1929 |
| realistic | 0 | 2392 | 0.297 | 299 | 2392 | 2392 |
| realistic | 8 | 2392 | 0.297 | 299 | 2392 | 2392 |
| realistic | 16 | 2392 | 0.297 | 299 | 2392 | 2392 |
| realistic | 32 | 2327 | 0.305 | 299 | 2327 | 2327 |
| realistic | 64 | 2060 | 0.337 | 293 | 2060 | 2060 |
| realistic | 128 | 1477 | 0.410 | 260 | 1477 | 1477 |
