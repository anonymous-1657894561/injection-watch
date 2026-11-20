# InjecForks v1.2 · test · e79-final-test-vs-monitor

Methods: probe, detect, length, monitor, selfreport · baseline monitor

## Case-mean within-case AUROC (95 % case-bootstrap CI)

| config / cut | N | probe | detect | length | monitor | selfreport | rows | mixed cases |
|---|---|---|---|---|---|---|---|---|
| realistic/all | 0 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 2392 | 299 |
| realistic/all | 8 | 0.692 [0.661, 0.719] | 0.457 [0.425, 0.490] | 0.671 [0.640, 0.700] | 0.555 [0.522, 0.587] | 0.599 [0.566, 0.632] | 2392 | 299 |
| realistic/all | 16 | 0.752 [0.726, 0.777] ← | 0.543 [0.511, 0.575] ← | 0.679 [0.649, 0.708] ← | 0.692 [0.662, 0.720] ← | 0.611 [0.578, 0.641] ← | 2392 | 299 |
| realistic/all | 32 | 0.808 [0.784, 0.832] | 0.565 [0.533, 0.596] | 0.700 [0.670, 0.728] | 0.801 [0.775, 0.826] | 0.698 [0.666, 0.727] | 2327 | 299 |
| realistic/all | 64 | 0.846 [0.822, 0.870] | 0.578 [0.544, 0.614] | 0.733 [0.703, 0.763] | 0.867 [0.843, 0.888] | 0.801 [0.773, 0.829] | 2059 | 293 |
| realistic/all | 128 | 0.831 [0.800, 0.861] | 0.536 [0.497, 0.575] | 0.654 [0.611, 0.695] | 0.903 [0.880, 0.923] | 0.845 [0.815, 0.873] | 1477 | 260 |
| realistic/reach64 | 0 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 2060 | 293 |
| realistic/reach64 | 8 | 0.688 [0.655, 0.719] | 0.449 [0.417, 0.483] | 0.664 [0.633, 0.694] | 0.554 [0.521, 0.587] | 0.602 [0.570, 0.636] | 2060 | 293 |
| realistic/reach64 | 16 | 0.743 [0.715, 0.770] | 0.528 [0.496, 0.560] | 0.649 [0.617, 0.680] | 0.667 [0.633, 0.699] | 0.607 [0.572, 0.640] | 2060 | 293 |
| realistic/reach64 | 32 | 0.801 [0.774, 0.826] | 0.548 [0.515, 0.584] | 0.679 [0.646, 0.708] | 0.781 [0.752, 0.810] | 0.702 [0.668, 0.734] | 2060 | 293 |
| realistic/reach64 | 64 | 0.846 [0.822, 0.869] | 0.578 [0.545, 0.614] | 0.733 [0.703, 0.760] | 0.867 [0.844, 0.888] | 0.801 [0.773, 0.829] | 2059 | 293 |
| realistic/reach64 | 128 | 0.831 [0.799, 0.859] | 0.536 [0.497, 0.575] | 0.654 [0.612, 0.698] | 0.903 [0.881, 0.923] | 0.845 [0.815, 0.873] | 1477 | 260 |

## Paired difference vs monitor (case-mean within-case AUROC)

| config / cut | N | method | Δ | 95 % CI | CI excludes 0 | Δ ≥ 0.05 |
|---|---|---|---|---|---|---|
| realistic/all | 0 | probe | +0.000 | [+0.000, +0.000] | False | False |
| realistic/all | 0 | detect | +0.000 | [+0.000, +0.000] | False | False |
| realistic/all | 0 | length | +0.000 | [+0.000, +0.000] | False | False |
| realistic/all | 0 | selfreport | +0.000 | [+0.000, +0.000] | False | False |
| realistic/all | 8 | probe | +0.137 | [+0.091, +0.182] | True | True |
| realistic/all | 8 | detect | -0.098 | [-0.141, -0.056] | True | False |
| realistic/all | 8 | length | +0.116 | [+0.069, +0.163] | True | True |
| realistic/all | 8 | selfreport | +0.044 | [-0.003, +0.091] | False | False |
| realistic/all | 16 | probe | +0.060 | [+0.029, +0.093] | True | True |
| realistic/all | 16 | detect | -0.149 | [-0.190, -0.106] | True | False |
| realistic/all | 16 | length | -0.013 | [-0.047, +0.022] | False | False |
| realistic/all | 16 | selfreport | -0.081 | [-0.119, -0.042] | True | False |
| realistic/all | 32 | probe | +0.007 | [-0.020, +0.035] | False | False |
| realistic/all | 32 | detect | -0.236 | [-0.274, -0.199] | True | False |
| realistic/all | 32 | length | -0.101 | [-0.136, -0.067] | True | False |
| realistic/all | 32 | selfreport | -0.103 | [-0.137, -0.071] | True | False |
| realistic/all | 64 | probe | -0.021 | [-0.044, +0.003] | False | False |
| realistic/all | 64 | detect | -0.288 | [-0.328, -0.247] | True | False |
| realistic/all | 64 | length | -0.133 | [-0.166, -0.100] | True | False |
| realistic/all | 64 | selfreport | -0.065 | [-0.092, -0.038] | True | False |
| realistic/all | 128 | probe | -0.072 | [-0.098, -0.046] | True | False |
| realistic/all | 128 | detect | -0.367 | [-0.408, -0.325] | True | False |
| realistic/all | 128 | length | -0.249 | [-0.292, -0.206] | True | False |
| realistic/all | 128 | selfreport | -0.058 | [-0.088, -0.030] | True | False |
| realistic/reach64 | 0 | probe | +0.000 | [+0.000, +0.000] | False | False |
| realistic/reach64 | 0 | detect | +0.000 | [+0.000, +0.000] | False | False |
| realistic/reach64 | 0 | length | +0.000 | [+0.000, +0.000] | False | False |
| realistic/reach64 | 0 | selfreport | +0.000 | [+0.000, +0.000] | False | False |
| realistic/reach64 | 8 | probe | +0.134 | [+0.088, +0.181] | True | True |
| realistic/reach64 | 8 | detect | -0.105 | [-0.148, -0.060] | True | False |
| realistic/reach64 | 8 | length | +0.110 | [+0.063, +0.159] | True | True |
| realistic/reach64 | 8 | selfreport | +0.048 | [-0.001, +0.096] | False | False |
| realistic/reach64 | 16 | probe | +0.076 | [+0.042, +0.113] | True | True |
| realistic/reach64 | 16 | detect | -0.139 | [-0.185, -0.093] | True | False |
| realistic/reach64 | 16 | length | -0.019 | [-0.058, +0.021] | False | False |
| realistic/reach64 | 16 | selfreport | -0.061 | [-0.102, -0.018] | True | False |
| realistic/reach64 | 32 | probe | +0.019 | [-0.011, +0.049] | False | False |
| realistic/reach64 | 32 | detect | -0.233 | [-0.275, -0.191] | True | False |
| realistic/reach64 | 32 | length | -0.103 | [-0.140, -0.065] | True | False |
| realistic/reach64 | 32 | selfreport | -0.080 | [-0.117, -0.044] | True | False |
| realistic/reach64 | 64 | probe | -0.021 | [-0.043, +0.002] | False | False |
| realistic/reach64 | 64 | detect | -0.288 | [-0.329, -0.246] | True | False |
| realistic/reach64 | 64 | length | -0.133 | [-0.168, -0.102] | True | False |
| realistic/reach64 | 64 | selfreport | -0.065 | [-0.092, -0.038] | True | False |
| realistic/reach64 | 128 | probe | -0.072 | [-0.099, -0.047] | True | False |
| realistic/reach64 | 128 | detect | -0.367 | [-0.410, -0.324] | True | False |
| realistic/reach64 | 128 | length | -0.249 | [-0.292, -0.204] | True | False |
| realistic/reach64 | 128 | selfreport | -0.058 | [-0.089, -0.028] | True | False |

## Average precision (95 % case-bootstrap CI; chance level in brackets)

| config / cut | N | probe: within | probe: pooled | detect: within | detect: pooled | length: within | length: pooled | monitor: within | monitor: pooled | selfreport: within | selfreport: pooled | chance within | chance pooled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| realistic/all | 0 | 0.297 [0.278, 0.316] | 0.455 [0.423, 0.485] | 0.297 [0.278, 0.316] | 0.327 [0.300, 0.355] | 0.297 [0.278, 0.316] | 0.420 [0.389, 0.450] | 0.297 [0.278, 0.316] | 0.293 [0.270, 0.318] | 0.297 [0.278, 0.316] | 0.320 [0.293, 0.351] | 0.297 | 0.297 |
| realistic/all | 8 | 0.634 [0.602, 0.665] | 0.540 [0.504, 0.578] | 0.439 [0.411, 0.469] | 0.292 [0.268, 0.320] | 0.610 [0.579, 0.640] | 0.477 [0.442, 0.516] | 0.523 [0.489, 0.557] | 0.417 [0.383, 0.455] | 0.578 [0.545, 0.609] | 0.332 [0.305, 0.364] | 0.297 | 0.297 |
| realistic/all | 16 | 0.705 [0.672, 0.735] | 0.620 [0.586, 0.656] | 0.495 [0.467, 0.527] | 0.312 [0.288, 0.341] | 0.625 [0.593, 0.657] | 0.475 [0.439, 0.513] | 0.635 [0.600, 0.668] | 0.554 [0.517, 0.590] | 0.575 [0.542, 0.608] | 0.349 [0.320, 0.386] | 0.297 | 0.297 |
| realistic/all | 32 | 0.777 [0.748, 0.807] | 0.711 [0.679, 0.745] | 0.525 [0.494, 0.555] | 0.342 [0.315, 0.373] | 0.655 [0.623, 0.687] | 0.504 [0.468, 0.545] | 0.772 [0.743, 0.802] | 0.703 [0.673, 0.734] | 0.679 [0.648, 0.711] | 0.435 [0.401, 0.474] | 0.305 | 0.305 |
| realistic/all | 64 | 0.835 [0.808, 0.861] | 0.789 [0.764, 0.815] | 0.589 [0.558, 0.622] | 0.367 [0.339, 0.402] | 0.720 [0.689, 0.750] | 0.535 [0.500, 0.576] | 0.839 [0.812, 0.866] | 0.803 [0.776, 0.828] | 0.786 [0.756, 0.817] | 0.617 [0.575, 0.660] | 0.340 | 0.337 |
| realistic/all | 128 | 0.849 [0.821, 0.876] | 0.809 [0.776, 0.840] | 0.611 [0.579, 0.641] | 0.414 [0.383, 0.450] | 0.716 [0.682, 0.748] | 0.579 [0.536, 0.621] | 0.893 [0.868, 0.916] | 0.885 [0.862, 0.904] | 0.855 [0.827, 0.882] | 0.729 [0.690, 0.770] | 0.411 | 0.410 |
| realistic/reach64 | 0 | 0.339 [0.319, 0.359] | 0.473 [0.442, 0.502] | 0.339 [0.319, 0.359] | 0.348 [0.320, 0.378] | 0.339 [0.319, 0.359] | 0.433 [0.402, 0.461] | 0.339 [0.319, 0.359] | 0.334 [0.311, 0.360] | 0.339 [0.319, 0.359] | 0.350 [0.321, 0.382] | 0.339 | 0.337 |
| realistic/reach64 | 8 | 0.669 [0.637, 0.699] | 0.552 [0.517, 0.587] | 0.474 [0.446, 0.501] | 0.312 [0.285, 0.341] | 0.641 [0.611, 0.670] | 0.489 [0.452, 0.526] | 0.556 [0.525, 0.586] | 0.452 [0.419, 0.491] | 0.616 [0.586, 0.647] | 0.370 [0.342, 0.402] | 0.339 | 0.337 |
| realistic/reach64 | 16 | 0.727 [0.697, 0.755] | 0.624 [0.588, 0.660] | 0.530 [0.500, 0.558] | 0.334 [0.308, 0.364] | 0.643 [0.612, 0.675] | 0.481 [0.444, 0.520] | 0.657 [0.624, 0.689] | 0.572 [0.533, 0.608] | 0.614 [0.583, 0.646] | 0.389 [0.358, 0.426] | 0.339 | 0.337 |
| realistic/reach64 | 32 | 0.790 [0.761, 0.819] | 0.715 [0.679, 0.748] | 0.550 [0.519, 0.582] | 0.362 [0.334, 0.397] | 0.667 [0.634, 0.699] | 0.509 [0.471, 0.551] | 0.780 [0.750, 0.810] | 0.707 [0.674, 0.738] | 0.706 [0.672, 0.738] | 0.465 [0.429, 0.508] | 0.339 | 0.337 |
| realistic/reach64 | 64 | 0.835 [0.808, 0.862] | 0.789 [0.764, 0.813] | 0.589 [0.557, 0.622] | 0.367 [0.338, 0.400] | 0.720 [0.689, 0.749] | 0.535 [0.497, 0.575] | 0.839 [0.811, 0.866] | 0.803 [0.775, 0.827] | 0.786 [0.755, 0.816] | 0.617 [0.578, 0.660] | 0.340 | 0.337 |
| realistic/reach64 | 128 | 0.849 [0.821, 0.875] | 0.809 [0.775, 0.837] | 0.611 [0.580, 0.641] | 0.414 [0.384, 0.450] | 0.716 [0.684, 0.751] | 0.579 [0.538, 0.623] | 0.893 [0.868, 0.916] | 0.885 [0.861, 0.905] | 0.855 [0.829, 0.882] | 0.729 [0.689, 0.768] | 0.411 | 0.410 |

## Paired difference vs monitor (average precision)

| config / cut | N | method | metric | Δ | 95 % CI | CI excludes 0 |
|---|---|---|---|---|---|---|
| realistic/all | 0 | probe | ap_cont | +0.162 | [+0.131, +0.191] | True |
| realistic/all | 0 | probe | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| realistic/all | 0 | detect | ap_cont | +0.034 | [+0.008, +0.058] | True |
| realistic/all | 0 | detect | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| realistic/all | 0 | length | ap_cont | +0.127 | [+0.096, +0.154] | True |
| realistic/all | 0 | length | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| realistic/all | 0 | selfreport | ap_cont | +0.028 | [+0.005, +0.051] | True |
| realistic/all | 0 | selfreport | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| realistic/all | 8 | probe | ap_cont | +0.122 | [+0.089, +0.155] | True |
| realistic/all | 8 | probe | ap_within_case_mean | +0.111 | [+0.073, +0.149] | True |
| realistic/all | 8 | detect | ap_cont | -0.125 | [-0.162, -0.090] | True |
| realistic/all | 8 | detect | ap_within_case_mean | -0.084 | [-0.116, -0.051] | True |
| realistic/all | 8 | length | ap_cont | +0.059 | [+0.019, +0.100] | True |
| realistic/all | 8 | length | ap_within_case_mean | +0.087 | [+0.048, +0.126] | True |
| realistic/all | 8 | selfreport | ap_cont | -0.085 | [-0.120, -0.050] | True |
| realistic/all | 8 | selfreport | ap_within_case_mean | +0.055 | [+0.015, +0.095] | True |
| realistic/all | 16 | probe | ap_cont | +0.066 | [+0.035, +0.097] | True |
| realistic/all | 16 | probe | ap_within_case_mean | +0.069 | [+0.035, +0.103] | True |
| realistic/all | 16 | detect | ap_cont | -0.242 | [-0.279, -0.204] | True |
| realistic/all | 16 | detect | ap_within_case_mean | -0.140 | [-0.179, -0.099] | True |
| realistic/all | 16 | length | ap_cont | -0.079 | [-0.117, -0.039] | True |
| realistic/all | 16 | length | ap_within_case_mean | -0.010 | [-0.044, +0.024] | False |
| realistic/all | 16 | selfreport | ap_cont | -0.205 | [-0.238, -0.169] | True |
| realistic/all | 16 | selfreport | ap_within_case_mean | -0.060 | [-0.100, -0.021] | True |
| realistic/all | 32 | probe | ap_cont | +0.008 | [-0.018, +0.034] | False |
| realistic/all | 32 | probe | ap_within_case_mean | +0.004 | [-0.024, +0.033] | False |
| realistic/all | 32 | detect | ap_cont | -0.362 | [-0.396, -0.323] | True |
| realistic/all | 32 | detect | ap_within_case_mean | -0.247 | [-0.284, -0.209] | True |
| realistic/all | 32 | length | ap_cont | -0.199 | [-0.238, -0.157] | True |
| realistic/all | 32 | length | ap_within_case_mean | -0.117 | [-0.155, -0.082] | True |
| realistic/all | 32 | selfreport | ap_cont | -0.268 | [-0.301, -0.231] | True |
| realistic/all | 32 | selfreport | ap_within_case_mean | -0.093 | [-0.129, -0.058] | True |
| realistic/all | 64 | probe | ap_cont | -0.014 | [-0.036, +0.010] | False |
| realistic/all | 64 | probe | ap_within_case_mean | -0.004 | [-0.031, +0.024] | False |
| realistic/all | 64 | detect | ap_cont | -0.436 | [-0.470, -0.397] | True |
| realistic/all | 64 | detect | ap_within_case_mean | -0.250 | [-0.289, -0.211] | True |
| realistic/all | 64 | length | ap_cont | -0.268 | [-0.306, -0.225] | True |
| realistic/all | 64 | length | ap_within_case_mean | -0.120 | [-0.156, -0.083] | True |
| realistic/all | 64 | selfreport | ap_cont | -0.186 | [-0.221, -0.148] | True |
| realistic/all | 64 | selfreport | ap_within_case_mean | -0.053 | [-0.085, -0.023] | True |
| realistic/all | 128 | probe | ap_cont | -0.076 | [-0.101, -0.054] | True |
| realistic/all | 128 | probe | ap_within_case_mean | -0.044 | [-0.069, -0.019] | True |
| realistic/all | 128 | detect | ap_cont | -0.471 | [-0.504, -0.434] | True |
| realistic/all | 128 | detect | ap_within_case_mean | -0.282 | [-0.317, -0.245] | True |
| realistic/all | 128 | length | ap_cont | -0.306 | [-0.343, -0.266] | True |
| realistic/all | 128 | length | ap_within_case_mean | -0.177 | [-0.213, -0.140] | True |
| realistic/all | 128 | selfreport | ap_cont | -0.156 | [-0.190, -0.121] | True |
| realistic/all | 128 | selfreport | ap_within_case_mean | -0.037 | [-0.065, -0.011] | True |
| realistic/reach64 | 0 | probe | ap_cont | +0.139 | [+0.107, +0.169] | True |
| realistic/reach64 | 0 | probe | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| realistic/reach64 | 0 | detect | ap_cont | +0.014 | [-0.012, +0.040] | False |
| realistic/reach64 | 0 | detect | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| realistic/reach64 | 0 | length | ap_cont | +0.098 | [+0.068, +0.127] | True |
| realistic/reach64 | 0 | length | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| realistic/reach64 | 0 | selfreport | ap_cont | +0.016 | [-0.007, +0.043] | False |
| realistic/reach64 | 0 | selfreport | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| realistic/reach64 | 8 | probe | ap_cont | +0.100 | [+0.068, +0.131] | True |
| realistic/reach64 | 8 | probe | ap_within_case_mean | +0.114 | [+0.075, +0.152] | True |
| realistic/reach64 | 8 | detect | ap_cont | -0.140 | [-0.177, -0.103] | True |
| realistic/reach64 | 8 | detect | ap_within_case_mean | -0.082 | [-0.113, -0.048] | True |
| realistic/reach64 | 8 | length | ap_cont | +0.037 | [-0.005, +0.077] | False |
| realistic/reach64 | 8 | length | ap_within_case_mean | +0.086 | [+0.048, +0.123] | True |
| realistic/reach64 | 8 | selfreport | ap_cont | -0.082 | [-0.118, -0.043] | True |
| realistic/reach64 | 8 | selfreport | ap_within_case_mean | +0.060 | [+0.022, +0.099] | True |
| realistic/reach64 | 16 | probe | ap_cont | +0.051 | [+0.022, +0.085] | True |
| realistic/reach64 | 16 | probe | ap_within_case_mean | +0.069 | [+0.037, +0.103] | True |
| realistic/reach64 | 16 | detect | ap_cont | -0.238 | [-0.274, -0.196] | True |
| realistic/reach64 | 16 | detect | ap_within_case_mean | -0.127 | [-0.168, -0.090] | True |
| realistic/reach64 | 16 | length | ap_cont | -0.092 | [-0.128, -0.051] | True |
| realistic/reach64 | 16 | length | ap_within_case_mean | -0.015 | [-0.049, +0.020] | False |
| realistic/reach64 | 16 | selfreport | ap_cont | -0.183 | [-0.218, -0.143] | True |
| realistic/reach64 | 16 | selfreport | ap_within_case_mean | -0.044 | [-0.082, -0.003] | True |
| realistic/reach64 | 32 | probe | ap_cont | +0.008 | [-0.018, +0.035] | False |
| realistic/reach64 | 32 | probe | ap_within_case_mean | +0.010 | [-0.018, +0.039] | False |
| realistic/reach64 | 32 | detect | ap_cont | -0.345 | [-0.381, -0.304] | True |
| realistic/reach64 | 32 | detect | ap_within_case_mean | -0.231 | [-0.268, -0.191] | True |
| realistic/reach64 | 32 | length | ap_cont | -0.198 | [-0.238, -0.154] | True |
| realistic/reach64 | 32 | length | ap_within_case_mean | -0.113 | [-0.148, -0.078] | True |
| realistic/reach64 | 32 | selfreport | ap_cont | -0.242 | [-0.278, -0.204] | True |
| realistic/reach64 | 32 | selfreport | ap_within_case_mean | -0.074 | [-0.108, -0.039] | True |
| realistic/reach64 | 64 | probe | ap_cont | -0.014 | [-0.035, +0.008] | False |
| realistic/reach64 | 64 | probe | ap_within_case_mean | -0.004 | [-0.031, +0.023] | False |
| realistic/reach64 | 64 | detect | ap_cont | -0.436 | [-0.469, -0.399] | True |
| realistic/reach64 | 64 | detect | ap_within_case_mean | -0.250 | [-0.289, -0.211] | True |
| realistic/reach64 | 64 | length | ap_cont | -0.268 | [-0.308, -0.225] | True |
| realistic/reach64 | 64 | length | ap_within_case_mean | -0.120 | [-0.158, -0.084] | True |
| realistic/reach64 | 64 | selfreport | ap_cont | -0.186 | [-0.222, -0.147] | True |
| realistic/reach64 | 64 | selfreport | ap_within_case_mean | -0.053 | [-0.083, -0.022] | True |
| realistic/reach64 | 128 | probe | ap_cont | -0.076 | [-0.100, -0.053] | True |
| realistic/reach64 | 128 | probe | ap_within_case_mean | -0.044 | [-0.069, -0.019] | True |
| realistic/reach64 | 128 | detect | ap_cont | -0.471 | [-0.501, -0.434] | True |
| realistic/reach64 | 128 | detect | ap_within_case_mean | -0.282 | [-0.317, -0.246] | True |
| realistic/reach64 | 128 | length | ap_cont | -0.306 | [-0.343, -0.264] | True |
| realistic/reach64 | 128 | length | ap_within_case_mean | -0.177 | [-0.212, -0.139] | True |
| realistic/reach64 | 128 | selfreport | ap_cont | -0.156 | [-0.191, -0.120] | True |
| realistic/reach64 | 128 | selfreport | ap_within_case_mean | -0.037 | [-0.064, -0.010] | True |

Pair-weighted `auroc_within` and across-row `auroc_cont` are in metrics.csv (secondary). Scores of detect, length, probe were rank-mapped into (0, 1): their Brier columns mean nothing.

## Coverage

| config | N | rows | violating_share | mixed_cases | scored_probe | scored_detect | scored_length | scored_monitor | scored_selfreport |
|---|---|---|---|---|---|---|---|---|---|
| smoke | 0 | 2392 | 0.340 | 299 | 2392 | 2392 | 2392 | 0 | 0 |
| smoke | 8 | 2392 | 0.340 | 299 | 2392 | 2392 | 2392 | 0 | 0 |
| smoke | 16 | 2392 | 0.340 | 299 | 2392 | 2392 | 2392 | 0 | 0 |
| smoke | 32 | 2381 | 0.342 | 299 | 2381 | 2381 | 2381 | 0 | 0 |
| smoke | 64 | 2315 | 0.348 | 297 | 2315 | 2315 | 2315 | 0 | 0 |
| smoke | 128 | 1929 | 0.385 | 290 | 1929 | 1929 | 1929 | 0 | 0 |
| realistic | 0 | 2392 | 0.297 | 299 | 2392 | 2392 | 2392 | 2392 | 2392 |
| realistic | 8 | 2392 | 0.297 | 299 | 2392 | 2392 | 2392 | 2392 | 2392 |
| realistic | 16 | 2392 | 0.297 | 299 | 2392 | 2392 | 2392 | 2392 | 2392 |
| realistic | 32 | 2327 | 0.305 | 299 | 2327 | 2327 | 2327 | 2327 | 2327 |
| realistic | 64 | 2060 | 0.337 | 293 | 2060 | 2060 | 2060 | 2059 | 2060 |
| realistic | 128 | 1477 | 0.410 | 260 | 1477 | 1477 | 1477 | 1477 | 1477 |
