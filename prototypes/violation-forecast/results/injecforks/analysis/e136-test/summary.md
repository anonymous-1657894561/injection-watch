# InjecForks v1.2 · test · e136-test

Methods: ft, probe, emb, comb, length · baseline probe

## Case-mean within-case AUROC (95 % case-bootstrap CI)

| config / cut | N | ft | probe | emb | comb | length | rows | mixed cases |
|---|---|---|---|---|---|---|---|---|
| realistic/all | 0 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 2392 | 299 |
| realistic/all | 8 | 0.647 [0.614, 0.678] | 0.692 [0.661, 0.719] | 0.641 [0.608, 0.672] | 0.680 [0.648, 0.712] | 0.671 [0.640, 0.700] | 2392 | 299 |
| realistic/all | 16 | 0.709 [0.678, 0.740] ← | 0.752 [0.726, 0.777] ← | 0.700 [0.669, 0.729] ← | 0.740 [0.712, 0.767] ← | 0.679 [0.649, 0.708] ← | 2392 | 299 |
| realistic/all | 32 | 0.810 [0.785, 0.836] | 0.808 [0.784, 0.832] | 0.804 [0.777, 0.828] | 0.833 [0.811, 0.855] | 0.700 [0.670, 0.728] | 2327 | 299 |
| realistic/all | 64 | 0.875 [0.852, 0.897] | 0.846 [0.822, 0.869] | 0.846 [0.820, 0.871] | 0.888 [0.866, 0.909] | 0.733 [0.702, 0.762] | 2060 | 293 |
| realistic/all | 128 | 0.899 [0.872, 0.923] | 0.831 [0.800, 0.861] | 0.873 [0.847, 0.899] | 0.906 [0.879, 0.929] | 0.654 [0.611, 0.695] | 1477 | 260 |
| realistic/reach64 | 0 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 2060 | 293 |
| realistic/reach64 | 8 | 0.646 [0.611, 0.680] | 0.688 [0.655, 0.719] | 0.634 [0.599, 0.671] | 0.677 [0.641, 0.710] | 0.664 [0.633, 0.694] | 2060 | 293 |
| realistic/reach64 | 16 | 0.700 [0.666, 0.733] | 0.743 [0.715, 0.770] | 0.700 [0.669, 0.730] | 0.737 [0.707, 0.765] | 0.649 [0.617, 0.680] | 2060 | 293 |
| realistic/reach64 | 32 | 0.805 [0.778, 0.829] | 0.801 [0.774, 0.826] | 0.793 [0.765, 0.819] | 0.824 [0.799, 0.847] | 0.679 [0.646, 0.708] | 2060 | 293 |
| realistic/reach64 | 64 | 0.875 [0.851, 0.896] | 0.846 [0.821, 0.869] | 0.846 [0.820, 0.870] | 0.888 [0.867, 0.909] | 0.733 [0.702, 0.760] | 2060 | 293 |
| realistic/reach64 | 128 | 0.899 [0.875, 0.923] | 0.831 [0.799, 0.859] | 0.873 [0.848, 0.899] | 0.906 [0.882, 0.928] | 0.654 [0.612, 0.698] | 1477 | 260 |
| smoke/all | 0 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 2392 | 299 |
| smoke/all | 8 | 0.629 [0.598, 0.659] | 0.639 [0.608, 0.671] | 0.589 [0.558, 0.620] | 0.648 [0.616, 0.678] | 0.605 [0.576, 0.634] | 2392 | 299 |
| smoke/all | 16 | 0.669 [0.639, 0.696] ← | 0.696 [0.670, 0.722] ← | 0.657 [0.629, 0.686] ← | 0.705 [0.679, 0.732] ← | 0.644 [0.617, 0.671] ← | 2392 | 299 |
| smoke/all | 32 | 0.762 [0.737, 0.787] | 0.746 [0.716, 0.772] | 0.696 [0.668, 0.721] | 0.779 [0.756, 0.802] | 0.674 [0.644, 0.701] | 2381 | 299 |
| smoke/all | 64 | 0.831 [0.807, 0.854] | 0.803 [0.778, 0.827] | 0.795 [0.768, 0.820] | 0.847 [0.826, 0.868] | 0.680 [0.651, 0.707] | 2315 | 297 |
| smoke/all | 128 | 0.879 [0.855, 0.902] | 0.797 [0.768, 0.825] | 0.851 [0.828, 0.874] | 0.875 [0.850, 0.899] | 0.642 [0.610, 0.676] | 1929 | 290 |
| smoke/reach64 | 0 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 2315 | 297 |
| smoke/reach64 | 8 | 0.622 [0.590, 0.652] | 0.639 [0.607, 0.668] | 0.593 [0.563, 0.624] | 0.648 [0.617, 0.676] | 0.597 [0.568, 0.626] | 2315 | 297 |
| smoke/reach64 | 16 | 0.664 [0.635, 0.692] | 0.691 [0.663, 0.718] | 0.653 [0.623, 0.682] | 0.699 [0.671, 0.725] | 0.634 [0.607, 0.661] | 2315 | 297 |
| smoke/reach64 | 32 | 0.756 [0.730, 0.781] | 0.741 [0.714, 0.768] | 0.694 [0.667, 0.721] | 0.774 [0.750, 0.797] | 0.668 [0.637, 0.696] | 2315 | 297 |
| smoke/reach64 | 64 | 0.831 [0.807, 0.854] | 0.803 [0.778, 0.827] | 0.795 [0.768, 0.820] | 0.847 [0.826, 0.868] | 0.680 [0.651, 0.708] | 2315 | 297 |
| smoke/reach64 | 128 | 0.879 [0.857, 0.900] | 0.797 [0.769, 0.823] | 0.851 [0.829, 0.873] | 0.875 [0.852, 0.897] | 0.642 [0.608, 0.675] | 1929 | 290 |

## Paired difference vs probe (case-mean within-case AUROC)

| config / cut | N | method | Δ | 95 % CI | CI excludes 0 | Δ ≥ 0.05 |
|---|---|---|---|---|---|---|
| realistic/all | 0 | ft | +0.000 | [+0.000, +0.000] | False | False |
| realistic/all | 0 | emb | +0.000 | [+0.000, +0.000] | False | False |
| realistic/all | 0 | comb | +0.000 | [+0.000, +0.000] | False | False |
| realistic/all | 0 | length | +0.000 | [+0.000, +0.000] | False | False |
| realistic/all | 8 | ft | -0.044 | [-0.075, -0.015] | True | False |
| realistic/all | 8 | emb | -0.051 | [-0.088, -0.014] | True | False |
| realistic/all | 8 | comb | -0.012 | [-0.030, +0.007] | False | False |
| realistic/all | 8 | length | -0.020 | [-0.046, +0.005] | False | False |
| realistic/all | 16 | ft | -0.042 | [-0.072, -0.012] | True | False |
| realistic/all | 16 | emb | -0.052 | [-0.081, -0.023] | True | False |
| realistic/all | 16 | comb | -0.012 | [-0.034, +0.010] | False | False |
| realistic/all | 16 | length | -0.073 | [-0.099, -0.045] | True | False |
| realistic/all | 32 | ft | +0.002 | [-0.024, +0.028] | False | False |
| realistic/all | 32 | emb | -0.004 | [-0.034, +0.024] | False | False |
| realistic/all | 32 | comb | +0.025 | [+0.009, +0.042] | True | False |
| realistic/all | 32 | length | -0.108 | [-0.136, -0.082] | True | False |
| realistic/all | 64 | ft | +0.029 | [+0.011, +0.048] | True | False |
| realistic/all | 64 | emb | +0.000 | [-0.021, +0.020] | False | False |
| realistic/all | 64 | comb | +0.042 | [+0.027, +0.059] | True | False |
| realistic/all | 64 | length | -0.113 | [-0.142, -0.086] | True | False |
| realistic/all | 128 | ft | +0.068 | [+0.042, +0.096] | True | True |
| realistic/all | 128 | emb | +0.042 | [+0.013, +0.071] | True | False |
| realistic/all | 128 | comb | +0.075 | [+0.051, +0.099] | True | True |
| realistic/all | 128 | length | -0.177 | [-0.222, -0.133] | True | False |
| realistic/reach64 | 0 | ft | +0.000 | [+0.000, +0.000] | False | False |
| realistic/reach64 | 0 | emb | +0.000 | [+0.000, +0.000] | False | False |
| realistic/reach64 | 0 | comb | +0.000 | [+0.000, +0.000] | False | False |
| realistic/reach64 | 0 | length | +0.000 | [+0.000, +0.000] | False | False |
| realistic/reach64 | 8 | ft | -0.042 | [-0.073, -0.011] | True | False |
| realistic/reach64 | 8 | emb | -0.054 | [-0.093, -0.017] | True | False |
| realistic/reach64 | 8 | comb | -0.011 | [-0.032, +0.009] | False | False |
| realistic/reach64 | 8 | length | -0.024 | [-0.052, +0.005] | False | False |
| realistic/reach64 | 16 | ft | -0.043 | [-0.076, -0.012] | True | False |
| realistic/reach64 | 16 | emb | -0.043 | [-0.076, -0.013] | True | False |
| realistic/reach64 | 16 | comb | -0.006 | [-0.030, +0.016] | False | False |
| realistic/reach64 | 16 | length | -0.094 | [-0.125, -0.066] | True | False |
| realistic/reach64 | 32 | ft | +0.004 | [-0.023, +0.031] | False | False |
| realistic/reach64 | 32 | emb | -0.007 | [-0.039, +0.021] | False | False |
| realistic/reach64 | 32 | comb | +0.023 | [+0.003, +0.042] | True | False |
| realistic/reach64 | 32 | length | -0.122 | [-0.150, -0.095] | True | False |
| realistic/reach64 | 64 | ft | +0.029 | [+0.010, +0.049] | True | False |
| realistic/reach64 | 64 | emb | +0.000 | [-0.021, +0.021] | False | False |
| realistic/reach64 | 64 | comb | +0.042 | [+0.026, +0.059] | True | False |
| realistic/reach64 | 64 | length | -0.113 | [-0.141, -0.085] | True | False |
| realistic/reach64 | 128 | ft | +0.068 | [+0.041, +0.097] | True | True |
| realistic/reach64 | 128 | emb | +0.042 | [+0.012, +0.072] | True | False |
| realistic/reach64 | 128 | comb | +0.075 | [+0.052, +0.099] | True | True |
| realistic/reach64 | 128 | length | -0.177 | [-0.221, -0.133] | True | False |
| smoke/all | 0 | ft | +0.000 | [+0.000, +0.000] | False | False |
| smoke/all | 0 | emb | +0.000 | [+0.000, +0.000] | False | False |
| smoke/all | 0 | comb | +0.000 | [+0.000, +0.000] | False | False |
| smoke/all | 0 | length | +0.000 | [+0.000, +0.000] | False | False |
| smoke/all | 8 | ft | -0.010 | [-0.042, +0.021] | False | False |
| smoke/all | 8 | emb | -0.050 | [-0.085, -0.014] | True | False |
| smoke/all | 8 | comb | +0.009 | [-0.004, +0.022] | False | False |
| smoke/all | 8 | length | -0.034 | [-0.061, -0.007] | True | False |
| smoke/all | 16 | ft | -0.027 | [-0.056, -0.001] | True | False |
| smoke/all | 16 | emb | -0.039 | [-0.068, -0.009] | True | False |
| smoke/all | 16 | comb | +0.010 | [-0.011, +0.029] | False | False |
| smoke/all | 16 | length | -0.052 | [-0.077, -0.028] | True | False |
| smoke/all | 32 | ft | +0.016 | [-0.010, +0.043] | False | False |
| smoke/all | 32 | emb | -0.050 | [-0.084, -0.016] | True | False |
| smoke/all | 32 | comb | +0.033 | [+0.014, +0.051] | True | False |
| smoke/all | 32 | length | -0.072 | [-0.098, -0.047] | True | False |
| smoke/all | 64 | ft | +0.028 | [-0.000, +0.054] | False | False |
| smoke/all | 64 | emb | -0.008 | [-0.035, +0.019] | False | False |
| smoke/all | 64 | comb | +0.045 | [+0.029, +0.060] | True | False |
| smoke/all | 64 | length | -0.123 | [-0.151, -0.096] | True | False |
| smoke/all | 128 | ft | +0.082 | [+0.058, +0.107] | True | True |
| smoke/all | 128 | emb | +0.054 | [+0.025, +0.083] | True | True |
| smoke/all | 128 | comb | +0.078 | [+0.059, +0.099] | True | True |
| smoke/all | 128 | length | -0.155 | [-0.189, -0.120] | True | False |
| smoke/reach64 | 0 | ft | +0.000 | [+0.000, +0.000] | False | False |
| smoke/reach64 | 0 | emb | +0.000 | [+0.000, +0.000] | False | False |
| smoke/reach64 | 0 | comb | +0.000 | [+0.000, +0.000] | False | False |
| smoke/reach64 | 0 | length | +0.000 | [+0.000, +0.000] | False | False |
| smoke/reach64 | 8 | ft | -0.017 | [-0.048, +0.013] | False | False |
| smoke/reach64 | 8 | emb | -0.046 | [-0.079, -0.011] | True | False |
| smoke/reach64 | 8 | comb | +0.009 | [-0.004, +0.022] | False | False |
| smoke/reach64 | 8 | length | -0.042 | [-0.069, -0.015] | True | False |
| smoke/reach64 | 16 | ft | -0.027 | [-0.054, +0.002] | False | False |
| smoke/reach64 | 16 | emb | -0.038 | [-0.067, -0.009] | True | False |
| smoke/reach64 | 16 | comb | +0.007 | [-0.013, +0.028] | False | False |
| smoke/reach64 | 16 | length | -0.057 | [-0.081, -0.032] | True | False |
| smoke/reach64 | 32 | ft | +0.014 | [-0.014, +0.042] | False | False |
| smoke/reach64 | 32 | emb | -0.047 | [-0.080, -0.012] | True | False |
| smoke/reach64 | 32 | comb | +0.033 | [+0.014, +0.052] | True | False |
| smoke/reach64 | 32 | length | -0.074 | [-0.098, -0.049] | True | False |
| smoke/reach64 | 64 | ft | +0.028 | [+0.002, +0.055] | True | False |
| smoke/reach64 | 64 | emb | -0.008 | [-0.035, +0.021] | False | False |
| smoke/reach64 | 64 | comb | +0.045 | [+0.030, +0.060] | True | False |
| smoke/reach64 | 64 | length | -0.123 | [-0.152, -0.094] | True | False |
| smoke/reach64 | 128 | ft | +0.082 | [+0.059, +0.107] | True | True |
| smoke/reach64 | 128 | emb | +0.054 | [+0.027, +0.083] | True | True |
| smoke/reach64 | 128 | comb | +0.078 | [+0.058, +0.099] | True | True |
| smoke/reach64 | 128 | length | -0.155 | [-0.190, -0.120] | True | False |

## Average precision (95 % case-bootstrap CI; chance level in brackets)

| config / cut | N | ft: within | ft: pooled | probe: within | probe: pooled | emb: within | emb: pooled | comb: within | comb: pooled | length: within | length: pooled | chance within | chance pooled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| realistic/all | 0 | 0.297 [0.278, 0.316] | 0.297 [0.278, 0.316] | 0.297 [0.278, 0.316] | 0.455 [0.423, 0.485] | 0.297 [0.278, 0.316] | 0.297 [0.278, 0.316] | 0.297 [0.278, 0.316] | 0.455 [0.423, 0.485] | 0.297 [0.278, 0.316] | 0.420 [0.389, 0.450] | 0.297 | 0.297 |
| realistic/all | 8 | 0.613 [0.578, 0.646] | 0.486 [0.449, 0.528] | 0.634 [0.602, 0.665] | 0.540 [0.504, 0.578] | 0.618 [0.583, 0.652] | 0.469 [0.433, 0.506] | 0.651 [0.617, 0.683] | 0.552 [0.516, 0.591] | 0.610 [0.579, 0.640] | 0.477 [0.442, 0.516] | 0.297 | 0.297 |
| realistic/all | 16 | 0.665 [0.631, 0.697] | 0.594 [0.554, 0.632] | 0.705 [0.672, 0.735] | 0.620 [0.586, 0.656] | 0.658 [0.625, 0.690] | 0.553 [0.516, 0.592] | 0.688 [0.655, 0.719] | 0.635 [0.599, 0.671] | 0.625 [0.593, 0.657] | 0.475 [0.439, 0.513] | 0.297 | 0.297 |
| realistic/all | 32 | 0.775 [0.746, 0.803] | 0.722 [0.687, 0.753] | 0.777 [0.748, 0.807] | 0.711 [0.679, 0.745] | 0.774 [0.744, 0.803] | 0.692 [0.658, 0.725] | 0.800 [0.773, 0.827] | 0.741 [0.709, 0.772] | 0.655 [0.623, 0.687] | 0.504 [0.468, 0.545] | 0.305 | 0.305 |
| realistic/all | 64 | 0.861 [0.837, 0.885] | 0.817 [0.791, 0.842] | 0.835 [0.808, 0.861] | 0.789 [0.764, 0.815] | 0.837 [0.811, 0.864] | 0.788 [0.755, 0.818] | 0.877 [0.853, 0.900] | 0.829 [0.804, 0.853] | 0.719 [0.688, 0.750] | 0.535 [0.500, 0.575] | 0.339 | 0.337 |
| realistic/all | 128 | 0.909 [0.886, 0.931] | 0.893 [0.870, 0.914] | 0.849 [0.821, 0.876] | 0.809 [0.776, 0.840] | 0.880 [0.855, 0.904] | 0.843 [0.814, 0.869] | 0.916 [0.891, 0.937] | 0.893 [0.869, 0.915] | 0.716 [0.682, 0.748] | 0.579 [0.536, 0.621] | 0.411 | 0.410 |
| realistic/reach64 | 0 | 0.339 [0.319, 0.359] | 0.337 [0.316, 0.357] | 0.339 [0.319, 0.359] | 0.473 [0.442, 0.502] | 0.339 [0.319, 0.359] | 0.337 [0.316, 0.357] | 0.339 [0.319, 0.359] | 0.473 [0.442, 0.502] | 0.339 [0.319, 0.359] | 0.433 [0.402, 0.461] | 0.339 | 0.337 |
| realistic/reach64 | 8 | 0.646 [0.613, 0.678] | 0.506 [0.468, 0.546] | 0.669 [0.637, 0.699] | 0.552 [0.517, 0.587] | 0.649 [0.614, 0.683] | 0.500 [0.462, 0.537] | 0.678 [0.646, 0.709] | 0.565 [0.529, 0.601] | 0.641 [0.611, 0.670] | 0.489 [0.452, 0.526] | 0.339 | 0.337 |
| realistic/reach64 | 16 | 0.693 [0.661, 0.725] | 0.610 [0.571, 0.648] | 0.727 [0.697, 0.755] | 0.624 [0.588, 0.660] | 0.695 [0.662, 0.725] | 0.569 [0.532, 0.606] | 0.715 [0.685, 0.745] | 0.643 [0.606, 0.678] | 0.643 [0.612, 0.675] | 0.481 [0.444, 0.520] | 0.339 | 0.337 |
| realistic/reach64 | 32 | 0.793 [0.766, 0.820] | 0.727 [0.692, 0.760] | 0.790 [0.761, 0.819] | 0.715 [0.679, 0.748] | 0.788 [0.759, 0.815] | 0.699 [0.663, 0.732] | 0.815 [0.788, 0.842] | 0.743 [0.711, 0.774] | 0.667 [0.634, 0.699] | 0.509 [0.471, 0.551] | 0.339 | 0.337 |
| realistic/reach64 | 64 | 0.861 [0.834, 0.885] | 0.817 [0.790, 0.841] | 0.835 [0.807, 0.862] | 0.789 [0.764, 0.813] | 0.837 [0.810, 0.864] | 0.788 [0.756, 0.818] | 0.877 [0.852, 0.900] | 0.829 [0.804, 0.852] | 0.719 [0.688, 0.748] | 0.535 [0.497, 0.575] | 0.339 | 0.337 |
| realistic/reach64 | 128 | 0.909 [0.887, 0.930] | 0.893 [0.870, 0.913] | 0.849 [0.821, 0.875] | 0.809 [0.775, 0.837] | 0.880 [0.854, 0.904] | 0.843 [0.815, 0.869] | 0.916 [0.894, 0.937] | 0.893 [0.868, 0.915] | 0.716 [0.684, 0.751] | 0.579 [0.538, 0.623] | 0.411 | 0.410 |
| smoke/all | 0 | 0.340 [0.321, 0.360] | 0.340 [0.321, 0.360] | 0.340 [0.321, 0.360] | 0.507 [0.474, 0.537] | 0.340 [0.321, 0.360] | 0.340 [0.321, 0.360] | 0.340 [0.321, 0.360] | 0.507 [0.474, 0.537] | 0.340 [0.321, 0.360] | 0.469 [0.437, 0.499] | 0.340 | 0.340 |
| smoke/all | 8 | 0.612 [0.580, 0.643] | 0.492 [0.459, 0.529] | 0.627 [0.594, 0.659] | 0.548 [0.513, 0.583] | 0.574 [0.542, 0.604] | 0.427 [0.395, 0.462] | 0.634 [0.601, 0.664] | 0.551 [0.516, 0.588] | 0.578 [0.549, 0.607] | 0.479 [0.447, 0.513] | 0.340 | 0.340 |
| smoke/all | 16 | 0.647 [0.615, 0.679] | 0.569 [0.535, 0.602] | 0.669 [0.642, 0.698] | 0.603 [0.571, 0.636] | 0.644 [0.614, 0.675] | 0.534 [0.500, 0.569] | 0.677 [0.647, 0.708] | 0.614 [0.582, 0.645] | 0.602 [0.573, 0.630] | 0.479 [0.449, 0.514] | 0.340 | 0.340 |
| smoke/all | 32 | 0.722 [0.691, 0.751] | 0.661 [0.627, 0.696] | 0.725 [0.693, 0.755] | 0.679 [0.646, 0.712] | 0.670 [0.641, 0.701] | 0.580 [0.548, 0.617] | 0.739 [0.709, 0.769] | 0.693 [0.660, 0.728] | 0.651 [0.620, 0.682] | 0.530 [0.495, 0.568] | 0.342 | 0.342 |
| smoke/all | 64 | 0.807 [0.780, 0.835] | 0.782 [0.754, 0.808] | 0.780 [0.751, 0.809] | 0.753 [0.722, 0.784] | 0.766 [0.737, 0.795] | 0.705 [0.672, 0.740] | 0.821 [0.794, 0.847] | 0.800 [0.771, 0.828] | 0.657 [0.626, 0.687] | 0.534 [0.499, 0.573] | 0.349 | 0.348 |
| smoke/all | 128 | 0.877 [0.852, 0.901] | 0.854 [0.828, 0.877] | 0.804 [0.774, 0.833] | 0.772 [0.739, 0.802] | 0.844 [0.818, 0.869] | 0.793 [0.764, 0.822] | 0.873 [0.847, 0.897] | 0.859 [0.832, 0.882] | 0.663 [0.634, 0.694] | 0.530 [0.494, 0.567] | 0.391 | 0.385 |
| smoke/reach64 | 0 | 0.349 [0.329, 0.370] | 0.348 [0.328, 0.369] | 0.349 [0.329, 0.370] | 0.509 [0.477, 0.537] | 0.349 [0.329, 0.370] | 0.348 [0.328, 0.369] | 0.349 [0.329, 0.370] | 0.509 [0.477, 0.537] | 0.349 [0.329, 0.370] | 0.470 [0.437, 0.501] | 0.349 | 0.348 |
| smoke/reach64 | 8 | 0.617 [0.585, 0.648] | 0.493 [0.458, 0.529] | 0.634 [0.604, 0.664] | 0.548 [0.512, 0.584] | 0.586 [0.556, 0.619] | 0.435 [0.404, 0.472] | 0.640 [0.610, 0.671] | 0.551 [0.515, 0.588] | 0.582 [0.553, 0.611] | 0.480 [0.448, 0.516] | 0.349 | 0.348 |
| smoke/reach64 | 16 | 0.651 [0.621, 0.682] | 0.568 [0.535, 0.603] | 0.676 [0.647, 0.704] | 0.601 [0.569, 0.635] | 0.649 [0.617, 0.681] | 0.536 [0.502, 0.571] | 0.678 [0.649, 0.709] | 0.611 [0.581, 0.644] | 0.605 [0.577, 0.633] | 0.479 [0.448, 0.516] | 0.349 | 0.348 |
| smoke/reach64 | 32 | 0.723 [0.694, 0.752] | 0.659 [0.624, 0.694] | 0.726 [0.696, 0.756] | 0.676 [0.641, 0.712] | 0.674 [0.645, 0.704] | 0.583 [0.550, 0.618] | 0.740 [0.711, 0.769] | 0.690 [0.655, 0.725] | 0.655 [0.623, 0.685] | 0.531 [0.496, 0.571] | 0.349 | 0.348 |
| smoke/reach64 | 64 | 0.807 [0.780, 0.834] | 0.782 [0.756, 0.808] | 0.780 [0.751, 0.808] | 0.753 [0.723, 0.784] | 0.766 [0.735, 0.795] | 0.705 [0.673, 0.738] | 0.821 [0.794, 0.846] | 0.800 [0.772, 0.827] | 0.657 [0.626, 0.687] | 0.534 [0.499, 0.573] | 0.349 | 0.348 |
| smoke/reach64 | 128 | 0.877 [0.852, 0.900] | 0.854 [0.829, 0.878] | 0.804 [0.776, 0.832] | 0.772 [0.740, 0.800] | 0.844 [0.818, 0.868] | 0.793 [0.766, 0.820] | 0.873 [0.848, 0.897] | 0.859 [0.834, 0.882] | 0.663 [0.633, 0.693] | 0.530 [0.494, 0.567] | 0.391 | 0.385 |

## Paired difference vs probe (average precision)

| config / cut | N | method | metric | Δ | 95 % CI | CI excludes 0 |
|---|---|---|---|---|---|---|
| realistic/all | 0 | ft | ap_cont | -0.158 | [-0.181, -0.134] | True |
| realistic/all | 0 | ft | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| realistic/all | 0 | emb | ap_cont | -0.158 | [-0.181, -0.134] | True |
| realistic/all | 0 | emb | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| realistic/all | 0 | comb | ap_cont | +0.000 | [+0.000, +0.000] | False |
| realistic/all | 0 | comb | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| realistic/all | 0 | length | ap_cont | -0.035 | [-0.052, -0.020] | True |
| realistic/all | 0 | length | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| realistic/all | 8 | ft | ap_cont | -0.053 | [-0.080, -0.027] | True |
| realistic/all | 8 | ft | ap_within_case_mean | -0.022 | [-0.049, +0.006] | False |
| realistic/all | 8 | emb | ap_cont | -0.070 | [-0.102, -0.041] | True |
| realistic/all | 8 | emb | ap_within_case_mean | -0.016 | [-0.049, +0.016] | False |
| realistic/all | 8 | comb | ap_cont | +0.013 | [-0.004, +0.029] | False |
| realistic/all | 8 | comb | ap_within_case_mean | +0.017 | [-0.003, +0.038] | False |
| realistic/all | 8 | length | ap_cont | -0.063 | [-0.089, -0.033] | True |
| realistic/all | 8 | length | ap_within_case_mean | -0.024 | [-0.050, +0.001] | False |
| realistic/all | 16 | ft | ap_cont | -0.027 | [-0.057, +0.004] | False |
| realistic/all | 16 | ft | ap_within_case_mean | -0.040 | [-0.072, -0.006] | True |
| realistic/all | 16 | emb | ap_cont | -0.067 | [-0.098, -0.034] | True |
| realistic/all | 16 | emb | ap_within_case_mean | -0.047 | [-0.078, -0.015] | True |
| realistic/all | 16 | comb | ap_cont | +0.015 | [-0.007, +0.036] | False |
| realistic/all | 16 | comb | ap_within_case_mean | -0.017 | [-0.043, +0.009] | False |
| realistic/all | 16 | length | ap_cont | -0.145 | [-0.172, -0.117] | True |
| realistic/all | 16 | length | ap_within_case_mean | -0.080 | [-0.105, -0.053] | True |
| realistic/all | 32 | ft | ap_cont | +0.010 | [-0.013, +0.034] | False |
| realistic/all | 32 | ft | ap_within_case_mean | -0.002 | [-0.030, +0.027] | False |
| realistic/all | 32 | emb | ap_cont | -0.019 | [-0.045, +0.009] | False |
| realistic/all | 32 | emb | ap_within_case_mean | -0.003 | [-0.036, +0.030] | False |
| realistic/all | 32 | comb | ap_cont | +0.029 | [+0.014, +0.045] | True |
| realistic/all | 32 | comb | ap_within_case_mean | +0.024 | [+0.004, +0.045] | True |
| realistic/all | 32 | length | ap_cont | -0.207 | [-0.242, -0.171] | True |
| realistic/all | 32 | length | ap_within_case_mean | -0.122 | [-0.152, -0.093] | True |
| realistic/all | 64 | ft | ap_cont | +0.028 | [+0.006, +0.048] | True |
| realistic/all | 64 | ft | ap_within_case_mean | +0.026 | [+0.001, +0.050] | True |
| realistic/all | 64 | emb | ap_cont | -0.001 | [-0.027, +0.023] | False |
| realistic/all | 64 | emb | ap_within_case_mean | +0.003 | [-0.022, +0.027] | False |
| realistic/all | 64 | comb | ap_cont | +0.040 | [+0.023, +0.056] | True |
| realistic/all | 64 | comb | ap_within_case_mean | +0.042 | [+0.020, +0.064] | True |
| realistic/all | 64 | length | ap_cont | -0.255 | [-0.289, -0.217] | True |
| realistic/all | 64 | length | ap_within_case_mean | -0.116 | [-0.146, -0.085] | True |
| realistic/all | 128 | ft | ap_cont | +0.084 | [+0.058, +0.111] | True |
| realistic/all | 128 | ft | ap_within_case_mean | +0.060 | [+0.036, +0.084] | True |
| realistic/all | 128 | emb | ap_cont | +0.034 | [+0.005, +0.066] | True |
| realistic/all | 128 | emb | ap_within_case_mean | +0.031 | [+0.006, +0.054] | True |
| realistic/all | 128 | comb | ap_cont | +0.085 | [+0.067, +0.104] | True |
| realistic/all | 128 | comb | ap_within_case_mean | +0.067 | [+0.046, +0.089] | True |
| realistic/all | 128 | length | ap_cont | -0.230 | [-0.264, -0.193] | True |
| realistic/all | 128 | length | ap_within_case_mean | -0.133 | [-0.169, -0.097] | True |
| realistic/reach64 | 0 | ft | ap_cont | -0.136 | [-0.159, -0.112] | True |
| realistic/reach64 | 0 | ft | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| realistic/reach64 | 0 | emb | ap_cont | -0.136 | [-0.159, -0.112] | True |
| realistic/reach64 | 0 | emb | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| realistic/reach64 | 0 | comb | ap_cont | +0.000 | [+0.000, +0.000] | False |
| realistic/reach64 | 0 | comb | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| realistic/reach64 | 0 | length | ap_cont | -0.040 | [-0.058, -0.024] | True |
| realistic/reach64 | 0 | length | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| realistic/reach64 | 8 | ft | ap_cont | -0.046 | [-0.073, -0.019] | True |
| realistic/reach64 | 8 | ft | ap_within_case_mean | -0.024 | [-0.053, +0.004] | False |
| realistic/reach64 | 8 | emb | ap_cont | -0.052 | [-0.082, -0.025] | True |
| realistic/reach64 | 8 | emb | ap_within_case_mean | -0.021 | [-0.053, +0.012] | False |
| realistic/reach64 | 8 | comb | ap_cont | +0.012 | [-0.004, +0.028] | False |
| realistic/reach64 | 8 | comb | ap_within_case_mean | +0.008 | [-0.012, +0.030] | False |
| realistic/reach64 | 8 | length | ap_cont | -0.063 | [-0.091, -0.034] | True |
| realistic/reach64 | 8 | length | ap_within_case_mean | -0.028 | [-0.054, -0.001] | True |
| realistic/reach64 | 16 | ft | ap_cont | -0.013 | [-0.045, +0.016] | False |
| realistic/reach64 | 16 | ft | ap_within_case_mean | -0.034 | [-0.066, -0.004] | True |
| realistic/reach64 | 16 | emb | ap_cont | -0.055 | [-0.089, -0.022] | True |
| realistic/reach64 | 16 | emb | ap_within_case_mean | -0.032 | [-0.061, -0.003] | True |
| realistic/reach64 | 16 | comb | ap_cont | +0.020 | [-0.002, +0.039] | False |
| realistic/reach64 | 16 | comb | ap_within_case_mean | -0.012 | [-0.038, +0.013] | False |
| realistic/reach64 | 16 | length | ap_cont | -0.143 | [-0.171, -0.115] | True |
| realistic/reach64 | 16 | length | ap_within_case_mean | -0.084 | [-0.109, -0.059] | True |
| realistic/reach64 | 32 | ft | ap_cont | +0.013 | [-0.012, +0.037] | False |
| realistic/reach64 | 32 | ft | ap_within_case_mean | +0.003 | [-0.023, +0.029] | False |
| realistic/reach64 | 32 | emb | ap_cont | -0.015 | [-0.042, +0.011] | False |
| realistic/reach64 | 32 | emb | ap_within_case_mean | -0.003 | [-0.035, +0.027] | False |
| realistic/reach64 | 32 | comb | ap_cont | +0.028 | [+0.013, +0.044] | True |
| realistic/reach64 | 32 | comb | ap_within_case_mean | +0.025 | [+0.005, +0.044] | True |
| realistic/reach64 | 32 | length | ap_cont | -0.206 | [-0.239, -0.169] | True |
| realistic/reach64 | 32 | length | ap_within_case_mean | -0.123 | [-0.153, -0.095] | True |
| realistic/reach64 | 64 | ft | ap_cont | +0.028 | [+0.005, +0.050] | True |
| realistic/reach64 | 64 | ft | ap_within_case_mean | +0.026 | [+0.002, +0.051] | True |
| realistic/reach64 | 64 | emb | ap_cont | -0.001 | [-0.028, +0.025] | False |
| realistic/reach64 | 64 | emb | ap_within_case_mean | +0.003 | [-0.022, +0.027] | False |
| realistic/reach64 | 64 | comb | ap_cont | +0.040 | [+0.022, +0.057] | True |
| realistic/reach64 | 64 | comb | ap_within_case_mean | +0.042 | [+0.020, +0.065] | True |
| realistic/reach64 | 64 | length | ap_cont | -0.255 | [-0.289, -0.217] | True |
| realistic/reach64 | 64 | length | ap_within_case_mean | -0.116 | [-0.145, -0.086] | True |
| realistic/reach64 | 128 | ft | ap_cont | +0.084 | [+0.059, +0.111] | True |
| realistic/reach64 | 128 | ft | ap_within_case_mean | +0.060 | [+0.036, +0.085] | True |
| realistic/reach64 | 128 | emb | ap_cont | +0.034 | [+0.006, +0.064] | True |
| realistic/reach64 | 128 | emb | ap_within_case_mean | +0.031 | [+0.006, +0.056] | True |
| realistic/reach64 | 128 | comb | ap_cont | +0.085 | [+0.067, +0.104] | True |
| realistic/reach64 | 128 | comb | ap_within_case_mean | +0.067 | [+0.047, +0.089] | True |
| realistic/reach64 | 128 | length | ap_cont | -0.230 | [-0.266, -0.191] | True |
| realistic/reach64 | 128 | length | ap_within_case_mean | -0.133 | [-0.168, -0.096] | True |
| smoke/all | 0 | ft | ap_cont | -0.166 | [-0.190, -0.139] | True |
| smoke/all | 0 | ft | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| smoke/all | 0 | emb | ap_cont | -0.166 | [-0.190, -0.139] | True |
| smoke/all | 0 | emb | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| smoke/all | 0 | comb | ap_cont | +0.000 | [+0.000, +0.000] | False |
| smoke/all | 0 | comb | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| smoke/all | 0 | length | ap_cont | -0.038 | [-0.054, -0.023] | True |
| smoke/all | 0 | length | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| smoke/all | 8 | ft | ap_cont | -0.056 | [-0.084, -0.024] | True |
| smoke/all | 8 | ft | ap_within_case_mean | -0.015 | [-0.044, +0.012] | False |
| smoke/all | 8 | emb | ap_cont | -0.121 | [-0.158, -0.085] | True |
| smoke/all | 8 | emb | ap_within_case_mean | -0.054 | [-0.086, -0.023] | True |
| smoke/all | 8 | comb | ap_cont | +0.004 | [-0.007, +0.015] | False |
| smoke/all | 8 | comb | ap_within_case_mean | +0.006 | [-0.005, +0.019] | False |
| smoke/all | 8 | length | ap_cont | -0.069 | [-0.097, -0.040] | True |
| smoke/all | 8 | length | ap_within_case_mean | -0.049 | [-0.073, -0.024] | True |
| smoke/all | 16 | ft | ap_cont | -0.034 | [-0.064, -0.005] | True |
| smoke/all | 16 | ft | ap_within_case_mean | -0.023 | [-0.050, +0.004] | False |
| smoke/all | 16 | emb | ap_cont | -0.069 | [-0.100, -0.037] | True |
| smoke/all | 16 | emb | ap_within_case_mean | -0.025 | [-0.052, +0.002] | False |
| smoke/all | 16 | comb | ap_cont | +0.011 | [-0.007, +0.029] | False |
| smoke/all | 16 | comb | ap_within_case_mean | +0.008 | [-0.015, +0.030] | False |
| smoke/all | 16 | length | ap_cont | -0.124 | [-0.148, -0.096] | True |
| smoke/all | 16 | length | ap_within_case_mean | -0.067 | [-0.092, -0.045] | True |
| smoke/all | 32 | ft | ap_cont | -0.018 | [-0.044, +0.009] | False |
| smoke/all | 32 | ft | ap_within_case_mean | -0.003 | [-0.029, +0.025] | False |
| smoke/all | 32 | emb | ap_cont | -0.098 | [-0.133, -0.063] | True |
| smoke/all | 32 | emb | ap_within_case_mean | -0.054 | [-0.087, -0.023] | True |
| smoke/all | 32 | comb | ap_cont | +0.014 | [-0.003, +0.033] | False |
| smoke/all | 32 | comb | ap_within_case_mean | +0.015 | [-0.006, +0.036] | False |
| smoke/all | 32 | length | ap_cont | -0.148 | [-0.179, -0.116] | True |
| smoke/all | 32 | length | ap_within_case_mean | -0.073 | [-0.100, -0.047] | True |
| smoke/all | 64 | ft | ap_cont | +0.029 | [+0.006, +0.051] | True |
| smoke/all | 64 | ft | ap_within_case_mean | +0.028 | [-0.003, +0.058] | False |
| smoke/all | 64 | emb | ap_cont | -0.048 | [-0.079, -0.017] | True |
| smoke/all | 64 | emb | ap_within_case_mean | -0.014 | [-0.042, +0.015] | False |
| smoke/all | 64 | comb | ap_cont | +0.047 | [+0.034, +0.061] | True |
| smoke/all | 64 | comb | ap_within_case_mean | +0.041 | [+0.021, +0.061] | True |
| smoke/all | 64 | length | ap_cont | -0.219 | [-0.252, -0.187] | True |
| smoke/all | 64 | length | ap_within_case_mean | -0.123 | [-0.153, -0.094] | True |
| smoke/all | 128 | ft | ap_cont | +0.082 | [+0.058, +0.106] | True |
| smoke/all | 128 | ft | ap_within_case_mean | +0.073 | [+0.050, +0.098] | True |
| smoke/all | 128 | emb | ap_cont | +0.022 | [-0.006, +0.050] | False |
| smoke/all | 128 | emb | ap_within_case_mean | +0.040 | [+0.013, +0.066] | True |
| smoke/all | 128 | comb | ap_cont | +0.087 | [+0.067, +0.107] | True |
| smoke/all | 128 | comb | ap_within_case_mean | +0.069 | [+0.049, +0.091] | True |
| smoke/all | 128 | length | ap_cont | -0.242 | [-0.276, -0.207] | True |
| smoke/all | 128 | length | ap_within_case_mean | -0.141 | [-0.171, -0.110] | True |
| smoke/reach64 | 0 | ft | ap_cont | -0.160 | [-0.184, -0.134] | True |
| smoke/reach64 | 0 | ft | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| smoke/reach64 | 0 | emb | ap_cont | -0.160 | [-0.184, -0.134] | True |
| smoke/reach64 | 0 | emb | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| smoke/reach64 | 0 | comb | ap_cont | +0.000 | [+0.000, +0.000] | False |
| smoke/reach64 | 0 | comb | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| smoke/reach64 | 0 | length | ap_cont | -0.039 | [-0.055, -0.025] | True |
| smoke/reach64 | 0 | length | ap_within_case_mean | +0.000 | [+0.000, +0.000] | False |
| smoke/reach64 | 8 | ft | ap_cont | -0.055 | [-0.084, -0.024] | True |
| smoke/reach64 | 8 | ft | ap_within_case_mean | -0.017 | [-0.043, +0.009] | False |
| smoke/reach64 | 8 | emb | ap_cont | -0.113 | [-0.150, -0.074] | True |
| smoke/reach64 | 8 | emb | ap_within_case_mean | -0.049 | [-0.079, -0.016] | True |
| smoke/reach64 | 8 | comb | ap_cont | +0.003 | [-0.008, +0.014] | False |
| smoke/reach64 | 8 | comb | ap_within_case_mean | +0.005 | [-0.006, +0.018] | False |
| smoke/reach64 | 8 | length | ap_cont | -0.068 | [-0.096, -0.039] | True |
| smoke/reach64 | 8 | length | ap_within_case_mean | -0.052 | [-0.076, -0.028] | True |
| smoke/reach64 | 16 | ft | ap_cont | -0.034 | [-0.063, -0.004] | True |
| smoke/reach64 | 16 | ft | ap_within_case_mean | -0.025 | [-0.051, +0.003] | False |
| smoke/reach64 | 16 | emb | ap_cont | -0.066 | [-0.098, -0.034] | True |
| smoke/reach64 | 16 | emb | ap_within_case_mean | -0.027 | [-0.052, +0.001] | False |
| smoke/reach64 | 16 | comb | ap_cont | +0.010 | [-0.009, +0.029] | False |
| smoke/reach64 | 16 | comb | ap_within_case_mean | +0.003 | [-0.020, +0.025] | False |
| smoke/reach64 | 16 | length | ap_cont | -0.122 | [-0.147, -0.096] | True |
| smoke/reach64 | 16 | length | ap_within_case_mean | -0.071 | [-0.094, -0.047] | True |
| smoke/reach64 | 32 | ft | ap_cont | -0.017 | [-0.044, +0.010] | False |
| smoke/reach64 | 32 | ft | ap_within_case_mean | -0.004 | [-0.031, +0.025] | False |
| smoke/reach64 | 32 | emb | ap_cont | -0.093 | [-0.129, -0.057] | True |
| smoke/reach64 | 32 | emb | ap_within_case_mean | -0.052 | [-0.083, -0.023] | True |
| smoke/reach64 | 32 | comb | ap_cont | +0.014 | [-0.004, +0.033] | False |
| smoke/reach64 | 32 | comb | ap_within_case_mean | +0.014 | [-0.006, +0.034] | False |
| smoke/reach64 | 32 | length | ap_cont | -0.145 | [-0.175, -0.111] | True |
| smoke/reach64 | 32 | length | ap_within_case_mean | -0.072 | [-0.097, -0.045] | True |
| smoke/reach64 | 64 | ft | ap_cont | +0.029 | [+0.005, +0.054] | True |
| smoke/reach64 | 64 | ft | ap_within_case_mean | +0.028 | [-0.003, +0.057] | False |
| smoke/reach64 | 64 | emb | ap_cont | -0.048 | [-0.077, -0.015] | True |
| smoke/reach64 | 64 | emb | ap_within_case_mean | -0.014 | [-0.043, +0.016] | False |
| smoke/reach64 | 64 | comb | ap_cont | +0.047 | [+0.033, +0.063] | True |
| smoke/reach64 | 64 | comb | ap_within_case_mean | +0.041 | [+0.022, +0.061] | True |
| smoke/reach64 | 64 | length | ap_cont | -0.219 | [-0.253, -0.186] | True |
| smoke/reach64 | 64 | length | ap_within_case_mean | -0.123 | [-0.153, -0.092] | True |
| smoke/reach64 | 128 | ft | ap_cont | +0.082 | [+0.058, +0.107] | True |
| smoke/reach64 | 128 | ft | ap_within_case_mean | +0.073 | [+0.051, +0.097] | True |
| smoke/reach64 | 128 | emb | ap_cont | +0.022 | [-0.006, +0.050] | False |
| smoke/reach64 | 128 | emb | ap_within_case_mean | +0.040 | [+0.014, +0.066] | True |
| smoke/reach64 | 128 | comb | ap_cont | +0.087 | [+0.067, +0.108] | True |
| smoke/reach64 | 128 | comb | ap_within_case_mean | +0.069 | [+0.050, +0.090] | True |
| smoke/reach64 | 128 | length | ap_cont | -0.242 | [-0.275, -0.206] | True |
| smoke/reach64 | 128 | length | ap_within_case_mean | -0.141 | [-0.171, -0.110] | True |

Pair-weighted `auroc_within` and across-row `auroc_cont` are in metrics.csv (secondary). Scores of comb, emb, ft, length, probe were rank-mapped into (0, 1): their Brier columns mean nothing.

## Coverage

| config | N | rows | violating_share | mixed_cases | scored_ft | scored_probe | scored_emb | scored_comb | scored_length |
|---|---|---|---|---|---|---|---|---|---|
| smoke | 0 | 2392 | 0.340 | 299 | 2392 | 2392 | 2392 | 2392 | 2392 |
| smoke | 8 | 2392 | 0.340 | 299 | 2392 | 2392 | 2392 | 2392 | 2392 |
| smoke | 16 | 2392 | 0.340 | 299 | 2392 | 2392 | 2392 | 2392 | 2392 |
| smoke | 32 | 2381 | 0.342 | 299 | 2381 | 2381 | 2381 | 2381 | 2381 |
| smoke | 64 | 2315 | 0.348 | 297 | 2315 | 2315 | 2315 | 2315 | 2315 |
| smoke | 128 | 1929 | 0.385 | 290 | 1929 | 1929 | 1929 | 1929 | 1929 |
| realistic | 0 | 2392 | 0.297 | 299 | 2392 | 2392 | 2392 | 2392 | 2392 |
| realistic | 8 | 2392 | 0.297 | 299 | 2392 | 2392 | 2392 | 2392 | 2392 |
| realistic | 16 | 2392 | 0.297 | 299 | 2392 | 2392 | 2392 | 2392 | 2392 |
| realistic | 32 | 2327 | 0.305 | 299 | 2327 | 2327 | 2327 | 2327 | 2327 |
| realistic | 64 | 2060 | 0.337 | 293 | 2060 | 2060 | 2060 | 2060 | 2060 |
| realistic | 128 | 1477 | 0.410 | 260 | 1477 | 1477 | 1477 | 1477 | 1477 |
