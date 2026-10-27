DEV, 5-fold GroupKFold by game, seed 0: 501 games, 200131 passes, 3927 failures

### discrimination

| model | ROC-AUC [95% CI] | PR-AUC [95% CI] | fold ROC mean+-sd | fold PR mean+-sd | fit s (all folds) |
|---|---|---|---|---|---|
| no-skill | 0.500 [0.500, 0.500] | 0.0196 [0.0190, 0.0203] |  |  |  |
| logistic | 0.728 [0.721, 0.737] | 0.0508 [0.0481, 0.0541] | 0.728+-0.003 | 0.0513+-0.0028 | 2.0 |
| gboost | 0.733 [0.725, 0.740] | 0.0487 [0.0464, 0.0514] | 0.733+-0.006 | 0.0491+-0.0028 | 75.0 |

### paired

| difference | ROC-AUC diff [95% CI] | PR-AUC diff [95% CI] |
|---|---|---|
| gboost - logistic | 0.004 [0.001, 0.007] | -0.0020 [-0.0040, -0.0006] |

### calibration

| model | mean pred raw / cal | Brier raw | Brier cal [95% CI] | BSS raw | BSS cal [95% CI] | logloss raw | logloss cal | ROC cal | PR raw -> cal |
|---|---|---|---|---|---|---|---|---|---|
| no-skill |  |  | 0.01924 |  | +0.000 |  | 0.0966 |  |  |
| logistic | 0.4237 / 0.0197 | 0.21389 | 0.01895 [0.01837, 0.01961] | -10.119 | 0.015 [0.013, 0.017] | 0.6106 | 0.0902 | 0.725 | 0.0508 -> 0.0496 |
| gboost | 0.0196 / 0.0198 | 0.01962 | 0.01896 [0.01837, 0.01960] | -0.020 | 0.015 [0.013, 0.016] | 0.0922 | 0.0898 | 0.731 | 0.0487 -> 0.0484 |

### by_failure_type

| failure_type | n | logistic ROC | logistic PR | gboost ROC | gboost PR | PR no-skill |
|---|---|---|---|---|---|---|
| fumbled | 475 | 0.736 | 0.006 | 0.742 | 0.006 | 0.002 |
| intercepted | 2838 | 0.723 | 0.036 | 0.727 | 0.035 | 0.014 |
| out_of_bounds | 614 | 0.748 | 0.011 | 0.752 | 0.010 | 0.003 |

