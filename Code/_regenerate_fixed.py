"""
One-off regeneration script: reruns the pass pipeline with the bug fixes
applied and writes output under a '_fixed' suffix so the original
(pre-fix) CSVs in Tracking_Data/passes/ are preserved untouched.
"""
from passtracking import PassTracking, features

DATE, HOME, AWAY = '12.12.2015', 'MIL', 'GSW'
OUT_DIR = './../Tracking_Data/passes'

passes = PassTracking(DATE, HOME, AWAY)
passes.track_passes()
passes.moments.to_csv(f'{OUT_DIR}/moments_fixed.csv')
passes.pbp.to_csv(f'{OUT_DIR}/pbp_fixed.csv')
passes.pass_attempt.to_csv(f'{OUT_DIR}/passes_fixed.csv')

feature = features(DATE, HOME, AWAY)
feature.features_extraction()
feature.features.to_csv(f'{OUT_DIR}/features_fixed.csv')

print("Done. Wrote moments_fixed.csv, pbp_fixed.csv, passes_fixed.csv, features_fixed.csv")
