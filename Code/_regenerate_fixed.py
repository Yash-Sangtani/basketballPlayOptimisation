"""
One-off regeneration script: reruns the pass pipeline on the sample game and
writes output under a '_v2' suffix so the original (pre-fix) CSVs in
Tracking_Data/passes/ are preserved untouched.

Run from Code/:  ./run_logged.sh regen python3 _regenerate_fixed.py
"""
from passtracking import features

DATE, HOME, AWAY = '12.12.2015', 'MIL', 'GSW'
OUT_DIR = './../Tracking_Data/passes'

feature = features(DATE, HOME, AWAY)
passes = feature.passes
passes.moments.to_csv(f'{OUT_DIR}/moments_v2.csv')
passes.pbp.to_csv(f'{OUT_DIR}/pbp_v2.csv')
passes.pass_attempt.to_csv(f'{OUT_DIR}/passes_v2.csv')

feature.features_extraction()
feature.features.to_csv(f'{OUT_DIR}/features_v2.csv')
feature.meta.to_csv(f'{OUT_DIR}/features_meta_v2.csv')

print("Failure matching:", passes.failure_report)
print("Done. Wrote moments_v2.csv, pbp_v2.csv, passes_v2.csv, features_v2.csv, features_meta_v2.csv")
