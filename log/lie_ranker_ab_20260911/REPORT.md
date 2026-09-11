# Lie identity ranker online A/B report

Date: 2026-09-11

## Configuration

- Baseline: multi-hypothesis identity + stale coast recovery, ranker disabled.
- Online candidate: the same configuration plus `models/lie_identity_ranker.txt`.
- Ranker minimum margin: `2.0`.
- Healthy-target switch votes: 5.
- Lost/predicted-target recovery votes: 3.
- Videos: all 19 recordings found in `ml/videos`.

The `2.0` margin was selected after a full replay comparison against `0.5`.
The lower threshold reached 68.04% active-frame 40px accuracy but caused 128
identity switches. Margin `2.0` retained 67.54% accuracy while reducing total
switches to 82 and removing the major regressions on recordings 11 and 13.

## Aggregate results

| Metric | Ranker off | Ranker on | Delta |
| --- | ---: | ---: | ---: |
| Active frames | 6849 | 6849 | 0 |
| Active frames within 40px | 4386 | 4626 | +240 |
| Active-frame 40px accuracy | 64.04% | 67.54% | +3.50pp |
| Prediction coverage | 84.96% | 84.96% | 0.00pp |
| Actionable-frame ratio | 84.49% | 84.83% | +0.34pp |
| Mean error on covered frames | 47.73px | 37.85px | -9.89px |
| Median error on covered frames | 22.68px | 21.53px | -1.15px |
| P90 error on covered frames | 91.52px | 67.34px | -24.18px |
| P95 error on covered frames | 178.29px | 124.19px | -54.10px |
| Total identity switches | 59 | 82 | +23 |
| Ranker-triggered active switches | 0 | 42 | +42 |

The active-frame accuracy denominator includes frames where no prediction was
available. Error percentiles use the 5819 frames with both a prediction and
cursor ground truth.

## Per-video active-frame 40px accuracy

| Video | Ranker off | Ranker on | Delta | Switches off/on |
| --- | ---: | ---: | ---: | ---: |
| 1 | 78.21% | 88.55% | +10.34pp | 2 / 2 |
| 2 | 79.36% | 93.59% | +14.23pp | 6 / 5 |
| 3 | 73.94% | 76.36% | +2.42pp | 8 / 14 |
| 4 | 98.73% | 98.73% | 0.00pp | 3 / 3 |
| 5 | 0.00% | 0.00% | 0.00pp | 0 / 0 |
| 6 | 50.69% | 51.79% | +1.10pp | 5 / 8 |
| 7 | 0.00% | 0.00% | 0.00pp | 0 / 0 |
| 8 | 71.55% | 69.25% | -2.30pp | 1 / 4 |
| 9 | 73.09% | 63.46% | -9.63pp | 2 / 11 |
| 10 | 57.50% | 69.72% | +12.22pp | 5 / 5 |
| 11 | 89.01% | 89.01% | 0.00pp | 3 / 3 |
| 12 | 0.00% | 0.00% | 0.00pp | 0 / 0 |
| 13 | 89.63% | 89.63% | 0.00pp | 2 / 2 |
| 14 | 89.70% | 92.32% | +2.63pp | 3 / 6 |
| 15 | 82.17% | 84.30% | +2.13pp | 5 / 7 |
| 16 | 80.67% | 87.73% | +7.06pp | 2 / 2 |
| 17 | 0.00% | 0.00% | 0.00pp | 0 / 0 |
| 19 | 55.58% | 66.24% | +10.66pp | 6 / 5 |
| 20 | 66.07% | 76.28% | +10.20pp | 6 / 5 |

Recordings 5 and 7 have no evaluated prediction coverage. Recordings 12 and
17 have only about 8% prediction coverage and remain detector/acquisition
failures rather than ranker failures.

## Switch-event audit

For each active ranker switch, median cursor error over up to five frames before
and after the switch was compared. All 42 events had usable ground truth:

- 30 improved and 12 regressed.
- 14 crossed from above 40px to at most 40px.
- 1 crossed from at most 40px to above 40px.
- Median event-window error changed from 36.59px to 19.14px.

This is a local diagnostic, not an independent causal metric, because each
switch changes the subsequent tracker trajectory.

## Artifacts

- `baseline_summary.json`: final ranker-off aggregate input.
- `ranker_final_summary.json`: final online aggregate input.
- `baseline/`: ranker-off per-frame CSV files.
- `ranker_final/`: final online per-frame CSV and annotated MP4 files.
- `ranker_final_switch_events.json`: final active ranker switch audit.
- `ranker_summary.json` and `ranker/`: exploratory margin `0.5` run.
- `ranker_margin_2_summary.json` and `ranker_margin_2/`: no-video threshold run.

All 19 final MP4 files were checked by decoding both their first and last frame.
