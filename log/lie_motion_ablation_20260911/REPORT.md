# Lie motion-feature ablation report

Date: 2026-09-11

## Outcome

The relative-motion idea is useful for candidate ranking, but the experimental
motion ranker does not replace the current online model.  Its strongest
leave-one-video-out result is substantially better than the 43-feature
baseline, while its guarded end-to-end replay remains below the current online
ranker.

- Keep `models/lie_identity_ranker.txt` and margin `2.0` online.
- Keep the motion feature implementation and experimental model for the next
  iteration.
- Do not add local kNN graph strain to the production feature set in its
  current form.

## Experimental controls

- Dataset: 2,991 listwise queries, 43,472 candidate rows, 19 videos.
- Split, LightGBM parameters, seed (`20260911`), and early stopping are held
  fixed across stages.
- Canonical base features are loaded from the original CSV rather than the
  six-decimal JSONL copy.  This exactly reproduces the existing 43-feature
  baseline (`best_iteration=60`, validation Top-1 `0.906883`).
- Green cursor pixels are removed before pixel motion extraction.  In addition,
  every candidate uses the same cursor-blind centre mask: optical flow samples
  only an annulus and polar descriptors suppress their inner radial bins.
- All caches, models, replay CSVs, and videos are written outside the production
  model path.

## Ordered cumulative ablation

| Stage | Features | Validation Top-1 | Hard Top-1 | Delta vs baseline |
| --- | ---: | ---: | ---: | ---: |
| Existing ranker | 43 | 90.69% | 75.82% | -- |
| + robust global similarity transform | 47 | 90.69% | 74.73% | +0.00pp |
| + local kNN graph strain | 51 | 88.26% | 75.82% | -2.43pp |
| + cursor-blind annular optical flow | 54 | 93.52% | 83.52% | +2.83pp |
| + multi-lag polar rotation | 57 | 93.93% | 87.91% | +3.24pp |

The similarity residual is individually informative but does not improve the
fixed validation split on its own.  Graph strain overfits and lowers validation
accuracy.  The first repeatable gain comes from annular optical-flow motion
consistency; multi-lag polar rotation adds most of its value on hard frames.

## Combination and leakage controls

| Combination | Validation Top-1 | Hard Top-1 |
| --- | ---: | ---: |
| Baseline + optical flow + multi-lag | **95.55%** | 87.91% |
| Similarity + optical flow + multi-lag | 94.74% | **90.11%** |
| Baseline + optical angle only + multi-lag | 93.52% | 82.42% |
| Baseline + multi-lag only | 90.28% | 74.73% |

Removing optical-flow fit confidence leaves a smaller gain, proving that the
angle signal has independent value.  The larger gain from the full flow group
shows that the useful cue is broader: the target's local motion is harder to
explain as a stable rigid transform.  That matches the suggested "spider web"
intuition more closely than a rotation-angle-only detector.

## Leave-one-video-out

| Model | Top-1 | Hard Top-1 | Worst video |
| --- | ---: | ---: | ---: |
| Existing 43 features | 86.19% | 77.94% | 54.17% |
| Baseline + optical + multi-lag | 90.64% | 85.00% | 74.87% |
| Angle-only control + multi-lag | 86.59% | 78.71% | 58.85% |
| Similarity + optical + multi-lag | **92.58%** | **88.53%** | **79.17%** |

The selected offline candidate therefore improves weighted LOVO Top-1 by
`+6.39pp` and hard Top-1 by `+10.59pp`.  The minimum held-out-video score also
rises by `+25.00pp`.

## End-to-end 19-video replay

The selected model is `ml/lie_motion_ablation/models/`
`similarity_optical_multilag.txt`, guarded by five-vote switching with margin
`0.5`.

| Metric | Ranker off | Current online ranker | Motion candidate |
| --- | ---: | ---: | ---: |
| Active-frame 40px accuracy | 64.04% | **67.54%** | 65.76% |
| Covered-frame 40px accuracy | 75.37% | **79.50%** | 77.40% |
| Correct active frames | 4,386 | **4,626** | 4,504 |
| Identity switches | **59** | 82 | 86 |
| Ranker-triggered switches | 0 | **42** | 45 |

The motion candidate gains `+118` correct active frames (`+1.72pp`) over no
ranker, but loses `122` frames (`-1.78pp`) to the current online ranker and
switches slightly more often.  An exploratory `0.6` replay before the final
online/offline eligibility alignment reached 65.18%; it is retained as a
diagnostic run rather than treated as a directly comparable final result.

The offline/online gap indicates that per-frame ranking quality is not enough:
online switches feed back into candidate histories and interact with voting,
reassociation, and current-target stickiness.  The next useful experiment is to
use motion features as corroborating evidence for the current ranker's switch
decision, or train directly on switch-event outcomes, rather than replacing the
whole listwise score.

## Artifacts

- `ml/lie_motion_ablation/ablation.json`: fixed-split, combination, and LOVO
  metrics.
- `ml/lie_motion_ablation/train_augmented.csv` and `val_augmented.csv`: cached
  augmented rows.
- `ml/lie_motion_ablation/models/`: isolated experimental models.
- `log/lie_motion_ablation_20260911/final_visualizations_summary.json`: final
  end-to-end summary.
- `log/lie_motion_ablation_20260911/final_visualizations/`: 19 annotated MP4s
  and 19 per-frame CSVs.
- `log/lie_motion_ablation_20260911/margin_0_5_switch_events.json`: exploratory
  switch-event audit.

All 19 final MP4 files passed first-frame and last-frame decode checks.  The
full unit-test suite passes: 80 tests.
