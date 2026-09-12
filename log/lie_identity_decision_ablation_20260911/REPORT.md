# 测谎身份决策与鼠标安全门消融报告

日期：2026-09-11

## 1. 结论

本轮按“正式基线 → Safety-only → State-aware switching → State-aware + motion”完成了当前 21 段录像的离线消融。结论是：

1. **Safety-only 已达到进入线上灰度的条件。** 它不改变目标轨迹和身份切换，只改变鼠标是否执行；正确动作保留率为 `93.13%`，错误动作拦截率为 `45.99%`，严重错误动作从 `513` 帧降到 `134` 帧，减少 `73.88%`。
2. **State-aware switching 方向有效，但当前版本不能直接上线。** 全有效帧 40px 准确率从 `68.18%` 提升到 `71.63%`，有输出帧准确率从 `78.70%` 提升到 `82.68%`；但切换数从 `94` 增到 `155`，录屏8、11、13出现 `3.95–4.95pp` 回归。
3. **Motion corroboration 暂不采用。** 它相对 State-aware 少 `39` 个正确帧、增加 `23` 次身份切换，并扩大录屏8、11、13、16的回归。
4. **推荐分两步推进：** 先只灰度 Safety-only；State-aware 保留为实验开关，下一轮加入稳定轨迹保护或直接训练 switch-event 模型后再复测。正式默认配置本轮保持不变。

## 2. 实验设置

正式基线：

```yaml
multi_hypothesis_identity: true
stale_coast_recovery: true
identity_ranker_model: models/lie_identity_ranker.txt
identity_ranker_min_margin: 2.0
```

消融层级：

| 层级 | 配置 | 目的 |
| --- | --- | --- |
| Baseline | 正式 ranker | 当前线上对照 |
| Safety-only | Baseline + `--identity-safety` | 身份置信度接入鼠标 HOLD，不改变轨迹 |
| State-aware | Baseline + `--state-aware-ranker` | 衰减证据、按当前身份状态调整切换门槛 |
| State + Motion | State-aware + motion model | 以相对运动模型作为切换佐证 |
| State + Safety | State-aware + Safety-only | 评估两个方向组合后的上限和安全性 |

评估覆盖 `测谎录屏1–17（缺18）、19、20、22、23`，共 `7,706` 个有效帧。绿色光标只用于离线真值，进入 tracker 前会被遮罩。

Safety-only 当前策略：

- 新切换观察 1 帧；
- ranker 与当前身份一致后立即恢复；
- 连续 2 帧与当前身份不一致才 HOLD；
- 已建立身份锁定后，允许最多 14 帧、且位置不确定度仍合格的短时 coast；
- ranker 最多允许短暂 3 帧不可用；
- 位置置信度和身份置信度必须同时满足才允许鼠标动作。

State-aware 当前策略：

- challenger 证据按 `0.82` 衰减，不因一个模糊帧立刻清零；
- recovery / suspicious / healthy 使用不同证据阈值；
- 短 coast 不再自动视为身份丢失，避免过早跳到附近背景；
- identity path 不再是绝对硬门，而是证据折扣；
- motion 模型只增加佐证，不替代正式 43 特征 ranker。

## 3. 全量结果

| 模式 | 全有效帧40px | 有输出帧40px | 正确帧 | 错误帧 | 身份切换 | Ranker切换 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 68.18% | 78.70% | 5,254 | 1,422 | 94 | 47 |
| Safety-only | 68.18% | 78.70% | 5,254 | 1,422 | 94 | 47 |
| State-aware | **71.63%** | **82.68%** | **5,520** | **1,156** | 155 | 118 |
| State + Motion | 71.13% | 82.10% | 5,481 | 1,195 | 178 | 145 |
| State + Safety | 71.63% | 82.68% | 5,520 | 1,156 | 155 | 118 |

Safety-only 不改变轨迹，因此它与 Baseline 的准确率和切换数完全相同。State + Safety 同理，与 State-aware 的轨迹结果相同。四种身份决策模式的无输出帧均为 `1,030`，说明本轮没有改变录屏5、7、12、17的冷启动瓶颈。

## 4. 鼠标安全效果

| 模式 | 正确动作 | 错误动作 | 动作精确率 | 正确动作召回 | 错误动作拦截 | >100px错误动作 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 5,246 | 1,416 | 78.75% | 99.85% | 0.42% | 513 |
| **Safety-only** | **4,893** | **768** | **86.43%** | **93.13%** | **45.99%** | **134** |
| State-aware | 5,520 | 1,156 | 82.68% | 100.00% | 0.00% | 290 |
| State + Safety | 5,226 | 841 | 86.14% | 94.67% | 27.25% | 173 |

Safety-only 相对 Baseline：

- 错误动作减少 `648` 帧，`1416 → 768`；
- 严重错误动作减少 `379` 帧，`513 → 134`；
- 代价是 HOLD `361` 个原本正确的帧，正确动作召回仍为 `93.13%`；
- 稳定样本录屏1、2、4合计正确动作召回约 `96.6%`。

State + Safety 的错误拦截率反而低于 Safety-only，原因不是安全门失效，而是 State-aware 更频繁地切换到 ranker top1，减少了 ranker 与当前身份不一致的可观察窗口。这说明后续不能只用“当前是否为 top1”评估新身份，还需要独立的切换后验证特征。

## 5. State-aware 逐视频变化

| 视频 | Baseline | State-aware | 增减 | State+Motion | State / Motion切换数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 录屏1 | 88.55% | 86.03% | -2.51pp | 85.47% | 4 / 6 |
| 录屏2 | 93.59% | 93.59% | +0.00pp | 95.02% | 5 / 5 |
| 录屏3 | 76.36% | 84.24% | +7.88pp | 88.79% | 17 / 18 |
| 录屏4 | 98.73% | 96.52% | -2.22pp | 96.52% | 8 / 8 |
| 录屏5 | 0.00% | 0.00% | +0.00pp | 0.00% | 0 / 0 |
| 录屏6 | 51.79% | 69.70% | +17.91pp | 69.97% | 10 / 10 |
| 录屏7 | 0.00% | 0.00% | +0.00pp | 0.00% | 0 / 0 |
| 录屏8 | 69.25% | 64.94% | -4.31pp | 61.78% | 13 / 15 |
| 录屏9 | 63.46% | 74.09% | +10.63pp | 75.42% | 11 / 14 |
| 录屏10 | 69.72% | 72.78% | +3.06pp | 73.33% | 11 / 11 |
| 录屏11 | 89.01% | 84.07% | -4.95pp | 76.92% | 7 / 13 |
| 录屏12 | 0.00% | 0.00% | +0.00pp | 1.34% | 0 / 1 |
| 录屏13 | 89.63% | 85.68% | -3.95pp | 81.98% | 8 / 12 |
| 录屏14 | 92.32% | 91.31% | -1.01pp | 91.52% | 10 / 10 |
| 录屏15 | 84.30% | 87.79% | +3.49pp | 88.57% | 10 / 10 |
| 录屏16 | 87.73% | 85.13% | -2.60pp | 81.04% | 6 / 10 |
| 录屏17 | 0.00% | 0.00% | +0.00pp | 0.00% | 0 / 0 |
| 录屏19 | 66.24% | 80.46% | +14.21pp | 80.46% | 9 / 9 |
| 录屏20 | 76.28% | 81.89% | +5.61pp | 82.91% | 9 / 9 |
| 录屏22 | 64.86% | 86.32% | +21.46pp | 86.08% | 8 / 8 |
| 录屏23 | 81.52% | 88.22% | +6.70pp | 87.76% | 9 / 9 |

State-aware 在主要困难视频组上有明显净收益，尤其是录屏6、9、19、22；录屏22、23两个新增视频合计从 `73.28%` 提升到 `87.28%`，说明不是只记住旧视频。

但回归也有共同特征：当前 REAL 短时 coast 或检测抖动时，challenger 在 ranker 中领先并累积证据，随后发生不必要切换。录屏8、11、13最明显。State-aware 的总身份切换从 `94` 增至 `155`，当前门槛仍偏激进。

## 6. Motion corroboration 为什么不采用

Motion 层只在 challenger 与正式 ranker top1 一致时增加证据。它在录屏3、9有小幅附加收益，但总体弱于 State-only：

- 正确帧 `5,520 → 5,481`，减少39帧；
- 身份切换 `155 → 178`；
- ranker触发切换 `118 → 145`；
- 录屏8 `64.94% → 61.78%`；
- 录屏11 `84.07% → 76.92%`；
- 录屏13 `85.68% → 81.98%`；
- 录屏16 `85.13% → 81.04%`。

这说明运动模型有候选排序信息，但当前“加分式佐证”把它当成了过强的独立证据。更合适的方式是将运动特征直接放入 switch-event 模型，学习它在不同状态下是否真的提升切换成功率。

## 7. 推荐方案

### 7.1 推荐灰度：Safety-only

建议先在小比例线上实例启用：

```yaml
lie_detector:
  identity_safety: true
```

灰度时监控：

- `LIE: HOLD` 连续时长；
- HOLD 后是否恢复到正确目标；
- 错误鼠标移动数；
- 正确动作召回，尤其是长 coast 场景；
- 录屏22类场景中 `target_predicted` HOLD 的占比。

Safety-only 是闭环风险最小的一层：它不改变 tracker 状态，也不影响下一帧候选和 ranker 历史，关闭开关即可完全恢复正式行为。

### 7.2 暂不灰度：State-aware switching

进入下一轮前建议增加：

1. **稳定轨迹保护：** 当前 REAL 在最近窗口内长期正确、位置连续、外观连续时，提高 challenger 阈值。
2. **切换事件模型：** 训练目标从 listwise top1 改为 `switch / keep`，输入 current 与 challenger 的分差、持续时间、coast状态、路径支持、运动佐证和切换后短窗结果。
3. **切换后验证：** 新身份不能仅因自己成为 top1 就立即视为安全，需要用后续2–3帧的独立证据确认。
4. **视频级留一验证：** 以最大单视频回归和切换增量作为硬约束，而不只优化总体正确帧。

建议进入下一阶段的门槛：总体至少 `+2pp`，稳定样本单视频回归不超过 `1pp`，身份切换增量不超过 `20%`。当前 State-aware 虽有 `+3.45pp`，但最大回归 `-4.95pp`、切换增量 `+64.9%`，未达标。

## 8. 复现命令

Safety-only：

```bash
python3 tools/replay_all_lie_videos.py \
  --tag safety_v3 \
  --out-dir log/lie_identity_decision_ablation_20260911/safety \
  --summary log/lie_identity_decision_ablation_20260911/safety_summary.json \
  --multi-hypothesis-identity \
  --stale-coast-recovery \
  --identity-ranker-model models/lie_identity_ranker.txt \
  --identity-ranker-min-margin 2.0 \
  --identity-safety
```

State-aware：

```bash
python3 tools/replay_all_lie_videos.py \
  --tag state_aware_v3 \
  --out-dir log/lie_identity_decision_ablation_20260911/state \
  --summary log/lie_identity_decision_ablation_20260911/state_summary.json \
  --multi-hypothesis-identity \
  --stale-coast-recovery \
  --identity-ranker-model models/lie_identity_ranker.txt \
  --identity-ranker-min-margin 2.0 \
  --state-aware-ranker
```

Motion 对照在 State-aware 命令后增加：

```bash
--motion-corroboration-model \
  ml/lie_motion_ablation/models/similarity_optical_multilag.txt
```

汇总：

```bash
python3 tools/summarize_lie_identity_ablation.py \
  log/lie_identity_decision_ablation_20260911
```

## 9. 产物

- `aggregate.json`：所有层级聚合指标及逐视频 summary。
- `baseline_summary.json`：正式基线。
- `safety_summary.json`：Safety-only。
- `state_summary.json`：State-aware。
- `state_safety_summary.json`：State-aware + Safety。
- `motion_summary.json`：State-aware + Motion。
- 各模式目录：逐帧 CSV，属于可再生诊断产物，已加入 `.gitignore`。

本轮没有生成21段可视化 MP4；消融阶段仅生成逐帧 CSV 和 JSON，以避免增加大体积重复产物。需要复查单个 badcase 时，可对指定视频去掉 `--no-video` 或直接用 `tools/lie_detector_replay.py` 生成可视化回放。
