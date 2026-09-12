# State-aware / Switch-event 优化报告

日期：2026-09-12

## 1. 结论

本轮将原先基于手写衰减证据的 State-aware 切换，升级为“State-aware 提案 + switch-event 模型否决 + 正式旧策略兜底 + 短窗快速回滚”。最终采用端到端校准阈值 `0.45`。完成实验验收后，`state_aware_ranker` 与 switch-event 模型已写入默认配置；Safety 和 motion corroboration 仍默认关闭。

21段视频最终结果通过预设门禁：

- 全有效帧40px：`68.18% -> 72.88%`，提升 `+4.70pp`。
- 有输出帧40px：`78.70% -> 84.12%`。
- 身份切换：`94 -> 110`，增加 `+17.0%`，低于20%门槛；原 State-aware 为 `155` 次。
- 稳定视频最大回归：`录屏4 -0.95pp`，未超过1pp。
- 错误 actionable：`1416 -> 1058`，减少 `358` 帧。
- 严重错误动作（>100px）：`513 -> 225`，减少 `288` 帧。
- 未参与训练的录屏22、23合计：`73.28% -> 87.75%`。

这版已通过预设门禁，并已按当前项目决策设为默认身份切换路径。

## 2. 为什么重做 switch-event 数据

最初从 listwise ranker 数据构造 `current vs challenger` 样本，并对 ranker 分数做按视频交叉拟合。该数据只有15个“ranker建议切换、实际应保持”的有效负例，无法覆盖 State-aware 自己切错后产生的闭环状态，离线得到的100% precision不可信。

因此改为重放旧19段视频，直接采集 State-aware 每次决策前的 on-policy 快照。监督标签只由绿色光标生成，模型输入严格限制为线上可见信息：

- current/challenger 的轨迹健康度、外观、运动、关联质量；
- ranker分差、当前排名、证据和路径支持；
- recovery、short coast、current suspicious 等状态。

标签采用更保守的间隔：一方误差不超过40px，另一方至少60px；两者都对、都错或距离接近的事件全部丢弃。最终得到234个 switch、302个 keep 的有效 basic-qualified 样本，按连续事件降权，并用按视频留一验证。录屏22、23没有参与训练或阈值选择。

## 3. 模型与决策链

离线LOVO在原 State-aware 已达到提交门槛的严格事件上：

- precision：97.44%；
- switch recall：95.00%；
- false switch：1 / 50。

LOVO自动选择的分类阈值为0.335，但闭环回放显示边缘概率会伤害稳定样本，因此端到端校准到0.45。最终决策链为：

```text
ranker产生challenger
  -> State-aware累计证据
  -> visible current要求ranker delta >= 0.50
  -> switch-event probability >= 0.45
  -> 允许提前切换
  -> 若模型否决，正式margin + 连续票 + path规则仍可兜底
  -> 提前切换后旧身份短窗重新成为top1时允许快速回滚
```

兜底规则很重要：模型只负责批准更快的恢复，不会永久删除正式旧策略本来能够完成的切换。快速回滚则解决“切换当帧challenger正确、下一帧旧track重新关联真值”的瞬时标签问题。

## 4. 21段逐视频结果

| 视频 | Baseline | 原State-aware | Switch-gated | 相对Baseline | 切换数 B / State / Final |
| --- | ---: | ---: | ---: | ---: | ---: |
| 录屏1 | 88.55% | 86.03% | 88.55% | +0.00pp | 2 / 4 / 2 |
| 录屏2 | 93.59% | 93.59% | 93.59% | +0.00pp | 5 / 5 / 5 |
| 录屏3 | 76.36% | 84.24% | 86.36% | +10.00pp | 14 / 17 / 15 |
| 录屏4 | 98.73% | 96.52% | 97.78% | -0.95pp | 3 / 8 / 5 |
| 录屏5 | 0.00% | 0.00% | 0.00% | +0.00pp | 0 / 0 / 0 |
| 录屏6 | 51.79% | 69.70% | 68.60% | +16.80pp | 8 / 10 / 9 |
| 录屏7 | 0.00% | 0.00% | 0.00% | +0.00pp | 0 / 0 / 0 |
| 录屏8 | 69.25% | 64.94% | 69.54% | +0.29pp | 4 / 13 / 5 |
| 录屏9 | 63.46% | 74.09% | 73.75% | +10.30pp | 11 / 11 / 11 |
| 录屏10 | 69.72% | 72.78% | 73.89% | +4.17pp | 5 / 11 / 9 |
| 录屏11 | 89.01% | 84.07% | 88.46% | -0.55pp | 3 / 7 / 3 |
| 录屏12 | 0.00% | 0.00% | 0.00% | +0.00pp | 0 / 0 / 0 |
| 录屏13 | 89.63% | 85.68% | 90.12% | +0.49pp | 2 / 8 / 2 |
| 录屏14 | 92.32% | 91.31% | 91.72% | -0.61pp | 6 / 10 / 10 |
| 录屏15 | 84.30% | 87.79% | 88.57% | +4.26pp | 7 / 10 / 8 |
| 录屏16 | 87.73% | 85.13% | 86.99% | -0.74pp | 2 / 6 / 4 |
| 录屏17 | 0.00% | 0.00% | 0.00% | +0.00pp | 0 / 0 / 0 |
| 录屏19 | 66.24% | 80.46% | 79.70% | +13.45pp | 5 / 9 / 5 |
| 录屏20 | 76.28% | 81.89% | 84.44% | +8.16pp | 5 / 9 / 7 |
| 录屏22 | 64.86% | 86.32% | 87.50% | +22.64pp | 5 / 8 / 3 |
| 录屏23 | 81.52% | 88.22% | 87.99% | +6.47pp | 7 / 9 / 7 |

冷启动失败的录屏5、7、12、17仍为0%，本轮只优化身份切换，不解决未观察到白色种子的初始化缺口。

## 5. 门禁结果

| 门禁 | 要求 | 结果 |
| --- | --- | --- |
| 总体40px | 至少 +2pp | `+4.70pp`，通过 |
| 稳定样本最大回归 | 不超过1pp | `录屏4 -0.95pp`，通过 |
| 身份切换增量 | 不超过20% | `94 -> 110`，`+17.0%`，通过 |
| 录屏22、23泛化 | 不明显回归 | `73.28% -> 87.75%`，通过 |

## 6. 启用方式

在线配置示例：

```yaml
lie_detector:
  identity_ranker_model: "models/lie_identity_ranker.txt"
  identity_ranker_min_margin: 2.0
  multi_hypothesis_identity: true
  stale_coast_recovery: true
  state_aware_ranker: true
  switch_event_model: "models/lie_switch_event_model.txt"
  switch_event_min_probability: 0.45
```

单视频回放：

```bash
python3 tools/lie_detector_replay.py   "ml/videos/测谎录屏22.mp4"   --multi-hypothesis-identity   --stale-coast-recovery   --identity-ranker-model models/lie_identity_ranker.txt   --identity-ranker-min-margin 2.0   --state-aware-ranker   --switch-event-model models/lie_switch_event_model.txt   --switch-event-min-probability 0.45   --csv log/switch_gate_22.csv   --output log/switch_gate_22.mp4
```

本轮全量回放没有生成MP4，只生成逐帧CSV和汇总JSON；逐帧目录已加入 `.gitignore`。

## 7. 产物

- `models/lie_switch_event_model.txt`：实验switch-event模型。
- `models/lie_switch_event_model.metrics.json`：LOVO指标、阈值扫描和特征重要性。
- `aggregate.json`：三组聚合、逐视频结果和门禁状态。
- `final_summary.json`：最终0.45阈值的21段原始summary。
- `capture_summary.json`：on-policy事件采集回放摘要。
- `default_compat_summary.json`：默认开关关闭时的历史Baseline兼容性复核。
- `raw/`、`gate*/`、`final/`：可再生逐帧诊断目录，已忽略。
