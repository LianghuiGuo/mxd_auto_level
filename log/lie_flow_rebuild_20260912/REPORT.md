# YOLO + 稀疏光流重构实验报告

日期：2026-09-12

## 1. 结论

本轮完成了 `YOLO + 前关联稀疏 Lucas-Kanade` 的可运行实现、回放消融和闭环验证。YOLO仍负责候选发现、冷启动和重捕获；LK只提供短时运动观测，不使用绿色光标，也不替代检测器。

最终结论：

- 保留稀疏LK观察器、lost-only关联、严格target-only flow-coast、诊断字段和Switch-event特征接口。
- 三个线上开关继续默认关闭，不替换现有Identity Ranker和正式Switch-event模型。
- lost-only光流关联在21段闭环中净回归，不上线。
- 严格target-only flow-coast通过精度和安全门禁，但收益只有2帧，平均处理延迟增加约3.8%，不值得默认开启。
- preflow Switch-event临时模型在焦点集明显回归，不复制到 `models/`。
- FlowNet和PWC-Net未引入；RAFT只保留为未来离线教师候选。

当前默认配置保持：

```yaml
lie_detector:
  preassociation_flow: false
  flow_association: false
  flow_coast: false
```

## 2. 实现内容

### 2.1 前关联稀疏LK观察器

新增 `src/engine/LiePreAssociationFlow.py`：

1. 在上一帧每条轨迹的环形局部区域选取角点，主动排除中心区域，避免依赖绿色光标。
2. 批量执行正向和反向Pyramidal Lucas-Kanade。
3. 使用forward-backward error、LK error过滤漂移点。
4. 对每条轨迹用RANSAC拟合局部相似变换，输出中心位移、旋转、尺度、覆盖率、内点率和局部拟合误差。
5. 对所有物体中心再拟合群体相似变换，输出每条轨迹相对“蜘蛛网整体运动”的残差。

这对应“整体相对静止，找不合理运动点”的思路，但只把光流作为观测证据，不直接赋予REAL身份。

### 2.2 光流关联

光流只允许帮助已经lost的轨迹重捕获：

- 可见轨迹不允许LK改写正常Hungarian关联。
- 光流中心只以有限权重参与距离代价。
- 置信度和光流/Kalman分歧必须通过门控。

首版让光流参与所有关联时，重复星形纹理会产生“自信但错误”的跟随；录屏20曾回归6.38pp。因此正式实验收紧为lost-only。

### 2.3 target-only flow-coast

flow-coast只允许当前REAL在YOLO短暂漏检时使用，并同时满足：

- 最多3个lost帧；
- flow confidence至少0.42；
- forward-backward error不超过0.80；
- inlier ratio至少0.65；
- coverage至少0.35；
- local fit error不超过1.50；
- LK中心与Kalman预测相差不超过14px；
- LK中心附近不存在YOLO detection；
- 预测中心仍在画面内。

光流只改写当前 `statePre` 的中心，不调用Kalman `correct()`，不写回 `statePost`，不降低协方差；轨迹仍保持 `predicted_only=true`，鼠标安全继续受原位置不确定性控制。

### 2.4 Switch-event接口与诊断

Switch-event可按模型feature schema自动启用preflow观察，并读取current/challenger/delta三组特征：

- confidence、forward-backward error、inlier ratio、coverage；
- local fit error、group residual、group confidence；
- 是否使用flow association、是否flow-coast。

回放工具新增三个开关及逐帧/汇总诊断：

```text
--preassociation-flow
--flow-association
--flow-coast
```

## 3. 21段闭环消融

正式基准为当前Identity Ranker + State-aware + Switch-event，21段共7,706个评估帧。录屏5和7有评估帧但没有有效输出，仍按正式口径计入总分母。

| 方案 | 40px正确帧 | 全评估帧40px | 身份切换 | Ranker切换 | 错误actionable | 严重错误 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 正式基准 | 5,616 | 72.878% | 110 | 80 | 1,058 | 225 | 保留 |
| lost-only flow association | 5,613 | 72.839% | 109 | 79 | 1,061 | 225 | 净回归 |
| target-only flow-coast | 5,618 | 72.904% | 110 | 80 | 1,056 | 225 | 小幅正向但不默认开 |

lost-only关联的唯一精度变化为录屏20 `-0.765pp`，没有任何视频获得净提升。

target-only flow-coast只有三个视频改变40px结果：

| 视频 | 40px变化 | 正确帧变化 | 错误actionable变化 | 严重错误变化 |
| --- | ---: | ---: | ---: | ---: |
| 录屏10 | -0.556pp | -2 | +2 | 0 |
| 录屏13 | +0.494pp | +2 | -2 | 0 |
| 录屏22 | +0.472pp | +2 | -2 | 0 |

它满足单视频最大回归不超过1pp、安全指标不恶化和录屏22/23泛化门禁，但全量只增加2个正确帧。

## 4. 延迟

为了避免跨轮次机器状态不可比，使用当前代码、同机串行补跑了21段正式基准。平均处理时间按全部11,042个回放帧加权；P95采用逐视频P95的中位数和最大值描述。

| 方案 | 平均处理时间 | 相对变化 | 视频P95中位数 | 最大视频P95 |
| --- | ---: | ---: | ---: | ---: |
| 当前正式基准 | 61.16ms/frame | - | 73.46ms | 79.32ms |
| target-only flow-coast | 63.51ms/frame | +2.35ms / +3.8% | 75.33ms | 82.32ms |
| lost-only flow association | 69.96ms/frame | +8.80ms / +14.4% | 80.94ms | 92.05ms |

flow-coast虽然只维护REAL的下一帧角点，但为了在漏检前建立观察状态，每帧仍需计算该目标的LK；其2帧收益不足以覆盖延迟和维护成本。

## 5. preflow Switch-event模型

在19段训练视频上采集observer-only特征，录屏22/23只用于独立闭环。future-window LOVO结果：

| 特征组 | 阈值 | 提案范围precision | 提案范围switch recall | 全basic precision | 全basic recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy | 0.693 | 92.86% | 36.62% | 97.92% | 36.15% |
| Rotation | 0.700 | 92.59% | 35.21% | 97.83% | 34.62% |
| Preflow | 0.741 | 92.00% | 32.39% | 98.20% | 41.92% |

Preflow提高了全basic范围的离线召回，但没有改善当前State-aware提案范围。使用阈值0.741对焦点集 `1,2,4,6,11,19,20,22,23` 做闭环：

| 指标 | 正式基准 | Preflow模型 | 变化 |
| --- | ---: | ---: | ---: |
| 40px正确帧 | 2,857 | 2,843 | -14 |
| 40px准确率 | 85.925% | 85.504% | -0.421pp |
| 身份切换 | 46 | 44 | -2 |
| Ranker切换 | 31 | 26 | -5 |
| 错误actionable | 466 | 476 | +10 |
| 严重错误 | 25 | 30 | +5 |

其中录屏22回归 `-1.887pp`，录屏6回归 `-1.377pp`且严重错误增加15帧，已触发止损条件，因此没有继续跑全21段，也没有替换正式模型。

离线指标与闭环相反，说明当前瓶颈不是“有没有光流特征”，而是逐事件future-window标签没有稳定表达一次切换对后续轨迹和鼠标动作的累计收益。

## 6. 默认配置决策

本轮没有把任何光流路径默认上线：

- `preassociation_flow=false`：单独观察不改变输出，只增加计算；当前正式模型也不需要其schema。
- `flow_association=false`：精度和错误actionable均回归，延迟增加明显。
- `flow_coast=false`：安全版方向正确，但仅增加2帧，平均延迟增加3.8%，性价比不足。
- 正式 `models/lie_identity_ranker.txt` 和 `models/lie_switch_event_model.txt` 均保持不变。

需要手工复现实验时，可以显式启用：

```bash
python3 tools/lie_detector_replay.py \
  "ml/videos/测谎录屏22.mp4" \
  --multi-hypothesis-identity \
  --stale-coast-recovery \
  --identity-ranker-model models/lie_identity_ranker.txt \
  --identity-ranker-min-margin 2.0 \
  --state-aware-ranker \
  --switch-event-model models/lie_switch_event_model.txt \
  --switch-event-min-probability 0.45 \
  --flow-coast \
  --output log/lie_flow_coast_22.mp4 \
  --csv log/lie_flow_coast_22.csv
```

`--flow-coast`和`--flow-association`都会隐式启用前关联观察器。只采集特征而不改变轨迹时使用 `--preassociation-flow`。

## 7. 后续优先级

1. 把训练单位从单帧event改为switch episode，目标直接预测“未来累计误差：keep减switch”，并对严重错误动作增加损失权重。
2. 先让episode模型在不放宽State-aware提案范围时通过LOVO，再单独研究如何利用Preflow提高的全basic召回。
3. 仅在离线标签不足时考虑RAFT教师：生成更密集、更稳的伪标签来蒸馏当前稀疏LK特征；不把RAFT直接放进线上热路径。
4. 扩充录屏6、19、22、23这类不同失败形态，现有19段训练集不足以支撑高维preflow模型稳定泛化。
5. 保持绿色光标只用于标签和评估，任何端到端模型都不得读取该信号。

## 8. 验证

- 完整单元测试：`91`项通过。
- Python语法检查通过。
- `git diff --check`通过。
- 所有实验CSV、临时模型和汇总均为可再生产物并被 `.gitignore` 忽略；本报告保留。
