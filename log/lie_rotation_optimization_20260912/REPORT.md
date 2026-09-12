# 图像旋转检测与 Switch-event 优化报告

日期：2026-09-12

## 1. 结论

本轮完成了旋转观测链修复、current/challenger 图像运动特征接入、瞬时标签和未来窗口标签两组消融，以及闭环回放。最终**不替换当前线上 Switch-event 模型**，默认配置继续使用：

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

原因不是旋转信号无效，而是现有19段训练视频中，旋转特征没有在闭环门禁上形成稳定净增益：

- 瞬时标签的 optical + multi-lag 模型，21段40px从 `72.88%` 降至 `72.80%`，切换从 `110` 增至 `123`，严重错误动作从 `225` 增至 `241`。
- 未来窗口 rotation-only 模型在困难集阈值0.55时总体正确帧与当前模型持平，但录屏1回归 `-1.68pp`、录屏23回归 `-2.08pp`。
- 阈值提高到0.65后，所测困难集正确帧增加5帧、错误 actionable 减少9帧、严重错误减少5帧、切换减少7次；但录屏6仍回归 `-1.38pp`，录屏19回归 `-1.02pp`，超过单视频1pp门禁。

因此本轮保留通用实现和训练能力，但不发布任何实验模型到 `models/`，也不修改默认配置。

## 2. 实现改动

### 2.1 关联后旋转观测

原来的 `collective_rotation_residual` 在 track 关联前计算，而真实 YOLO/轮廓 detection 的方向只会在关联时通过极坐标描述子匹配获得，因此线上真实路径无法可靠得到本帧 peer rotation residual。

现在每个 track 明确保存：

- `last_rotation_delta`：本次有效关联得到的旋转量；
- `rotation_confidence`：极坐标相关峰置信度；
- `rotation_valid`：本帧旋转是否可观测；
- `peer_rotation_residual`：本轨迹相对至少3条有效轨迹的置信度加权旋转中位数的偏差。

该 residual 使用独立字段，只进入实验 Switch-event 特征；没有覆盖正式43维 Identity Ranker 的旧字段，也没有直接加入手写四线索分数。track 丢失时这些瞬时量会清零，避免旧观测泄漏到 coast 帧。

### 2.2 Switch-event 图像运动特征

新增 `SwitchMotionFeatureHistory`，按帧同步保存去除绿色光标后的灰度图与可见 track 位置，只为 current 和 challenger 计算：

- `optical_spin_abs_norm/confidence/evidence`；
- `multilag_spin_abs_norm/confidence/consistency`；
- `spin_estimator_agreement`；
- `spin_joint_confidence`。

Switch-event 同时获得 current、challenger 和 challenger-current 差值。模型按 feature name 自动开启所需计算：当前正式模型没有这些特征，因此默认运行不会增加光流或 multi-lag 开销。

回放工具增加 `--switch-motion-features`，用于在没有加载新模型时旁路采集全部运动特征。

### 2.3 未来窗口 outcome 标签

瞬时标签只能回答“当前帧哪条轨迹更接近绿色光标”，无法惩罚切换后2至5帧马上回滚的短命切换。回放工具现在额外写入未来12帧的离线统计：

- current/challenger 可见帧数；
- 40px内正确率；
- 60px外错误率；
- 中位误差。

训练脚本增加 `--label-policy future-window`。当前实验标签要求：

- keep：未来 current 正确率至少60%，challenger 正确率不超过25%；
- switch：未来 challenger 正确率至少60%，current 正确率不超过25%；
- 至少5个未来帧；其余事件排除。

绿色光标仍然只用于离线监督，所有 future 字段都不属于模型输入。

## 3. 离线消融

### 3.1 瞬时标签

训练集：549个有效 basic-qualified 事件，其中247 switch、302 keep。

| 特征组 | 阈值 | 提案 Precision | 提案 Recall | 提案 False Switch | 全事件 Recall | Keep Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy | 0.322 | 94.87% | 90.24% | 2 | 91.90% | 90.40% |
| + tracker rotation | 0.346 | 94.87% | 90.24% | 2 | 91.90% | 90.40% |
| + optical | 0.403 | 97.37% | 90.24% | 1 | **93.93%** | 90.40% |
| + optical + multi-lag | 0.438 | 97.30% | 87.80% | 1 | 92.71% | **91.06%** |

光流置信度是最强新增特征，说明“局部点阵能否被稳定刚体变换解释”比角度值本身更有效。multi-lag 更偏向抑制误切，但会损失部分召回。

### 3.2 未来窗口标签

训练集：655个有效事件，其中330 switch、325 keep。

| 特征组 | 阈值 | 提案 Precision | 提案 Recall | 提案 False Switch | 全事件 Precision | 全事件 Recall | Keep Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy | 0.575 | 94.87% | 72.55% | 2 | **94.47%** | 67.27% | **96.00%** |
| + tracker rotation | 0.547 | **95.00%** | **74.51%** | 2 | 92.24% | 68.48% | 94.15% |
| + optical | 0.641 | 94.87% | 72.55% | 2 | 90.46% | 71.82% | 92.31% |
| + optical + multi-lag | 0.641 | 94.87% | 72.55% | 2 | 91.39% | **73.94%** | 92.92% |

未来窗口标签显著降低了对短命切换的偏好。严格提案范围内，成本最低的 tracker rotation 组表现最好；光流和 multi-lag 的主要收益出现在更宽的 basic-qualified 事件，而当前运行时仍要求 State-aware 证据先达到阈值，因此这些收益没有充分转化为闭环收益。

## 4. 闭环结果

### 4.1 瞬时标签 optical + multi-lag，阈值0.45，全21段

| 指标 | 当前默认 | 实验模型 | 差异 |
| --- | ---: | ---: | ---: |
| 全有效帧40px | 72.88% | 72.80% | -0.08pp |
| 有输出帧40px | 84.12% | 84.03% | -0.09pp |
| 正确40px帧 | 5,616 | 5,610 | -6 |
| 身份切换 | 110 | 123 | +13 |
| 错误 actionable | 1,058 | 1,066 | +8 |
| 严重错误动作 | 225 | 241 | +16 |

该模型没有通过门禁。

### 4.2 未来窗口 rotation-only，阈值0.55，13段困难/盲测集

| 指标 | 当前默认 | 实验模型 | 差异 |
| --- | ---: | ---: | ---: |
| 正确40px帧 | 4,365 | 4,365 | 0 |
| 身份切换 | 81 | 81 | 0 |
| 错误 actionable | 837 | 835 | -2 |
| 严重错误动作 | 109 | 108 | -1 |

总体持平，但录屏1 `-1.68pp`、录屏23 `-2.08pp`，未通过稳定性门禁。

### 4.3 未来窗口 rotation-only，阈值0.65，9段回归/盲测集

| 指标 | 当前默认 | 实验模型 | 差异 |
| --- | ---: | ---: | ---: |
| 40px准确率 | 84.26% | 84.39% | +0.13pp |
| 正确40px帧 | 3,142 | 3,147 | +5 |
| 身份切换 | 52 | 45 | -7 |
| 错误 actionable | 585 | 576 | -9 |
| 严重错误动作 | 53 | 48 | -5 |

总体方向正确，但录屏6 `-1.38pp`、录屏19 `-1.02pp`，仍未通过单视频最大回归1pp门禁。

## 5. 性能与兼容性

- 当前正式模型不声明新增 motion 特征，因此不会启用额外光流/multi-lag 计算。
- 新模型会根据 feature names 自动开启 optical 或 multi-lag。
- 单视频烟测录屏22在同时采集 optical + multi-lag 时平均处理约 `63.49ms/frame`；这是离线调试路径，不是当前正式路径的新增成本。
- 完整测试：87项通过。
- Python语法检查和 `git diff --check` 通过。

## 6. 下一步建议

当前瓶颈已经从“是否检测到旋转”进一步收敛到“如何为闭环切换定义监督目标”。下一轮优先级建议：

1. 扩充视频，特别是录屏1、6、19、23这类不同失败形态；当前19段训练视频不足以稳定学习跨视频的 episode outcome。
2. 将训练单位从逐帧事件改为 switch episode，每个 current/challenger 连续区间只产生一个样本，标签使用切换后12至20帧的累计收益、回滚次数和严重错误帧。
3. 训练直接预测 `future_error_keep - future_error_switch` 的收益模型，而不是只做当前帧二分类。
4. 只有 episode 模型在LOVO和21段闭环都通过门禁后，再考虑替换 `models/lie_switch_event_model.txt`。
5. 不建议继续单纯调阈值，也不建议把 motion ranker 直接替换 Identity Ranker。

## 7. 复现入口

旁路采集旋转/光流特征：

```bash
python3 tools/replay_all_lie_videos.py \
  --tag rotation_switch_future_capture \
  --clips 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20 \
  --out-dir log/lie_rotation_optimization_20260912/future_capture/replay \
  --summary log/lie_rotation_optimization_20260912/future_capture_summary.json \
  --switch-events-dir log/lie_rotation_optimization_20260912/future_capture/raw \
  --switch-label-window 12 \
  --multi-hypothesis-identity \
  --stale-coast-recovery \
  --identity-ranker-model models/lie_identity_ranker.txt \
  --identity-ranker-min-margin 2.0 \
  --state-aware-ranker \
  --switch-motion-features
```

训练未来窗口 rotation-only 消融模型：

```bash
python3 ml/train_lie_switch_event_from_replays.py \
  --events log/lie_rotation_optimization_20260912/future_capture/raw \
  --label-policy future-window \
  --feature-set rotation \
  --out log/lie_rotation_optimization_20260912/future_models/switch_rotation.txt
```

实验模型与逐帧文件均为可再生产物，已加入 `.gitignore`；报告保留在版本库中。
