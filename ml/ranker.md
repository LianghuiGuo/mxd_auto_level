# Lie Identity Ranker 使用说明

## 当前状态

当前 43 特征的 Lie Identity Ranker 已接入项目主运行链路，并在默认配置中启用。

- 当前分支：`main`
- 接入提交：`decae3b add lie detector`
- 远端状态：该提交已同步至 `origin/main`
- 正式模型：`models/lie_identity_ranker.txt`
- 默认切换分差阈值：`2.00`

这里的“上线”表示代码和模型已经进入本项目的主分支，并被默认配置启用。其他机器或运行环境是否已部署，需要确认对应环境是否已经拉取到提交 `decae3b` 或更新版本。

本轮实验产生的 motion ranker 没有替换正式模型。它的端到端结果为 `65.76%`，低于当前 ranker 的 `67.54%`。因此目前仍建议使用 `models/lie_identity_ranker.txt`，不要把实验模型 `similarity_optical_multilag.txt` 配入正式运行环境。

2026-09-11 的身份决策消融验证了 Safety-only，但原始 State-aware 因切换过多、稳定视频回归而不建议单独启用。2026-09-12 新增 switch-event 门控、正式规则兜底和短窗快速回滚后，21段回放通过全部门禁；当前默认配置已启用该组合，详见 `log/lie_switch_event_20260912/REPORT.md`。

## 默认配置

配置位于 `config/config_default.yaml`：

```yaml
lie_detector:
  enabled: true
  model: "models/lie_shape_yolo_manual.pt"
  multi_hypothesis_identity: true
  stale_coast_recovery: true
  identity_ranker_model: "models/lie_identity_ranker.txt"
  identity_ranker_min_margin: 2.00
  identity_safety: false
  state_aware_ranker: true
  motion_corroboration_model: ""
  switch_event_model: "models/lie_switch_event_model.txt"
  switch_event_min_probability: 0.45
```

运行时由 `src/engine/LieDetectorRuntime.py` 读取上述配置、加载检测器和 ranker，并创建 `LieDetectorTracker`。

## 日常使用

### 使用 UI

在项目根目录执行：

```bash
python3 -m src.main
```

进入界面后点击 `Start`，或者按 `F1` 启动自动挂机。测谎窗口出现后，程序会自动进入测谎处理流程。

### 使用命令行

```bash
python3 -m src.engine.MapleStoryAutoLevelUp
```

命令行默认使用 `custom` 配置，并依次合并基础配置、平台配置和自定义配置。自定义配置没有填写 `lie_detector` 时，会继承默认的 ranker 配置；填写同名字段时，则以自定义值覆盖默认值。

指定其他自定义配置，例如 `config/config_cleric.yaml`：

```bash
python3 -m src.engine.MapleStoryAutoLevelUp --cfg cleric
```

## 运行行为

检测到“谎言探测仪”窗口后，程序会：

1. 连续确认测谎面板确实存在。
2. 暂停正常的键盘挂机操作。
3. 使用 YOLO 检测当前画面中的所有候选图形。
4. 由 tracker 对候选进行跨帧关联并维护身份轨迹。
5. 由 ranker 根据每条候选轨迹的历史特征进行评分。
6. 只有新候选连续多帧领先，且与第二名的分差达到 `2.0`，才允许切换目标。
7. 只在目标位置可信且可操作时移动鼠标。
8. 面板结束后自动寻找并点击“确认”。
9. 恢复正常挂机。

Ranker 并不会在每一帧直接强制覆盖 tracker。它提供的是带有 margin 和连续投票保护的身份切换信号。

### 实验性 Safety 灰度

如果需要灰度身份安全门，在自定义配置中设置：

```yaml
lie_detector:
  identity_safety: true
```

该开关只改变鼠标是否执行，不改变跟踪轨迹和 ranker 切换。21段离线结果中，它保留 `93.13%` 的正确动作，拦截 `45.99%` 的错误动作，并将超过100px的严重错误动作从513帧降到134帧。

不要单独启用原始 State-aware 或把 motion 实验模型直接配入正式环境：

```yaml
lie_detector:
  state_aware_ranker: true
  motion_corroboration_model: "ml/lie_motion_ablation/models/similarity_optical_multilag.txt"
```

需要灰度优化后的 State-aware / switch-event 组合时，使用：

```yaml
lie_detector:
  state_aware_ranker: true
  switch_event_model: "models/lie_switch_event_model.txt"
  switch_event_min_probability: 0.45
```

该组合的21段全有效帧40px准确率从 `68.18%` 提升到 `72.88%`，身份切换从 `94` 增至 `110`；相比原始 State-aware 的 `155` 次明显降低。稳定视频最大回归为 `-0.95pp`，录屏22、23两个未参与训练的视频合计从 `73.28%` 提升到 `87.75%`。

switch-event 模型只门控 State-aware 的提前切换。若模型否决，正式的 margin、连续票和 identity path 规则仍可作为慢速兜底，因此不会永久删除正式策略原本可以完成的恢复。该组合现在是项目默认配置；如需回退，可将 `state_aware_ranker` 设为 `false` 并清空 `switch_event_model`。

`identity_ranker_min_margin` 控制切换的保守程度：

- 增大阈值：切换更保守，误切换较少，但可能错过正确切换。
- 减小阈值：切换更敏感，目标恢复更积极，但误切换风险更高。
- 当前正式配置推荐保持为 `2.00`。

## 确认是否加载成功

正常初始化后会输出：

```text
[Lie Detector] Mouse-follow runtime ready.
```

如果模型文件缺失、模型加载失败或初始化过程中出现其他异常，会输出：

```text
[Lie Detector] Disabled because initialization failed: ...
```

出现上述错误时，整个测谎运行时会被禁用，而不只是 ranker 被禁用。

诊断回放中还可以观察：

- 每个候选的 `rk=...` ranker 分数
- 当前 top1 与 top2 的 margin
- ranker 当前推荐的候选
- 是否实际触发 ranker switch

## 单视频离线验证

可以对录像进行验证，而不实际控制鼠标：

```bash
python3 tools/lie_detector_replay.py \
  "ml/videos/测谎录屏1.mp4" \
  --identity-ranker-model models/lie_identity_ranker.txt \
  --identity-ranker-min-margin 2.0 \
  --multi-hypothesis-identity \
  --stale-coast-recovery \
  --output log/ranker_check.mp4 \
  --csv log/ranker_check_22.csv
```

生成结果：

- `log/ranker_check.mp4`：包含检测框、轨迹和 ranker 信息的可视化视频。
- `log/ranker_check_22.csv`：逐帧诊断数据。

## 关闭 Ranker

仅关闭 ranker、保留其余测谎逻辑：

```yaml
lie_detector:
  identity_ranker_model: ""
```

完全关闭自动测谎处理：

```yaml
lie_detector:
  enabled: false
```

## 相关文件

- `models/lie_identity_ranker.txt`：当前正式使用的 LightGBM 文本模型。
- `models/lie_identity_ranker.metrics.json`：模型评估指标。
- `models/lie_identity_ranker.lovo.json`：留一视频评估结果。
- `models/lie_switch_event_model.txt`：实验性 current-vs-challenger 切换门控模型。
- `models/lie_switch_event_model.metrics.json`：switch-event LOVO、阈值和特征重要性。
- `src/engine/LieIdentityRanker.py`：特征计算和模型推理逻辑。
- `src/engine/LieSwitchEventModel.py`：switch-event 特征和纯 Python LightGBM 推理。
- `src/engine/LieDetectorTracker.py`：轨迹维护以及 ranker 切换决策。
- `src/engine/LieDetectorRuntime.py`：在线运行时配置和初始化入口。
- `tools/lie_detector_replay.py`：单视频诊断回放工具。
- `tools/replay_all_lie_videos.py`：全量视频回放评估工具。
- `log/lie_ranker_ab_20260911/REPORT.md`：当前 ranker 的 A/B 评估报告。
- `log/lie_motion_ablation_20260911/REPORT.md`：运动特征消融实验报告。
- `log/lie_switch_event_20260912/REPORT.md`：State-aware / switch-event 最终实验报告。
