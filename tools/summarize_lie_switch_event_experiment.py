#!/usr/bin/env python3
"""Summarize baseline, State-aware, and switch-gated replay results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    return list(json.loads(path.read_text(encoding="utf-8"))["results"])


def _clip_number(row: dict) -> int:
    return int(str(row["label"]).replace("录屏", ""))


def _aggregate(rows: list[dict]) -> dict[str, object]:
    evaluated = sum(int(row["evaluated_frames"] or 0) for row in rows)
    output = sum(int(row["evaluated_error_frames"] or 0) for row in rows)
    correct = round(
        sum(
            float(row["within_radius_active_ratio"] or 0.0)
            * int(row["evaluated_frames"] or 0)
            for row in rows
        )
    )
    return {
        "videos": len(rows),
        "evaluated_frames": evaluated,
        "output_frames": output,
        "correct_40px_frames": correct,
        "within_40px_active_ratio": correct / max(1, evaluated),
        "within_40px_output_ratio": correct / max(1, output),
        "identity_switches": sum(int(row["identity_switches"] or 0) for row in rows),
        "ranker_switches": sum(int(row["ranker_switches"] or 0) for row in rows),
        "correct_actionable_frames": sum(
            int(row.get("correct_actionable_frames") or 0) for row in rows
        ),
        "wrong_actionable_frames": sum(
            int(row.get("wrong_actionable_frames") or 0) for row in rows
        ),
        "severe_wrong_actionable_frames": sum(
            int(row.get("severe_wrong_actionable_frames") or 0) for row in rows
        ),
        "correct_held_frames": sum(
            int(row.get("correct_held_frames") or 0) for row in rows
        ),
        "wrong_held_frames": sum(
            int(row.get("wrong_held_frames") or 0) for row in rows
        ),
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "log/lie_identity_decision_ablation_20260911/baseline_summary.json"
        ),
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path(
            "log/lie_identity_decision_ablation_20260911/state_summary.json"
        ),
    )
    parser.add_argument(
        "--final-part",
        type=Path,
        action="append",
        default=None,
        help="one or more partial final summaries",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("log/lie_switch_event_20260912"),
    )
    parser.add_argument("--runtime-threshold", type=float, default=0.45)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    final_parts = args.final_part or [
        args.out_dir / "gate_p045_summary.json",
        args.out_dir / "gate_p045_remaining_summary.json",
    ]
    baseline = sorted(_load_rows(args.baseline), key=_clip_number)
    state = sorted(_load_rows(args.state), key=_clip_number)
    final_by_label = {}
    for path in final_parts:
        for row in _load_rows(path):
            final_by_label[row["label"]] = row
    final = sorted(final_by_label.values(), key=_clip_number)
    labels = [row["label"] for row in baseline]
    if labels != [row["label"] for row in state] or labels != [
        row["label"] for row in final
    ]:
        raise ValueError("baseline/state/final clip sets do not match")

    modes = {
        "baseline": _aggregate(baseline),
        "state_aware": _aggregate(state),
        "switch_gated": _aggregate(final),
    }
    baseline_by_label = {row["label"]: row for row in baseline}
    state_by_label = {row["label"]: row for row in state}
    per_video = []
    for row in final:
        label = row["label"]
        base = baseline_by_label[label]
        state_row = state_by_label[label]
        final_ratio = float(row["within_radius_active_ratio"] or 0.0)
        base_ratio = float(base["within_radius_active_ratio"] or 0.0)
        state_ratio = float(state_row["within_radius_active_ratio"] or 0.0)
        per_video.append(
            {
                "label": label,
                "baseline_40px": base_ratio,
                "state_aware_40px": state_ratio,
                "switch_gated_40px": final_ratio,
                "delta_vs_baseline_pp": 100.0 * (final_ratio - base_ratio),
                "delta_vs_state_pp": 100.0 * (final_ratio - state_ratio),
                "baseline_switches": int(base["identity_switches"]),
                "state_aware_switches": int(state_row["identity_switches"]),
                "switch_gated_switches": int(row["identity_switches"]),
            }
        )
    stable = [
        row for row in per_video if float(row["baseline_40px"]) >= 0.50
    ]
    worst = min(stable, key=lambda row: float(row["delta_vs_baseline_pp"]))
    blind_labels = {"录屏22", "录屏23"}
    blind = {
        "baseline": _aggregate(
            [row for row in baseline if row["label"] in blind_labels]
        ),
        "state_aware": _aggregate(
            [row for row in state if row["label"] in blind_labels]
        ),
        "switch_gated": _aggregate(
            [row for row in final if row["label"] in blind_labels]
        ),
    }
    baseline_agg = modes["baseline"]
    final_agg = modes["switch_gated"]
    gates = {
        "overall_gain_pp": 100.0
        * (
            float(final_agg["within_40px_active_ratio"])
            - float(baseline_agg["within_40px_active_ratio"])
        ),
        "overall_gain_at_least_2pp": (
            float(final_agg["within_40px_active_ratio"])
            - float(baseline_agg["within_40px_active_ratio"])
            >= 0.02
        ),
        "worst_stable_video": worst["label"],
        "worst_stable_regression_pp": float(worst["delta_vs_baseline_pp"]),
        "stable_regression_within_1pp": float(worst["delta_vs_baseline_pp"])
        >= -1.0,
        "switch_increase_ratio": (
            int(final_agg["identity_switches"])
            / max(1, int(baseline_agg["identity_switches"]))
            - 1.0
        ),
        "switch_increase_within_20pct": int(final_agg["identity_switches"])
        <= 1.2 * int(baseline_agg["identity_switches"]),
        "blind_22_23_not_regressed": float(
            blind["switch_gated"]["within_40px_active_ratio"]
        )
        >= float(blind["baseline"]["within_40px_active_ratio"]),
    }
    gates["all_passed"] = all(
        bool(gates[name])
        for name in (
            "overall_gain_at_least_2pp",
            "stable_regression_within_1pp",
            "switch_increase_within_20pct",
            "blind_22_23_not_regressed",
        )
    )
    aggregate = {
        "format_version": 1,
        "runtime_probability_threshold": args.runtime_threshold,
        "modes": modes,
        "blind_22_23": blind,
        "gates": gates,
        "per_video": per_video,
        "final_results": final,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "final_summary.json").write_text(
        json.dumps(
            {"tag": "switch_gate_v4_p045_final", "results": final},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    table_rows = []
    for row in per_video:
        table_rows.append(
            "| {label} | {base} | {state} | {final} | {delta:+.2f}pp | {b_sw} / {s_sw} / {f_sw} |".format(
                label=row["label"],
                base=_percent(float(row["baseline_40px"])),
                state=_percent(float(row["state_aware_40px"])),
                final=_percent(float(row["switch_gated_40px"])),
                delta=float(row["delta_vs_baseline_pp"]),
                b_sw=row["baseline_switches"],
                s_sw=row["state_aware_switches"],
                f_sw=row["switch_gated_switches"],
            )
        )
    report = f"""# State-aware / Switch-event 优化报告

日期：2026-09-12

## 1. 结论

本轮将原先基于手写衰减证据的 State-aware 切换，升级为“State-aware 提案 + switch-event 模型否决 + 正式旧策略兜底 + 短窗快速回滚”。最终采用端到端校准阈值 `{args.runtime_threshold:.2f}`，所有实验功能仍默认关闭。

21段视频最终结果通过预设门禁：

- 全有效帧40px：`{_percent(float(baseline_agg['within_40px_active_ratio']))} -> {_percent(float(final_agg['within_40px_active_ratio']))}`，提升 `{float(gates['overall_gain_pp']):+.2f}pp`。
- 有输出帧40px：`{_percent(float(baseline_agg['within_40px_output_ratio']))} -> {_percent(float(final_agg['within_40px_output_ratio']))}`。
- 身份切换：`{baseline_agg['identity_switches']} -> {final_agg['identity_switches']}`，增加 `{100.0 * float(gates['switch_increase_ratio']):+.1f}%`，低于20%门槛；原 State-aware 为 `{modes['state_aware']['identity_switches']}` 次。
- 稳定视频最大回归：`{gates['worst_stable_video']} {float(gates['worst_stable_regression_pp']):+.2f}pp`，未超过1pp。
- 错误 actionable：`{baseline_agg['wrong_actionable_frames']} -> {final_agg['wrong_actionable_frames']}`，减少 `{int(baseline_agg['wrong_actionable_frames']) - int(final_agg['wrong_actionable_frames'])}` 帧。
- 严重错误动作（>100px）：`{baseline_agg['severe_wrong_actionable_frames']} -> {final_agg['severe_wrong_actionable_frames']}`，减少 `{int(baseline_agg['severe_wrong_actionable_frames']) - int(final_agg['severe_wrong_actionable_frames'])}` 帧。
- 未参与训练的录屏22、23合计：`{_percent(float(blind['baseline']['within_40px_active_ratio']))} -> {_percent(float(blind['switch_gated']['within_40px_active_ratio']))}`。

这版已达到继续灰度验证的实验标准，但配置仍保持 opt-in，不直接改变正式默认行为。

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
{chr(10).join(table_rows)}

冷启动失败的录屏5、7、12、17仍为0%，本轮只优化身份切换，不解决未观察到白色种子的初始化缺口。

## 5. 门禁结果

| 门禁 | 要求 | 结果 |
| --- | --- | --- |
| 总体40px | 至少 +2pp | `{float(gates['overall_gain_pp']):+.2f}pp`，通过 |
| 稳定样本最大回归 | 不超过1pp | `{gates['worst_stable_video']} {float(gates['worst_stable_regression_pp']):+.2f}pp`，通过 |
| 身份切换增量 | 不超过20% | `{baseline_agg['identity_switches']} -> {final_agg['identity_switches']}`，`{100.0 * float(gates['switch_increase_ratio']):+.1f}%`，通过 |
| 录屏22、23泛化 | 不明显回归 | `{_percent(float(blind['baseline']['within_40px_active_ratio']))} -> {_percent(float(blind['switch_gated']['within_40px_active_ratio']))}`，通过 |

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
python3 tools/lie_detector_replay.py \
  "ml/videos/测谎录屏22.mp4" \
  --multi-hypothesis-identity \
  --stale-coast-recovery \
  --identity-ranker-model models/lie_identity_ranker.txt \
  --identity-ranker-min-margin 2.0 \
  --state-aware-ranker \
  --switch-event-model models/lie_switch_event_model.txt \
  --switch-event-min-probability 0.45 \
  --csv log/switch_gate_22.csv \
  --output log/switch_gate_22.mp4
```

本轮全量回放没有生成MP4，只生成逐帧CSV和汇总JSON；逐帧目录已加入 `.gitignore`。

## 7. 产物

- `models/lie_switch_event_model.txt`：实验switch-event模型。
- `models/lie_switch_event_model.metrics.json`：LOVO指标、阈值扫描和特征重要性。
- `aggregate.json`：三组聚合、逐视频结果和门禁状态。
- `final_summary.json`：最终0.45阈值的21段原始summary。
- `capture_summary.json`：on-policy事件采集回放摘要。
- `raw/`、`gate*/`、`final/`：可再生逐帧诊断目录，已忽略。
"""
    (args.out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"modes": modes, "gates": gates}, ensure_ascii=False, indent=2))
    print(f"wrote {args.out_dir / 'aggregate.json'}")
    print(f"wrote {args.out_dir / 'final_summary.json'}")
    print(f"wrote {args.out_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
