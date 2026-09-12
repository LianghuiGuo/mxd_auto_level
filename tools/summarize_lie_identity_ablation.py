#!/usr/bin/env python3
"""Aggregate lie identity-decision ablation summary JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COUNT_KEYS = (
    "evaluated_frames",
    "evaluated_error_frames",
    "correct_actionable_frames",
    "wrong_actionable_frames",
    "severe_wrong_actionable_frames",
    "correct_held_frames",
    "wrong_held_frames",
    "identity_switches",
    "ranker_switches",
)


def aggregate(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    results = json.loads(path.read_text(encoding="utf-8"))["results"]
    totals = {key: sum(int(row.get(key) or 0) for row in results) for key in COUNT_KEYS}
    correct = sum(
        round(float(row.get("within_radius_active_ratio") or 0.0) * int(row["evaluated_frames"]))
        for row in results
    )
    evaluated = totals["evaluated_frames"]
    covered = totals["evaluated_error_frames"]
    wrong = covered - correct
    correct_actionable = totals["correct_actionable_frames"]
    wrong_actionable = totals["wrong_actionable_frames"]
    totals.update(
        {
            "correct_frames": correct,
            "wrong_frames": wrong,
            "no_output_frames": evaluated - covered,
            "active_accuracy": correct / max(1, evaluated),
            "covered_accuracy": correct / max(1, covered),
            "actionable_precision": correct_actionable
            / max(1, correct_actionable + wrong_actionable),
            "correct_actionable_recall": correct_actionable / max(1, correct),
            "wrong_action_block_rate": totals["wrong_held_frames"]
            / max(1, wrong),
        }
    )
    return totals, results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--output", type=Path, default=None, help="default: <run_dir>/aggregate.json"
    )
    args = parser.parse_args()
    output = args.output or args.run_dir / "aggregate.json"
    modes = ("baseline", "safety", "state", "state_safety", "motion")
    aggregate_rows: dict[str, object] = {}
    per_video: dict[str, object] = {}
    for mode in modes:
        summary = args.run_dir / f"{mode}_summary.json"
        if not summary.is_file():
            continue
        aggregate_rows[mode], per_video[mode] = aggregate(summary)
    payload = {"aggregate": aggregate_rows, "per_video": per_video}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate_rows, ensure_ascii=False, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
