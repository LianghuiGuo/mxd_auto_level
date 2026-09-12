"""Serially replay every lie-detector clip and write one comparable summary.

Runs are serial on purpose: sharing the YOLO/ONNX session across parallel
processes makes per-frame detections non-deterministic, which shows up as
several points of noise in the aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

METRIC_KEYS = (
    "within_40px_ratio",
    "within_radius_active_ratio",
    "median_error_px",
    "p95_error_px",
    "identity_switches",
    "ranker_switches",
    "evaluated_frames",
    "evaluated_error_frames",
    "prediction_coverage_ratio",
    "acquired_frame_ratio",
    "actionable_frame_ratio",
    "position_actionable_frame_ratio",
    "correct_actionable_frames",
    "wrong_actionable_frames",
    "severe_wrong_actionable_frames",
    "correct_held_frames",
    "wrong_held_frames",
    "actionable_precision",
    "correct_actionable_recall",
    "mean_processing_ms",
    "p95_processing_ms",
    "flow_observation_frames",
    "flow_association_matches",
    "flow_coast_frames",
)


def discover_clip_numbers(videos: Path) -> list[int]:
    numbers = []
    for path in videos.glob("测谎录屏*.mp4"):
        match = re.fullmatch(r"测谎录屏(\d+)", path.stem)
        if match is not None:
            numbers.append(int(match.group(1)))
    return sorted(set(numbers))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="iteration name stored in the summary")
    parser.add_argument("--videos", type=Path, default=Path("ml/videos"))
    parser.add_argument("--out-dir", type=Path, default=Path("log/lie_vis"))
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--video",
        action="store_true",
        help="also render the annotated replay mp4 for each clip",
    )
    parser.add_argument("--shadow-switch", action="store_true")
    parser.add_argument("--provisional-reassociation", action="store_true")
    parser.add_argument("--multi-hypothesis-identity", action="store_true")
    parser.add_argument("--background-certificates", action="store_true")
    parser.add_argument("--stale-coast-recovery", action="store_true")
    parser.add_argument("--identity-ranker-model", type=Path, default=None)
    parser.add_argument("--identity-ranker-min-margin", type=float, default=2.00)
    parser.add_argument("--identity-safety", action="store_true")
    parser.add_argument("--state-aware-ranker", action="store_true")
    parser.add_argument("--motion-corroboration-model", type=Path, default=None)
    parser.add_argument("--switch-event-model", type=Path, default=None)
    parser.add_argument("--switch-event-min-probability", type=float, default=0.5)
    parser.add_argument("--switch-motion-features", action="store_true")
    parser.add_argument("--preassociation-flow", action="store_true")
    parser.add_argument("--flow-association", action="store_true")
    parser.add_argument("--flow-coast", action="store_true")
    parser.add_argument("--switch-label-window", type=int, default=12)
    parser.add_argument(
        "--switch-events-dir",
        type=Path,
        default=None,
        help="optional directory for per-clip switch-event JSONL diagnostics",
    )
    parser.add_argument(
        "--clips",
        default=None,
        help="comma-separated clip numbers; default: every recording in --videos",
    )
    args = parser.parse_args()
    clip_numbers = (
        [int(value) for value in args.clips.split(",") if value.strip()]
        if args.clips is not None
        else discover_clip_numbers(args.videos)
    )
    if not clip_numbers:
        parser.error(f"no lie-detector recordings found in {args.videos}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for number in clip_numbers:
        video = args.videos / f"测谎录屏{number}.mp4"
        if not video.exists():
            print(f"skip missing {video}", flush=True)
            continue
        label = f"录屏{number}"
        command = [
            sys.executable,
            "tools/lie_detector_replay.py",
            str(video),
            "--csv",
            str(args.out_dir / f"{label}_fourcue.csv"),
        ]
        if args.switch_events_dir is not None:
            command += [
                "--switch-events",
                str(args.switch_events_dir / f"{label}_switch_events.jsonl"),
                "--switch-label-window",
                str(args.switch_label_window),
            ]
        if args.video:
            command += ["--output", str(args.out_dir / f"{label}_fourcue.mp4")]
        else:
            command.append("--no-video")
        if args.shadow_switch:
            command.append("--shadow-switch")
        if args.provisional_reassociation:
            command.append("--provisional-reassociation")
        if args.multi_hypothesis_identity:
            command.append("--multi-hypothesis-identity")
        if args.background_certificates:
            command.append("--background-certificates")
        if args.stale_coast_recovery:
            command.append("--stale-coast-recovery")
        if args.identity_ranker_model is not None:
            command += [
                "--identity-ranker-model",
                str(args.identity_ranker_model),
                "--identity-ranker-min-margin",
                str(args.identity_ranker_min_margin),
            ]
        if args.identity_safety:
            command.append("--identity-safety")
        if args.state_aware_ranker:
            command.append("--state-aware-ranker")
        if args.motion_corroboration_model is not None:
            command += [
                "--motion-corroboration-model",
                str(args.motion_corroboration_model),
            ]
        if args.switch_event_model is not None:
            command += [
                "--switch-event-model",
                str(args.switch_event_model),
                "--switch-event-min-probability",
                str(args.switch_event_min_probability),
            ]
        if args.switch_motion_features:
            command.append("--switch-motion-features")
        if args.preassociation_flow:
            command.append("--preassociation-flow")
        if args.flow_association:
            command.append("--flow-association")
        if args.flow_coast:
            command.append("--flow-coast")
        finished = subprocess.run(command, capture_output=True, text=True)
        if finished.returncode != 0:
            print(finished.stdout[-2000:], flush=True)
            print(finished.stderr[-2000:], flush=True)
            raise SystemExit(f"replay failed for {label}")
        payload = json.loads(finished.stdout)
        row = {"label": label}
        row.update({key: payload.get(key) for key in METRIC_KEYS})
        results.append(row)
        ratio = row["within_40px_ratio"]
        median = row["median_error_px"]
        print(
            f"{label:8} within40="
            + ("n/a" if ratio is None else f"{ratio:.4f}")
            + " median="
            + ("n/a" if median is None else f"{median:.2f}")
            + f" switches={row['identity_switches']}",
            flush=True,
        )

    args.summary.write_text(
        json.dumps({"tag": args.tag, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
