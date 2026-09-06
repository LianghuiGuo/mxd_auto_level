#!/usr/bin/env python3
"""Evaluate classical and YOLO-assisted trackers on all real recordings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.LieDetectorTracker import LieDetectorTracker
from src.engine.LieShapeYoloDetector import LieShapeYoloDetector
from tools.lie_detector_replay import cursor_ground_truth, parse_roi


def _metrics(errors: list[float | None], times_ms: list[float]) -> dict[str, object]:
    available = np.asarray([value for value in errors if value is not None], dtype=float)
    active_count = len(errors)
    return {
        "active_frames": active_count,
        "prediction_coverage": (
            float(len(available) / active_count) if active_count else 0.0
        ),
        "within_30px_active": (
            float(np.sum(available <= 30.0) / active_count) if active_count else 0.0
        ),
        "within_40px_active": (
            float(np.sum(available <= 40.0) / active_count) if active_count else 0.0
        ),
        "within_50px_active": (
            float(np.sum(available <= 50.0) / active_count) if active_count else 0.0
        ),
        "mean_error_px": float(np.mean(available)) if len(available) else None,
        "median_error_px": float(np.median(available)) if len(available) else None,
        "p90_error_px": (
            float(np.percentile(available, 90)) if len(available) else None
        ),
        "mean_processing_ms": float(np.mean(times_ms)) if times_ms else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=Path("ml/videos"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/lie_videos_config.json"),
        help=(
            "panel config; when present, only the recordings listed under 'val' "
            "are evaluated, each with its own ROI (avoids self-evaluating on "
            "training clips)."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("val", "train", "all"),
        default="val",
        help="which split to evaluate when --config is available",
    )
    parser.add_argument("--model", type=Path, default=Path("models/lie_shape_yolo.pt"))
    parser.add_argument("--roi", type=parse_roi, default=(290, 109, 700, 464))
    parser.add_argument("--yolo-conf", type=float, default=0.01)
    parser.add_argument("--yolo-stride", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("log/lie_eval/summary.json"))
    parser.add_argument("--csv", type=Path, default=Path("log/lie_eval/frames.csv"))
    args = parser.parse_args()

    # Build the list of (video_path, roi) to evaluate.  When a panel config is
    # available we honour per-recording ROIs and, by default, evaluate only the
    # held-out 'val' split so we never self-evaluate on training clips.
    roi_by_name: dict[str, tuple[int, int, int, int]] = {}
    selected_names: set[str] | None = None
    if args.config.is_file():
        config = json.loads(args.config.read_text(encoding="utf-8"))
        val_names = set(config.get("val", []))
        for entry in config["recordings"]:
            name = entry["filename"]
            roi_by_name[name] = tuple(int(v) for v in entry["roi"])
            if args.split == "all":
                continue
            is_val = name in val_names
            if (args.split == "val") == is_val:
                selected_names = (selected_names or set()) | {name}
        if args.split == "all":
            selected_names = set(roi_by_name)

    videos = sorted(args.videos.glob("*.mp4"))
    if selected_names is not None:
        videos = [v for v in videos if v.name in selected_names]
    if not videos:
        parser.error(
            f"no mp4 recordings to evaluate under {args.videos} "
            f"(split={args.split})"
        )

    learned_detector = LieShapeYoloDetector(
        args.model,
        confidence=args.yolo_conf,
        inference_stride=args.yolo_stride,
    )
    summaries: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    aggregate: dict[str, dict[str, list]] = {
        "classical": {"errors": [], "times": []},
        "hybrid": {"errors": [], "times": []},
    }
    for video in videos:
        x, y, width, height = roi_by_name.get(video.name, args.roi)
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"unable to open {video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        trackers = {
            "classical": LieDetectorTracker(),
            "hybrid": LieDetectorTracker(candidate_detector=learned_detector),
        }
        errors: dict[str, list[float | None]] = {name: [] for name in trackers}
        times: dict[str, list[float]] = {name: [] for name in trackers}
        first_acquired_seconds: dict[str, float | None] = {
            name: None for name in trackers
        }
        frame_index = 0
        while True:
            ok, full_frame = capture.read()
            if not ok:
                break
            if y + height > full_frame.shape[0] or x + width > full_frame.shape[1]:
                raise ValueError(
                    f"ROI {(x, y, width, height)} exceeds {video.name} frame "
                    f"{full_frame.shape[1]}x{full_frame.shape[0]}"
                )
            frame = full_frame[y:y + height, x:x + width]
            ground_truth = cursor_ground_truth(frame)
            for name, tracker in trackers.items():
                started = time.perf_counter()
                result = tracker.update(frame, frame_index / fps)
                times[name].append((time.perf_counter() - started) * 1000.0)
                if result.acquired and first_acquired_seconds[name] is None:
                    first_acquired_seconds[name] = frame_index / fps
                if ground_truth is None:
                    continue
                error = (
                    None
                    if result.center is None
                    else float(np.linalg.norm(np.subtract(result.center, ground_truth)))
                )
                errors[name].append(error)
                frame_rows.append(
                    {
                        "video": video.name,
                        "frame": frame_index,
                        "timestamp": f"{frame_index / fps:.6f}",
                        "mode": name,
                        "acquired": int(result.acquired),
                        "predicted_only": int(result.predicted_only),
                        "error_px": "" if error is None else f"{error:.3f}",
                        "detection_count": len(result.detections),
                    }
                )
            frame_index += 1
        capture.release()

        video_summary: dict[str, object] = {
            "video": video.name,
            "fps": fps,
            "frames": frame_index,
            "first_acquired_seconds": first_acquired_seconds["classical"],
            "has_timely_initialization": (
                first_acquired_seconds["classical"] is not None
                and first_acquired_seconds["classical"] <= 2.0
            ),
        }
        for name in trackers:
            video_summary[name] = _metrics(errors[name], times[name])
            aggregate[name]["errors"].extend(errors[name])
            aggregate[name]["times"].extend(times[name])
        summaries.append(video_summary)
        print(json.dumps(video_summary, ensure_ascii=False))

    output = {
        "videos": summaries,
        "aggregate": {
            name: _metrics(values["errors"], values["times"])
            for name, values in aggregate.items()
        },
        "notes": {
            "ground_truth": "strict green cursor component; evaluation only",
            "missing_prediction": "counted as failure in within_*_active",
            "model": str(args.model),
            "yolo_confidence": args.yolo_conf,
            "yolo_stride": args.yolo_stride,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    print(json.dumps(output["aggregate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
