#!/usr/bin/env python3
"""Evaluate the standalone transparent tracker on lie-detector recordings.

The recording's strict green cursor component is used only as ground truth.
It is never passed back into the evaluator as a target hint; the tracker itself
applies its own green-UI suppression when processing each panel frame.
"""

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
TRACKER_SRC = PROJECT_ROOT / "unet-transparent-tracker" / "src"
for path in (PROJECT_ROOT, TRACKER_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.lie_detector_replay import (  # noqa: E402
    EvaluationPhase,
    cursor_ground_truth,
    evaluation_phase,
)
from ml.detect_lie_panels import detect_recording  # noqa: E402
from unet_tracker import UniversalUNetTracker  # noqa: E402


def _metrics(errors: list[float], processing_ms: list[float]) -> dict[str, object]:
    values = np.asarray(errors, dtype=np.float64)
    result: dict[str, object] = {
        "evaluated_frames": int(len(values)),
        "mean_processing_ms": (
            float(np.mean(processing_ms)) if processing_ms else None
        ),
    }
    if not len(values):
        result.update(
            {
                "within_10px": 0.0,
                "within_20px": 0.0,
                "within_30px": 0.0,
                "within_40px": 0.0,
                "within_50px": 0.0,
                "mean_error_px": None,
                "median_error_px": None,
                "p90_error_px": None,
                "max_error_px": None,
            }
        )
        return result
    for threshold in (10, 20, 30, 40, 50):
        result[f"within_{threshold}px"] = float(np.mean(values <= threshold))
    result.update(
        {
            "mean_error_px": float(np.mean(values)),
            "median_error_px": float(np.median(values)),
            "p90_error_px": float(np.percentile(values, 90)),
            "max_error_px": float(np.max(values)),
        }
    )
    return result


def _draw_overlay(
    frame: np.ndarray,
    state: dict[str, object],
    truth: tuple[float, float] | None,
    error: float | None,
) -> np.ndarray:
    canvas = frame.copy()
    prediction = (int(round(float(state["cx"]))), int(round(float(state["cy"]))))
    cv2.drawMarker(
        canvas, prediction, (0, 0, 255), cv2.MARKER_CROSS, 30, 3, cv2.LINE_AA
    )
    if truth is not None:
        target = (int(round(truth[0])), int(round(truth[1])))
        cv2.circle(canvas, target, 18, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.line(canvas, prediction, target, (0, 165, 255), 2, cv2.LINE_AA)

    phase = str(state.get("phase", "?"))
    confidence = float(state.get("conf", 0.0))
    snr = float(state.get("snr", 0.0))
    error_text = "n/a" if error is None else f"{error:.1f}px"
    lines = (
        "red: prediction | green: cursor truth",
        f"phase={phase} conf={confidence:.2f} snr={snr:.2f} error={error_text}",
    )
    for index, line in enumerate(lines):
        origin = (12, 28 + index * 26)
        cv2.putText(
            canvas, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.58,
            (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.58,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    return canvas


def _selected_recordings(config: dict, split: str, names: list[str]) -> list[dict]:
    available = {entry["filename"]: entry for entry in config["recordings"]}
    if names:
        missing = [name for name in names if name not in available]
        if missing:
            raise ValueError(f"recordings not found in config: {missing}")
        return [available[name] for name in names]

    val_names = set(config.get("val", []))
    if split == "all":
        return list(config["recordings"])
    want_val = split == "val"
    return [
        entry
        for entry in config["recordings"]
        if (entry["filename"] in val_names) == want_val
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "ml/lie_videos_config.json"
    )
    parser.add_argument(
        "--videos-dir", type=Path, default=PROJECT_ROOT / "ml/videos"
    )
    parser.add_argument("--split", choices=("val", "train", "all"), default="val")
    parser.add_argument("--video", action="append", default=[], help="exact filename; repeatable")
    parser.add_argument(
        "--auto-video",
        action="append",
        type=Path,
        default=[],
        help="video path absent from config; repeatable, with panel ROI auto-detected",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", default="ultra")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "log/unet_transparent_eval"
    )
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.auto_video:
        recordings = []
        for video_path in args.auto_video:
            resolved = video_path.resolve()
            recording = detect_recording(resolved, probe_count=30)
            if recording is None:
                raise ValueError(f"unable to detect panel ROI in {resolved}")
            recording["_video_path"] = str(resolved)
            recordings.append(recording)
    else:
        recordings = _selected_recordings(config, args.split, args.video)
        recordings = [
            entry for entry in recordings if (args.videos_dir / entry["filename"]).is_file()
        ]
    if not recordings:
        parser.error("no configured video files selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_errors: list[float] = []
    all_times: list[float] = []
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for recording in recordings:
        video_path = Path(
            recording.get("_video_path", args.videos_dir / recording["filename"])
        )
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"unable to open {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        x, y, width, height = (int(value) for value in recording["roi"])
        active_start = float(recording.get("active_start", 0.0))
        active_end = float(recording.get("active_end", float("inf")))
        tracker = UniversalUNetTracker(
            fps=fps, perf_mode=args.mode, device=args.device
        )

        writer = None
        output_video = args.output_dir / f"{video_path.stem}_unet_tracker.mp4"
        if not args.no_video:
            writer = cv2.VideoWriter(
                str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )

        errors: list[float] = []
        times_ms: list[float] = []
        phases: dict[str, int] = {}
        evaluated_phases: dict[str, int] = {}
        truth_frames = 0
        frame_index = 0
        while True:
            ok, full_frame = capture.read()
            if not ok:
                break
            if x + width > full_frame.shape[1] or y + height > full_frame.shape[0]:
                raise ValueError(
                    f"ROI {(x, y, width, height)} exceeds {video_path.name} frame "
                    f"{full_frame.shape[1]}x{full_frame.shape[0]}"
                )
            frame = full_frame[y:y + height, x:x + width]
            timestamp = frame_index / fps
            truth = cursor_ground_truth(frame)
            if truth is not None:
                truth_frames += 1

            started = time.perf_counter()
            state = tracker.update(frame, dt_sec=1.0 / fps)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            times_ms.append(elapsed_ms)
            phase_name = str(state.get("phase", "UNKNOWN"))
            phases[phase_name] = phases.get(phase_name, 0) + 1

            evaluation = evaluation_phase(timestamp, active_start, active_end)
            error = None
            if truth is not None and evaluation is EvaluationPhase.ACTIVE:
                error = float(
                    np.linalg.norm(
                        np.subtract((float(state["cx"]), float(state["cy"])), truth)
                    )
                )
                errors.append(error)
                evaluated_phases[phase_name] = evaluated_phases.get(phase_name, 0) + 1

            all_rows.append(
                {
                    "video": video_path.name,
                    "frame": frame_index,
                    "timestamp": f"{timestamp:.6f}",
                    "evaluation_phase": evaluation.value,
                    "tracker_phase": phase_name,
                    "prediction_x": f"{float(state['cx']):.3f}",
                    "prediction_y": f"{float(state['cy']):.3f}",
                    "truth_x": "" if truth is None else f"{truth[0]:.3f}",
                    "truth_y": "" if truth is None else f"{truth[1]:.3f}",
                    "error_px": "" if error is None else f"{error:.3f}",
                    "confidence": f"{float(state.get('conf', 0.0)):.6f}",
                    "snr": f"{float(state.get('snr', 0.0)):.6f}",
                    "latency_ms": f"{elapsed_ms:.3f}",
                }
            )
            if writer is not None:
                writer.write(_draw_overlay(frame, state, truth, error))
            frame_index += 1

        capture.release()
        if writer is not None:
            writer.release()

        summary = {
            "video": video_path.name,
            "fps": fps,
            "frames": frame_index,
            "active_window": [active_start, active_end],
            "truth_frames": truth_frames,
            "tracker_phases": phases,
            "evaluated_tracker_phases": evaluated_phases,
            **_metrics(errors, times_ms),
            "output_video": None if args.no_video else str(output_video),
        }
        summaries.append(summary)
        all_errors.extend(errors)
        all_times.extend(times_ms)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    output = {
        "tracker": "unet-transparent-tracker V3.7 CPU",
        "evaluation": (
            "Euclidean center error against strict green cursor, only inside each "
            "recording's configured ACTIVE window"
        ),
        "videos": summaries,
        "aggregate": _metrics(all_errors, all_times),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = args.output_dir / "frames.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps({"aggregate": output["aggregate"]}, ensure_ascii=False, indent=2))
    print(f"summary: {summary_path}")
    print(f"frames: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
