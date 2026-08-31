#!/usr/bin/env python3
"""Replay and evaluate the classical lie-detector tracker on a recording."""

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

from src.engine.LieDetectorTracker import LieDetectorTracker, green_cursor_mask


def parse_roi(value: str) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h") from exc
    if len(values) != 4 or any(part < 0 for part in values):
        raise argparse.ArgumentTypeError("ROI must contain four non-negative integers")
    return values


def cursor_ground_truth(frame_bgr: np.ndarray) -> tuple[float, float] | None:
    """Extract the green cursor centroid for evaluation only."""

    mask = green_cursor_mask(frame_bgr)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if 80 <= area <= 2500 and 10 <= width <= 80 and 10 <= height <= 80:
            candidates.append((area, centroids[index]))
    if not candidates:
        return None
    center = max(candidates, key=lambda item: item[0])[1]
    return float(center[0]), float(center[1])


def percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(values, q))


def draw_result(frame: np.ndarray, result, ground_truth, error) -> np.ndarray:
    debug = frame.copy()
    for detection in result.detections:
        cx, cy = (int(round(v)) for v in detection.center)
        cv2.circle(debug, (cx, cy), max(2, int(round(detection.radius))), (80, 150, 80), 1)
    for track_id, center, radius, lost in result.tracks:
        cx, cy = (int(round(v)) for v in center)
        cv2.putText(
            debug,
            f"{track_id}{'*' if track_id == result.target_id else ''}",
            (cx + 4, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (100, 220, 220) if not lost else (100, 100, 220),
            1,
            cv2.LINE_AA,
        )
    if result.center is not None:
        predicted = tuple(int(round(v)) for v in result.center)
        cv2.drawMarker(debug, predicted, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.circle(debug, predicted, max(2, int(round(result.radius or 1))), (0, 0, 255), 2)
    if ground_truth is not None:
        gt = tuple(int(round(v)) for v in ground_truth)
        cv2.drawMarker(debug, gt, (0, 255, 0), cv2.MARKER_TILTED_CROSS, 20, 2)
        if result.center is not None:
            cv2.line(debug, tuple(int(round(v)) for v in result.center), gt, (255, 0, 255), 1)
    label = (
        f"target={result.target_id} conf={result.confidence:.2f} "
        f"lost={result.lost_frames} H={result.hypothesis_count} "
        f"G={int(result.collective_promoted)} err={error:.1f}px"
        if error is not None
        else (
            f"target={result.target_id} conf={result.confidence:.2f} "
            f"lost={result.lost_frames} H={result.hypothesis_count} "
            f"G={int(result.collective_promoted)}"
        )
    )
    cv2.rectangle(debug, (0, 0), (min(debug.shape[1], 620), 34), (0, 0, 0), -1)
    cv2.putText(debug, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    return debug


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--roi", type=parse_roi, default=(418, 198, 1012, 676))
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=13.5)
    parser.add_argument("--eval-start", type=float, default=1.8)
    parser.add_argument("--eval-end", type=float, default=12.8)
    parser.add_argument("--inside-radius", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=Path("log/lie_detector_replay.mp4"))
    parser.add_argument("--csv", type=Path, default=Path("log/lie_detector_replay.csv"))
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        parser.error(f"unable to open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    x, y, width, height = args.roi
    tracker = LieDetectorTracker()
    rows: list[dict[str, object]] = []
    errors: list[float] = []
    covered: list[bool] = []
    processing_ms: list[float] = []
    evaluated_frames = 0
    acquired_frames = 0
    predicted_only_frames = 0

    writer = None
    if not args.no_video:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"unable to create video writer: {args.output}")

    frame_index = 0
    while True:
        ok, full_frame = capture.read()
        if not ok:
            break
        timestamp = frame_index / fps
        frame_index += 1
        if timestamp < args.start:
            continue
        if timestamp > args.end:
            break
        if y + height > full_frame.shape[0] or x + width > full_frame.shape[1]:
            raise ValueError(
                f"ROI {args.roi} exceeds video frame {full_frame.shape[1]}x{full_frame.shape[0]}"
            )
        frame = full_frame[y:y + height, x:x + width]
        ground_truth = cursor_ground_truth(frame)
        started = time.perf_counter()
        result = tracker.update(frame, timestamp)
        processing_ms.append((time.perf_counter() - started) * 1000.0)
        error = None
        in_eval_window = args.eval_start <= timestamp <= args.eval_end
        if in_eval_window:
            evaluated_frames += 1
            if result.acquired:
                acquired_frames += 1
            if result.predicted_only:
                predicted_only_frames += 1
            if result.center is not None and ground_truth is not None:
                error = float(np.linalg.norm(np.subtract(result.center, ground_truth)))
                errors.append(error)
                covered.append(error <= args.inside_radius)
        rows.append(
            {
                "frame": frame_index - 1,
                "timestamp": f"{timestamp:.6f}",
                "target_id": "" if result.target_id is None else result.target_id,
                "pred_x": "" if result.center is None else f"{result.center[0]:.3f}",
                "pred_y": "" if result.center is None else f"{result.center[1]:.3f}",
                "radius": "" if result.radius is None else f"{result.radius:.3f}",
                "confidence": f"{result.confidence:.4f}",
                "lost_frames": result.lost_frames,
                "predicted_only": int(result.predicted_only),
                "gt_x": "" if ground_truth is None else f"{ground_truth[0]:.3f}",
                "gt_y": "" if ground_truth is None else f"{ground_truth[1]:.3f}",
                "error_px": "" if error is None else f"{error:.3f}",
                "detection_count": len(result.detections),
                "track_count": len(result.tracks),
                "hypothesis_count": result.hypothesis_count,
                "recovery_active": int(result.recovery_active),
                "collective_promoted": int(result.collective_promoted),
                "collective_dx": f"{result.collective_delta[0]:.4f}",
                "collective_dy": f"{result.collective_delta[1]:.4f}",
                "collective_rotation_deg": (
                    f"{result.collective_rotation_degrees:.4f}"
                ),
            }
        )
        if writer is not None:
            writer.write(draw_result(frame, result, ground_truth, error))

    capture.release()
    if writer is not None:
        writer.release()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0]) if rows else []
        csv_writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            csv_writer.writeheader()
            csv_writer.writerows(rows)

    summary = {
        "video": str(args.video),
        "roi": args.roi,
        "fps": fps,
        "evaluated_frames": evaluated_frames,
        "acquired_frame_ratio": acquired_frames / evaluated_frames if evaluated_frames else 0.0,
        "predicted_only_frame_ratio": predicted_only_frames / evaluated_frames if evaluated_frames else 0.0,
        "evaluated_error_frames": len(errors),
        "mean_error_px": None if not errors else float(np.mean(errors)),
        "median_error_px": percentile(errors, 50),
        "p90_error_px": percentile(errors, 90),
        "p95_error_px": percentile(errors, 95),
        "max_error_px": None if not errors else max(errors),
        "within_30px_ratio": None if not errors else float(np.mean(np.array(errors) <= 30.0)),
        "within_40px_ratio": None if not errors else float(np.mean(np.array(errors) <= 40.0)),
        "within_50px_ratio": None if not errors else float(np.mean(np.array(errors) <= 50.0)),
        "within_radius_ratio": None if not covered else float(np.mean(covered)),
        "inside_radius_px": args.inside_radius,
        "mean_processing_ms": None if not processing_ms else float(np.mean(processing_ms)),
        "p95_processing_ms": percentile(processing_ms, 95),
        "output_video": None if args.no_video else str(args.output),
        "output_csv": str(args.csv),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
