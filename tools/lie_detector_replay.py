#!/usr/bin/env python3
"""Replay and evaluate the hybrid lie-detector tracker on a recording."""

from __future__ import annotations

import argparse
import csv
from enum import Enum
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.LieDetectorTracker import LieDetectorTracker, green_cursor_mask
from src.engine.LieShapeYoloDetector import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_WEIGHTS,
    LieShapeYoloDetector,
)


def parse_roi(value: str) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h") from exc
    if len(values) != 4 or any(part < 0 for part in values):
        raise argparse.ArgumentTypeError("ROI must contain four non-negative integers")
    return values


def cursor_ground_truth(
    frame_bgr: np.ndarray,
    reference_size: tuple[int, int] = (690, 463),
) -> tuple[float, float] | None:
    """Extract the green cursor centroid for evaluation only."""

    scale = math.sqrt(
        frame_bgr.shape[1] * frame_bgr.shape[0]
        / float(reference_size[0] * reference_size[1])
    )
    mask = green_cursor_mask(frame_bgr)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        # Scale the strict ring limits with the panel ROI. This preserves the
        # rejection of green UI text while supporting higher-resolution clips.
        if (
            620.0 * scale * scale <= area <= 1000.0 * scale * scale
            and 30.0 * scale <= width <= 44.0 * scale
            and 30.0 * scale <= height <= 44.0 * scale
        ):
            candidates.append((area, centroids[index]))
    if not candidates:
        return None
    center = max(candidates, key=lambda item: item[0])[1]
    return float(center[0]), float(center[1])


def percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(values, q))


def _motion_arrow_tip(
    center: tuple[int, int],
    velocity: tuple[float, float],
    *,
    radius: int,
    min_length: float = 12.0,
    max_length: float = 56.0,
    scale: float = 4.0,
) -> tuple[int, int] | None:
    """Map a velocity vector to a visible arrow tip around the box."""

    vx, vy = float(velocity[0]), float(velocity[1])
    speed = math.hypot(vx, vy)
    if speed < 0.35:
        return None
    length = float(np.clip(speed * scale, min_length, max(max_length, radius * 1.8)))
    ux, uy = vx / speed, vy / speed
    return (
        int(round(center[0] + ux * length)),
        int(round(center[1] + uy * length)),
    )


def draw_result(frame: np.ndarray, result, ground_truth, error) -> np.ndarray:
    debug = frame.copy()
    constellation = getattr(result, "constellation", None) or []
    if constellation:
        for view in constellation:
            cx, cy = (int(round(v)) for v in view.center)
            radius = max(2, int(round(view.radius)))
            is_real = view.role == "real" or view.track_id == result.target_id
            color = (0, 0, 255) if is_real else (220, 140, 40)
            cv2.rectangle(
                debug,
                (cx - radius, cy - radius),
                (cx + radius, cy + radius),
                color,
                2 if is_real else 1,
            )
            label = "REAL" if is_real else f"BG:{view.track_id}"
            cv2.putText(
                debug,
                label,
                (cx - radius, cy - radius - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                color,
                1,
                cv2.LINE_AA,
            )
            # Green: pose/orientation. Cyan: predicted move direction.
            if view.orientation is not None:
                tip = (
                    int(round(cx + math.cos(view.orientation) * radius * 0.95)),
                    int(round(cy + math.sin(view.orientation) * radius * 0.95)),
                )
                cv2.arrowedLine(debug, (cx, cy), tip, (40, 220, 40), 1, tipLength=0.30)
            velocity = getattr(view, "velocity", None) or (0.0, 0.0)
            motion_tip = _motion_arrow_tip((cx, cy), velocity, radius=radius)
            if motion_tip is not None:
                # BGR: orange for REAL, cyan for BG.
                motion_color = (0, 140, 255) if is_real else (255, 255, 0)
                cv2.arrowedLine(
                    debug,
                    (cx, cy),
                    motion_tip,
                    motion_color,
                    2 if is_real else 1,
                    tipLength=0.28,
                )
                # Small predicted next-center mark along the motion ray.
                cv2.circle(debug, motion_tip, 2, motion_color, -1, cv2.LINE_AA)
    else:
        for detection in result.detections:
            cx, cy = (int(round(v)) for v in detection.center)
            radius = float(detection.observed_radius or detection.radius)
            color = (
                (40, 200, 40)
                if detection.source == "yolo" or detection.yolo_confidence > 0
                else (80, 150, 80)
            )
            x, y, w, h = detection.bbox
            if detection.source == "yolo" and w > 0 and h > 0:
                cv2.rectangle(debug, (x, y), (x + w, y + h), color, 1)
            else:
                cv2.circle(debug, (cx, cy), max(2, int(round(radius))), color, 1)
    if result.center is not None:
        predicted = tuple(int(round(v)) for v in result.center)
        cv2.drawMarker(debug, predicted, (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.circle(debug, predicted, max(2, int(round(result.radius or 1))), (0, 0, 255), 2)
    if ground_truth is not None:
        gt = tuple(int(round(v)) for v in ground_truth)
        cv2.drawMarker(debug, gt, (0, 255, 0), cv2.MARKER_TILTED_CROSS, 20, 2)
        if result.center is not None:
            cv2.line(debug, tuple(int(round(v)) for v in result.center), gt, (255, 0, 255), 1)
    n_bg = sum(1 for view in constellation if view.role == "bg")
    target_view = next(
        (
            view
            for view in constellation
            if view.track_id == result.target_id
        ),
        None,
    )
    label = (
        f"REAL={result.target_id} BG={n_bg} conf={result.confidence:.2f} "
        f"lost={result.lost_frames} "
        f"sw={int(getattr(result, 'identity_switched', False))} "
        f"err={error:.1f}px"
        if error is not None
        else (
            f"REAL={result.target_id} BG={n_bg} conf={result.confidence:.2f} "
            f"lost={result.lost_frames} "
            f"sw={int(getattr(result, 'identity_switched', False))}"
        )
    )
    score_label = (
        "score=-- R=-- D=-- V=-- G=-- rel=-- vis=--"
        if target_view is None
        else (
            f"score={target_view.real_score:.1f} "
            f"R={target_view.rotation_score:.1f} "
            f"D={target_view.direction_score:.1f} "
            f"V={target_view.speed_score:.1f} "
            f"G={target_view.rigidity_score:.1f} "
            f"rel={target_view.motion_reliability:.2f} "
            f"vis={target_view.visible_streak}"
        )
    )
    cv2.rectangle(debug, (0, 0), (min(debug.shape[1], 760), 54), (0, 0, 0), -1)
    cv2.putText(debug, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(
        debug,
        score_label,
        (8, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    # Legend for the two arrow styles.
    cv2.putText(
        debug,
        "green=orient  orange=REAL move  cyan=BG move",
        (8, debug.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return debug


def load_recording_from_config(
    video: Path,
    config_path: Path = Path("ml/lie_videos_config.json"),
) -> dict[str, object] | None:
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for entry in config.get("recordings", []):
        if entry.get("filename") == video.name:
            return dict(entry)
    return None


def load_roi_from_config(
    video: Path,
    config_path: Path = Path("ml/lie_videos_config.json"),
) -> tuple[int, int, int, int] | None:
    recording = load_recording_from_config(video, config_path)
    if recording is None:
        return None
    roi = recording.get("roi")
    if isinstance(roi, list) and len(roi) == 4:
        return tuple(int(v) for v in roi)
    return None


class EvaluationPhase(str, Enum):
    SEED = "seed"
    ACTIVE = "active"
    DONE = "done"


def evaluation_phase(
    timestamp: float,
    active_start: float,
    active_end: float,
) -> EvaluationPhase:
    if timestamp < active_start:
        return EvaluationPhase.SEED
    if timestamp <= active_end:
        return EvaluationPhase.ACTIVE
    return EvaluationPhase.DONE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--roi",
        type=parse_roi,
        default=None,
        help="panel ROI as x,y,w,h; default: lookup video in lie_videos_config.json",
    )
    parser.add_argument("--config", type=Path, default=Path("ml/lie_videos_config.json"))
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=float("inf"))
    parser.add_argument("--eval-start", type=float, default=None)
    parser.add_argument("--eval-end", type=float, default=None)
    parser.add_argument("--inside-radius", type=float, default=40.0)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"lie-shape YOLO weights (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument("--no-yolo", action="store_true", help="classical tracker only")
    parser.add_argument(
        "--shadow-switch",
        action="store_true",
        help="delay REAL switches until a mature shadow candidate has two cue families",
    )
    parser.add_argument(
        "--provisional-reassociation",
        action="store_true",
        help="validate a temporary child track before committing a long-gap REAL match",
    )
    parser.add_argument(
        "--multi-hypothesis-identity",
        action="store_true",
        help="select REAL from a beam of continuous identity paths",
    )
    parser.add_argument("--yolo-conf", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--yolo-imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--yolo-stride", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("log/lie_detector_replay.mp4"))
    parser.add_argument("--csv", type=Path, default=Path("log/lie_detector_replay.csv"))
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    recording = load_recording_from_config(args.video, args.config)
    roi = args.roi or load_roi_from_config(args.video, args.config)
    if roi is None:
        parser.error(
            "ROI not provided and video not found in config; pass --roi x,y,w,h"
        )
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        parser.error(f"unable to open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    x, y, width, height = roi
    eval_start = (
        float(args.eval_start)
        if args.eval_start is not None
        else float((recording or {}).get("active_start", 0.0))
    )
    eval_end = (
        float(args.eval_end)
        if args.eval_end is not None
        else float((recording or {}).get("active_end", float("inf")))
    )
    detector = None
    if not args.no_yolo:
        if not args.model.is_file():
            parser.error(f"YOLO weights not found: {args.model}")
        detector = LieShapeYoloDetector(
            args.model,
            confidence=args.yolo_conf,
            image_size=args.yolo_imgsz,
            inference_stride=args.yolo_stride,
        )
    tracker = LieDetectorTracker(
        candidate_detector=detector,
        shadow_switch=args.shadow_switch,
        provisional_reassociation=args.provisional_reassociation,
        multi_hypothesis_identity=args.multi_hypothesis_identity,
    )
    rows: list[dict[str, object]] = []
    errors: list[float] = []
    covered: list[bool] = []
    processing_ms: list[float] = []
    evaluated_frames = 0
    acquired_frames = 0
    actionable_frames = 0
    predicted_only_frames = 0
    # D: segmented metrics after cursor catch-up / after white fade.
    catchup_errors: list[float] = []
    postfade_errors: list[float] = []
    catchup_streak = 0
    catchup_started = False
    saw_white = False
    white_faded = False
    identity_switches = 0

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
                f"ROI {roi} exceeds video frame {full_frame.shape[1]}x{full_frame.shape[0]}"
            )
        frame = full_frame[y:y + height, x:x + width]
        ground_truth = cursor_ground_truth(frame)
        phase = evaluation_phase(timestamp, eval_start, eval_end)
        started = time.perf_counter()
        result = tracker.update(frame, timestamp)
        processing_ms.append((time.perf_counter() - started) * 1000.0)
        error = None
        # A strict cursor component is the offline-only indication that the
        # challenge is active.  It is never passed into the tracker.
        in_eval_window = phase is EvaluationPhase.ACTIVE and ground_truth is not None
        if getattr(result, "white_active", False):
            saw_white = True
        elif saw_white:
            white_faded = True
        if (
            phase is EvaluationPhase.ACTIVE
            and getattr(result, "identity_switched", False)
        ):
            identity_switches += 1
        if in_eval_window:
            evaluated_frames += 1
            if result.acquired:
                acquired_frames += 1
            if getattr(result, "actionable", False):
                actionable_frames += 1
            if result.predicted_only:
                predicted_only_frames += 1
            if result.center is not None:
                error = float(np.linalg.norm(np.subtract(result.center, ground_truth)))
                errors.append(error)
                covered.append(error <= args.inside_radius)
                # Catch-up: human cursor stays within 40px for 5 frames.
                if error <= 40.0:
                    catchup_streak += 1
                    if catchup_streak >= 5:
                        catchup_started = True
                else:
                    catchup_streak = 0
                if catchup_started:
                    catchup_errors.append(error)
                if white_faded and catchup_started:
                    postfade_errors.append(error)
        rows.append(
            {
                "frame": frame_index - 1,
                "timestamp": f"{timestamp:.6f}",
                "evaluation_phase": phase.value,
                "target_id": "" if result.target_id is None else result.target_id,
                "pred_x": "" if result.center is None else f"{result.center[0]:.3f}",
                "pred_y": "" if result.center is None else f"{result.center[1]:.3f}",
                "radius": "" if result.radius is None else f"{result.radius:.3f}",
                "confidence": f"{result.confidence:.4f}",
                "actionable": int(getattr(result, "actionable", False)),
                "position_uncertainty_px": (
                    f"{getattr(result, 'position_uncertainty_px', 0.0):.3f}"
                ),
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
                "white_active": int(getattr(result, "white_active", False)),
                "identity_window_active": int(
                    getattr(result, "identity_window_active", False)
                ),
                "identity_switched": int(getattr(result, "identity_switched", False)),
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
        "roi": list(roi),
        "fps": fps,
        "eval_start": eval_start,
        "eval_end": eval_end,
        "evaluated_frames": evaluated_frames,
        "acquired_frame_ratio": acquired_frames / evaluated_frames if evaluated_frames else 0.0,
        "actionable_frame_ratio": (
            actionable_frames / evaluated_frames if evaluated_frames else 0.0
        ),
        "predicted_only_frame_ratio": predicted_only_frames / evaluated_frames if evaluated_frames else 0.0,
        "evaluated_error_frames": len(errors),
        "prediction_coverage_ratio": (
            len(errors) / evaluated_frames if evaluated_frames else 0.0
        ),
        "mean_error_px": None if not errors else float(np.mean(errors)),
        "median_error_px": percentile(errors, 50),
        "p90_error_px": percentile(errors, 90),
        "p95_error_px": percentile(errors, 95),
        "max_error_px": None if not errors else max(errors),
        "within_30px_ratio": None if not errors else float(np.mean(np.array(errors) <= 30.0)),
        "within_40px_ratio": None if not errors else float(np.mean(np.array(errors) <= 40.0)),
        "within_50px_ratio": None if not errors else float(np.mean(np.array(errors) <= 50.0)),
        "within_radius_ratio": None if not covered else float(np.mean(covered)),
        "within_radius_active_ratio": (
            float(np.sum(covered)) / evaluated_frames if evaluated_frames else 0.0
        ),
        "catchup_frames": len(catchup_errors),
        "catchup_mean_error_px": (
            None if not catchup_errors else float(np.mean(catchup_errors))
        ),
        "catchup_median_error_px": percentile(catchup_errors, 50),
        "catchup_within_40px_ratio": (
            None
            if not catchup_errors
            else float(np.mean(np.asarray(catchup_errors) <= 40.0))
        ),
        "postfade_frames": len(postfade_errors),
        "postfade_mean_error_px": (
            None if not postfade_errors else float(np.mean(postfade_errors))
        ),
        "postfade_median_error_px": percentile(postfade_errors, 50),
        "postfade_within_40px_ratio": (
            None
            if not postfade_errors
            else float(np.mean(np.asarray(postfade_errors) <= 40.0))
        ),
        "identity_switches": identity_switches,
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
