#!/usr/bin/env python3
"""Compare the production lie tracker with an OC-SORT identity baseline.

The experiment deliberately runs YOLO once per source frame and caches the raw
detections.  Both identity trackers then consume the same radius-filtered boxes
and the same opening white-highlight observation.  Replaying the cache at the
source frame rate and at a real-time target rate isolates association/identity
behaviour from detector noise and inference latency.

The OC-SORT baseline follows the anti-detection-test policy exactly: bind the
nearest active OC-SORT ID during the opening white highlight, then keep that ID
for the rest of the clip.  It does not use ground truth, ranker output, or any
identity repair heuristic.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass, field
import gzip
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterable, Optional

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.LieDetectorTracker import (  # noqa: E402
    LieDetectorTracker,
    ShapeDetection,
)
from src.engine.LieShapeYoloDetector import (  # noqa: E402
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_WEIGHTS,
    LieShapeYoloDetector,
)
from tools.lie_detector_replay import (  # noqa: E402
    EvaluationPhase,
    cursor_ground_truth,
    evaluation_phase,
)


CACHE_VERSION = 2
DEFAULT_TARGET_FPS = 16.5
CORRECT_RADIUS_PX = 40.0
SEVERE_ERROR_PX = 100.0


def _clone_detection(item: ShapeDetection) -> ShapeDetection:
    """Return an unannotated copy safe for one tracker to mutate."""

    return copy.deepcopy(item)


class CachedCandidateDetector:
    """Expose one externally selected cached detection set to the tracker."""

    def __init__(self) -> None:
        self.current: list[ShapeDetection] = []

    def set_current(self, detections: Iterable[ShapeDetection]) -> None:
        self.current = [_clone_detection(item) for item in detections]

    def detect(
        self,
        frame_bgr: np.ndarray,
        reference_radius: Optional[float] = None,
    ) -> list[ShapeDetection]:
        del frame_bgr, reference_radius
        return [_clone_detection(item) for item in self.current]

    def reset(self) -> None:
        self.current = []


class DetectionBatch:
    """Minimal Results-like container accepted by Ultralytics OC-SORT."""

    def __init__(self, detections: Iterable[ShapeDetection]) -> None:
        values = list(detections)
        xywh = []
        confidence = []
        for item in values:
            x, y, width, height = item.bbox
            xywh.append(
                (
                    float(x) + 0.5 * float(width),
                    float(y) + 0.5 * float(height),
                    float(width),
                    float(height),
                )
            )
            confidence.append(float(item.yolo_confidence or item.score))
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(confidence, dtype=np.float32)
        self.cls = np.zeros(len(values), dtype=np.float32)

    def __len__(self) -> int:
        return int(len(self.conf))

    def __getitem__(self, index) -> "DetectionBatch":
        selected = DetectionBatch(())
        selected.xywh = np.asarray(self.xywh[index], dtype=np.float32).reshape(-1, 4)
        selected.conf = np.asarray(self.conf[index], dtype=np.float32).reshape(-1)
        selected.cls = np.asarray(self.cls[index], dtype=np.float32).reshape(-1)
        return selected


@dataclass
class SharedObservationState:
    """Reproduce the production pre-association radius and white phases."""

    target_radius: Optional[float] = None
    white_seen: bool = False
    white_absent_streak: int = 0
    white_phase_completed: bool = False

    def prepare(
        self,
        cleaned: np.ndarray,
        raw_detections: Iterable[ShapeDetection],
    ) -> tuple[list[ShapeDetection], Optional[ShapeDetection], bool]:
        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(cleaned, cv2.COLOR_BGR2HSV)
        bright = LieDetectorTracker._initial_bright_shape(gray, hsv)
        bright_valid = bool(
            bright is not None and bright.score * 255.0 >= 205.0
        )
        allow_bright_seed = bool(
            bright_valid and bright is not None and not self.white_phase_completed
        )
        if allow_bright_seed and bright is not None:
            self.white_seen = True
            self.white_absent_streak = 0
            if self.target_radius is None:
                self.target_radius = float(bright.radius)
        else:
            if self.white_seen and not self.white_phase_completed:
                self.white_absent_streak += 1
                if self.white_absent_streak >= 30:
                    self.white_phase_completed = True

        learned = []
        for item in raw_detections:
            observed_radius = float(item.observed_radius or item.radius)
            if self.target_radius is not None and not (
                0.30 * self.target_radius
                <= observed_radius
                <= 3.20 * self.target_radius
            ):
                continue
            learned.append(_clone_detection(item))

        learned_radii = [
            float(item.observed_radius or item.radius) for item in learned
        ]
        if learned_radii:
            median_radius = float(np.median(np.asarray(learned_radii)))
            if self.target_radius is None:
                self.target_radius = median_radius
            else:
                self.target_radius = 0.65 * self.target_radius + 0.35 * median_radius

        # Return YOLO observations only.  The production tracker applies its
        # own bright-contour gap fill internally; OC-SORT, matching the
        # anti-detection-test policy, uses white only to choose among YOLO IDs.
        # Feeding the temporary bright contour into OC-SORT would create a
        # short-lived non-YOLO track and unfairly bind REAL to that ID.
        return learned, bright, allow_bright_seed


@dataclass
class OCSortIdentityBaseline:
    """Freeze REAL to the OC-SORT ID nearest the opening white seed."""

    tracker: object
    locked_id: Optional[int] = None
    seed_frame: Optional[int] = None
    seed_timestamp: Optional[float] = None
    first_white_frame: Optional[int] = None
    last_center: Optional[tuple[float, float]] = None

    @classmethod
    def create(cls) -> "OCSortIdentityBaseline":
        try:
            from ultralytics.cfg import get_cfg
            from ultralytics.trackers.oc_sort import OCSORT
            import ultralytics
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "OC-SORT baseline requires an Ultralytics build with "
                "ultralytics.trackers.oc_sort"
            ) from exc
        config_path = (
            Path(ultralytics.__file__).resolve().parent
            / "cfg"
            / "trackers"
            / "ocsort.yaml"
        )
        return cls(tracker=OCSORT(get_cfg(config_path)))

    def update(
        self,
        detections: Iterable[ShapeDetection],
        bright: Optional[ShapeDetection],
        allow_bright_seed: bool,
        frame_index: int,
        timestamp: float,
    ) -> tuple[Optional[tuple[float, float]], Optional[int], int]:
        batch = DetectionBatch(detections)
        tracked = self.tracker.update(batch)
        by_id: dict[int, tuple[float, float]] = {}
        for row in tracked:
            x1, y1, x2, y2 = (float(value) for value in row[:4])
            track_id = int(round(float(row[4])))
            by_id[track_id] = (0.5 * (x1 + x2), 0.5 * (y1 + y2))

        if allow_bright_seed and bright is not None:
            if self.first_white_frame is None:
                self.first_white_frame = frame_index
            if self.locked_id is None and by_id:
                nearest_id, nearest_center = min(
                    by_id.items(),
                    key=lambda item: math.hypot(
                        item[1][0] - bright.center[0],
                        item[1][1] - bright.center[1],
                    ),
                )
                distance = math.hypot(
                    nearest_center[0] - bright.center[0],
                    nearest_center[1] - bright.center[1],
                )
                if distance <= max(40.0, float(bright.radius) * 0.9):
                    self.locked_id = nearest_id
                    self.seed_frame = frame_index
                    self.seed_timestamp = timestamp

        center = None if self.locked_id is None else by_id.get(self.locked_id)
        if center is not None:
            self.last_center = center
        return center, self.locked_id, len(by_id)


@dataclass
class MethodMetrics:
    method: str
    evaluated_frames: int = 0
    visible_frames: int = 0
    correct_frames: int = 0
    severe_frames: int = 0
    errors: list[float] = field(default_factory=list)
    states: list[str] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    ids: list[Optional[int]] = field(default_factory=list)
    identity_switch_signals: int = 0
    ranker_switch_signals: int = 0

    def add(
        self,
        timestamp: float,
        center: Optional[tuple[float, float]],
        target_id: Optional[int],
        truth: tuple[float, float],
        *,
        identity_switched: bool = False,
        ranker_switched: bool = False,
    ) -> tuple[Optional[float], str]:
        self.evaluated_frames += 1
        self.timestamps.append(timestamp)
        self.ids.append(target_id)
        self.identity_switch_signals += int(identity_switched)
        self.ranker_switch_signals += int(ranker_switched)
        if center is None:
            state = "missing"
            error = None
        else:
            self.visible_frames += 1
            error = float(np.linalg.norm(np.subtract(center, truth)))
            self.errors.append(error)
            if error <= CORRECT_RADIUS_PX:
                self.correct_frames += 1
                state = "correct"
            else:
                state = "wrong"
                if error > SEVERE_ERROR_PX:
                    self.severe_frames += 1
        self.states.append(state)
        return error, state

    def summary(self, processed_frames: int) -> dict[str, object]:
        errors = np.asarray(self.errors, dtype=np.float64)
        bad_runs: list[tuple[str, int, float]] = []
        recoveries = 0
        failures_after_correct = 0
        saw_correct = False
        index = 0
        while index < len(self.states):
            state = self.states[index]
            end = index + 1
            while end < len(self.states) and self.states[end] == state:
                end += 1
            if state == "correct":
                if saw_correct and index > 0 and self.states[index - 1] != "correct":
                    recoveries += 1
                saw_correct = True
            elif saw_correct:
                failures_after_correct += 1
                duration = max(
                    0.0, self.timestamps[end - 1] - self.timestamps[index]
                )
                bad_runs.append((state, end - index, duration))
            index = end

        id_changes = 0
        previous_id = None
        for target_id in self.ids:
            if target_id is None:
                continue
            if previous_id is not None and target_id != previous_id:
                id_changes += 1
            previous_id = target_id

        return {
            "processed_frames": processed_frames,
            "evaluated_frames": self.evaluated_frames,
            "visible_frames": self.visible_frames,
            "correct_frames": self.correct_frames,
            "prediction_coverage": (
                self.visible_frames / self.evaluated_frames
                if self.evaluated_frames
                else 0.0
            ),
            "within_40px_active": (
                self.correct_frames / self.evaluated_frames
                if self.evaluated_frames
                else 0.0
            ),
            "visible_identity_precision": (
                self.correct_frames / self.visible_frames
                if self.visible_frames
                else 0.0
            ),
            "missing_frames": self.evaluated_frames - self.visible_frames,
            "wrong_visible_frames": self.visible_frames - self.correct_frames,
            "severe_error_frames": self.severe_frames,
            "mean_error_px": float(np.mean(errors)) if len(errors) else None,
            "median_error_px": float(np.median(errors)) if len(errors) else None,
            "p90_error_px": (
                float(np.percentile(errors, 90)) if len(errors) else None
            ),
            "target_id_changes": id_changes,
            "identity_switch_signals": self.identity_switch_signals,
            "ranker_switch_signals": self.ranker_switch_signals,
            "failure_runs_after_first_correct": failures_after_correct,
            "recoveries_after_failure": recoveries,
            "longest_bad_run_frames": max(
                (item[1] for item in bad_runs), default=0
            ),
            "longest_bad_run_seconds": max(
                (item[2] for item in bad_runs), default=0.0
            ),
            "ended_in_failure": bool(self.states and self.states[-1] != "correct"),
        }


def _serialize_detection(item: ShapeDetection) -> list[float]:
    x, y, width, height = item.bbox
    return [
        float(item.center[0]),
        float(item.center[1]),
        float(item.observed_radius or item.radius),
        float(x),
        float(y),
        float(width),
        float(height),
        float(item.yolo_confidence or item.score),
    ]


def _deserialize_detection(values: list[float]) -> ShapeDetection:
    cx, cy, radius, x, y, width, height, confidence = (
        float(value) for value in values
    )
    return ShapeDetection(
        center=(cx, cy),
        radius=radius,
        bbox=(int(round(x)), int(round(y)), int(round(width)), int(round(height))),
        score=confidence,
        source="yolo",
        observed_radius=radius,
        yolo_confidence=confidence,
    )


def _cache_signature(
    video: Path,
    roi: tuple[int, int, int, int],
    model: Path,
    confidence: float,
    image_size: int,
) -> dict[str, object]:
    video_stat = video.stat()
    model_stat = model.stat()
    return {
        "version": CACHE_VERSION,
        "video_name": video.name,
        "video_size": video_stat.st_size,
        "video_mtime_ns": video_stat.st_mtime_ns,
        "roi": list(roi),
        "model": str(model.resolve()),
        "model_size": model_stat.st_size,
        "model_mtime_ns": model_stat.st_mtime_ns,
        "confidence": confidence,
        "image_size": image_size,
    }


def load_or_build_cache(
    video: Path,
    roi: tuple[int, int, int, int],
    cache_path: Path,
    model: Path,
    confidence: float,
    image_size: int,
    device: Optional[str],
) -> dict[str, object]:
    signature = _cache_signature(video, roi, model, confidence, image_size)
    if cache_path.is_file():
        with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("signature") == signature:
            print(f"cache hit: {video.name}", flush=True)
            return cached

    print(f"building YOLO cache: {video.name}", flush=True)
    detector = LieShapeYoloDetector(
        model,
        confidence=confidence,
        image_size=image_size,
        device=device,
        inference_stride=1,
    )
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    x, y, width, height = roi
    frames: list[list[list[float]]] = []
    started = time.perf_counter()
    while True:
        ok, full_frame = capture.read()
        if not ok:
            break
        if x + width > full_frame.shape[1] or y + height > full_frame.shape[0]:
            raise ValueError(
                f"ROI {roi} exceeds {video.name} frame "
                f"{full_frame.shape[1]}x{full_frame.shape[0]}"
            )
        frame = full_frame[y:y + height, x:x + width]
        cleaned, _ = LieDetectorTracker._remove_cursor(frame)
        detections = detector.detect(cleaned, reference_radius=None)
        frames.append([_serialize_detection(item) for item in detections])
        if len(frames) % 250 == 0:
            print(f"  {video.name}: {len(frames)} frames", flush=True)
    capture.release()
    payload = {
        "signature": signature,
        "fps": fps,
        "frame_count": len(frames),
        "build_seconds": time.perf_counter() - started,
        "frames": frames,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    return payload


def _selected_frame(
    timestamp: float,
    target_fps: Optional[float],
    next_timestamp: float,
) -> tuple[bool, float]:
    if target_fps is None:
        return True, next_timestamp
    if timestamp + 1e-9 < next_timestamp:
        return False, next_timestamp
    period = 1.0 / target_fps
    while next_timestamp <= timestamp + 1e-9:
        next_timestamp += period
    return True, next_timestamp


def _production_tracker(detector: CachedCandidateDetector) -> LieDetectorTracker:
    return LieDetectorTracker(
        candidate_detector=detector,
        multi_hypothesis_identity=True,
        stale_coast_recovery=True,
        identity_ranker_model="models/lie_identity_ranker.txt",
        identity_ranker_min_margin=2.0,
        identity_safety=False,
        state_aware_ranker=True,
        switch_event_model="models/lie_switch_event_model.txt",
        switch_event_min_probability=0.45,
    )


def evaluate_mode(
    video: Path,
    roi: tuple[int, int, int, int],
    active_start: float,
    active_end: float,
    cache: dict[str, object],
    target_fps: Optional[float],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    source_fps = float(cache["fps"])
    mode_name = "source_fps" if target_fps is None else f"{target_fps:g}_fps"
    detector = CachedCandidateDetector()
    current = _production_tracker(detector)
    ocsort = OCSortIdentityBaseline.create()
    shared = SharedObservationState()
    metrics = {
        "current_tracker": MethodMetrics("current_tracker"),
        "ocsort_locked_id": MethodMetrics("ocsort_locked_id"),
    }

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open {video}")
    x, y, width, height = roi
    frame_index = 0
    processed_frames = 0
    next_timestamp = 0.0
    rows: list[dict[str, object]] = []
    current_seed_frame = None
    current_seed_timestamp = None
    first_white_frame = None
    cached_frames = cache["frames"]
    while True:
        ok, full_frame = capture.read()
        if not ok:
            break
        if frame_index >= len(cached_frames):
            raise RuntimeError(f"cache ended early at frame {frame_index}")
        timestamp = frame_index / source_fps
        selected, next_timestamp = _selected_frame(
            timestamp, target_fps, next_timestamp
        )
        if not selected:
            frame_index += 1
            continue

        processed_frames += 1
        frame = full_frame[y:y + height, x:x + width]
        truth = cursor_ground_truth(frame)
        cleaned, _ = LieDetectorTracker._remove_cursor(frame)
        raw = [
            _deserialize_detection(values) for values in cached_frames[frame_index]
        ]
        learned, bright, allow_bright_seed = shared.prepare(cleaned, raw)
        if allow_bright_seed and first_white_frame is None:
            first_white_frame = frame_index

        detector.set_current(learned)
        current_result = current.update(frame, timestamp)
        if current_result.target_id is not None and current_seed_frame is None:
            current_seed_frame = frame_index
            current_seed_timestamp = timestamp

        oc_center, oc_id, oc_track_count = ocsort.update(
            learned, bright, allow_bright_seed, frame_index, timestamp
        )
        phase = evaluation_phase(timestamp, active_start, active_end)
        if truth is not None and phase is EvaluationPhase.ACTIVE:
            current_error, current_state = metrics["current_tracker"].add(
                timestamp,
                current_result.center,
                current_result.target_id,
                truth,
                identity_switched=current_result.identity_switched,
                ranker_switched=current_result.ranker_switched,
            )
            oc_error, oc_state = metrics["ocsort_locked_id"].add(
                timestamp, oc_center, oc_id, truth
            )
            for method, center, target_id, error, state, track_count in (
                (
                    "current_tracker",
                    current_result.center,
                    current_result.target_id,
                    current_error,
                    current_state,
                    len(current_result.tracks),
                ),
                (
                    "ocsort_locked_id",
                    oc_center,
                    oc_id,
                    oc_error,
                    oc_state,
                    oc_track_count,
                ),
            ):
                rows.append(
                    {
                        "video": video.name,
                        "mode": mode_name,
                        "method": method,
                        "source_frame": frame_index,
                        "timestamp": f"{timestamp:.6f}",
                        "target_id": "" if target_id is None else target_id,
                        "pred_x": "" if center is None else f"{center[0]:.3f}",
                        "pred_y": "" if center is None else f"{center[1]:.3f}",
                        "truth_x": f"{truth[0]:.3f}",
                        "truth_y": f"{truth[1]:.3f}",
                        "error_px": "" if error is None else f"{error:.3f}",
                        "state": state,
                        "detection_count": len(learned),
                        "track_count": track_count,
                    }
                )
        frame_index += 1
    capture.release()

    output = {
        "mode": mode_name,
        "source_fps": source_fps,
        "target_fps": source_fps if target_fps is None else target_fps,
        "source_frames": frame_index,
        "processed_frames": processed_frames,
        "first_white_frame": first_white_frame,
        "current_seed_frame": current_seed_frame,
        "current_seed_timestamp": current_seed_timestamp,
        "ocsort_seed_frame": ocsort.seed_frame,
        "ocsort_seed_timestamp": ocsort.seed_timestamp,
        "current_tracker": metrics["current_tracker"].summary(processed_frames),
        "ocsort_locked_id": metrics["ocsort_locked_id"].summary(processed_frames),
    }
    return output, rows


def _aggregate(results: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for mode in sorted({str(item["mode"]) for item in results}):
        mode_rows = [item for item in results if item["mode"] == mode]
        methods = {}
        for method in ("current_tracker", "ocsort_locked_id"):
            values = [item[method] for item in mode_rows]
            evaluated = sum(int(value["evaluated_frames"]) for value in values)
            visible = sum(int(value["visible_frames"]) for value in values)
            correct = sum(int(value["correct_frames"]) for value in values)
            methods[method] = {
                "videos": len(values),
                "evaluated_frames": evaluated,
                "prediction_coverage": visible / evaluated if evaluated else 0.0,
                "within_40px_active": correct / evaluated if evaluated else 0.0,
                "visible_identity_precision": correct / visible if visible else 0.0,
                "missing_frames": sum(int(value["missing_frames"]) for value in values),
                "wrong_visible_frames": sum(
                    int(value["wrong_visible_frames"]) for value in values
                ),
                "severe_error_frames": sum(
                    int(value["severe_error_frames"]) for value in values
                ),
                "target_id_changes": sum(
                    int(value["target_id_changes"]) for value in values
                ),
                "recoveries_after_failure": sum(
                    int(value["recoveries_after_failure"]) for value in values
                ),
                "videos_ended_in_failure": sum(
                    int(bool(value["ended_in_failure"])) for value in values
                ),
            }
        output[mode] = methods
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("ml/lie_videos_config.json")
    )
    parser.add_argument("--videos-dir", type=Path, default=Path("ml/videos"))
    parser.add_argument(
        "--auto-video",
        action="append",
        type=Path,
        default=[],
        help="video absent from config; repeatable, with ROI/window auto-detected",
    )
    parser.add_argument(
        "--clips",
        default=None,
        help="comma-separated recording numbers; default: every available configured clip",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--yolo-conf", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--yolo-imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--device", default="")
    parser.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS)
    parser.add_argument(
        "--modes",
        choices=("both", "source", "realtime"),
        default="both",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("log/lie_ocsort_baseline/cache")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("log/lie_ocsort_baseline/summary.json"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("log/lie_ocsort_baseline/frames.csv"),
    )
    args = parser.parse_args()
    if args.target_fps <= 0:
        parser.error("--target-fps must be positive")
    if not args.config.is_file():
        parser.error(f"config not found: {args.config}")
    if not args.model.is_file():
        parser.error(f"model not found: {args.model}")

    recordings = []
    if args.auto_video:
        from ml.detect_lie_panels import detect_recording

        for video in args.auto_video:
            resolved = video.resolve()
            entry = detect_recording(resolved, probe_count=30)
            if entry is None:
                parser.error(f"unable to detect panel ROI in {resolved}")
            recordings.append((resolved, entry))
    else:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        requested = (
            {int(value.strip()) for value in args.clips.split(",") if value.strip()}
            if args.clips
            else None
        )
        for entry in config.get("recordings", []):
            stem = Path(str(entry["filename"])).stem
            suffix = stem.removeprefix("测谎录屏")
            if requested is not None and (not suffix.isdigit() or int(suffix) not in requested):
                continue
            video = args.videos_dir / str(entry["filename"])
            if video.is_file():
                recordings.append((video, entry))
    if not recordings:
        parser.error("no requested configured recordings are available")

    mode_values: list[Optional[float]] = []
    if args.modes in ("both", "source"):
        mode_values.append(None)
    if args.modes in ("both", "realtime"):
        mode_values.append(args.target_fps)

    summaries = []
    frame_rows: list[dict[str, object]] = []
    experiment_started = time.perf_counter()
    for video, entry in recordings:
        roi = tuple(int(value) for value in entry["roi"])
        cache_path = args.cache_dir / f"{video.stem}.json.gz"
        cache = load_or_build_cache(
            video,
            roi,
            cache_path,
            args.model,
            args.yolo_conf,
            args.yolo_imgsz,
            args.device or None,
        )
        for target_fps in mode_values:
            summary, rows = evaluate_mode(
                video,
                roi,
                float(entry.get("active_start", 0.0)),
                float(entry.get("active_end", float("inf"))),
                cache,
                target_fps,
            )
            summary["video"] = video.name
            summaries.append(summary)
            frame_rows.extend(rows)
            current_ratio = summary["current_tracker"]["within_40px_active"]
            oc_ratio = summary["ocsort_locked_id"]["within_40px_active"]
            print(
                f"{video.name} {summary['mode']}: "
                f"current={current_ratio:.3f} ocsort={oc_ratio:.3f}",
                flush=True,
            )

    output = {
        "experiment": "shared-yolo shared-white-seed current-vs-ocsort",
        "model": str(args.model),
        "yolo_confidence": args.yolo_conf,
        "yolo_image_size": args.yolo_imgsz,
        "realtime_target_fps": args.target_fps,
        "runtime_seconds": time.perf_counter() - experiment_started,
        "recordings": summaries,
        "aggregate": _aggregate(summaries),
        "notes": {
            "ground_truth": "strict green cursor; evaluation only and removed before YOLO",
            "detections": "one cached raw YOLO pass, shared by both trackers and replay rates",
            "white_seed": "same production bright-shape observation; nearest active track binds REAL",
            "ocsort_policy": "bind once, preserve ID, no identity repair or ground-truth access",
            "missing_prediction": "counted as incorrect in within_40px_active",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    if frame_rows:
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
            writer.writeheader()
            writer.writerows(frame_rows)
    print(json.dumps(output["aggregate"], ensure_ascii=False, indent=2))
    print(f"summary: {args.output}")
    print(f"frames: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
