"""Classical computer-vision tracker for the lie-detector mini-game.

The tracker deliberately does not move the mouse.  It consumes an already
cropped mini-game frame and returns a target point that a caller may visualize
or, in an explicitly enabled test environment, pass to an input controller.

The implementation follows the classical detection/tracking pipeline:

* acquire the initially opaque target from a low-saturation bright mask;
* detect all shape candidates with OpenCV (Hough circles for circular targets,
  contour candidates otherwise);
* predict every track with a Kalman filter;
* associate detections to tracks with the Hungarian algorithm;
* retain the ID selected during acquisition when the target fades;
* for non-circular targets, keep a small beam of motion/shape hypotheses so a
  single missed contour or ambiguous decoy cannot permanently switch the ID.

The green in-game cursor is removed before feature extraction.  Its position is
never used as a target measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Iterable, Optional, Protocol

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - exercised only on minimal installs
    linear_sum_assignment = None


@dataclass
class ShapeDetection:
    center: tuple[float, float]
    radius: float
    bbox: tuple[int, int, int, int]
    score: float = 1.0
    circularity: float = 0.0
    source: str = "contour"
    shape_distance: float = 0.0
    observed_radius: Optional[float] = None
    collective_residual: float = 0.0
    orientation: Optional[float] = None
    collective_rotation_residual: float = 0.0
    yolo_confidence: float = 0.0
    contour: Optional[np.ndarray] = field(default=None, repr=False, compare=False)


class ShapeCandidateDetector(Protocol):
    """Optional learned detector interface; receives cursor-cleaned frames."""

    def detect(
        self,
        frame_bgr: np.ndarray,
        reference_radius: Optional[float] = None,
    ) -> list[ShapeDetection]: ...

    def reset(self) -> None: ...


@dataclass
class LieDetectorTrackingResult:
    acquired: bool
    target_id: Optional[int]
    center: Optional[tuple[float, float]]
    radius: Optional[float]
    confidence: float
    predicted_only: bool
    lost_frames: int
    detections: list[ShapeDetection]
    tracks: list[tuple[int, tuple[float, float], float, int]]
    hypothesis_count: int = 0
    recovery_active: bool = False
    collective_promoted: bool = False
    collective_delta: tuple[float, float] = (0.0, 0.0)
    collective_rotation_degrees: float = 0.0


@dataclass
class _TargetHypothesis:
    """One branch in the contour target's bounded multi-hypothesis tracker."""

    center: np.ndarray
    velocity: np.ndarray
    radius: float
    radius_velocity: float
    cost: float
    missed: int
    age: int
    last_timestamp: float
    predicted_only: bool
    group_evidence: float = 0.0
    last_collective_residual: float = 0.0

    def predict(self, timestamp: float) -> tuple[np.ndarray, float, float]:
        dt = float(np.clip(timestamp - self.last_timestamp, 1.0 / 120.0, 0.25))
        center = self.center + self.velocity * dt
        radius = max(1.0, self.radius + self.radius_velocity * dt)
        return center, radius, dt


def green_cursor_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Return a mask for the vivid green lie-detector cursor.

    This is used only to stop the cursor graphic from becoming a contour or a
    Hough circle.  The cursor centroid is intentionally not exposed to the
    tracker.
    """

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([35, 100, 50], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8),
    )
    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )


class _ShapeTrack:
    """One six-state Kalman track: x, y, vx, vy, radius, radius velocity."""

    def __init__(self, track_id: int, detection: ShapeDetection, timestamp: float):
        self.id = track_id
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0]],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = np.diag(
            [1.0, 1.0, 20.0, 20.0, 1.0, 5.0]
        ).astype(np.float32)
        self.kf.measurementNoiseCov = np.diag([15.0, 15.0, 9.0]).astype(
            np.float32
        )
        self.kf.errorCovPost = np.diag(
            [10.0, 10.0, 100.0, 100.0, 10.0, 20.0]
        ).astype(np.float32)
        x, y = detection.center
        self.kf.statePost = np.array(
            [[x], [y], [0.0], [0.0], [detection.radius], [0.0]],
            dtype=np.float32,
        )
        self.last_timestamp = timestamp
        self.age = 1
        self.hits = 1
        self.lost_frames = 0
        self.last_detection = detection
        self.predicted_only = False

    def _set_transition(self, dt: float) -> None:
        dt = float(np.clip(dt, 1.0 / 120.0, 0.25))
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0, 0, 0],
             [0, 1, 0, dt, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, dt],
             [0, 0, 0, 0, 0, 1]],
            dtype=np.float32,
        )

    def predict(self, timestamp: float) -> tuple[float, float, float]:
        self._set_transition(timestamp - self.last_timestamp)
        state = self.kf.predict().reshape(-1)
        self.last_timestamp = timestamp
        self.age += 1
        self.predicted_only = True
        return float(state[0]), float(state[1]), max(1.0, float(state[4]))

    def update(
        self,
        detection: ShapeDetection,
        timestamp: float,
    ) -> tuple[float, float, float]:
        measurement = np.array(
            [[detection.center[0]], [detection.center[1]], [detection.radius]],
            dtype=np.float32,
        )
        state = self.kf.correct(measurement).reshape(-1)
        self.last_detection = detection
        self.hits += 1
        self.lost_frames = 0
        self.predicted_only = False
        return float(state[0]), float(state[1]), max(1.0, float(state[4]))

    def mark_missed(self) -> None:
        self.lost_frames += 1

    @property
    def state(self) -> tuple[float, float, float]:
        # statePre is the current frame's estimate while a track is coasting;
        # statePost is the corrected estimate after a matched detection.
        state = self.kf.statePre if self.predicted_only else self.kf.statePost
        s = state.reshape(-1)
        return float(s[0]), float(s[1]), max(1.0, float(s[4]))


class LieDetectorTracker:
    """Track the initially highlighted shape among moving decoys."""

    def __init__(
        self,
        *,
        min_radius: int = 35,
        max_radius: int = 90,
        hough_param2: float = 30.0,
        max_match_distance: float = 55.0,
        max_lost_frames: int = 15,
        acquire_brightness: float = 205.0,
        hypothesis_count: int = 12,
        candidate_detector: Optional[ShapeCandidateDetector] = None,
    ) -> None:
        self.min_radius = int(min_radius)
        self.max_radius = int(max_radius)
        self.hough_param2 = float(hough_param2)
        self.max_match_distance = float(max_match_distance)
        self.max_lost_frames = int(max_lost_frames)
        self.acquire_brightness = float(acquire_brightness)
        self.hypothesis_count = max(1, int(hypothesis_count))
        self.candidate_detector = candidate_detector
        self.reset()

    def reset(self) -> None:
        self.tracks: dict[int, _ShapeTrack] = {}
        self.target_id: Optional[int] = None
        self.target_kind: Optional[str] = None
        self.target_contour: Optional[np.ndarray] = None
        self.target_radius: Optional[float] = None
        self.next_track_id = 1
        self.last_timestamp: Optional[float] = None
        self.target_hypotheses: list[_TargetHypothesis] = []
        self.target_recovery_active = False
        self.target_recovery_stable_frames = 0
        self.previous_contour_centers: Optional[np.ndarray] = None
        self.previous_contour_orientations: Optional[np.ndarray] = None
        self.collective_delta = np.zeros(2, dtype=np.float64)
        self.collective_rotation_delta = 0.0
        self.collective_motion_votes = 0
        if self.candidate_detector is not None:
            self.candidate_detector.reset()

    @staticmethod
    def _remove_cursor(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = green_cursor_mask(frame_bgr)
        expanded = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        )
        cleaned = cv2.inpaint(frame_bgr, expanded, 5, cv2.INPAINT_TELEA)
        return cleaned, mask

    @staticmethod
    def _initial_bright_shape(gray: np.ndarray, hsv: np.ndarray) -> Optional[ShapeDetection]:
        mask = np.where(
            (gray >= 210) & (hsv[:, :, 1] <= 75), 255, 0
        ).astype(np.uint8)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: Optional[tuple[float, ShapeDetection]] = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            x, y, w, h = cv2.boundingRect(contour)
            if not (700 <= area <= 40000 and 35 <= w <= 240 and 35 <= h <= 240):
                continue
            aspect = w / max(1.0, float(h))
            perimeter = float(cv2.arcLength(contour, True))
            circularity = 0.0 if perimeter <= 0 else 4.0 * math.pi * area / (perimeter * perimeter)
            component_mask = np.zeros_like(gray)
            cv2.drawContours(component_mask, [contour], -1, 255, -1)
            mean_brightness = float(cv2.mean(gray, mask=component_mask)[0])
            compactness_penalty = abs(math.log(max(aspect, 1e-3))) * 25.0
            rank = mean_brightness + circularity * 45.0 - compactness_penalty
            moments = cv2.moments(contour)
            if moments["m00"]:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
            else:
                cx, cy = x + w / 2.0, y + h / 2.0
            radius = 0.25 * (w + h)
            det = ShapeDetection(
                center=(float(cx), float(cy)),
                radius=float(radius),
                bbox=(x, y, w, h),
                score=mean_brightness / 255.0,
                circularity=float(circularity),
                source="bright",
                contour=contour.copy(),
            )
            if best is None or rank > best[0]:
                best = (rank, det)
        return None if best is None else best[1]

    def _circle_candidates(self, gray: np.ndarray) -> list[ShapeDetection]:
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=45,
            param1=90,
            param2=self.hough_param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )
        if circles is None:
            return []
        height, width = gray.shape
        detections: list[ShapeDetection] = []
        for cx, cy, radius in circles[0]:
            r = float(radius)
            x = max(0, int(round(cx - r)))
            y = max(0, int(round(cy - r)))
            w = min(width - x, int(round(2 * r)))
            h = min(height - y, int(round(2 * r)))
            detections.append(
                ShapeDetection(
                    center=(float(cx), float(cy)),
                    radius=r,
                    bbox=(x, y, w, h),
                    score=1.0,
                    circularity=1.0,
                    source="hough_circle",
                )
            )
        return detections

    def _contour_candidates(
        self,
        gray: np.ndarray,
        reference_contour: Optional[np.ndarray] = None,
    ) -> list[ShapeDetection]:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 35, 80)
        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[ShapeDetection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            x, y, w, h = cv2.boundingRect(contour)
            if not (900 <= area <= 35000 and 45 <= w <= 220 and 45 <= h <= 220):
                continue
            aspect = w / max(1.0, float(h))
            if not 0.35 <= aspect <= 2.8:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            circularity = 0.0 if perimeter <= 0 else 4.0 * math.pi * area / (perimeter * perimeter)
            moments = cv2.moments(contour)
            if not moments["m00"]:
                continue
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            observed_radius = 0.25 * (w + h)
            points = contour.reshape(-1, 2).astype(np.float64)
            relative = points - np.array([cx, cy], dtype=np.float64)
            point_radii = np.linalg.norm(relative, axis=1)
            angles = np.arctan2(relative[:, 1], relative[:, 0])
            weights = np.maximum(point_radii, 1.0) ** 2
            fifth_moment = np.sum(weights * np.exp(5j * angles))
            orientation = (
                None
                if abs(fifth_moment) <= 1e-6
                else float(np.angle(fifth_moment) / 5.0)
            )
            shape_distance = (
                0.0
                if reference_contour is None
                else float(
                    cv2.matchShapes(
                        reference_contour,
                        contour,
                        cv2.CONTOURS_MATCH_I1,
                        0.0,
                    )
                )
            )
            if reference_contour is not None and shape_distance > 1.35:
                continue
            detections.append(
                ShapeDetection(
                    center=(float(cx), float(cy)),
                    radius=float(observed_radius),
                    bbox=(x, y, w, h),
                    circularity=float(circularity),
                    source="contour",
                    shape_distance=shape_distance,
                    observed_radius=float(observed_radius),
                    orientation=orientation,
                    contour=contour.copy(),
                )
            )
        detections = self._deduplicate(detections, self.target_radius)
        if self.target_radius is not None:
            fixed_radius = float(self.target_radius)
            fixed_size = max(2, int(round(2.0 * fixed_radius)))
            for detection in detections:
                cx, cy = detection.center
                detection.radius = fixed_radius
                detection.bbox = (
                    int(round(cx - fixed_radius)),
                    int(round(cy - fixed_radius)),
                    fixed_size,
                    fixed_size,
                )
        return detections

    @staticmethod
    def _deduplicate(
        detections: Iterable[ShapeDetection],
        reference_radius: Optional[float] = None,
    ) -> list[ShapeDetection]:
        kept: list[ShapeDetection] = []
        for det in sorted(
            detections,
            key=lambda item: item.observed_radius or item.radius,
            reverse=True,
        ):
            if any(
                np.linalg.norm(np.subtract(det.center, other.center))
                < (
                    0.45 * reference_radius
                    if reference_radius is not None
                    else 0.35
                    * min(
                        det.observed_radius or det.radius,
                        other.observed_radius or other.radius,
                    )
                )
                for other in kept
            ):
                continue
            kept.append(det)
        return kept

    def _fuse_learned_candidates(
        self,
        classical: list[ShapeDetection],
        learned: list[ShapeDetection],
    ) -> list[ShapeDetection]:
        """Fuse YOLO boxes as evidence without losing classical recall.

        The synthetic-only model currently misses some real shapes, so it must
        not be used as a hard gate.  A nearby YOLO box annotates a classical
        candidate; only strong unmatched boxes are allowed to create a new
        candidate.
        """

        if not learned:
            return classical
        fused = list(classical)
        matched_classical: set[int] = set()
        for learned_detection in sorted(
            learned, key=lambda item: item.yolo_confidence, reverse=True
        ):
            if classical:
                distances = np.asarray(
                    [
                        np.linalg.norm(
                            np.subtract(item.center, learned_detection.center)
                        )
                        for item in classical
                    ]
                )
                index = int(np.argmin(distances))
                gate = max(
                    18.0,
                    0.70
                    * float(
                        self.target_radius
                        or classical[index].observed_radius
                        or classical[index].radius
                    ),
                )
                if distances[index] <= gate and index not in matched_classical:
                    classical[index].yolo_confidence = max(
                        classical[index].yolo_confidence,
                        learned_detection.yolo_confidence,
                    )
                    matched_classical.add(index)
                    continue
            if learned_detection.yolo_confidence >= 0.10:
                fused.append(learned_detection)
        return self._deduplicate(fused, self.target_radius)

    def _annotate_collective_motion(
        self,
        detections: list[ShapeDetection],
    ) -> None:
        """Estimate the displacement shared by the decoy constellation.

        The recording's decoys use the same per-frame trajectory. The mode of
        all short point-set displacements therefore estimates that motion even
        without knowing individual decoy IDs. A target candidate is unusual
        when no previous contour, shifted by that group motion, explains it.
        """

        contour_detections = [
            detection
            for detection in detections
            if detection.source in ("contour", "yolo")
            and detection.shape_distance <= 1.35
        ]
        current = np.asarray(
            [detection.center for detection in contour_detections],
            dtype=np.float64,
        )
        current_orientations = np.asarray(
            [
                np.nan if detection.orientation is None else detection.orientation
                for detection in contour_detections
            ],
            dtype=np.float64,
        )
        if (
            self.previous_contour_centers is None
            or not len(self.previous_contour_centers)
            or not len(current)
        ):
            self.previous_contour_centers = current
            self.previous_contour_orientations = current_orientations
            return

        differences = (
            current[:, None, :] - self.previous_contour_centers[None, :, :]
        ).reshape(-1, 2)
        differences = differences[np.linalg.norm(differences, axis=1) <= 18.0]
        if len(differences):
            bin_size = 2.0
            bins = np.rint(differences / bin_size).astype(np.int32)
            keys, counts = np.unique(bins, axis=0, return_counts=True)
            # Prefer a well-supported displacement close to the previous
            # group motion; this suppresses stationary background contours.
            support = counts.astype(np.float64) - 0.12 * np.linalg.norm(
                keys * bin_size - self.collective_delta,
                axis=1,
            )
            best = int(np.argmax(support))
            seed = keys[best].astype(np.float64) * bin_size
            nearby = differences[np.linalg.norm(differences - seed, axis=1) <= 2.5]
            measured = np.median(nearby, axis=0)
            self.collective_delta = 0.35 * self.collective_delta + 0.65 * measured
            self.collective_motion_votes = int(counts[best])

        expected = self.previous_contour_centers + self.collective_delta
        distances = np.linalg.norm(current[:, None, :] - expected[None, :, :], axis=2)
        nearest_previous = np.argmin(distances, axis=1)
        position_residuals = distances[np.arange(len(current)), nearest_previous]

        period = 2.0 * math.pi / 5.0
        rotation_differences: list[float] = []
        if self.previous_contour_orientations is not None:
            for index, previous_index in enumerate(nearest_previous):
                current_angle = current_orientations[index]
                previous_angle = self.previous_contour_orientations[previous_index]
                if (
                    position_residuals[index] <= 8.0
                    and np.isfinite(current_angle)
                    and np.isfinite(previous_angle)
                ):
                    difference = (current_angle - previous_angle + period / 2.0) % period
                    rotation_differences.append(float(difference - period / 2.0))
        if rotation_differences:
            rotation_array = np.asarray(rotation_differences)
            bins = np.rint(np.degrees(rotation_array) / 2.0).astype(np.int32)
            keys, counts = np.unique(bins, return_counts=True)
            seed = math.radians(float(keys[int(np.argmax(counts))]) * 2.0)
            nearby = rotation_array[
                np.abs(
                    (rotation_array - seed + period / 2.0) % period - period / 2.0
                )
                <= math.radians(3.0)
            ]
            measured_rotation = float(np.median(nearby))
            self.collective_rotation_delta = (
                0.35 * self.collective_rotation_delta + 0.65 * measured_rotation
            )

        for index, detection in enumerate(contour_detections):
            rotation_residual_degrees = 0.0
            if self.previous_contour_orientations is not None:
                previous_angle = self.previous_contour_orientations[nearest_previous[index]]
                current_angle = current_orientations[index]
                if np.isfinite(previous_angle) and np.isfinite(current_angle):
                    residual = (
                        current_angle
                        - previous_angle
                        - self.collective_rotation_delta
                        + period / 2.0
                    ) % period - period / 2.0
                    rotation_residual_degrees = abs(math.degrees(float(residual)))
            detection.collective_rotation_residual = rotation_residual_degrees
            detection.collective_residual = float(
                position_residuals[index] + 0.55 * rotation_residual_degrees
            )
        self.previous_contour_centers = current
        self.previous_contour_orientations = current_orientations

    @staticmethod
    def _hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
        if cost.size == 0:
            return []
        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(cost)
            return list(zip(rows.tolist(), cols.tolist()))
        # Deterministic greedy fallback.  The regular project environment uses
        # SciPy; this keeps the module importable in reduced deployments.
        pairs: list[tuple[int, int]] = []
        available_rows = set(range(cost.shape[0]))
        available_cols = set(range(cost.shape[1]))
        while available_rows and available_cols:
            row, col = min(
                ((r, c) for r in available_rows for c in available_cols),
                key=lambda rc: cost[rc],
            )
            pairs.append((row, col))
            available_rows.remove(row)
            available_cols.remove(col)
        return pairs

    def _spawn(self, detection: ShapeDetection, timestamp: float) -> _ShapeTrack:
        track = _ShapeTrack(self.next_track_id, detection, timestamp)
        self.tracks[track.id] = track
        self.next_track_id += 1
        return track

    def _associate(self, detections: list[ShapeDetection], timestamp: float) -> None:
        track_list = list(self.tracks.values())
        predictions = [track.predict(timestamp) for track in track_list]
        if not track_list:
            for detection in detections:
                self._spawn(detection, timestamp)
            return
        if not detections:
            for track in track_list:
                track.mark_missed()
            self._drop_stale_tracks()
            return

        large = 1e6
        cost = np.full((len(track_list), len(detections)), large, dtype=np.float32)
        for row, (track, prediction) in enumerate(zip(track_list, predictions)):
            px, py, pr = prediction
            gate = self.max_match_distance + min(track.lost_frames, 5) * 8.0
            if track.id == self.target_id and self.target_kind == "contour":
                # A wide reacquisition gate eagerly jumps to one of the many
                # identical stars. The hypothesis beam handles recovery while
                # this conservative primary association protects identity.
                gate = 35.0
            for col, detection in enumerate(detections):
                distance = math.hypot(
                    detection.center[0] - px,
                    detection.center[1] - py,
                )
                radius_delta = abs(detection.radius - pr)
                if distance <= gate and radius_delta <= max(30.0, pr * 0.75):
                    if track.id == self.target_id and self.target_kind == "contour":
                        if (
                            detection.source in ("contour", "yolo")
                            and detection.shape_distance > 1.0
                        ):
                            continue
                    shape_cost = (
                        80.0 * detection.shape_distance
                        if track.id == self.target_id
                        and self.target_kind == "contour"
                        and detection.source in ("contour", "yolo")
                        else 0.0
                    )
                    learned_bonus = 10.0 * detection.yolo_confidence
                    cost[row, col] = (
                        distance + 0.25 * radius_delta + shape_cost - learned_bonus
                    )

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        # Protect the identity selected during the opaque acquisition phase.
        # A single global Hungarian solve is allowed to sacrifice that track
        # when doing so lowers the sum for dozens of visually identical
        # decoys.  For the mini-game the selected identity is semantically
        # special, so match it first and run Hungarian on everything left.
        target_row = next(
            (row for row, track in enumerate(track_list) if track.id == self.target_id),
            None,
        )
        if target_row is not None:
            bright_cols = [
                col
                for col, detection in enumerate(detections)
                if detection.source == "bright" and cost[target_row, col] < large
            ]
            target_col = (
                min(bright_cols, key=lambda col: cost[target_row, col])
                if bright_cols
                else int(np.argmin(cost[target_row]))
            )
            if cost[target_row, target_col] < large:
                track_list[target_row].update(
                    detections[target_col],
                    timestamp,
                )
                matched_tracks.add(target_row)
                matched_detections.add(target_col)

        remaining_rows = [row for row in range(len(track_list)) if row not in matched_tracks]
        remaining_cols = [col for col in range(len(detections)) if col not in matched_detections]
        reduced_cost = cost[np.ix_(remaining_rows, remaining_cols)]
        for reduced_row, reduced_col in self._hungarian(reduced_cost):
            row = remaining_rows[reduced_row]
            col = remaining_cols[reduced_col]
            if cost[row, col] >= large:
                continue
            track_list[row].update(detections[col], timestamp)
            matched_tracks.add(row)
            matched_detections.add(col)

        for row, track in enumerate(track_list):
            if row not in matched_tracks:
                track.mark_missed()
        for col, detection in enumerate(detections):
            if col not in matched_detections:
                self._spawn(detection, timestamp)
        self._drop_stale_tracks()

    def _drop_stale_tracks(self) -> None:
        stale = [
            track_id
            for track_id, track in self.tracks.items()
            if track.lost_frames > self.max_lost_frames and track_id != self.target_id
        ]
        for track_id in stale:
            del self.tracks[track_id]

    @staticmethod
    def _clamp_velocity(velocity: np.ndarray, limit: float = 650.0) -> np.ndarray:
        speed = float(np.linalg.norm(velocity))
        if speed <= limit or speed <= 1e-6:
            return velocity
        return velocity * (limit / speed)

    def _seed_target_hypothesis(
        self,
        detection: ShapeDetection,
        timestamp: float,
    ) -> None:
        """Start or authoritatively correct the beam from an opaque target."""

        center = np.asarray(detection.center, dtype=np.float64)
        velocity = np.zeros(2, dtype=np.float64)
        radius_velocity = 0.0
        if self.target_hypotheses:
            previous = self.target_hypotheses[0]
            dt = float(np.clip(timestamp - previous.last_timestamp, 1.0 / 120.0, 0.25))
            measured_velocity = (center - previous.center) / dt
            velocity = self._clamp_velocity(
                0.55 * previous.velocity + 0.45 * measured_velocity
            )
            radius_velocity = float(
                np.clip(
                    0.55 * previous.radius_velocity
                    + 0.45 * (detection.radius - previous.radius) / dt,
                    -250.0,
                    250.0,
                )
            )
        self.target_hypotheses = [
            _TargetHypothesis(
                center=center,
                velocity=velocity,
                radius=float(detection.radius),
                radius_velocity=radius_velocity,
                cost=0.0,
                missed=0,
                age=1,
                last_timestamp=timestamp,
                predicted_only=False,
                group_evidence=0.0,
                last_collective_residual=0.0,
            )
        ]

    @staticmethod
    def _hypothesis_is_distinct(
        candidate: _TargetHypothesis,
        kept: list[_TargetHypothesis],
    ) -> bool:
        """NMS in position/velocity space, preserving genuinely different paths."""

        for other in kept:
            center_distance = float(np.linalg.norm(candidate.center - other.center))
            velocity_distance = float(np.linalg.norm(candidate.velocity - other.velocity))
            if center_distance < 14.0 and velocity_distance < 90.0:
                return False
        return True

    def _update_target_hypotheses(
        self,
        detections: list[ShapeDetection],
        timestamp: float,
        bright: Optional[ShapeDetection],
    ) -> Optional[_TargetHypothesis]:
        """Advance a bounded Top-K beam for an initially highlighted contour.

        Each prior branch emits a coast branch and zero or more detection
        branches.  Keeping both is important: near a crossing, the locally
        cheapest star is often the wrong star, while the coast branch retains
        enough history to recover after the shapes separate again.
        """

        if bright is not None:
            self._seed_target_hypothesis(bright, timestamp)
            return self.target_hypotheses[0]
        if not self.target_hypotheses:
            return None

        contour_detections = [
            detection
            for detection in detections
            if detection.source in ("contour", "yolo")
            and detection.shape_distance <= 1.35
        ]
        branches: list[_TargetHypothesis] = []
        for hypothesis in self.target_hypotheses:
            predicted_center, predicted_radius, dt = hypothesis.predict(timestamp)

            # Never discard the motion-only explanation just because a nearby
            # decoy exists. Its penalty grows so a later consistent contour
            # sequence can overtake it.
            branches.append(
                _TargetHypothesis(
                    center=predicted_center,
                    velocity=hypothesis.velocity * 0.985,
                    radius=predicted_radius,
                    radius_velocity=hypothesis.radius_velocity * 0.95,
                    cost=0.965 * hypothesis.cost + 1.35 + 0.22 * hypothesis.missed,
                    missed=hypothesis.missed + 1,
                    age=hypothesis.age + 1,
                    last_timestamp=timestamp,
                    predicted_only=True,
                    group_evidence=0.96 * hypothesis.group_evidence,
                    last_collective_residual=0.0,
                )
            )

            gate = min(145.0, 40.0 + 9.0 * hypothesis.missed)
            for detection in contour_detections:
                measured_center = np.asarray(detection.center, dtype=np.float64)
                innovation = measured_center - predicted_center
                distance = float(np.linalg.norm(innovation))
                radius_delta = abs(float(detection.radius) - predicted_radius)
                if distance > gate or radius_delta > max(38.0, predicted_radius * 0.85):
                    continue

                measured_velocity = (measured_center - hypothesis.center) / dt
                measured_velocity = self._clamp_velocity(measured_velocity)
                new_velocity = self._clamp_velocity(
                    0.62 * hypothesis.velocity + 0.38 * measured_velocity
                )
                acceleration = float(np.linalg.norm(new_velocity - hypothesis.velocity))
                new_radius_velocity = float(
                    np.clip(
                        0.65 * hypothesis.radius_velocity
                        + 0.35 * (detection.radius - hypothesis.radius) / dt,
                        -250.0,
                        250.0,
                    )
                )
                position_cost = 0.5 * (distance / 22.0) ** 2
                shape_cost = 1.8 * min(2.0, detection.shape_distance)
                radius_cost = 0.15 * (radius_delta / 18.0) ** 2
                acceleration_cost = 0.18 * (acceleration / 180.0) ** 2
                evidence_increment = float(
                    np.clip((detection.collective_residual - 8.0) / 10.0, -0.5, 2.5)
                )
                group_evidence = 0.90 * hypothesis.group_evidence + evidence_increment
                branches.append(
                    _TargetHypothesis(
                        center=measured_center,
                        velocity=new_velocity,
                        radius=float(detection.radius),
                        radius_velocity=new_radius_velocity,
                        cost=(
                            0.965 * hypothesis.cost
                            + position_cost
                            + shape_cost
                            + radius_cost
                            + acceleration_cost
                            - 0.45 * max(0.0, min(6.0, group_evidence))
                        ),
                        missed=0,
                        age=hypothesis.age + 1,
                        last_timestamp=timestamp,
                        predicted_only=False,
                        group_evidence=group_evidence,
                        last_collective_residual=detection.collective_residual,
                    )
                )

        kept: list[_TargetHypothesis] = []
        for candidate in sorted(branches, key=lambda item: item.cost):
            if self._hypothesis_is_distinct(candidate, kept):
                kept.append(candidate)
                if len(kept) >= self.hypothesis_count:
                    break
        self.target_hypotheses = kept or sorted(branches, key=lambda item: item.cost)[:1]
        return self.target_hypotheses[0] if self.target_hypotheses else None

    def update(
        self,
        frame_bgr: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> LieDetectorTrackingResult:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("frame_bgr must be a non-empty BGR image")
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        cleaned, _ = self._remove_cursor(frame_bgr)
        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(cleaned, cv2.COLOR_BGR2HSV)

        initial = self._initial_bright_shape(gray, hsv)
        initial_is_valid = (
            initial is not None
            and initial.score * 255.0 >= self.acquire_brightness
        )
        if self.target_id is None:
            if initial_is_valid:
                aspect = initial.bbox[2] / max(1.0, initial.bbox[3])
                self.target_kind = (
                    "circle"
                    if initial.circularity >= 0.72 and 0.75 <= aspect <= 1.33
                    else "contour"
                )
                self.target_contour = initial.contour.copy()
                self.target_radius = float(initial.radius)

        detections = (
            self._circle_candidates(gray)
            if self.target_kind in (None, "circle")
            else self._contour_candidates(gray, self.target_contour)
        )
        if self.candidate_detector is not None and self.target_kind is not None:
            learned = self.candidate_detector.detect(cleaned, self.target_radius)
            detections = self._fuse_learned_candidates(detections, learned)
        if self.target_kind == "contour" and initial_is_valid:
            detections.append(initial)
        if self.target_kind == "contour":
            self._annotate_collective_motion(detections)
        self._associate(detections, timestamp)

        if self.target_id is None and initial_is_valid and self.tracks:
            nearest = min(
                self.tracks.values(),
                key=lambda track: math.hypot(
                    track.state[0] - initial.center[0],
                    track.state[1] - initial.center[1],
                ),
            )
            if math.hypot(
                nearest.state[0] - initial.center[0],
                nearest.state[1] - initial.center[1],
            ) <= max(30.0, initial.radius * 0.6):
                self.target_id = nearest.id

        contour_hypothesis: Optional[_TargetHypothesis] = None
        if self.target_kind == "contour" and self.target_id is not None:
            contour_target = self.tracks.get(self.target_id)
            last_detection = None if contour_target is None else contour_target.last_detection
            suspicious_match = (
                contour_target is None
                or contour_target.lost_frames >= 3
            )
            if not self.target_recovery_active and suspicious_match:
                self.target_recovery_active = True
                self.target_recovery_stable_frames = 0

            if contour_target is not None and not self.target_recovery_active:
                # The conservative primary association remains the committed
                # path while measurements are strong. Its state continuously
                # seeds velocity for a future ambiguous interval.
                tx, ty, tradius = contour_target.state
                anchor = ShapeDetection(
                    center=(tx, ty),
                    radius=tradius,
                    bbox=(0, 0, 0, 0),
                    source="anchor",
                )
                self._seed_target_hypothesis(anchor, timestamp)
            else:
                contour_hypothesis = self._update_target_hypotheses(
                    detections,
                    timestamp,
                    None,
                )
                if (
                    contour_target is not None
                    and not contour_target.predicted_only
                    and last_detection is not None
                    and last_detection.shape_distance <= 0.60
                ):
                    self.target_recovery_stable_frames += 1
                    if self.target_recovery_stable_frames >= 6:
                        tx, ty, tradius = contour_target.state
                        anchor = ShapeDetection(
                            center=(tx, ty),
                            radius=tradius,
                            bbox=(0, 0, 0, 0),
                            source="anchor",
                        )
                        self._seed_target_hypothesis(anchor, timestamp)
                        self.target_recovery_active = False
                        self.target_recovery_stable_frames = 0
                        contour_hypothesis = None
                elif self.target_recovery_active:
                    self.target_recovery_stable_frames = 0

        target = self.tracks.get(self.target_id) if self.target_id is not None else None
        if target is None:
            return LieDetectorTrackingResult(
                acquired=False,
                target_id=self.target_id,
                center=None,
                radius=None,
                confidence=0.0,
                predicted_only=True,
                lost_frames=0,
                detections=detections,
                tracks=[
                    (track.id, track.state[:2], track.state[2], track.lost_frames)
                    for track in self.tracks.values()
                ],
                hypothesis_count=len(self.target_hypotheses),
                recovery_active=self.target_recovery_active,
                collective_delta=tuple(float(value) for value in self.collective_delta),
                collective_rotation_degrees=math.degrees(
                    self.collective_rotation_delta
                ),
            )

        collective_promoted = False
        if contour_hypothesis is not None:
            collective_candidate = min(
                self.target_hypotheses,
                key=lambda hypothesis: hypothesis.cost
                - 0.75 * hypothesis.group_evidence,
            )
            primary_center = np.asarray(target.state[:2], dtype=np.float64)
            disagreement = float(
                np.linalg.norm(collective_candidate.center - primary_center)
            )
            primary_residual = target.last_detection.collective_residual
            collective_promoted = (
                collective_candidate.group_evidence >= 10.0
                and (
                    target.lost_frames >= 3
                    or (primary_residual <= 8.0 and disagreement >= 100.0)
                )
            )
            contour_hypothesis = collective_candidate if collective_promoted else None

        if contour_hypothesis is not None:
            x, y = (float(value) for value in contour_hypothesis.center)
            radius = float(contour_hypothesis.radius)
            predicted_only = contour_hypothesis.predicted_only
            lost_frames = contour_hypothesis.missed
            confidence = math.exp(-0.16 * contour_hypothesis.missed) * math.exp(
                -0.035 * min(20.0, contour_hypothesis.cost)
            )
        else:
            x, y, radius = target.state
            predicted_only = target.predicted_only
            lost_frames = target.lost_frames
            confidence = min(1.0, target.hits / 5.0) * math.exp(
                -0.18 * target.lost_frames
            )
        debug_tracks = [
            (track.id, track.state[:2], track.state[2], track.lost_frames)
            for track in self.tracks.values()
        ]
        if contour_hypothesis is not None:
            debug_tracks = [
                (
                    track_id,
                    (x, y) if track_id == target.id else center,
                    radius if track_id == target.id else track_radius,
                    lost_frames if track_id == target.id else lost,
                )
                for track_id, center, track_radius, lost in debug_tracks
            ]
        return LieDetectorTrackingResult(
            acquired=True,
            target_id=target.id,
            center=(x, y),
            radius=radius,
            confidence=float(confidence),
            predicted_only=predicted_only,
            lost_frames=lost_frames,
            detections=detections,
            tracks=debug_tracks,
            hypothesis_count=len(self.target_hypotheses),
            recovery_active=self.target_recovery_active,
            collective_promoted=collective_promoted,
            collective_delta=tuple(float(value) for value in self.collective_delta),
            collective_rotation_degrees=math.degrees(self.collective_rotation_delta),
        )
