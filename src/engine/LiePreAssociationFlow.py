"""Sparse optical-flow observations computed before detection association.

The observer tracks cursor-blind image points from the previous frame into the
current frame.  A robust local similarity transform predicts each track's new
centre, while a second robust transform across track centres estimates the
shared "constellation" motion.  No target identity or evaluation cursor is
used here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Protocol

import cv2
import numpy as np


class FlowTrackView(Protocol):
    id: int
    lost_frames: int
    predicted_only: bool
    flow_coasted: bool

    @property
    def state(self) -> tuple[float, float, float]: ...


@dataclass(frozen=True)
class FlowTrackSnapshot:
    center: tuple[float, float]
    radius: float


@dataclass
class FlowTrackObservation:
    track_id: int
    source_center: tuple[float, float]
    predicted_center: tuple[float, float]
    displacement: tuple[float, float]
    confidence: float
    forward_backward_error: float
    inlier_ratio: float
    coverage: float
    point_count: int
    rotation_degrees: float = 0.0
    scale: float = 1.0
    local_fit_error: float = 0.0
    group_residual: float = 0.0
    group_confidence: float = 0.0


class PreAssociationFlow:
    """Maintain one-frame sparse LK state for all visible tracks."""

    MAX_CORNERS_PER_TRACK = 40
    MIN_TRACK_POINTS = 6
    MAX_FORWARD_BACKWARD_ERROR = 1.75
    MAX_LK_ERROR = 28.0
    LOCAL_RANSAC_ERROR = 1.8
    GROUP_RANSAC_ERROR = 3.5

    def __init__(self, *, max_coast_frames: int = 3) -> None:
        self.max_coast_frames = max(0, int(max_coast_frames))
        self.reset()

    def reset(self) -> None:
        self.previous_gray: np.ndarray | None = None
        self.previous_tracks: dict[int, FlowTrackSnapshot] = {}

    @staticmethod
    def _feature_points(
        gray: np.ndarray, snapshot: FlowTrackSnapshot
    ) -> np.ndarray | None:
        height, width = gray.shape[:2]
        cx, cy = snapshot.center
        radius = max(8.0, float(snapshot.radius))
        outer = max(10, int(round(1.10 * radius)))
        x1 = max(0, int(math.floor(cx - outer)))
        y1 = max(0, int(math.floor(cy - outer)))
        x2 = min(width, int(math.ceil(cx + outer + 1)))
        y2 = min(height, int(math.ceil(cy + outer + 1)))
        if x2 - x1 < 12 or y2 - y1 < 12:
            return None
        crop = gray[y1:y2, x1:x2]
        local_center = (int(round(cx - x1)), int(round(cy - y1)))
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        inner = max(6, int(round(0.30 * radius)))
        cv2.circle(mask, local_center, outer, 255, -1)
        cv2.circle(mask, local_center, inner, 0, -1)
        points = cv2.goodFeaturesToTrack(
            crop,
            maxCorners=PreAssociationFlow.MAX_CORNERS_PER_TRACK,
            qualityLevel=0.018,
            minDistance=3.0,
            mask=mask,
            blockSize=5,
        )
        if points is None:
            return None
        points = points.reshape(-1, 2)
        points[:, 0] += x1
        points[:, 1] += y1
        return points.astype(np.float32).reshape(-1, 1, 2)

    @staticmethod
    def _coverage(points: np.ndarray, center: tuple[float, float]) -> float:
        relative = points - np.asarray(center, dtype=np.float32)
        angles = (np.arctan2(relative[:, 1], relative[:, 0]) + math.pi) / (2 * math.pi)
        occupied = len(np.unique(np.floor(angles * 8.0).astype(np.int32) % 8))
        angular = min(1.0, occupied / 6.0)
        count = min(1.0, len(points) / 18.0)
        return float(angular * count)

    @staticmethod
    def _transform_point(affine: np.ndarray, point: tuple[float, float]) -> np.ndarray:
        return affine[:, :2] @ np.asarray(point, dtype=np.float64) + affine[:, 2]

    def observe(self, current_gray: np.ndarray) -> dict[int, FlowTrackObservation]:
        previous_gray = self.previous_gray
        if previous_gray is None or previous_gray.shape != current_gray.shape:
            return {}

        point_sets: list[np.ndarray] = []
        owners: list[int] = []
        for track_id, snapshot in self.previous_tracks.items():
            points = self._feature_points(previous_gray, snapshot)
            if points is None or len(points) < self.MIN_TRACK_POINTS:
                continue
            point_sets.append(points)
            owners.extend([track_id] * len(points))
        if not point_sets:
            return {}

        source_all = np.concatenate(point_sets, axis=0)
        moved, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            source_all,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01),
        )
        if moved is None or forward_status is None:
            return {}
        returned, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            previous_gray,
            moved,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01),
        )
        if returned is None or backward_status is None:
            return {}

        source_flat = source_all.reshape(-1, 2)
        moved_flat = moved.reshape(-1, 2)
        returned_flat = returned.reshape(-1, 2)
        fb_errors = np.linalg.norm(returned_flat - source_flat, axis=1)
        valid = (forward_status.reshape(-1) > 0) & (backward_status.reshape(-1) > 0)
        valid &= np.isfinite(moved_flat).all(axis=1)
        valid &= fb_errors <= self.MAX_FORWARD_BACKWARD_ERROR
        if forward_error is not None:
            valid &= forward_error.reshape(-1) <= self.MAX_LK_ERROR

        owner_array = np.asarray(owners, dtype=np.int64)
        observations: dict[int, FlowTrackObservation] = {}
        for track_id, snapshot in self.previous_tracks.items():
            selected = valid & (owner_array == track_id)
            source = source_flat[selected]
            target = moved_flat[selected]
            if len(source) < self.MIN_TRACK_POINTS:
                continue
            affine, inliers = cv2.estimateAffinePartial2D(
                source.reshape(-1, 1, 2),
                target.reshape(-1, 1, 2),
                method=cv2.RANSAC,
                ransacReprojThreshold=self.LOCAL_RANSAC_ERROR,
                maxIters=160,
                confidence=0.99,
                refineIters=8,
            )
            if affine is None or inliers is None:
                continue
            inlier_mask = inliers.reshape(-1) > 0
            if np.count_nonzero(inlier_mask) < self.MIN_TRACK_POINTS:
                continue
            predicted = self._transform_point(affine, snapshot.center)
            displacement = predicted - np.asarray(snapshot.center, dtype=np.float64)
            matrix = np.asarray(affine[:, :2], dtype=np.float64)
            scale = math.sqrt(abs(float(np.linalg.det(matrix))))
            rotation = math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
            if (
                not 0.72 <= scale <= 1.32
                or abs(rotation) > 45.0
                or float(np.linalg.norm(displacement)) > max(100.0, 2.8 * snapshot.radius)
            ):
                continue
            transformed = source @ matrix.T + np.asarray(affine[:, 2])
            fit_errors = np.linalg.norm(transformed - target, axis=1)
            median_fit = float(np.median(fit_errors[inlier_mask]))
            median_fb = float(np.median(fb_errors[selected][inlier_mask]))
            inlier_ratio = float(np.mean(inlier_mask))
            coverage = self._coverage(source[inlier_mask], snapshot.center)
            confidence = (
                inlier_ratio
                * math.sqrt(max(0.0, coverage))
                * math.exp(-median_fb / 1.15)
                * math.exp(-median_fit / 2.0)
            )
            if confidence <= 0.02:
                continue
            observations[track_id] = FlowTrackObservation(
                track_id=track_id,
                source_center=snapshot.center,
                predicted_center=(float(predicted[0]), float(predicted[1])),
                displacement=(float(displacement[0]), float(displacement[1])),
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                forward_backward_error=median_fb,
                inlier_ratio=inlier_ratio,
                coverage=coverage,
                point_count=int(np.count_nonzero(inlier_mask)),
                rotation_degrees=float(rotation),
                scale=float(scale),
                local_fit_error=median_fit,
            )

        self._annotate_group_motion(observations)
        return observations

    def _annotate_group_motion(
        self, observations: Mapping[int, FlowTrackObservation]
    ) -> None:
        eligible = [item for item in observations.values() if item.confidence >= 0.18]
        if len(eligible) < 4:
            return
        source = np.asarray([item.source_center for item in eligible], dtype=np.float32)
        target = np.asarray([item.predicted_center for item in eligible], dtype=np.float32)
        affine, inliers = cv2.estimateAffinePartial2D(
            source.reshape(-1, 1, 2),
            target.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=self.GROUP_RANSAC_ERROR,
            maxIters=200,
            confidence=0.99,
            refineIters=10,
        )
        if affine is None or inliers is None:
            return
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < 3:
            return
        group_confidence = (
            inlier_count
            / len(eligible)
            * float(np.median([item.confidence for item in eligible]))
        )
        expected = np.column_stack(
            [source, np.ones(len(source), dtype=np.float32)]
        ) @ np.asarray(affine, dtype=np.float64).T
        residuals = np.linalg.norm(target - expected, axis=1)
        for item, residual in zip(eligible, residuals):
            item.group_residual = float(residual)
            item.group_confidence = float(np.clip(group_confidence, 0.0, 1.0))

    def commit(
        self, gray: np.ndarray, tracks: Iterable[FlowTrackView]
    ) -> None:
        snapshots: dict[int, FlowTrackSnapshot] = {}
        for track in tracks:
            eligible = bool(
                track.lost_frames == 0
                or (
                    track.lost_frames <= self.max_coast_frames
                    and (track.predicted_only or track.flow_coasted)
                )
            )
            if not eligible:
                continue
            x, y, radius = track.state
            snapshots[int(track.id)] = FlowTrackSnapshot(
                center=(float(x), float(y)), radius=float(radius)
            )
        self.previous_gray = gray.copy()
        self.previous_tracks = snapshots
