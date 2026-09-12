"""Lightweight trajectory feature extraction and LightGBM text inference."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


BASE_FEATURE_NAMES = (
    "x_norm", "y_norm", "radius_norm", "vx_norm", "vy_norm", "speed_norm",
    "orientation_sin", "orientation_cos", "orientation_known",
    "angular_velocity_norm", "translation_residual_norm",
    "rotation_residual_norm", "rotation_score_norm", "direction_score_norm",
    "speed_score_norm", "rigidity_score_norm", "motion_reliability",
    "visible_streak_norm", "lost_frames_norm", "appearance_distance",
    "association_quality", "yolo_confidence", "is_current_target",
    "bg_certified", "distance_to_centroid_norm",
    "speed_minus_peer_median_norm", "rotation_score_peer_z",
    "direction_score_peer_z", "rigidity_score_peer_z", "appearance_peer_z",
)

TEMPORAL_FEATURE_NAMES = (
    "speed_norm", "angular_velocity_norm", "translation_residual_norm",
    "rotation_score_norm", "direction_score_norm", "appearance_distance",
)

HISTORY_FEATURE_NAMES = (
    "x_norm", "y_norm", "vx_norm", "vy_norm", "speed_norm",
    "angular_velocity_norm", "translation_residual_norm",
    "rotation_residual_norm", "rotation_score_norm", "direction_score_norm",
    "speed_score_norm", "rigidity_score_norm", "appearance_distance",
    "association_quality", "yolo_confidence", "visible_streak_norm",
    "is_current_target",
)

FLAT_FEATURE_NAMES = (
    *BASE_FEATURE_NAMES,
    *(f"history_mean_{name}" for name in TEMPORAL_FEATURE_NAMES),
    *(f"history_std_{name}" for name in TEMPORAL_FEATURE_NAMES),
    "history_length_norm",
)

SIMILARITY_FEATURE_NAMES = (
    "global_similarity_residual_norm",
    "global_similarity_residual_peer_z",
    "history_mean_global_similarity_residual_norm",
    "history_std_global_similarity_residual_norm",
)

FLOW_FEATURE_NAMES = (
    "optical_spin_abs_norm",
    "optical_spin_confidence",
    "optical_spin_evidence",
)

MULTILAG_FEATURE_NAMES = (
    "multilag_spin_abs_norm",
    "multilag_spin_confidence",
    "multilag_spin_consistency",
)

SWITCH_MOTION_FEATURE_NAMES = (
    *FLOW_FEATURE_NAMES,
    *MULTILAG_FEATURE_NAMES,
    "spin_estimator_agreement",
    "spin_joint_confidence",
)

MOTION_FEATURE_NAMES = (
    *SIMILARITY_FEATURE_NAMES,
    *FLOW_FEATURE_NAMES,
    *MULTILAG_FEATURE_NAMES,
)

AVAILABLE_FEATURE_NAMES = (*FLAT_FEATURE_NAMES, *MOTION_FEATURE_NAMES)

_CURSOR_BLIND_RADIUS_FRACTION = 0.44
_FLOW_OUTER_RADIUS_FRACTION = 0.86
_POLAR_BLIND_RADIAL_BINS = 18
_ROTATION_ANGLE_BINS = 360
_ROTATION_RADIUS_BINS = 32


def _safe_z(value: float, values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    scale = float(np.std(array))
    if scale <= 1e-6:
        return 0.0
    return float(np.clip((value - float(np.mean(array))) / scale, -5.0, 5.0))


def _robust_z(values: np.ndarray) -> np.ndarray:
    if values.size < 2:
        return np.zeros_like(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1e-8, 1.4826 * mad)
    return np.clip((values - median) / scale, -8.0, 8.0)


def _fit_similarity(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Robustly fit translation, rotation, and isotropic scale."""
    count = len(previous)
    if count < 2:
        center = np.mean(previous, axis=0) if count else np.zeros(2)
        translation = (
            np.median(current - previous, axis=0) if count else np.zeros(2)
        )
        return center, np.asarray([*translation, 1.0, 0.0])
    center = np.median(previous, axis=0)
    centered = previous - center
    design = np.zeros((2 * count, 4), dtype=np.float64)
    design[0::2] = np.column_stack(
        [np.ones(count), np.zeros(count), centered[:, 0], -centered[:, 1]]
    )
    design[1::2] = np.column_stack(
        [np.zeros(count), np.ones(count), centered[:, 1], centered[:, 0]]
    )
    target = current.reshape(-1)
    weights = np.ones(count, dtype=np.float64)
    beta = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    for _ in range(iterations):
        row_weights = np.repeat(np.sqrt(weights), 2)
        beta = np.linalg.lstsq(
            design * row_weights[:, None],
            target * row_weights,
            rcond=None,
        )[0]
        residuals = (design @ beta - target).reshape(-1, 2)
        distances = np.linalg.norm(residuals, axis=1)
        median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median)))
        robust_scale = max(0.35, 1.4826 * mad)
        weights = np.minimum(
            1.0, 1.5 * robust_scale / np.maximum(distances, 1e-8)
        )
    return center, beta


def _predict_similarity(
    points: np.ndarray, center: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    relative = points - center
    tx, ty, scale_cos, scale_sin = beta
    return np.column_stack(
        [
            tx + scale_cos * relative[:, 0] - scale_sin * relative[:, 1],
            ty + scale_sin * relative[:, 0] + scale_cos * relative[:, 1],
        ]
    )


def extract_similarity_features(
    track_ids: Sequence[int],
    histories: dict[int, deque[dict[str, float]]],
    frame_width: int,
    frame_height: int,
    *,
    history_limit: int = 8,
) -> dict[int, dict[str, float]]:
    """Measure each track's residual from the peers' global similarity motion."""
    width = max(1.0, float(frame_width))
    height = max(1.0, float(frame_height))
    diagonal = max(1.0, math.hypot(width, height))
    positions = {
        track_id: np.asarray(
            [
                [item["x_norm"] * width, item["y_norm"] * height]
                for item in histories.get(track_id, ())
            ],
            dtype=np.float64,
        )
        for track_id in track_ids
    }
    residual_history: dict[int, list[float]] = defaultdict(list)
    for lag in range(history_limit):
        eligible = [
            track_id
            for track_id in track_ids
            if len(positions[track_id]) >= lag + 2
        ]
        if len(eligible) < 3:
            continue
        previous = np.asarray([positions[item][-lag - 2] for item in eligible])
        current = np.asarray([positions[item][-lag - 1] for item in eligible])
        center, beta = _fit_similarity(previous, current)
        predicted = _predict_similarity(previous, center, beta)
        residuals = np.linalg.norm(current - predicted, axis=1) / diagonal
        for track_id, residual in zip(eligible, residuals):
            residual_history[track_id].append(float(residual))

    current = np.asarray(
        [
            residual_history[track_id][0] if residual_history[track_id] else 0.0
            for track_id in track_ids
        ],
        dtype=np.float64,
    )
    peer_z = _robust_z(current)
    output = {}
    for index, track_id in enumerate(track_ids):
        values = residual_history[track_id]
        output[track_id] = {
            "global_similarity_residual_norm": float(current[index]),
            "global_similarity_residual_peer_z": float(peer_z[index]),
            "history_mean_global_similarity_residual_norm": (
                float(np.mean(values)) if values else 0.0
            ),
            "history_std_global_similarity_residual_norm": (
                float(np.std(values)) if values else 0.0
            ),
        }
    return output


def optical_spin_features(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_center: Sequence[float],
    current_center: Sequence[float],
    radius: float,
) -> tuple[float, float, float]:
    """Fit local annular optical flow while hiding the cursor-sized centre."""
    height, width = previous_gray.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    center = tuple(int(round(value)) for value in previous_center)
    inner_radius = max(7, int(round(_CURSOR_BLIND_RADIUS_FRACTION * radius)))
    outer_radius = max(
        inner_radius + 4, int(round(_FLOW_OUTER_RADIUS_FRACTION * radius))
    )
    cv2.circle(mask, center, outer_radius, 255, -1)
    cv2.circle(mask, center, inner_radius, 0, -1)
    corners = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=60,
        qualityLevel=0.025,
        minDistance=3.0,
        mask=mask,
        blockSize=5,
    )
    if corners is None or len(corners) < 6:
        return 0.0, 0.0, 0.0
    moved, status, errors = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        corners,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01),
    )
    if moved is None or status is None:
        return 0.0, 0.0, 0.0
    valid = status.reshape(-1) > 0
    if errors is not None:
        valid &= errors.reshape(-1) < 24.0
    source = corners.reshape(-1, 2)[valid]
    target = moved.reshape(-1, 2)[valid]
    if len(source) < 6:
        return 0.0, 0.0, 0.0
    next_center = np.asarray(current_center, dtype=np.float64)
    target_radius = np.linalg.norm(target - next_center, axis=1)
    valid_target = (
        (target_radius >= _CURSOR_BLIND_RADIUS_FRACTION * radius)
        & (target_radius <= 1.02 * radius)
    )
    source = source[valid_target]
    target = target[valid_target]
    if len(source) < 6:
        return 0.0, 0.0, 0.0
    matrix, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=1.7,
        maxIters=200,
        confidence=0.98,
        refineIters=10,
    )
    if matrix is None or inliers is None:
        return 0.0, 0.0, 0.0
    angle = math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
    inlier_ratio = float(np.mean(inliers.reshape(-1) > 0))
    coverage = min(1.0, len(source) / 24.0)
    confidence = inlier_ratio * coverage
    normalized = min(3.0, abs(angle) / 22.0)
    return float(normalized), float(confidence), float(normalized * confidence)


def _polar_descriptor(
    gray: np.ndarray, center: Sequence[float], radius: float
) -> np.ndarray | None:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.magnitude(grad_x, grad_y)
    sample_radius = max(8.0, 1.1 * float(radius))
    side = max(16, int(round(2.0 * sample_radius)))
    patch = cv2.getRectSubPix(
        edges, (side, side), (float(center[0]), float(center[1]))
    )
    if patch is None or patch.size == 0:
        return None
    magnitude = cv2.resize(patch, (96, 96), interpolation=cv2.INTER_AREA)
    polar = cv2.warpPolar(
        magnitude,
        (_ROTATION_RADIUS_BINS, _ROTATION_ANGLE_BINS),
        (47.5, 47.5),
        47.0,
        cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS,
    ).astype(np.float32)
    polar[:, :5] = 0.0
    floor = float(np.percentile(polar, 45.0))
    polar = np.maximum(polar - floor, 0.0)
    radial_position = np.linspace(
        0.0, 1.0, _ROTATION_RADIUS_BINS, dtype=np.float32
    )
    polar *= np.exp(
        -0.5 * ((radial_position - 0.72) / 0.30) ** 2
    )[None, :]
    raw_norm = float(np.linalg.norm(polar))
    if raw_norm <= 1e-5:
        return None
    polar -= np.mean(polar, axis=0, keepdims=True)
    angular_norm = float(np.linalg.norm(polar))
    if angular_norm / raw_norm < 0.45:
        return None
    polar /= angular_norm
    polar[:, :_POLAR_BLIND_RADIAL_BINS] = 0.0
    norm = float(np.linalg.norm(polar))
    return None if norm <= 1e-6 else polar / norm


def _match_multilag(
    previous: np.ndarray, current: np.ndarray, *, max_degrees: float
) -> tuple[float | None, float]:
    if previous.shape != current.shape or previous.ndim != 2:
        return None, 0.0
    degrees_per_bin = 360.0 / previous.shape[0]
    max_shift = max(1, int(round(max_degrees / degrees_per_bin)))
    shifts = np.arange(-max_shift, max_shift + 1, dtype=np.int32)
    correlations = np.asarray(
        [float(np.sum(previous * np.roll(current, int(shift), axis=0))) for shift in shifts]
    )
    best = int(np.argmax(correlations))
    peak = float(correlations[best])
    if peak < 0.25:
        return None, 0.0
    exclusion = max(2, int(round(5.0 / degrees_per_bin)))
    alternatives = np.ones(len(correlations), dtype=bool)
    alternatives[
        max(0, best - exclusion) : min(len(correlations), best + exclusion + 1)
    ] = False
    second = (
        float(np.max(correlations[alternatives])) if np.any(alternatives) else -1.0
    )
    confidence = float(
        np.clip((peak - 0.25) / 0.65, 0.0, 1.0)
        * np.clip((peak - second) / 0.10, 0.0, 1.0)
    )
    if confidence < 0.05:
        return None, 0.0
    return float(-shifts[best] * degrees_per_bin), confidence


def multilag_rotation_features(
    gray_history: Sequence[np.ndarray],
    position_history: Sequence[Sequence[float]],
    radius: float,
) -> tuple[float, float, float]:
    if not gray_history or not position_history:
        return 0.0, 0.0, 0.0
    current_descriptor = _polar_descriptor(
        gray_history[-1], position_history[-1], radius
    )
    if current_descriptor is None:
        return 0.0, 0.0, 0.0
    measurements: list[tuple[float, float]] = []
    for lag in (1, 2, 4):
        if len(gray_history) <= lag or len(position_history) <= lag:
            continue
        previous_descriptor = _polar_descriptor(
            gray_history[-lag - 1], position_history[-lag - 1], radius
        )
        if previous_descriptor is None:
            continue
        delta, confidence = _match_multilag(
            previous_descriptor,
            current_descriptor,
            max_degrees=min(88.0, 22.0 * lag),
        )
        if delta is not None:
            measurements.append((float(delta) / lag, confidence))
    if not measurements:
        return 0.0, 0.0, 0.0
    values = np.asarray([item[0] for item in measurements], dtype=np.float64)
    weights = np.asarray([item[1] for item in measurements], dtype=np.float64)
    signed = float(np.average(values, weights=np.maximum(weights, 1e-6)))
    sign_consistency = abs(float(np.sum(weights * np.sign(values)))) / max(
        float(np.sum(weights)), 1e-6
    )
    confidence = float(np.mean(weights)) * sign_consistency
    return min(3.0, abs(signed) / 22.0), confidence, sign_consistency


class SwitchMotionFeatureHistory:
    """Compute expensive image motion only for switch-event participants.

    Frames and visible track centres are kept aligned so a temporary missed
    detection cannot accidentally pair an old position with a newer image.
    The source frames have already had the green cursor removed by the tracker.
    """

    def __init__(self, *, history_size: int = 5):
        self.history_size = max(5, int(history_size))
        self.reset()

    def reset(self) -> None:
        self.frames: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.tracks: deque[dict[int, tuple[tuple[float, float], float]]] = deque(
            maxlen=self.history_size
        )

    def update(
        self,
        views: Sequence[Any],
        *,
        gray_frame: np.ndarray | None,
    ) -> None:
        if gray_frame is None:
            return
        visible = {
            int(view.track_id): (
                (float(view.center[0]), float(view.center[1])),
                float(view.radius),
            )
            for view in views
            if view.lost_frames == 0 and not view.predicted_only
        }
        self.frames.append(gray_frame.copy())
        self.tracks.append(visible)

    @staticmethod
    def _empty() -> dict[str, float]:
        return {name: 0.0 for name in SWITCH_MOTION_FEATURE_NAMES}

    def features(
        self,
        track_ids: Iterable[int],
        *,
        include_optical: bool = True,
        include_multilag: bool = True,
    ) -> dict[int, dict[str, float]]:
        output = {int(track_id): self._empty() for track_id in track_ids}
        if not self.frames or not self.tracks:
            return output
        current_tracks = self.tracks[-1]
        for track_id in output:
            current = current_tracks.get(track_id)
            if current is None:
                continue
            current_center, current_radius = current
            flow = (0.0, 0.0, 0.0)
            if include_optical and len(self.frames) >= 2:
                previous = self.tracks[-2].get(track_id)
                if previous is not None:
                    flow = optical_spin_features(
                        self.frames[-2],
                        self.frames[-1],
                        previous[0],
                        current_center,
                        0.5 * (previous[1] + current_radius),
                    )

            measurements: list[tuple[float, float]] = []
            current_descriptor = (
                _polar_descriptor(self.frames[-1], current_center, current_radius)
                if include_multilag
                else None
            )
            if current_descriptor is not None:
                for lag in (1, 2, 4):
                    if len(self.frames) <= lag:
                        continue
                    previous = self.tracks[-lag - 1].get(track_id)
                    if previous is None:
                        continue
                    previous_descriptor = _polar_descriptor(
                        self.frames[-lag - 1], previous[0], previous[1]
                    )
                    if previous_descriptor is None:
                        continue
                    delta, confidence = _match_multilag(
                        previous_descriptor,
                        current_descriptor,
                        max_degrees=min(88.0, 22.0 * lag),
                    )
                    if delta is not None:
                        measurements.append((float(delta) / lag, confidence))
            if measurements:
                values = np.asarray(
                    [item[0] for item in measurements], dtype=np.float64
                )
                weights = np.asarray(
                    [item[1] for item in measurements], dtype=np.float64
                )
                signed = float(
                    np.average(values, weights=np.maximum(weights, 1e-6))
                )
                sign_consistency = abs(
                    float(np.sum(weights * np.sign(values)))
                ) / max(float(np.sum(weights)), 1e-6)
                multilag = (
                    min(3.0, abs(signed) / 22.0),
                    float(np.mean(weights)) * sign_consistency,
                    sign_consistency,
                )
            else:
                multilag = (0.0, 0.0, 0.0)

            optical_confidence = float(flow[1])
            multilag_confidence = float(multilag[1])
            both_observed = optical_confidence > 0.0 and multilag_confidence > 0.0
            agreement = (
                max(0.0, 1.0 - abs(float(flow[0]) - float(multilag[0])) / 3.0)
                if both_observed
                else 0.0
            )
            joint_confidence = (
                math.sqrt(optical_confidence * multilag_confidence)
                if both_observed
                else 0.0
            )
            output[track_id].update(dict(zip(FLOW_FEATURE_NAMES, flow)))
            output[track_id].update(dict(zip(MULTILAG_FEATURE_NAMES, multilag)))
            output[track_id]["spin_estimator_agreement"] = float(agreement)
            output[track_id]["spin_joint_confidence"] = float(joint_confidence)
        return output


def extract_frame_features(
    views: Sequence[Any],
    *,
    target_id: int | None,
    frame_width: int,
    frame_height: int,
) -> dict[int, dict[str, float]]:
    """Convert tracker views into normalized per-track and peer features."""
    if not views:
        return {}
    width = max(1.0, float(frame_width))
    height = max(1.0, float(frame_height))
    diagonal = max(1.0, math.hypot(width, height))
    visible = [view for view in views if view.lost_frames == 0]
    peer_source = visible or list(views)
    centers = np.asarray([view.center for view in peer_source], dtype=np.float64)
    centroid = np.mean(centers, axis=0)
    speeds = [
        math.hypot(float(view.velocity[0]), float(view.velocity[1]))
        for view in peer_source
    ]
    peer_speed_median = float(np.median(np.asarray(speeds)))
    rotation_scores = [float(view.rotation_score) for view in peer_source]
    direction_scores = [float(view.direction_score) for view in peer_source]
    rigidity_scores = [float(view.rigidity_score) for view in peer_source]
    appearance_distances = [
        float(np.clip(view.appearance_distance, 0.0, 2.0))
        for view in peer_source
    ]

    result: dict[int, dict[str, float]] = {}
    for view in views:
        vx, vy = float(view.velocity[0]), float(view.velocity[1])
        speed = math.hypot(vx, vy)
        orientation_known = view.orientation is not None
        orientation = float(view.orientation or 0.0)
        appearance = float(np.clip(view.appearance_distance, 0.0, 2.0))
        result[view.track_id] = {
            "x_norm": float(view.center[0]) / width,
            "y_norm": float(view.center[1]) / height,
            "radius_norm": float(view.radius) / min(width, height),
            "vx_norm": vx / diagonal,
            "vy_norm": vy / diagonal,
            "speed_norm": speed / diagonal,
            "orientation_sin": math.sin(orientation) if orientation_known else 0.0,
            "orientation_cos": math.cos(orientation) if orientation_known else 0.0,
            "orientation_known": float(orientation_known),
            "angular_velocity_norm": float(
                np.clip(view.angular_velocity / (2.0 * math.pi), -5.0, 5.0)
            ),
            "translation_residual_norm": float(view.translation_residual) / diagonal,
            "rotation_residual_norm": float(view.rotation_residual) / 180.0,
            "rotation_score_norm": float(view.rotation_score) / 180.0,
            "direction_score_norm": float(view.direction_score) / 180.0,
            "speed_score_norm": float(view.speed_score) / diagonal,
            "rigidity_score_norm": float(view.rigidity_score) / diagonal,
            "motion_reliability": float(
                np.clip(view.motion_reliability, 0.0, 1.0)
            ),
            "visible_streak_norm": float(
                np.clip(view.visible_streak / 60.0, 0.0, 1.0)
            ),
            "lost_frames_norm": float(
                np.clip(view.lost_frames / 10.0, 0.0, 1.0)
            ),
            "appearance_distance": appearance,
            "association_quality": float(
                np.clip(view.association_quality, 0.0, 1.0)
            ),
            "yolo_confidence": float(
                np.clip(view.yolo_confidence, 0.0, 1.0)
            ),
            "is_current_target": float(view.track_id == target_id),
            "bg_certified": float(view.bg_certified),
            "distance_to_centroid_norm": math.hypot(
                float(view.center[0]) - float(centroid[0]),
                float(view.center[1]) - float(centroid[1]),
            )
            / diagonal,
            "speed_minus_peer_median_norm": (speed - peer_speed_median) / diagonal,
            "rotation_score_peer_z": _safe_z(
                float(view.rotation_score), rotation_scores
            ),
            "direction_score_peer_z": _safe_z(
                float(view.direction_score), direction_scores
            ),
            "rigidity_score_peer_z": _safe_z(
                float(view.rigidity_score), rigidity_scores
            ),
            "appearance_peer_z": _safe_z(appearance, appearance_distances),
        }
    return result


def aggregate_history(
    history: Sequence[dict[str, float]],
    *,
    history_size: int,
) -> dict[str, float]:
    if not history:
        raise ValueError("history must not be empty")
    current = dict(history[-1])
    for name in TEMPORAL_FEATURE_NAMES:
        values = np.asarray([item[name] for item in history], dtype=np.float64)
        current[f"history_mean_{name}"] = float(np.mean(values))
        current[f"history_std_{name}"] = float(np.std(values))
    current["history_length_norm"] = min(1.0, len(history) / max(1, history_size))
    return current


@dataclass(frozen=True)
class _Tree:
    split_feature: tuple[int, ...]
    threshold: tuple[float, ...]
    left_child: tuple[int, ...]
    right_child: tuple[int, ...]
    leaf_value: tuple[float, ...]

    def predict(self, features: Sequence[float]) -> float:
        node = 0
        while node >= 0:
            branch = (
                self.left_child[node]
                if features[self.split_feature[node]] <= self.threshold[node]
                else self.right_child[node]
            )
            node = branch
        return self.leaf_value[-node - 1]


class LightGbmTextModel:
    """Numeric-only LightGBM tree ensemble without the native runtime."""

    def __init__(self, feature_names: Sequence[str], trees: Sequence[_Tree]):
        self.feature_names = tuple(feature_names)
        self.trees = tuple(trees)

    @staticmethod
    def _numbers(block: dict[str, str], name: str, cast) -> tuple:
        return tuple(cast(value) for value in block[name].split())

    @classmethod
    def load(cls, path: str | Path) -> "LightGbmTextModel":
        text = Path(path).read_text(encoding="utf-8")
        header, *tree_chunks = text.split("\nTree=")
        header_fields = {}
        for line in header.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                header_fields[key] = value
        feature_names = header_fields.get("feature_names", "").split()
        if not feature_names:
            raise ValueError(f"feature_names missing from LightGBM model: {path}")
        trees = []
        for chunk in tree_chunks:
            block: dict[str, str] = {}
            for line in chunk.splitlines()[1:]:
                if "=" in line:
                    key, value = line.split("=", 1)
                    block[key] = value
            if not block or block.get("num_cat", "0") != "0":
                if block:
                    raise ValueError("categorical LightGBM trees are not supported")
                continue
            decision_types = block.get("decision_type", "").split()
            if any(value != "2" for value in decision_types):
                raise ValueError("only numeric <= LightGBM splits are supported")
            trees.append(
                _Tree(
                    split_feature=cls._numbers(block, "split_feature", int),
                    threshold=cls._numbers(block, "threshold", float),
                    left_child=cls._numbers(block, "left_child", int),
                    right_child=cls._numbers(block, "right_child", int),
                    leaf_value=cls._numbers(block, "leaf_value", float),
                )
            )
        if not trees:
            raise ValueError(f"no trees found in LightGBM model: {path}")
        return cls(feature_names, trees)

    def predict(self, features: Sequence[float]) -> float:
        if len(features) != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} features, got {len(features)}"
            )
        return float(sum(tree.predict(features) for tree in self.trees))


class TrajectoryIdentityRanker:
    def __init__(self, model_path: str | Path, *, history_size: int = 16):
        self.model = LightGbmTextModel.load(model_path)
        missing = set(self.model.feature_names) - set(AVAILABLE_FEATURE_NAMES)
        if missing:
            raise ValueError(f"ranker uses unavailable features: {sorted(missing)}")
        feature_set = set(self.model.feature_names)
        self.needs_similarity = bool(feature_set.intersection(SIMILARITY_FEATURE_NAMES))
        self.needs_flow = bool(feature_set.intersection(FLOW_FEATURE_NAMES))
        self.needs_multilag = bool(feature_set.intersection(MULTILAG_FEATURE_NAMES))
        self.history_size = max(2, int(history_size))
        self.reset()

    def reset(self) -> None:
        self.histories: dict[int, deque[dict[str, float]]] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )
        self.gray_history: deque[np.ndarray] = deque(maxlen=5)

    def update(
        self,
        views: Sequence[Any],
        *,
        target_id: int | None,
        frame_width: int,
        frame_height: int,
        gray_frame: np.ndarray | None = None,
    ) -> dict[int, float]:
        frame_features = extract_frame_features(
            views,
            target_id=target_id,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        live_ids = {view.track_id for view in views}
        for stale_id in set(self.histories) - live_ids:
            del self.histories[stale_id]
        for track_id, features in frame_features.items():
            self.histories[track_id].append(features)
        if gray_frame is not None and (self.needs_flow or self.needs_multilag):
            self.gray_history.append(gray_frame.copy())

        visible_ids = [
            int(view.track_id)
            for view in views
            if (
                view.lost_frames == 0
                and not view.predicted_only
                and view.visible_streak >= 2
            )
        ]
        similarity = (
            extract_similarity_features(
                visible_ids,
                self.histories,
                frame_width,
                frame_height,
            )
            if self.needs_similarity
            else {}
        )
        scores = {}
        for view in views:
            if (
                view.lost_frames != 0
                or view.predicted_only
                or view.visible_streak < 2
            ):
                continue
            flat = aggregate_history(
                list(self.histories[view.track_id]),
                history_size=self.history_size,
            )
            flat.update(
                similarity.get(
                    view.track_id,
                    {name: 0.0 for name in SIMILARITY_FEATURE_NAMES},
                )
            )
            motion = {name: 0.0 for name in (*FLOW_FEATURE_NAMES, *MULTILAG_FEATURE_NAMES)}
            history = list(self.histories[view.track_id])
            positions = [
                (
                    item["x_norm"] * frame_width,
                    item["y_norm"] * frame_height,
                )
                for item in history
            ]
            if (
                self.needs_flow
                and len(self.gray_history) >= 2
                and len(positions) >= 2
            ):
                flow = optical_spin_features(
                    self.gray_history[-2],
                    self.gray_history[-1],
                    positions[-2],
                    positions[-1],
                    float(view.radius),
                )
                motion.update(dict(zip(FLOW_FEATURE_NAMES, flow)))
            if self.needs_multilag and self.gray_history and positions:
                multilag = multilag_rotation_features(
                    list(self.gray_history),
                    positions,
                    float(view.radius),
                )
                motion.update(dict(zip(MULTILAG_FEATURE_NAMES, multilag)))
            flat.update(motion)
            vector = [flat[name] for name in self.model.feature_names]
            scores[view.track_id] = self.model.predict(vector)
        return scores
