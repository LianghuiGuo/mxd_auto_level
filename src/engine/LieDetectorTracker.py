"""Constellation tracker for the MapleStory lie-detector mini-game.

Design (aligned with the working third-party overlay):

* detect every visible star each frame (YOLO primary);
* keep a track for **every** star (BG:id), not only the target;
* seed REAL from the opening white highlight;
* after fade, score REAL by four cues: self-rotation, BG layout rigidity,
  BG speed consistency, and translation-direction disagreement with BG;
* emit the REAL center for the caller; never use the green cursor as input.

The tracker itself does not move the mouse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Iterable, Literal, Optional, Protocol

import cv2
import numpy as np

from src.engine.LieBackgroundCertificates import LieBackgroundCertificates
from src.engine.LieIdentityRanker import (
    SWITCH_MOTION_FEATURE_NAMES,
    SwitchMotionFeatureHistory,
    TrajectoryIdentityRanker,
)
from src.engine.LieSwitchEventModel import SwitchEventModel
from src.engine.LiePreAssociationFlow import (
    FlowTrackObservation,
    PreAssociationFlow,
)

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None

MAX_PATTERN_COUNT = 20
_DECOY_ROTATION_WEIGHT = 1.35
_MATCH_GATE = 55.0
_REAL_STICKY_BONUS = 18.0
_REAL_SWITCH_MARGIN = 10.0
_REAL_COAST_GRACE = 10
_REAL_OUTLIER_RATIO = 1.8
_ORIENT_HISTORY = 5
_ROTATION_ANGLE_BINS = 360
_ROTATION_RADIUS_BINS = 32
_MAX_ROTATION_DEGREES_PER_FRAME = 22.0
_FULL_ROTATION_SPEED_DEG_PER_SECOND = 120.0
_MAX_ROTATION_SCORE = 20.0
_MAX_LOST_DEFAULT = 20
# REAL may coast through short occlusions, but never indefinitely: past this
# budget the label is released so a live track can be re-acquired.
_REAL_MAX_COAST = 25
# A coasting centre allowed to sit this far outside the panel before the track
# is discarded instead of extrapolating off-screen forever.
_FRAME_EXIT_MARGIN = 6.0
# Border band (also scaled by the shape radius) where an outward-moving track
# is not allowed to become REAL.
_EDGE_SWITCH_MARGIN = 12.0
# Four-cue REAL score weights (leave-one-out over BG tracks).
_W_ROTATION = 1.60
_W_DIRECTION = 1.40
_W_SPEED = 1.10
_W_RIGIDITY = 2.20
_W_TRANSLATION = 0.45
_MIN_HEADING_SPEED = 0.8
_CHALLENGER_EVIDENCE_DECAY = 0.78
_CHALLENGER_EVIDENCE_NEEDED = 4.0
_WEAK_REAL_EVIDENCE_NEEDED = 2.0
_W_ANCHOR_APPEARANCE = 12.0
_SHADOW_SWITCH_WINDOW = 5
_SHADOW_SWITCH_VOTES = 3
_PROVISIONAL_GAP_FRAMES = 8
_PROVISIONAL_OBSERVATIONS = 4
_IDENTITY_PATH_DECAY = 0.86
_IDENTITY_EMISSION_CLIP = 3.0
_IDENTITY_HANDOFF_GATE = 105.0
_IDENTITY_HANDOFF_PENALTY = 0.8
_IDENTITY_HANDOFF_DISTANCE_WEIGHT = 1.4
_IDENTITY_MISSING_PENALTY = 0.4
_IDENTITY_MAX_MISSING = 10
_IDENTITY_SWITCH_MARGIN = 1.2
_IDENTITY_SWITCH_VOTES = 3
_IDENTITY_FAR_SWITCH_EXTRA_VOTES = 3
_IDENTITY_FAR_SWITCH_DISTANCE = 130.0
_IDENTITY_UNVALIDATED_RECOVERY_VOTES = 4
_IDENTITY_EXTREME_RECOVERY_DISTANCE = 400.0
_IDENTITY_EXTREME_RECOVERY_VOTES = 2
_STALE_RECOVERY_MAX_APPEARANCE_ERROR = 0.07
_STALE_RECOVERY_HOLD_FRAMES = 12
_STALE_RECOVERY_APPEARANCE_PROTECT_FRAMES = 60
_RANKER_MIN_MARGIN = 2.00
_RANKER_SWITCH_VOTES = 5
_RANKER_RECOVERY_VOTES = 3
_IDENTITY_SWITCH_COOLDOWN = 10
_IDENTITY_SAFETY_COAST_GRACE = 14
_STATE_AWARE_RECOVERY_COAST = 14
_SWITCH_EVENT_VISIBLE_MIN_DELTA = 0.50
_SWITCH_EVENT_ROLLBACK_WINDOW = 12
_SWITCH_EVENT_ROLLBACK_EVIDENCE = 1.20
_FLOW_ASSOCIATION_MIN_CONFIDENCE = 0.28
_FLOW_ASSOCIATION_MAX_WEIGHT = 0.35
_FLOW_COAST_MIN_CONFIDENCE = 0.42
_FLOW_COAST_MAX_FB_ERROR = 0.80
_FLOW_COAST_MIN_INLIER_RATIO = 0.65
_FLOW_COAST_MIN_COVERAGE = 0.35
_FLOW_COAST_MAX_FIT_ERROR = 1.50
_FLOW_COAST_MAX_KALMAN_DISAGREEMENT = 14.0
TrackRole = Literal["real", "bg", "unknown"]


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
    orientation_confidence: float = 0.0
    collective_rotation_residual: float = 0.0
    yolo_confidence: float = 0.0
    bg_certified: bool = False
    contour: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    appearance: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    rotation_descriptor: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False
    )


class ShapeCandidateDetector(Protocol):
    def detect(
        self,
        frame_bgr: np.ndarray,
        reference_radius: Optional[float] = None,
    ) -> list[ShapeDetection]: ...

    def reset(self) -> None: ...


@dataclass
class ConstellationTrackView:
    track_id: int
    center: tuple[float, float]
    radius: float
    role: TrackRole
    orientation: Optional[float]
    lost_frames: int
    translation_residual: float = 0.0
    rotation_residual: float = 0.0
    # Per-frame motion estimate used for direction prediction arrows.
    velocity: tuple[float, float] = (0.0, 0.0)
    real_score: float = 0.0
    rotation_score: float = 0.0
    direction_score: float = 0.0
    speed_score: float = 0.0
    rigidity_score: float = 0.0
    motion_reliability: float = 0.0
    visible_streak: int = 0
    angular_velocity: float = 0.0
    appearance_distance: float = 2.0
    association_quality: float = 0.0
    yolo_confidence: float = 0.0
    predicted_only: bool = False
    bg_certified: bool = False
    ranker_score: float = 0.0
    flow_confidence: float = 0.0
    flow_forward_backward_error: float = 0.0
    flow_inlier_ratio: float = 0.0
    flow_coverage: float = 0.0
    flow_local_fit_error: float = 0.0
    flow_group_residual: float = 0.0
    flow_group_confidence: float = 0.0
    flow_association_used: bool = False
    flow_coasted: bool = False


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
    white_active: bool = False
    identity_window_active: bool = False
    identity_switched: bool = False
    actionable: bool = False
    position_uncertainty_px: float = 0.0
    bg_registry_state: str = "dormant"
    bg_certified_count: int = 0
    stale_recovery_committed: bool = False
    ranker_candidate_id: Optional[int] = None
    ranker_margin: float = 0.0
    ranker_switched: bool = False
    ranker_top_id: Optional[int] = None
    ranker_current_rank: int = 0
    ranker_current_score: float = 0.0
    ranker_switch_delta: float = 0.0
    ranker_evidence: float = 0.0
    motion_corroborated: bool = False
    identity_confidence: float = 0.0
    identity_state: str = "uninitialized"
    identity_safe: bool = False
    hold_reason: Optional[str] = None
    position_actionable: bool = False
    # Pre-commit current/challenger snapshot for offline switch-event
    # supervision. It contains tracker/ranker state only; the replay tool adds
    # the cursor-derived label after ``update`` returns.
    switch_event: Optional[dict[str, object]] = None
    switch_event_probability: float = 0.0
    switch_event_approved: bool = False
    flow_observation_count: int = 0
    flow_association_count: int = 0
    flow_coast_count: int = 0
    constellation: list[ConstellationTrackView] = field(default_factory=list)


@dataclass
class _PendingReassociation:
    old_track_id: int
    child_track_id: int
    initial_signature: Optional[np.ndarray]
    initial_position_error: float


@dataclass
class _IdentityHypothesis:
    track_id: int
    score: float
    center: tuple[float, float]
    radius: float
    age: int = 1
    missing: int = 0
    handoffs: int = 0


def green_cursor_mask(frame_bgr: np.ndarray) -> np.ndarray:
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
    def __init__(self, track_id: int, detection: ShapeDetection, timestamp: float):
        self.id = track_id
        self.role: TrackRole = "unknown"
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
        self.kf.measurementNoiseCov = np.diag([12.0, 12.0, 8.0]).astype(np.float32)
        self.kf.errorCovPost = np.diag(
            [8.0, 8.0, 80.0, 80.0, 8.0, 16.0]
        ).astype(np.float32)
        x, y = detection.center
        self.kf.statePost = np.array(
            [[x], [y], [0.0], [0.0], [detection.radius], [0.0]],
            dtype=np.float32,
        )
        self.last_timestamp = timestamp
        self.age = 1
        self.hits = 1
        self.visible_streak = 1
        self.lost_frames = 0
        self.last_detection = detection
        self.predicted_only = False
        self.rotation_descriptor = (
            None
            if detection.rotation_descriptor is None
            else detection.rotation_descriptor.astype(np.float32, copy=True)
        )
        self.orientation = (
            detection.orientation
            if detection.orientation is not None
            else (0.0 if self.rotation_descriptor is not None else None)
        )
        self.orientation_phase = self.orientation
        self.orientation_history: list[float] = (
            [] if self.orientation is None else [float(self.orientation)]
        )
        self.orientation_quality_history: list[float] = (
            []
            if self.orientation is None
            else [float(detection.orientation_confidence or 1.0)]
        )
        self.rotation_delta_history: list[float] = []
        self.rotation_dt_history: list[float] = []
        self.angular_velocity = 0.0
        self.last_rotation_delta = 0.0
        self.rotation_confidence = 0.0
        self.rotation_valid = False
        self.appearance = (
            None
            if detection.appearance is None
            else detection.appearance.astype(np.float32, copy=True)
        )
        self.translation_residual = 0.0
        self.rotation_residual = 0.0
        self.peer_rotation_residual = 0.0
        self.real_score = 0.0
        self.prev_center: Optional[tuple[float, float]] = None
        self.velocity: tuple[float, float] = (0.0, 0.0)
        self.motion_history: list[tuple[float, float]] = []
        self.association_quality_history: list[float] = [0.0]
        self.last_dt = 1.0 / 30.0
        self.rotation_score = 0.0
        self.direction_score = 0.0
        self.speed_score = 0.0
        self.rigidity_score = 0.0
        self.appearance_score = 0.0
        self.ranker_score = 0.0
        self.flow_predicted_center: Optional[tuple[float, float]] = None
        self.flow_confidence = 0.0
        self.flow_forward_backward_error = 0.0
        self.flow_inlier_ratio = 0.0
        self.flow_coverage = 0.0
        self.flow_local_fit_error = 0.0
        self.flow_group_residual = 0.0
        self.flow_group_confidence = 0.0
        self.flow_association_used = False
        self.flow_coasted = False

    def _set_transition(self, dt: float) -> None:
        dt = float(np.clip(dt, 1.0 / 120.0, 0.25))
        self.last_dt = dt
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
        *,
        association_quality: float = 1.0,
        flow_association_used: bool = False,
    ) -> None:
        if self.last_detection is not None:
            self.prev_center = (
                float(self.last_detection.center[0]),
                float(self.last_detection.center[1]),
            )
            self.velocity = (
                float(detection.center[0] - self.prev_center[0]),
                float(detection.center[1] - self.prev_center[1]),
            )
            self.motion_history.append(self.velocity)
            if len(self.motion_history) > 7:
                self.motion_history = self.motion_history[-7:]
        measurement = np.array(
            [[detection.center[0]], [detection.center[1]], [detection.radius]],
            dtype=np.float32,
        )
        self.kf.correct(measurement)
        self.last_detection = detection
        self.hits += 1
        self.visible_streak += 1
        self.lost_frames = 0
        self.predicted_only = False
        self.flow_coasted = False
        self.flow_association_used = bool(flow_association_used)
        self.association_quality_history.append(
            float(np.clip(association_quality, 0.0, 1.0))
        )
        if len(self.association_quality_history) > 5:
            self.association_quality_history = self.association_quality_history[-5:]
        self.last_timestamp = timestamp
        if detection.appearance is not None:
            incoming = detection.appearance.astype(np.float32, copy=False)
            if self.appearance is None or self.appearance.shape != incoming.shape:
                self.appearance = incoming.copy()
            else:
                updated = 0.85 * self.appearance + 0.15 * incoming
                norm = float(np.linalg.norm(updated))
                self.appearance = updated / norm if norm > 1e-6 else updated
        rotation_delta: Optional[float] = None
        rotation_quality = 0.0
        incoming_rotation = detection.rotation_descriptor
        if incoming_rotation is not None and self.rotation_descriptor is not None:
            predicted_delta = self.angular_velocity * self.last_dt
            rotation_delta, rotation_quality = (
                LieDetectorTracker._match_rotation_descriptors(
                    self.rotation_descriptor,
                    incoming_rotation,
                    predicted_delta=predicted_delta,
                )
            )
        elif detection.orientation is not None:
            # Compatibility path for classical/synthetic detections carrying an
            # explicit full-turn angle. Real detections use descriptor matching.
            measured = float(detection.orientation)
            if self.orientation_phase is None:
                self.orientation_phase = measured
            else:
                rotation_delta = (
                    measured - float(self.orientation_phase) + math.pi
                ) % (2.0 * math.pi) - math.pi
                rotation_quality = float(detection.orientation_confidence or 1.0)

        if incoming_rotation is not None:
            self.rotation_descriptor = incoming_rotation.astype(
                np.float32, copy=True
            )
        if rotation_delta is not None:
            self.orientation_phase = float(self.orientation_phase or 0.0) + rotation_delta
            measured_velocity = rotation_delta / max(self.last_dt, 1e-6)
            self.angular_velocity = (
                0.65 * self.angular_velocity + 0.35 * measured_velocity
            )
            self.orientation = float(self.orientation_phase)
            self.orientation_history.append(float(self.orientation_phase))
            self.rotation_delta_history.append(float(rotation_delta))
            self.rotation_dt_history.append(float(self.last_dt))
            self.orientation_quality_history.append(float(rotation_quality))
            detection.orientation = float(self.orientation_phase)
            detection.orientation_confidence = float(rotation_quality)
            self.last_rotation_delta = float(rotation_delta)
            self.rotation_confidence = float(rotation_quality)
            self.rotation_valid = bool(rotation_quality > 0.0)
        elif incoming_rotation is not None:
            # Preserve a failed measurement in the reliability window so that
            # intermittent texture/occlusion cannot produce a high spin score.
            self.rotation_delta_history.append(0.0)
            self.rotation_dt_history.append(float(self.last_dt))
            self.orientation_quality_history.append(0.0)
            self.last_rotation_delta = 0.0
            self.rotation_confidence = 0.0
            self.rotation_valid = False
        else:
            self.last_rotation_delta = 0.0
            self.rotation_confidence = 0.0
            self.rotation_valid = False

        if len(self.orientation_history) > _ORIENT_HISTORY:
            self.orientation_history = self.orientation_history[-_ORIENT_HISTORY:]
        if len(self.rotation_delta_history) > _ORIENT_HISTORY - 1:
            self.rotation_delta_history = self.rotation_delta_history[
                -(_ORIENT_HISTORY - 1):
            ]
            self.rotation_dt_history = self.rotation_dt_history[
                -(_ORIENT_HISTORY - 1):
            ]
        if len(self.orientation_quality_history) > _ORIENT_HISTORY:
            self.orientation_quality_history = self.orientation_quality_history[
                -_ORIENT_HISTORY:
            ]

    def mark_missed(self) -> None:
        self.lost_frames += 1
        self.visible_streak = 0
        self.flow_coasted = False
        self.flow_association_used = False
        self.last_rotation_delta = 0.0
        self.rotation_confidence = 0.0
        self.rotation_valid = False
        self.peer_rotation_residual = 0.0

    def set_flow_observation(
        self, observation: Optional[FlowTrackObservation]
    ) -> None:
        if observation is None:
            self.flow_predicted_center = None
            self.flow_confidence = 0.0
            self.flow_forward_backward_error = 0.0
            self.flow_inlier_ratio = 0.0
            self.flow_coverage = 0.0
            self.flow_local_fit_error = 0.0
            self.flow_group_residual = 0.0
            self.flow_group_confidence = 0.0
            self.flow_association_used = False
            return
        self.flow_predicted_center = observation.predicted_center
        self.flow_confidence = float(observation.confidence)
        self.flow_forward_backward_error = float(
            observation.forward_backward_error
        )
        self.flow_inlier_ratio = float(observation.inlier_ratio)
        self.flow_coverage = float(observation.coverage)
        self.flow_local_fit_error = float(observation.local_fit_error)
        self.flow_group_residual = float(observation.group_residual)
        self.flow_group_confidence = float(observation.group_confidence)
        self.flow_association_used = False

    def apply_flow_coast(
        self, observation: FlowTrackObservation
    ) -> None:
        """Correct a missed track from strict optical flow but keep it predicted."""
        previous = np.asarray(self.state[:2], dtype=np.float64)
        x, y = observation.predicted_center
        # Do not call Kalman.correct(): an optical-only observation must never
        # shrink covariance or make a detector-missed position look safer.
        flow_state = self.kf.statePre.copy()
        flow_state[0, 0] = float(x)
        flow_state[1, 0] = float(y)
        self.kf.statePre = flow_state
        self.prev_center = (float(previous[0]), float(previous[1]))
        self.velocity = (float(x - previous[0]), float(y - previous[1]))
        self.motion_history.append(self.velocity)
        if len(self.motion_history) > 7:
            self.motion_history = self.motion_history[-7:]
        self.lost_frames += 1
        self.visible_streak = 0
        self.predicted_only = True
        self.flow_coasted = True
        self.flow_association_used = False
        self.last_rotation_delta = 0.0
        self.rotation_confidence = 0.0
        self.rotation_valid = False
        self.peer_rotation_residual = 0.0

    @property
    def state(self) -> tuple[float, float, float]:
        state = self.kf.statePre if self.predicted_only else self.kf.statePost
        s = state.reshape(-1)
        return float(s[0]), float(s[1]), max(1.0, float(s[4]))

    @property
    def predicted_velocity(self) -> tuple[float, float]:
        """Per-frame motion estimate.

        The Kalman state carries px/second (the transition matrix uses a
        seconds-based dt), while ``velocity`` is a raw per-frame centre
        displacement.  Both are reported in px/frame here so callers never mix
        the two scales.
        """
        state = self.kf.statePre if self.predicted_only else self.kf.statePost
        s = state.reshape(-1)
        vx = float(s[2]) * self.last_dt
        vy = float(s[3]) * self.last_dt
        if abs(vx) < 1e-3 and abs(vy) < 1e-3:
            return self.velocity
        return vx, vy

    @property
    def robust_velocity(self) -> tuple[float, float]:
        if not self.motion_history:
            return self.velocity
        recent = np.asarray(self.motion_history[-5:], dtype=np.float64)
        median = np.median(recent, axis=0)
        return float(median[0]), float(median[1])

    @property
    def motion_reliability(self) -> float:
        maturity = float(np.clip((self.visible_streak - 2) / 4.0, 0.0, 1.0))
        qualities = self.association_quality_history[-3:]
        association = float(np.median(np.asarray(qualities))) if qualities else 0.0
        return maturity * association

    @property
    def position_uncertainty(self) -> float:
        covariance = (
            self.kf.errorCovPre if self.predicted_only else self.kf.errorCovPost
        )
        variance = max(
            0.0,
            0.5 * float(covariance[0, 0] + covariance[1, 1]),
        )
        return float(math.sqrt(variance))


class LieDetectorTracker:
    """Track every star; label one as REAL via constellation geometry."""

    def __init__(
        self,
        *,
        min_radius: int = 35,
        max_radius: int = 90,
        hough_param2: float = 30.0,
        max_match_distance: float = _MATCH_GATE,
        max_lost_frames: int = _MAX_LOST_DEFAULT,
        acquire_brightness: float = 205.0,
        hypothesis_count: int = 12,
        candidate_detector: Optional[ShapeCandidateDetector] = None,
        classical_candidates: Optional[bool] = None,
        shadow_switch: bool = False,
        provisional_reassociation: bool = False,
        multi_hypothesis_identity: bool = False,
        background_certificates: bool = False,
        stale_coast_recovery: bool = False,
        identity_ranker_model: Optional[str] = None,
        identity_ranker_min_margin: float = _RANKER_MIN_MARGIN,
        identity_safety: bool = False,
        state_aware_ranker: bool = False,
        motion_corroboration_model: Optional[str] = None,
        switch_event_model: Optional[str] = None,
        switch_event_min_probability: float = 0.5,
        switch_motion_features: bool = False,
        preassociation_flow: bool = False,
        flow_association: bool = False,
        flow_coast: bool = False,
    ) -> None:
        self.min_radius = int(min_radius)
        self.max_radius = int(max_radius)
        self.hough_param2 = float(hough_param2)
        self.max_match_distance = float(max_match_distance)
        self.max_lost_frames = int(max_lost_frames)
        self.acquire_brightness = float(acquire_brightness)
        self.hypothesis_count = max(1, int(hypothesis_count))
        self.candidate_detector = candidate_detector
        self.classical_candidates = (
            candidate_detector is None
            if classical_candidates is None
            else bool(classical_candidates)
        )
        self.shadow_switch_enabled = bool(shadow_switch)
        self.provisional_reassociation_enabled = bool(provisional_reassociation)
        self.multi_hypothesis_identity_enabled = bool(
            multi_hypothesis_identity
        )
        self.background_certificates_enabled = bool(background_certificates)
        self.stale_coast_recovery_enabled = bool(stale_coast_recovery)
        self.identity_ranker = (
            None
            if not identity_ranker_model
            else TrajectoryIdentityRanker(identity_ranker_model)
        )
        self.identity_ranker_min_margin = max(
            0.0, float(identity_ranker_min_margin)
        )
        # Experimental identity-decision controls.  All are opt-in so the
        # production defaults keep their previous behaviour during ablation.
        self.identity_safety_enabled = bool(identity_safety)
        self.state_aware_ranker_enabled = bool(state_aware_ranker)
        self.motion_corroboration_ranker = (
            None
            if not motion_corroboration_model
            else TrajectoryIdentityRanker(motion_corroboration_model)
        )
        self.switch_event_model = (
            None
            if not switch_event_model
            else SwitchEventModel(
                switch_event_model,
                min_probability=switch_event_min_probability,
            )
        )
        self.switch_optical_features_enabled = bool(
            switch_motion_features
            or (
                self.switch_event_model is not None
                and self.switch_event_model.needs_optical_features
            )
        )
        self.switch_multilag_features_enabled = bool(
            switch_motion_features
            or (
                self.switch_event_model is not None
                and self.switch_event_model.needs_multilag_features
            )
        )
        self.switch_motion_features_enabled = bool(
            self.switch_optical_features_enabled
            or self.switch_multilag_features_enabled
        )
        self.switch_motion_history = (
            SwitchMotionFeatureHistory()
            if self.switch_motion_features_enabled
            else None
        )
        self.flow_association_enabled = bool(flow_association)
        self.flow_coast_enabled = bool(flow_coast)
        self.preassociation_flow_shadow_enabled = bool(
            preassociation_flow
            or (
                self.switch_event_model is not None
                and self.switch_event_model.needs_preflow_features
            )
        )
        self.preassociation_flow_enabled = bool(
            self.preassociation_flow_shadow_enabled
            or self.flow_association_enabled
            or self.flow_coast_enabled
        )
        self.preassociation_flow = (
            PreAssociationFlow() if self.preassociation_flow_enabled else None
        )
        self.reset()

    def reset(self) -> None:
        self.contamination_streak = 0
        self.contamination_candidate = None
        self.last_outlier_real_xy = None
        self.tracks: dict[int, _ShapeTrack] = {}
        self.target_id: Optional[int] = None
        self.target_kind: Optional[str] = None
        self.target_contour: Optional[np.ndarray] = None
        self.target_radius: Optional[float] = None
        self.target_appearance_anchor: Optional[np.ndarray] = None
        self.next_track_id = 1
        self.collective_delta = np.zeros(2, dtype=np.float64)
        self.previous_centers: Optional[np.ndarray] = None
        self.previous_orientations: Optional[np.ndarray] = None
        self.frame_size: Optional[tuple[int, int]] = None
        self.white_seen = False
        self.white_active = False
        self.white_absent_streak = 0
        self.white_phase_completed = False
        self.identity_switched = False
        self.contamination_streak = 0
        self.contamination_candidate: Optional[int] = None
        self.challenger_evidence: dict[int, float] = {}
        self.shadow_candidate_id: Optional[int] = None
        self.shadow_candidate_votes: list[bool] = []
        self.pending_reassociation: Optional[_PendingReassociation] = None
        self.reassociation_cooldown = 0
        self.target_hypotheses: dict[int, _IdentityHypothesis] = {}
        self.identity_path_candidate_id: Optional[int] = None
        self.identity_path_candidate_streak = 0
        self.identity_path_recommended_id: Optional[int] = None
        self.identity_switch_cooldown = 0
        self.identity_unvalidated_candidate_id: Optional[int] = None
        self.identity_unvalidated_candidate_streak = 0
        self.identity_unvalidated_center: Optional[tuple[float, float]] = None
        self.background_certificates = LieBackgroundCertificates()
        self.bg_registry_state = "dormant"
        self.bg_certified_count = 0
        self.stale_recovery_candidate_id: Optional[int] = None
        self.stale_recovery_streak = 0
        self.stale_recovery_committed = False
        self.stale_recovery_hold_frames = 0
        self.stale_recovery_appearance_protect_frames = 0
        self.ranker_candidate_id: Optional[int] = None
        self.ranker_candidate_streak = 0
        self.ranker_margin = 0.0
        self.ranker_switched = False
        self.ranker_top_id: Optional[int] = None
        self.ranker_current_rank = 0
        self.ranker_current_score = 0.0
        self.ranker_switch_delta = 0.0
        self.ranker_evidence = 0.0
        self.ranker_evidence_by_id: dict[int, float] = {}
        self.ranker_disagreement_streak = 0
        self.state_aware_legacy_candidate_id: Optional[int] = None
        self.state_aware_legacy_candidate_streak = 0
        self.motion_corroborated = False
        self.switch_event: Optional[dict[str, object]] = None
        self.switch_event_probability = 0.0
        self.switch_event_approved = False
        self.flow_observation_count = 0
        self.flow_association_count = 0
        self.flow_coast_count = 0
        self.recent_ranker_switch_previous_id: Optional[int] = None
        self.recent_ranker_switch_age = 0
        self.identity_confidence = 0.0
        self.identity_state = "uninitialized"
        self.identity_safe = False
        self.identity_match_streak = 0
        self.identity_mismatch_streak = 0
        self.identity_ranker_unavailable_frames = 0
        self.identity_probation_frames = 0
        self.hold_reason: Optional[str] = "uninitialized"
        if self.identity_ranker is not None:
            self.identity_ranker.reset()
        if self.motion_corroboration_ranker is not None:
            self.motion_corroboration_ranker.reset()
        if self.switch_motion_history is not None:
            self.switch_motion_history.reset()
        if self.preassociation_flow is not None:
            self.preassociation_flow.reset()
        if self.candidate_detector is not None:
            self.candidate_detector.reset()

    @staticmethod
    def _remove_cursor(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = green_cursor_mask(frame_bgr)
        expanded = cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        )
        cleaned = cv2.inpaint(frame_bgr, expanded, 5, cv2.INPAINT_TELEA)
        return cleaned, mask

    @staticmethod
    def _angle_delta_degrees(a: float, b: float) -> float:
        residual = (a - b + math.pi) % (2.0 * math.pi) - math.pi
        return abs(math.degrees(float(residual)))

    @staticmethod
    def _rotation_motion_score(
        deltas: list[float],
        dts: list[float],
        qualities: list[float],
    ) -> float:
        """Confidence-weighted sustained angular speed for an arbitrary shape."""
        count = min(len(deltas), len(dts), len(qualities))
        if count < 2:
            return 0.0
        delta_values = np.asarray(deltas[-count:], dtype=np.float64)
        dt_values = np.maximum(
            np.asarray(dts[-count:], dtype=np.float64), 1.0 / 120.0
        )
        quality_values = np.clip(
            np.asarray(qualities[-count:], dtype=np.float64), 0.0, 1.0
        )
        valid = quality_values > 0.0
        if np.count_nonzero(valid) < 2:
            return 0.0

        angular_speeds = np.degrees(delta_values[valid] / dt_values[valid])
        valid_quality = quality_values[valid]
        order = np.argsort(angular_speeds)
        ordered_speeds = angular_speeds[order]
        ordered_quality = valid_quality[order]
        midpoint = 0.5 * float(np.sum(ordered_quality))
        median_index = int(
            np.searchsorted(np.cumsum(ordered_quality), midpoint, side="left")
        )
        robust_speed = float(
            ordered_speeds[min(median_index, len(ordered_speeds) - 1)]
        )
        direction_consistency = abs(
            float(np.sum(valid_quality * np.sign(angular_speeds)))
        ) / max(float(np.sum(valid_quality)), 1e-6)
        reliability = (
            float(np.median(valid_quality))
            * direction_consistency
            * float(np.count_nonzero(valid)) / count
        )
        normalized_speed = min(
            1.0, abs(robust_speed) / _FULL_ROTATION_SPEED_DEG_PER_SECOND
        )
        return float(_MAX_ROTATION_SCORE * normalized_speed * reliability)

    @staticmethod
    def _rotation_edge_map(gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(grad_x, grad_y)

    @staticmethod
    def _rotation_descriptor(
        edge_magnitude: np.ndarray, detection: ShapeDetection
    ) -> Optional[np.ndarray]:
        """Polar edge signature whose angular shift measures frame-to-frame spin."""
        cx, cy = detection.center
        _, _, width, height = detection.bbox
        sample_radius = max(8.0, 0.55 * max(float(width), float(height)))
        side = max(16, int(round(2.0 * sample_radius)))
        if side < 16:
            return None
        patch = cv2.getRectSubPix(
            edge_magnitude, (side, side), (float(cx), float(cy))
        )
        if patch is None or patch.size == 0:
            return None
        magnitude = cv2.resize(
            patch, (96, 96), interpolation=cv2.INTER_AREA
        )
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
        radial_weight = np.exp(
            -0.5 * ((radial_position - 0.72) / 0.30) ** 2
        )
        polar *= radial_weight[None, :]
        raw_norm = float(np.linalg.norm(polar))
        if raw_norm <= 1e-5:
            return None
        # Remove rotationally invariant rings. A circle becomes nearly zero,
        # while corners and arbitrary boundary/texture details remain.
        polar -= np.mean(polar, axis=0, keepdims=True)
        angular_norm = float(np.linalg.norm(polar))
        if angular_norm / raw_norm < 0.45:
            return None
        return polar / angular_norm

    @staticmethod
    def _match_rotation_descriptors(
        previous: np.ndarray,
        current: np.ndarray,
        *,
        predicted_delta: float = 0.0,
    ) -> tuple[Optional[float], float]:
        """Match two polar signatures inside the physically plausible window."""
        if previous.shape != current.shape or previous.ndim != 2:
            return None, 0.0
        degrees_per_bin = 360.0 / previous.shape[0]
        max_shift = max(
            1,
            int(round(_MAX_ROTATION_DEGREES_PER_FRAME / degrees_per_bin)),
        )
        shifts = np.arange(-max_shift, max_shift + 1, dtype=np.int32)
        correlations = np.asarray(
            [
                float(np.sum(previous * np.roll(current, int(shift), axis=0)))
                for shift in shifts
            ],
            dtype=np.float64,
        )
        # A weak temporal prior resolves equal peaks from symmetric polygons
        # without overpowering actual image evidence.
        delta_candidates = -np.radians(shifts * degrees_per_bin)
        prior_distance = np.abs(delta_candidates - float(predicted_delta))
        objective = correlations - 0.015 * np.clip(
            prior_distance / math.radians(_MAX_ROTATION_DEGREES_PER_FRAME),
            0.0,
            1.0,
        )
        best_index = int(np.argmax(objective))
        best_correlation = float(correlations[best_index])
        if best_correlation < 0.25:
            return None, 0.0
        exclusion = max(2, int(round(5.0 / degrees_per_bin)))
        alternatives = np.ones(len(correlations), dtype=bool)
        alternatives[
            max(0, best_index - exclusion) : min(
                len(correlations), best_index + exclusion + 1
            )
        ] = False
        second_correlation = (
            float(np.max(correlations[alternatives]))
            if np.any(alternatives)
            else -1.0
        )
        peak_strength = float(
            np.clip((best_correlation - 0.25) / 0.65, 0.0, 1.0)
        )
        peak_uniqueness = float(
            np.clip((best_correlation - second_correlation) / 0.10, 0.0, 1.0)
        )
        confidence = peak_strength * peak_uniqueness
        if confidence < 0.05:
            return None, 0.0
        refined_shift = float(shifts[best_index])
        if 0 < best_index < len(correlations) - 1:
            left = float(correlations[best_index - 1])
            center = float(correlations[best_index])
            right = float(correlations[best_index + 1])
            curvature = left - 2.0 * center + right
            if curvature < -1e-6:
                refined_shift += float(
                    np.clip(0.5 * (left - right) / curvature, -1.0, 1.0)
                )
        refined_delta = -math.radians(refined_shift * degrees_per_bin)
        return float(refined_delta), float(confidence)

    @staticmethod
    def _appearance_descriptor(
        gray: np.ndarray, detection: ShapeDetection
    ) -> Optional[np.ndarray]:
        """Rotation/translation-tolerant frequency signature of a YOLO crop."""
        x, y, width, height = detection.bbox
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(gray.shape[1], x + width)
        y1 = min(gray.shape[0], y + height)
        if x1 - x0 < 12 or y1 - y0 < 12:
            return None
        patch = cv2.resize(
            gray[y0:y1, x0:x1],
            (40, 40),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        patch = (patch - float(np.mean(patch))) / (
            float(np.std(patch)) + 1e-5
        )
        patch *= np.outer(np.hanning(40), np.hanning(40)).astype(np.float32)
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(patch)))
        grid_y, grid_x = np.indices(spectrum.shape, dtype=np.float32)
        radius = np.sqrt((grid_x - 19.5) ** 2 + (grid_y - 19.5) ** 2)
        radial: list[float] = []
        # Radial pooling makes the signature insensitive to the REAL star's
        # self-rotation while retaining its texture-frequency distribution.
        for low, high in zip(np.linspace(1.0, 18.0, 10)[:-1], np.linspace(1.0, 18.0, 10)[1:]):
            values = spectrum[(radius >= low) & (radius < high)]
            radial.extend(
                (
                    float(np.mean(values)),
                    float(np.std(values)),
                    float(np.percentile(values, 75.0)),
                )
            )
        descriptor = np.log1p(np.asarray(radial, dtype=np.float32))
        norm = float(np.linalg.norm(descriptor))
        if norm <= 1e-6:
            return None
        return descriptor / norm

    @staticmethod
    def _initial_bright_shape(
        gray: np.ndarray, hsv: np.ndarray
    ) -> Optional[ShapeDetection]:
        mask = np.where((gray >= 210) & (hsv[:, :, 1] <= 75), 255, 0).astype(np.uint8)
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
            circularity = (
                0.0
                if perimeter <= 0
                else 4.0 * math.pi * area / (perimeter * perimeter)
            )
            component = np.zeros_like(gray)
            cv2.drawContours(component, [contour], -1, 255, -1)
            mean_brightness = float(cv2.mean(gray, mask=component)[0])
            rank = (
                mean_brightness
                + circularity * 45.0
                - abs(math.log(max(aspect, 1e-3))) * 25.0
            )
            moments = cv2.moments(contour)
            if moments["m00"]:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
            else:
                cx, cy = x + w / 2.0, y + h / 2.0
            det = ShapeDetection(
                center=(float(cx), float(cy)),
                radius=float(0.25 * (w + h)),
                bbox=(x, y, w, h),
                score=mean_brightness / 255.0,
                circularity=float(circularity),
                source="bright",
                observed_radius=float(0.25 * (w + h)),
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
                    observed_radius=r,
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
            circularity = (
                0.0
                if perimeter <= 0
                else 4.0 * math.pi * area / (perimeter * perimeter)
            )
            moments = cv2.moments(contour)
            if not moments["m00"]:
                continue
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            shape_distance = 0.0
            if reference_contour is not None:
                shape_distance = float(
                    cv2.matchShapes(
                        reference_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0
                    )
                )
                if shape_distance > 1.35:
                    continue
            detections.append(
                ShapeDetection(
                    center=(float(cx), float(cy)),
                    radius=float(0.25 * (w + h)),
                    bbox=(x, y, w, h),
                    circularity=float(circularity),
                    source="contour",
                    shape_distance=shape_distance,
                    observed_radius=float(0.25 * (w + h)),
                    contour=contour.copy(),
                )
            )
        return self._deduplicate(detections, self.target_radius)

    @staticmethod
    def _deduplicate(
        detections: Iterable[ShapeDetection],
        reference_radius: Optional[float] = None,
    ) -> list[ShapeDetection]:
        kept: list[ShapeDetection] = []
        for det in sorted(
            detections,
            key=lambda item: (
                float(item.yolo_confidence),
                item.observed_radius or item.radius,
            ),
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
        if not learned:
            return classical
        # YOLO boxes keep their native size; classical only annotates / gap-fills.
        fused = list(learned)
        if len(fused) >= 8:
            return self._deduplicate(fused, self.target_radius)
        for classical_det in classical:
            distances = [
                float(np.linalg.norm(np.subtract(item.center, classical_det.center)))
                for item in fused
            ]
            index = int(np.argmin(distances))
            gate = max(18.0, 0.70 * float(self.target_radius or fused[index].radius))
            if distances[index] <= gate:
                if classical_det.contour is not None:
                    fused[index].contour = classical_det.contour.copy()
                if classical_det.orientation is not None and fused[index].orientation is None:
                    fused[index].orientation = classical_det.orientation
                fused[index].shape_distance = min(
                    fused[index].shape_distance or 99.0,
                    classical_det.shape_distance,
                )
                continue
            if len(fused) < 8:
                fused.append(classical_det)
        return self._deduplicate(fused, self.target_radius)

    @staticmethod
    def _limit_patterns(
        detections: list[ShapeDetection],
        *,
        max_count: int = MAX_PATTERN_COUNT,
    ) -> list[ShapeDetection]:
        if len(detections) <= max_count:
            return detections
        ranked = sorted(
            detections,
            key=lambda item: (
                0 if item.source == "bright" else 1,
                0 if item.source == "yolo" or item.yolo_confidence > 0 else 1,
                -float(item.yolo_confidence),
                float(item.shape_distance),
                -float(item.score),
            ),
        )
        return ranked[:max_count]

    def _annotate_collective_motion(self, detections: list[ShapeDetection]) -> None:
        if not detections:
            return
        current = np.asarray([d.center for d in detections], dtype=np.float64)
        current_orient = np.asarray(
            [np.nan if d.orientation is None else float(d.orientation) for d in detections],
            dtype=np.float64,
        )
        if self.previous_centers is None or len(self.previous_centers) == 0:
            self.previous_centers = current
            self.previous_orientations = current_orient
            for detection in detections:
                detection.collective_residual = 0.0
                detection.collective_rotation_residual = 0.0
            return

        diffs = (current[:, None, :] - self.previous_centers[None, :, :]).reshape(-1, 2)
        diffs = diffs[np.linalg.norm(diffs, axis=1) <= 40.0]
        if len(diffs):
            bin_size = 2.0
            bins = np.rint(diffs / bin_size).astype(np.int32)
            keys, counts = np.unique(bins, axis=0, return_counts=True)
            support = counts.astype(np.float64) - 0.12 * np.linalg.norm(
                keys * bin_size - self.collective_delta, axis=1
            )
            best = int(np.argmax(support))
            seed = keys[best].astype(np.float64) * bin_size
            nearby = diffs[np.linalg.norm(diffs - seed, axis=1) <= 3.0]
            measured = np.median(nearby, axis=0)
            self.collective_delta = 0.30 * self.collective_delta + 0.70 * measured

        expected = self.previous_centers + self.collective_delta
        distances = np.linalg.norm(current[:, None, :] - expected[None, :, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        position_residuals = distances[np.arange(len(current)), nearest]
        for index, detection in enumerate(detections):
            rotation_deg = 0.0
            if self.previous_orientations is not None:
                prev = self.previous_orientations[nearest[index]]
                cur = current_orient[index]
                if np.isfinite(prev) and np.isfinite(cur):
                    rotation_deg = self._angle_delta_degrees(float(cur), float(prev))
            detection.collective_rotation_residual = rotation_deg
            # Keep translation residual pure; rotation is scored separately.
            detection.collective_residual = float(position_residuals[index])
        self.previous_centers = current
        self.previous_orientations = current_orient

    def _annotate_track_rotation_residuals(self) -> None:
        """Compare associated per-track spin against the visible peer consensus.

        Real image detections only obtain an orientation delta while they are
        associated with a track.  Computing this residual before association
        therefore silently produced zeros for the normal YOLO path.
        """
        visible = [
            track
            for track in self.tracks.values()
            if track.lost_frames == 0
            and not track.predicted_only
            and track.last_detection is not None
        ]
        measured = [track for track in visible if track.rotation_valid]
        consensus: Optional[float] = None
        if len(measured) >= 3:
            deltas = np.asarray(
                [track.last_rotation_delta for track in measured], dtype=np.float64
            )
            weights = np.asarray(
                [track.rotation_confidence for track in measured], dtype=np.float64
            )
            order = np.argsort(deltas)
            ordered = deltas[order]
            ordered_weights = weights[order]
            midpoint = 0.5 * float(np.sum(ordered_weights))
            index = int(
                np.searchsorted(
                    np.cumsum(ordered_weights), midpoint, side="left"
                )
            )
            consensus = float(ordered[min(index, len(ordered) - 1)])
        for track in visible:
            residual = (
                self._angle_delta_degrees(track.last_rotation_delta, consensus)
                if track.rotation_valid and consensus is not None
                else 0.0
            )
            track.peer_rotation_residual = float(residual)

    def _update_background_certificates(
        self,
        detections: list[ShapeDetection],
        bright: Optional[ShapeDetection],
        allow_bright_seed: bool,
    ) -> None:
        """Carry only unambiguous white-phase BG provenance forward."""
        for detection in detections:
            detection.bg_certified = False
        if not self.background_certificates_enabled:
            return
        target = (
            self.tracks.get(self.target_id)
            if self.target_id is not None
            else None
        )
        real_center = (
            bright.center
            if bright is not None and allow_bright_seed
            else (target.state[:2] if target is not None else None)
        )
        real_radius = (
            bright.radius
            if bright is not None and allow_bright_seed
            else (target.state[2] if target is not None else 0.0)
        )
        result = self.background_certificates.update(
            [detection.center for detection in detections],
            white_active=allow_bright_seed,
            real_center=real_center,
            real_radius=real_radius,
            frame_size=self.frame_size,
        )
        self.bg_registry_state = result.state
        self.bg_certified_count = len(result.certified_indices)
        for index in result.certified_indices:
            if 0 <= index < len(detections):
                detections[index].bg_certified = True

    @staticmethod
    def _hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
        if cost.size == 0:
            return []
        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(cost)
            return list(zip(rows.tolist(), cols.tolist()))
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

    @staticmethod
    def _appearance_distance(
        left: Optional[np.ndarray],
        right: Optional[np.ndarray],
    ) -> float:
        if left is None or right is None or left.shape != right.shape:
            return float("inf")
        return float(np.clip(1.0 - np.dot(left, right), 0.0, 2.0))

    @staticmethod
    def _local_signature(
        center: tuple[float, float],
        radius: float,
        other_centers: Iterable[tuple[float, float]],
    ) -> Optional[np.ndarray]:
        distances = sorted(
            math.hypot(center[0] - other[0], center[1] - other[1])
            / max(float(radius), 1.0)
            for other in other_centers
            if math.hypot(center[0] - other[0], center[1] - other[1]) > 1e-3
        )
        if len(distances) < 3:
            return None
        return np.asarray(distances[:5], dtype=np.float32)

    @staticmethod
    def _signature_error(
        left: Optional[np.ndarray],
        right: Optional[np.ndarray],
    ) -> float:
        if left is None or right is None:
            return float("inf")
        count = min(len(left), len(right))
        if count < 3:
            return float("inf")
        return float(np.median(np.abs(left[:count] - right[:count])))

    def _advance_pending_reassociation(self) -> bool:
        pending = self.pending_reassociation
        if pending is None:
            return False
        old = self.tracks.get(pending.old_track_id)
        child = self.tracks.get(pending.child_track_id)
        if old is None or child is None:
            self.pending_reassociation = None
            return False
        if child.lost_frames > 1:
            self.pending_reassociation = None
            self.reassociation_cooldown = 3
            return False
        if child.predicted_only or child.visible_streak < _PROVISIONAL_OBSERVATIONS:
            return False

        current_signature = self._local_signature(
            child.state[:2],
            child.state[2],
            (
                track.state[:2]
                for track in self.tracks.values()
                if track.id not in (old.id, child.id)
                and track.lost_frames == 0
                and not track.predicted_only
            ),
        )
        appearance = self._appearance_distance(
            (
                self.target_appearance_anchor
                if self.target_appearance_anchor is not None
                else old.appearance
            ),
            child.appearance,
        )
        topology = self._signature_error(
            pending.initial_signature,
            current_signature,
        )
        association = float(
            np.median(np.asarray(child.association_quality_history[-3:]))
        )
        accepted = (
            pending.initial_position_error <= max(45.0, old.state[2] * 1.5)
            and appearance <= 0.18
            and topology <= 0.55
            and association >= 0.20
        )
        self.pending_reassociation = None
        if not accepted:
            self.reassociation_cooldown = 3
            return False

        self.target_id = child.id
        child.role = "real"
        self.last_outlier_real_xy = child.state[:2]
        del self.tracks[old.id]
        return True

    def _spawn(self, detection: ShapeDetection, timestamp: float) -> _ShapeTrack:
        track = _ShapeTrack(self.next_track_id, detection, timestamp)
        self.tracks[track.id] = track
        self.next_track_id += 1
        return track

    def _flow_coast_is_safe(
        self,
        track: _ShapeTrack,
        observation: Optional[FlowTrackObservation],
        detections: list[ShapeDetection],
    ) -> bool:
        if (
            not self.flow_coast_enabled
            or track.id != self.target_id
            or observation is None
            or track.lost_frames >= 3
            or observation.confidence < _FLOW_COAST_MIN_CONFIDENCE
            or observation.forward_backward_error > _FLOW_COAST_MAX_FB_ERROR
            or observation.inlier_ratio < _FLOW_COAST_MIN_INLIER_RATIO
            or observation.coverage < _FLOW_COAST_MIN_COVERAGE
            or observation.local_fit_error > _FLOW_COAST_MAX_FIT_ERROR
            or math.hypot(
                observation.predicted_center[0] - track.state[0],
                observation.predicted_center[1] - track.state[1],
            )
            > _FLOW_COAST_MAX_KALMAN_DISAGREEMENT
            or self._outside_frame(observation.predicted_center)
        ):
            return False
        # If YOLO found a nearby centre, a duplicate flow-only track would make
        # identity ambiguity worse.  Flow coast is reserved for genuine short
        # detector gaps.
        nearest = min(
            (
                math.hypot(
                    detection.center[0] - observation.predicted_center[0],
                    detection.center[1] - observation.predicted_center[1],
                )
                for detection in detections
            ),
            default=float("inf"),
        )
        return nearest > max(14.0, 0.55 * track.state[2])

    def _associate(
        self,
        detections: list[ShapeDetection],
        timestamp: float,
        flow_observations: Optional[dict[int, FlowTrackObservation]] = None,
    ) -> bool:
        flow_observations = flow_observations or {}
        track_list = list(self.tracks.values())
        predictions = [track.predict(timestamp) for track in track_list]
        if not track_list:
            for detection in detections:
                self._spawn(detection, timestamp)
            return False
        if not detections:
            for track in track_list:
                observation = flow_observations.get(track.id)
                if self._flow_coast_is_safe(track, observation, detections):
                    assert observation is not None
                    track.apply_flow_coast(observation)
                    self.flow_coast_count += 1
                else:
                    track.mark_missed()
            committed = self._advance_pending_reassociation()
            self._drop_stale()
            return committed

        large = 1e6
        cost = np.full((len(track_list), len(detections)), large, dtype=np.float32)
        match_quality = np.zeros_like(cost)
        flow_used = np.zeros_like(cost, dtype=bool)
        reserved_col: Optional[int] = None
        reserved_position_error = float("inf")
        provisional_old: Optional[_ShapeTrack] = None
        if self.reassociation_cooldown > 0:
            self.reassociation_cooldown -= 1
        if self.provisional_reassociation_enabled and self.target_id is not None:
            provisional_old = self.tracks.get(self.target_id)
            if (
                provisional_old is not None
                and provisional_old.lost_frames >= _PROVISIONAL_GAP_FRAMES
                and self.pending_reassociation is None
                and self.reassociation_cooldown == 0
            ):
                old_prediction = predictions[track_list.index(provisional_old)]
                anchor = (
                    self.target_appearance_anchor
                    if self.target_appearance_anchor is not None
                    else provisional_old.appearance
                )
                candidates: list[tuple[float, int, float]] = []
                for col, detection in enumerate(detections):
                    position_error = math.hypot(
                        detection.center[0] - old_prediction[0],
                        detection.center[1] - old_prediction[1],
                    )
                    appearance_error = self._appearance_distance(
                        anchor, detection.appearance
                    )
                    if (
                        position_error
                        <= self.max_match_distance + 75.0
                        and appearance_error <= 0.30
                    ):
                        candidates.append(
                            (
                                position_error + 80.0 * appearance_error,
                                col,
                                position_error,
                            )
                        )
                if candidates:
                    candidates.sort()
                    best = candidates[0]
                    # Repeated star textures frequently produce several nearly
                    # equivalent matches. In that case keep coasting the old
                    # REAL instead of inventing certainty and committing a child.
                    unambiguous = (
                        len(candidates) == 1
                        or candidates[1][0] - best[0] >= 20.0
                    )
                    if unambiguous:
                        _, reserved_col, reserved_position_error = best

        for row, (track, prediction) in enumerate(zip(track_list, predictions)):
            if (
                self.provisional_reassociation_enabled
                and track.id == self.target_id
                and track.lost_frames >= _PROVISIONAL_GAP_FRAMES
                and (
                    reserved_col is not None
                    or (
                        self.pending_reassociation is not None
                        and self.pending_reassociation.old_track_id == track.id
                    )
                )
            ):
                continue
            px, py, pr = prediction
            flow = flow_observations.get(track.id)
            flow_weight = 0.0
            # Repeated star textures can produce confident-but-wrong LK tracks
            # during visible crossings.  Optical flow may rescue a track only
            # after the detector association has already missed it; it never
            # rewrites an otherwise live association.
            association_ambiguous = track.lost_frames > 0
            if (
                self.flow_association_enabled
                and association_ambiguous
                and flow is not None
                and flow.confidence >= _FLOW_ASSOCIATION_MIN_CONFIDENCE
                and math.hypot(
                    flow.predicted_center[0] - px,
                    flow.predicted_center[1] - py,
                )
                <= max(36.0, 0.90 * pr)
            ):
                flow_weight = min(
                    _FLOW_ASSOCIATION_MAX_WEIGHT,
                    0.10 + 0.30 * flow.confidence,
                )
            gate = self.max_match_distance + min(track.lost_frames, 6) * 10.0
            if track.id == self.target_id:
                gate += 15.0
            for col, detection in enumerate(detections):
                distance = math.hypot(
                    detection.center[0] - px, detection.center[1] - py
                )
                flow_distance = (
                    math.hypot(
                        detection.center[0] - flow.predicted_center[0],
                        detection.center[1] - flow.predicted_center[1],
                    )
                    if flow_weight > 0.0 and flow is not None
                    else distance
                )
                association_distance = (
                    (1.0 - flow_weight) * distance
                    + flow_weight * flow_distance
                )
                radius_delta = abs(detection.radius - pr)
                flow_rescue = bool(
                    flow is not None
                    and flow.confidence >= 0.55
                    and track.lost_frames > 0
                    and flow_distance <= gate
                )
                if (
                    (distance <= gate or flow_rescue)
                    and radius_delta <= max(30.0, pr * 0.85)
                ):
                    frame_gap = max(1, track.lost_frames + 1)
                    measured_velocity = (
                        (
                            detection.center[0] - track.last_detection.center[0]
                        ) / frame_gap,
                        (
                            detection.center[1] - track.last_detection.center[1]
                        ) / frame_gap,
                    )
                    expected_velocity = track.robust_velocity
                    acceleration_error = math.hypot(
                        measured_velocity[0] - expected_velocity[0],
                        measured_velocity[1] - expected_velocity[1],
                    )
                    cost[row, col] = (
                        association_distance
                        + 0.20 * radius_delta
                        - 10.0 * detection.yolo_confidence
                    )
                    innovation_quality = math.exp(
                        -0.5 * (association_distance / 30.0) ** 2
                    )
                    continuity_quality = (
                        math.exp(-max(0.0, acceleration_error - 4.0) / 10.0)
                        if track.motion_history
                        else 1.0
                    )
                    match_quality[row, col] = float(
                        innovation_quality
                        * continuity_quality
                        * (
                            0.75 + 0.25 * flow.confidence
                            if flow_weight > 0.0 and flow is not None
                            else 1.0
                        )
                    )
                    flow_used[row, col] = flow_weight > 0.0
        if reserved_col is not None:
            cost[:, reserved_col] = large

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for row, col in self._hungarian(cost):
            if cost[row, col] >= large:
                continue
            track_list[row].update(
                detections[col],
                timestamp,
                association_quality=float(match_quality[row, col]),
                flow_association_used=bool(flow_used[row, col]),
            )
            if flow_used[row, col]:
                self.flow_association_count += 1
            matched_tracks.add(row)
            matched_detections.add(col)
        for row, track in enumerate(track_list):
            if row not in matched_tracks:
                observation = flow_observations.get(track.id)
                if self._flow_coast_is_safe(track, observation, detections):
                    assert observation is not None
                    track.apply_flow_coast(observation)
                    self.flow_coast_count += 1
                else:
                    track.mark_missed()
        if reserved_col is not None and provisional_old is not None:
            detection = detections[reserved_col]
            child = self._spawn(detection, timestamp)
            signature = self._local_signature(
                detection.center,
                detection.radius,
                (
                    other.center
                    for col, other in enumerate(detections)
                    if col != reserved_col
                ),
            )
            self.pending_reassociation = _PendingReassociation(
                old_track_id=provisional_old.id,
                child_track_id=child.id,
                initial_signature=signature,
                initial_position_error=reserved_position_error,
            )
            matched_detections.add(reserved_col)
        for col, detection in enumerate(detections):
            if col not in matched_detections:
                self._spawn(detection, timestamp)
        committed = self._advance_pending_reassociation()
        self._drop_stale()
        return committed

    def _outside_frame(
        self, center: tuple[float, float], margin: float = _FRAME_EXIT_MARGIN
    ) -> bool:
        if self.frame_size is None:
            return False
        width, height = self.frame_size
        x, y = float(center[0]), float(center[1])
        return x < -margin or y < -margin or x > width + margin or y > height + margin

    def _leaving_frame(self, track: _ShapeTrack) -> bool:
        """True when the track sits in the border band and drifts outward."""
        if self.frame_size is None:
            return False
        width, height = self.frame_size
        x, y, radius = track.state
        vx, vy = track.predicted_velocity
        margin = max(_EDGE_SWITCH_MARGIN, radius * 0.5)
        return (
            (x <= margin and vx < 0.0)
            or (x >= width - margin and vx > 0.0)
            or (y <= margin and vy < 0.0)
            or (y >= height - margin and vy > 0.0)
        )

    def _drop_stale(self) -> None:
        stale: list[int] = []
        for track_id, track in self.tracks.items():
            if track.lost_frames == 0:
                continue
            # A coasting track that has left the panel can never be matched
            # again; keeping it alive only extrapolates it further off-screen.
            if self._outside_frame(track.state[:2]):
                stale.append(track_id)
                continue
            budget = (
                _REAL_MAX_COAST
                if track_id == self.target_id
                else self.max_lost_frames
            )
            if track.lost_frames > budget:
                stale.append(track_id)
        for track_id in stale:
            del self.tracks[track_id]
        if self.target_id is not None and self.target_id not in self.tracks:
            # Release the label so scoring can re-acquire a live track instead
            # of staying pinned to a ghost forever.
            self.target_id = None
            self.contamination_streak = 0
            self.contamination_candidate = None

    @staticmethod
    def _velocity_speed(velocity: tuple[float, float]) -> float:
        return float(math.hypot(velocity[0], velocity[1]))

    @staticmethod
    def _velocity_angle(velocity: tuple[float, float]) -> Optional[float]:
        if abs(velocity[0]) < 1e-6 and abs(velocity[1]) < 1e-6:
            return None
        return float(math.atan2(velocity[1], velocity[0]))

    @staticmethod
    def _circular_mean(angles: list[float]) -> Optional[float]:
        if not angles:
            return None
        sine = float(sum(math.sin(angle) for angle in angles))
        cosine = float(sum(math.cos(angle) for angle in angles))
        if abs(sine) < 1e-9 and abs(cosine) < 1e-9:
            return None
        return float(math.atan2(sine, cosine))

    @staticmethod
    def _heading_delta_degrees(a: float, b: float) -> float:
        residual = (a - b + math.pi) % (2.0 * math.pi) - math.pi
        return abs(math.degrees(float(residual)))

    @staticmethod
    def _pairwise_rigidity_error(
        centers: dict[int, tuple[float, float]],
        previous: dict[int, tuple[float, float]],
    ) -> float:
        ids = [track_id for track_id in centers if track_id in previous]
        if len(ids) < 2:
            return 0.0
        changes: list[float] = []
        for index, left in enumerate(ids):
            for right in ids[index + 1 :]:
                now = math.hypot(
                    centers[left][0] - centers[right][0],
                    centers[left][1] - centers[right][1],
                )
                then = math.hypot(
                    previous[left][0] - previous[right][0],
                    previous[left][1] - previous[right][1],
                )
                changes.append(abs(now - then))
        return float(np.mean(np.asarray(changes, dtype=np.float64))) if changes else 0.0

    def _score_real_candidates(self) -> dict[int, float]:
        """Score each track as REAL using four leave-one-out cues.

        1) self-rotation (REAL spins; BG does not)
        2) BG layout rigidity after excluding the candidate
        3) BG speed consistency after excluding the candidate
        4) translation-direction disagreement vs BG consensus
        """
        active = [
            track
            for track in self.tracks.values()
            if track.lost_frames == 0
            and not track.predicted_only
            and track.last_detection is not None
            and track.prev_center is not None
        ]
        scores: dict[int, float] = {}
        if len(active) < 2:
            for track in self.tracks.values():
                if track.lost_frames > 4 or track.last_detection is None:
                    continue
                detection = track.last_detection
                track.translation_residual = float(detection.collective_residual)
                track.rotation_residual = float(detection.collective_rotation_residual)
                spin = 0.0
                if len(track.rotation_delta_history) >= 2:
                    spin = self._rotation_motion_score(
                        track.rotation_delta_history,
                        track.rotation_dt_history,
                        track.orientation_quality_history[
                            -len(track.rotation_delta_history):
                        ],
                    )
                score = (
                    _W_TRANSLATION * float(detection.collective_residual)
                    + _W_ROTATION * spin
                )
                if track.id == self.target_id:
                    score += _REAL_STICKY_BONUS
                track.real_score = float(score)
                track.rotation_score = spin
                track.direction_score = 0.0
                track.speed_score = 0.0
                track.rigidity_score = 0.0
                track.appearance_score = 0.0
                scores[track.id] = score
            return scores

        centers = {
            track.id: (
                float(track.last_detection.center[0]),
                float(track.last_detection.center[1]),
            )
            for track in active
        }
        previous = {
            track.id: (float(track.prev_center[0]), float(track.prev_center[1]))
            for track in active
        }
        velocities = {track.id: track.velocity for track in active}
        speeds = {
            track_id: self._velocity_speed(velocity)
            for track_id, velocity in velocities.items()
        }
        headings = {
            track_id: (
                self._velocity_angle(velocity)
                if speeds[track_id] >= _MIN_HEADING_SPEED
                else None
            )
            for track_id, velocity in velocities.items()
        }
        full_rigidity = self._pairwise_rigidity_error(centers, previous)
        appearance_distances: dict[int, float] = {}
        if self.target_appearance_anchor is not None:
            for track in active:
                if track.appearance is None:
                    continue
                appearance_distances[track.id] = float(
                    np.clip(
                        1.0
                        - np.dot(
                            track.appearance,
                            self.target_appearance_anchor,
                        ),
                        0.0,
                        2.0,
                    )
                )
        appearance_median = (
            float(np.median(np.asarray(list(appearance_distances.values()))))
            if len(appearance_distances) >= 4
            else 0.0
        )
        appearance_mad = (
            float(
                np.median(
                    np.abs(
                        np.asarray(list(appearance_distances.values()))
                        - appearance_median
                    )
                )
            )
            if len(appearance_distances) >= 4
            else 0.0
        )
        current_appearance_distance = appearance_distances.get(self.target_id)
        best_appearance_distance = (
            min(appearance_distances.values())
            if appearance_distances
            else None
        )
        appearance_recovery_active = (
            current_appearance_distance is not None
            and best_appearance_distance is not None
            and best_appearance_distance <= 0.05
            and current_appearance_distance - best_appearance_distance >= 0.02
        )

        for track in active:
            detection = track.last_detection
            track.translation_residual = float(detection.collective_residual)
            track.rotation_residual = float(detection.collective_rotation_residual)

            # 1) self-rotation
            rotation = 0.0
            if len(track.rotation_delta_history) >= 2:
                rotation = self._rotation_motion_score(
                    track.rotation_delta_history,
                    track.rotation_dt_history,
                    track.orientation_quality_history[
                        -len(track.rotation_delta_history):
                    ],
                )
            rotation += float(detection.collective_rotation_residual)

            bg_ids = [other.id for other in active if other.id != track.id]
            bg_speeds = [speeds[track_id] for track_id in bg_ids]
            bg_heading_items = [
                headings[track_id]
                for track_id in bg_ids
                if headings[track_id] is not None
            ]
            bg_centers = {track_id: centers[track_id] for track_id in bg_ids}
            bg_previous = {track_id: previous[track_id] for track_id in bg_ids}

            # 2) relative-position rigidity of BG after excluding candidate
            bg_rigidity = self._pairwise_rigidity_error(bg_centers, bg_previous)
            rigidity = max(0.0, full_rigidity - bg_rigidity)

            # 3) BG speed consistency + candidate speed disagreement
            speed = speeds[track.id]
            if bg_speeds:
                bg_speed_med = float(np.median(np.asarray(bg_speeds)))
                bg_speed_std = float(np.std(np.asarray(bg_speeds)))
                speed_disagree = abs(speed - bg_speed_med)
                # Prefer candidates that make the remaining BG pack tighter.
                speed_pack = max(
                    0.0,
                    float(np.std(np.asarray(list(speeds.values())))) - bg_speed_std,
                )
            else:
                bg_speed_med = 0.0
                speed_disagree = speed
                speed_pack = 0.0
            speed_score = speed_disagree + 0.75 * speed_pack

            # 4) translation direction: REAL disagrees with BG consensus
            own_heading = headings[track.id]
            bg_mean = self._circular_mean(bg_heading_items)
            if (
                own_heading is None
                or bg_mean is None
                or len(bg_heading_items) < 3
            ):
                direction = 0.0
            else:
                direction = self._heading_delta_degrees(own_heading, bg_mean)

            appearance_score = 0.0
            if (
                appearance_recovery_active
                and track.id in appearance_distances
                and appearance_mad > 0.0
            ):
                appearance_score = float(
                    np.clip(
                        (
                            appearance_median
                            - appearance_distances[track.id]
                        )
                        / max(0.008, appearance_mad),
                        -2.0,
                        3.0,
                    )
                )
            score = (
                _W_ROTATION * rotation
                + _W_DIRECTION * direction
                + _W_SPEED * speed_score
                + _W_RIGIDITY * rigidity
                + _W_TRANSLATION * float(detection.collective_residual)
                + _W_ANCHOR_APPEARANCE * appearance_score
            )
            if track.id == self.target_id:
                score += _REAL_STICKY_BONUS
            track.rotation_score = float(rotation)
            track.direction_score = float(direction)
            track.speed_score = float(speed_score)
            track.rigidity_score = float(rigidity)
            track.appearance_score = float(appearance_score)
            track.real_score = float(score)
            scores[track.id] = float(score)
        return scores

    @staticmethod
    def _track_looks_like_background(track: _ShapeTrack) -> bool:
        """Whether the current label has lost the characteristic REAL cues."""
        return (
            track.rotation_score < 8.0
            and track.direction_score < 15.0
            and track.translation_residual < 8.0
        )

    @staticmethod
    def _identity_emissions(scores: dict[int, float]) -> dict[int, float]:
        """Robustly normalize per-frame cue scores for path accumulation."""
        if not scores:
            return {}
        values = np.asarray(list(scores.values()), dtype=np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = max(5.0, 1.4826 * mad)
        return {
            track_id: float(np.clip(
                (score - median) / scale,
                -_IDENTITY_EMISSION_CLIP,
                _IDENTITY_EMISSION_CLIP,
            ))
            for track_id, score in scores.items()
        }

    def _advance_identity_hypotheses(
        self,
        scores: dict[int, float],
        *,
        commit: bool = True,
    ) -> bool:
        """Advance a small beam of continuous REAL identity paths.

        Unlike greedy role switching, every hypothesis has to descend from the
        white-seeded path through spatially reachable ID hand-offs. Competing
        paths remain alive, so a brief cue spike cannot immediately teleport
        REAL to a distant decoy.
        """
        path_scores = dict(scores)
        if self.target_id is not None and self.target_id in path_scores:
            # Greedy scoring includes a public-target sticky bonus. It is
            # useful for the baseline decision but would make an independent
            # path validator circular, so remove it from beam emissions.
            path_scores[self.target_id] -= _REAL_STICKY_BONUS
        emissions = self._identity_emissions(path_scores)
        self.identity_path_recommended_id = None
        previous = self.target_hypotheses
        if not previous:
            seed_id = (
                self.target_id
                if self.target_id is not None and self.target_id in self.tracks
                else (max(scores, key=scores.get) if scores else None)
            )
            if seed_id is not None:
                track = self.tracks[seed_id]
                previous = {
                    seed_id: _IdentityHypothesis(
                        track_id=seed_id,
                        score=emissions.get(seed_id, 0.0),
                        center=track.state[:2],
                        radius=track.state[2],
                    )
                }

        advanced: dict[int, _IdentityHypothesis] = {}
        for track_id, emission in emissions.items():
            track = self.tracks.get(track_id)
            if track is None or track.lost_frames > 4:
                continue
            center = track.state[:2]
            radius = track.state[2]
            best: Optional[_IdentityHypothesis] = None
            best_score = -float("inf")
            for hypothesis in previous.values():
                distance = math.hypot(
                    center[0] - hypothesis.center[0],
                    center[1] - hypothesis.center[1],
                )
                if hypothesis.track_id == track_id:
                    transition_penalty = min(
                        1.5, 0.15 * distance / max(radius, 1.0)
                    )
                    handoffs = hypothesis.handoffs
                else:
                    gate = min(
                        160.0,
                        max(
                            _IDENTITY_HANDOFF_GATE,
                            1.25 * max(radius, hypothesis.radius),
                        )
                        + 12.0 * hypothesis.missing,
                    )
                    if distance > gate:
                        continue
                    transition_penalty = (
                        _IDENTITY_HANDOFF_PENALTY
                        + _IDENTITY_HANDOFF_DISTANCE_WEIGHT
                        * distance / max(gate, 1.0)
                    )
                    handoffs = hypothesis.handoffs + 1
                candidate_score = (
                    _IDENTITY_PATH_DECAY * hypothesis.score
                    + emission
                    - transition_penalty
                )
                if candidate_score > best_score:
                    best_score = candidate_score
                    best = _IdentityHypothesis(
                        track_id=track_id,
                        score=float(candidate_score),
                        center=(float(center[0]), float(center[1])),
                        radius=float(radius),
                        age=hypothesis.age + 1,
                        missing=0,
                        handoffs=handoffs,
                    )
            if best is not None:
                advanced[track_id] = best

        # Keep coasting paths alive through short detector gaps. They may hand
        # off to a newly spawned nearby ID on a later frame.
        for track_id, hypothesis in previous.items():
            if track_id in advanced or hypothesis.missing >= _IDENTITY_MAX_MISSING:
                continue
            track = self.tracks.get(track_id)
            center = hypothesis.center if track is None else track.state[:2]
            radius = hypothesis.radius if track is None else track.state[2]
            advanced[track_id] = _IdentityHypothesis(
                track_id=track_id,
                score=(
                    _IDENTITY_PATH_DECAY * hypothesis.score
                    - _IDENTITY_MISSING_PENALTY
                ),
                center=(float(center[0]), float(center[1])),
                radius=float(radius),
                age=hypothesis.age + 1,
                missing=hypothesis.missing + 1,
                handoffs=hypothesis.handoffs,
            )

        if not advanced and scores:
            # The complete beam can disappear after a long occlusion. Re-seed
            # only then; normal frames never create a disconnected path.
            seed_id = max(scores, key=scores.get)
            track = self.tracks[seed_id]
            advanced[seed_id] = _IdentityHypothesis(
                track_id=seed_id,
                score=emissions.get(seed_id, 0.0),
                center=track.state[:2],
                radius=track.state[2],
            )

        ordered = sorted(
            advanced.values(), key=lambda hypothesis: hypothesis.score, reverse=True
        )[: self.hypothesis_count]
        # Beam pruning must never silently discard the currently published
        # path; doing so would force an immediate, potentially distant switch.
        current_before_prune = advanced.get(self.target_id)
        if (
            current_before_prune is not None
            and all(
                item.track_id != current_before_prune.track_id
                for item in ordered
            )
        ):
            if len(ordered) >= self.hypothesis_count:
                ordered[-1] = current_before_prune
            else:
                ordered.append(current_before_prune)
        self.target_hypotheses = {
            hypothesis.track_id: hypothesis for hypothesis in ordered
        }
        if not ordered:
            return False

        best = ordered[0]
        current = self.target_hypotheses.get(self.target_id)
        if self.target_id is None or current is None:
            self.identity_path_recommended_id = best.track_id
            if not commit:
                return False
            old_id = self.target_id
            self.target_id = best.track_id
            self.identity_path_candidate_id = None
            self.identity_path_candidate_streak = 0
            return old_id != self.target_id
        if best.track_id == self.target_id:
            self.identity_path_candidate_id = None
            self.identity_path_candidate_streak = 0
            if self.identity_switch_cooldown > 0:
                self.identity_switch_cooldown -= 1
            return False

        distance = math.hypot(
            best.center[0] - current.center[0],
            best.center[1] - current.center[1],
        )
        if self.identity_switch_cooldown > 0:
            self.identity_switch_cooldown -= 1
            self.identity_path_candidate_id = None
            self.identity_path_candidate_streak = 0
            return False
        required_margin = _IDENTITY_SWITCH_MARGIN + min(3.0, distance / 100.0)
        if best.score - current.score < required_margin:
            self.identity_path_candidate_id = None
            self.identity_path_candidate_streak = 0
            return False
        if self.identity_path_candidate_id == best.track_id:
            self.identity_path_candidate_streak += 1
        else:
            self.identity_path_candidate_id = best.track_id
            self.identity_path_candidate_streak = 1
        votes_needed = _IDENTITY_SWITCH_VOTES + (
            _IDENTITY_FAR_SWITCH_EXTRA_VOTES
            if distance > _IDENTITY_FAR_SWITCH_DISTANCE
            else 0
        )
        if self.identity_path_candidate_streak < votes_needed:
            return False

        self.identity_path_recommended_id = best.track_id
        if not commit:
            return False
        self.target_id = best.track_id
        self.identity_switch_cooldown = _IDENTITY_SWITCH_COOLDOWN
        self.identity_path_candidate_id = None
        self.identity_path_candidate_streak = 0
        self.challenger_evidence.clear()
        self.contamination_streak = 0
        self.contamination_candidate = None
        self.last_outlier_real_xy = best.center
        return True

    def _assign_roles_multi_hypothesis(
        self,
        scores: dict[int, float],
    ) -> bool:
        switched = self._advance_identity_hypotheses(scores)
        for track in self.tracks.values():
            if track.id == self.target_id:
                track.role = "real"
            elif track.lost_frames <= 4:
                track.role = "bg"
            else:
                track.role = "unknown"
        return switched

    def _identity_paths_allow_switch(
        self,
        current: _ShapeTrack,
        challenger: _ShapeTrack,
    ) -> bool:
        distance = math.hypot(
            challenger.state[0] - current.state[0],
            challenger.state[1] - current.state[1],
        )
        if distance <= _IDENTITY_FAR_SWITCH_DISTANCE:
            self.identity_unvalidated_candidate_id = None
            self.identity_unvalidated_candidate_streak = 0
            self.identity_unvalidated_center = None
            return True
        if self.identity_path_recommended_id == challenger.id:
            self.identity_unvalidated_candidate_id = None
            self.identity_unvalidated_candidate_streak = 0
            self.identity_unvalidated_center = None
            return True
        center = challenger.state[:2]
        self.identity_unvalidated_candidate_streak += 1
        self.identity_unvalidated_candidate_id = challenger.id
        self.identity_unvalidated_center = (
            float(center[0]),
            float(center[1]),
        )
        votes_needed = (
            _IDENTITY_EXTREME_RECOVERY_VOTES
            if distance >= _IDENTITY_EXTREME_RECOVERY_DISTANCE
            else _IDENTITY_UNVALIDATED_RECOVERY_VOTES
        )
        return self.identity_unvalidated_candidate_streak >= votes_needed

    def _constellation_views(self) -> list[ConstellationTrackView]:
        return [
            ConstellationTrackView(
                track_id=track.id,
                center=(track.state[0], track.state[1]),
                radius=track.state[2],
                role=track.role,
                orientation=track.orientation,
                lost_frames=track.lost_frames,
                translation_residual=track.translation_residual,
                rotation_residual=track.rotation_residual,
                velocity=track.predicted_velocity,
                real_score=track.real_score,
                rotation_score=track.rotation_score,
                direction_score=track.direction_score,
                speed_score=track.speed_score,
                rigidity_score=track.rigidity_score,
                motion_reliability=track.motion_reliability,
                visible_streak=track.visible_streak,
                angular_velocity=track.angular_velocity,
                appearance_distance=min(
                    2.0,
                    self._appearance_distance(
                        self.target_appearance_anchor,
                        track.appearance,
                    ),
                ),
                association_quality=(
                    float(
                        np.median(
                            np.asarray(
                                track.association_quality_history[-3:],
                                dtype=np.float32,
                            )
                        )
                    )
                    if track.association_quality_history
                    else 0.0
                ),
                yolo_confidence=(
                    float(track.last_detection.yolo_confidence)
                    if track.last_detection is not None
                    else 0.0
                ),
                predicted_only=track.predicted_only,
                bg_certified=(
                    bool(track.last_detection.bg_certified)
                    if track.last_detection is not None
                    else False
                ),
                ranker_score=track.ranker_score,
                flow_confidence=track.flow_confidence,
                flow_forward_backward_error=(
                    track.flow_forward_backward_error
                ),
                flow_inlier_ratio=track.flow_inlier_ratio,
                flow_coverage=track.flow_coverage,
                flow_local_fit_error=track.flow_local_fit_error,
                flow_group_residual=track.flow_group_residual,
                flow_group_confidence=track.flow_group_confidence,
                flow_association_used=track.flow_association_used,
                flow_coasted=track.flow_coasted,
            )
            for track in self.tracks.values()
            if track.lost_frames <= self.max_lost_frames
        ]

    def _ranker_switch_candidate(
        self,
        current_track: Optional[_ShapeTrack],
        gray: Optional[np.ndarray] = None,
    ) -> Optional[_ShapeTrack]:
        self.ranker_margin = 0.0
        self.ranker_top_id = None
        self.ranker_current_rank = 0
        self.ranker_current_score = 0.0
        self.ranker_switch_delta = 0.0
        self.ranker_evidence = 0.0
        self.motion_corroborated = False
        self.switch_event = None
        self.switch_event_probability = 0.0
        self.switch_event_approved = False
        if self.identity_ranker is None or self.frame_size is None:
            self.state_aware_legacy_candidate_id = None
            self.state_aware_legacy_candidate_streak = 0
            return None
        width, height = self.frame_size
        views = self._constellation_views()
        ranker_scores = self.identity_ranker.update(
            views,
            target_id=self.target_id,
            frame_width=width,
            frame_height=height,
            gray_frame=gray,
        )
        for track in self.tracks.values():
            track.ranker_score = float(ranker_scores.get(track.id, 0.0))
        if len(ranker_scores) < 2:
            self.ranker_candidate_id = None
            self.ranker_candidate_streak = 0
            self.state_aware_legacy_candidate_id = None
            self.state_aware_legacy_candidate_streak = 0
            return None
        ranked = sorted(
            ranker_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        best_id, best_score = ranked[0]
        self.ranker_top_id = int(best_id)
        self.ranker_margin = float(best_score - ranked[1][1])
        if self.target_id in ranker_scores:
            self.ranker_current_score = float(ranker_scores[self.target_id])
            self.ranker_current_rank = 1 + next(
                index
                for index, (track_id, _score) in enumerate(ranked)
                if track_id == self.target_id
            )
            self.ranker_switch_delta = float(
                best_score - ranker_scores[self.target_id]
            )
        else:
            self.ranker_current_rank = len(ranked) + 1
            self.ranker_switch_delta = float(max(0.0, self.ranker_margin))

        if best_id == self.target_id:
            self.ranker_disagreement_streak = 0
            self.identity_mismatch_streak = 0
            self.identity_match_streak = getattr(
                self, "identity_match_streak", 0
            ) + 1
        else:
            self.ranker_disagreement_streak = getattr(
                self, "ranker_disagreement_streak", 0
            ) + 1
            self.identity_mismatch_streak = getattr(
                self, "identity_mismatch_streak", 0
            ) + 1
            self.identity_match_streak = 0

        motion_ranker = getattr(self, "motion_corroboration_ranker", None)
        if motion_ranker is not None:
            motion_scores = motion_ranker.update(
                views,
                target_id=self.target_id,
                frame_width=width,
                frame_height=height,
                gray_frame=gray,
            )
            if motion_scores:
                motion_best_id = max(motion_scores, key=motion_scores.get)
                motion_current = motion_scores.get(self.target_id, -float("inf"))
                self.motion_corroborated = bool(
                    motion_best_id == best_id
                    and best_id != self.target_id
                    and motion_scores[motion_best_id] - motion_current >= 0.25
                )

        if not getattr(self, "state_aware_ranker_enabled", False):
            if (
                best_id == self.target_id
                or self.ranker_margin < self.identity_ranker_min_margin
            ):
                self.ranker_candidate_id = None
                self.ranker_candidate_streak = 0
                return None
            candidate = self.tracks.get(best_id)
            qualifies = (
                candidate is not None
                and candidate.lost_frames == 0
                and not candidate.predicted_only
                and candidate.visible_streak >= 5
                and candidate.motion_reliability >= 0.10
                and not self._leaving_frame(candidate)
                and (
                    current_track is None
                    or not self.multi_hypothesis_identity_enabled
                    or self._identity_paths_allow_switch(current_track, candidate)
                )
            )
            if not qualifies:
                self.ranker_candidate_id = None
                self.ranker_candidate_streak = 0
                return None
            if self.ranker_candidate_id == best_id:
                self.ranker_candidate_streak += 1
            else:
                self.ranker_candidate_id = best_id
                self.ranker_candidate_streak = 1
            votes_needed = (
                _RANKER_RECOVERY_VOTES
                if current_track is None
                or current_track.lost_frames > _REAL_COAST_GRACE
                or current_track.predicted_only
                else _RANKER_SWITCH_VOTES
            )
            return (
                candidate
                if self.ranker_candidate_streak >= votes_needed
                else None
            )

        # Experimental state-aware mode. Evidence decays instead of resetting
        # on a single ambiguous frame, and identity-path disagreement is a soft
        # penalty except for geometrically impossible candidates.
        self.ranker_evidence_by_id = {
            track_id: evidence * 0.82
            for track_id, evidence in self.ranker_evidence_by_id.items()
            if track_id in ranker_scores and evidence * 0.82 >= 0.05
        }
        if best_id == self.target_id:
            self.ranker_candidate_id = None
            self.ranker_candidate_streak = 0
            self.state_aware_legacy_candidate_id = None
            self.state_aware_legacy_candidate_streak = 0
            return None
        candidate = self.tracks.get(best_id)
        basic_qualifies = (
            candidate is not None
            and candidate.lost_frames == 0
            and not candidate.predicted_only
            and candidate.visible_streak >= 3
            and candidate.motion_reliability >= 0.05
            and not self._leaving_frame(candidate)
        )
        event: dict[str, object] = {
            "current_id": self.target_id,
            "challenger_id": int(best_id),
            "candidate_count": len(ranker_scores),
            "ranker_switch_delta": float(self.ranker_switch_delta),
            "ranker_top_margin": float(self.ranker_margin),
            "ranker_current_score": float(self.ranker_current_score),
            "ranker_current_rank": int(self.ranker_current_rank),
            "ranker_disagreement_streak": int(self.ranker_disagreement_streak),
            "ranker_evidence_before": float(
                self.ranker_evidence_by_id.get(best_id, 0.0)
            ),
            "motion_corroborated": int(self.motion_corroborated),
            "basic_qualifies": int(basic_qualifies),
        }

        def _add_track(prefix: str, track: Optional[_ShapeTrack]) -> None:
            if track is None:
                event.update(
                    {
                        f"{prefix}_present": 0,
                        f"{prefix}_x": None,
                        f"{prefix}_y": None,
                    }
                )
                return
            diagonal = max(1.0, math.hypot(*self.frame_size))
            association = (
                float(np.median(track.association_quality_history[-3:]))
                if track.association_quality_history
                else 0.0
            )
            yolo_confidence = (
                float(track.last_detection.yolo_confidence)
                if track.last_detection is not None
                else 0.0
            )
            appearance = min(
                2.0,
                self._appearance_distance(
                    self.target_appearance_anchor, track.appearance
                ),
            )
            event.update(
                {
                    f"{prefix}_present": 1,
                    f"{prefix}_x": float(track.state[0]),
                    f"{prefix}_y": float(track.state[1]),
                    f"{prefix}_radius_norm": float(track.state[2])
                    / max(1.0, min(self.frame_size)),
                    f"{prefix}_speed_norm": float(
                        np.linalg.norm(track.predicted_velocity) / diagonal
                    ),
                    f"{prefix}_translation_residual_norm": float(
                        track.translation_residual / diagonal
                    ),
                    f"{prefix}_rotation_score_norm": float(
                        track.rotation_score / 180.0
                    ),
                    f"{prefix}_rotation_delta_abs_norm": float(
                        min(3.0, abs(math.degrees(track.last_rotation_delta)) / 22.0)
                    ),
                    f"{prefix}_peer_rotation_residual_norm": float(
                        track.peer_rotation_residual / 180.0
                    ),
                    f"{prefix}_rotation_confidence": float(
                        track.rotation_confidence
                    ),
                    f"{prefix}_rotation_valid": int(track.rotation_valid),
                    f"{prefix}_direction_score_norm": float(
                        track.direction_score / 180.0
                    ),
                    f"{prefix}_speed_score_norm": float(
                        track.speed_score / diagonal
                    ),
                    f"{prefix}_rigidity_score_norm": float(
                        track.rigidity_score / diagonal
                    ),
                    f"{prefix}_motion_reliability": float(
                        track.motion_reliability
                    ),
                    f"{prefix}_visible_streak_norm": float(
                        np.clip(track.visible_streak / 60.0, 0.0, 1.0)
                    ),
                    f"{prefix}_lost_frames_norm": float(
                        np.clip(track.lost_frames / 10.0, 0.0, 1.0)
                    ),
                    f"{prefix}_predicted_only": int(track.predicted_only),
                    f"{prefix}_appearance_distance": float(appearance),
                    f"{prefix}_association_quality": float(association),
                    f"{prefix}_yolo_confidence": float(yolo_confidence),
                    f"{prefix}_bg_certified": int(
                        bool(
                            track.last_detection is not None
                            and track.last_detection.bg_certified
                        )
                    ),
                    f"{prefix}_position_uncertainty_norm": float(
                        track.position_uncertainty / diagonal
                    ),
                    f"{prefix}_looks_background": int(
                        self._track_looks_like_background(track)
                    ),
                    f"{prefix}_preflow_confidence": float(
                        track.flow_confidence
                    ),
                    f"{prefix}_preflow_forward_backward_error_norm": float(
                        np.clip(track.flow_forward_backward_error / 3.0, 0.0, 3.0)
                    ),
                    f"{prefix}_preflow_inlier_ratio": float(
                        track.flow_inlier_ratio
                    ),
                    f"{prefix}_preflow_coverage": float(track.flow_coverage),
                    f"{prefix}_preflow_local_fit_error_norm": float(
                        np.clip(track.flow_local_fit_error / 5.0, 0.0, 3.0)
                    ),
                    f"{prefix}_preflow_group_residual_norm": float(
                        track.flow_group_residual / diagonal
                    ),
                    f"{prefix}_preflow_group_confidence": float(
                        track.flow_group_confidence
                    ),
                    f"{prefix}_preflow_association_used": int(
                        track.flow_association_used
                    ),
                    f"{prefix}_flow_coasted": int(track.flow_coasted),
                }
            )

        _add_track("current", current_track)
        _add_track("challenger", candidate)
        switch_motion = (
            {}
            if self.switch_motion_history is None
            else self.switch_motion_history.features(
                (
                    track.id
                    for track in (current_track, candidate)
                    if track is not None
                ),
                include_optical=self.switch_optical_features_enabled,
                include_multilag=self.switch_multilag_features_enabled,
            )
        )
        for prefix, track in (
            ("current", current_track),
            ("challenger", candidate),
        ):
            values = (
                {}
                if track is None
                else switch_motion.get(track.id, {})
            )
            for name in SWITCH_MOTION_FEATURE_NAMES:
                event[f"{prefix}_{name}"] = float(values.get(name, 0.0))
        if not basic_qualifies:
            event["path_support"] = 0.0
            event["ranker_evidence_after"] = float(
                event["ranker_evidence_before"]
            )
            event["threshold"] = None
            event["proposed"] = 0
            self.switch_event = event
            self.ranker_candidate_id = None
            self.ranker_candidate_streak = 0
            self.state_aware_legacy_candidate_id = None
            self.state_aware_legacy_candidate_streak = 0
            return None

        path_support = 1.0
        if current_track is not None and self.multi_hypothesis_identity_enabled:
            path_support = (
                1.0
                if self._identity_paths_allow_switch(current_track, candidate)
                else 0.55
            )
        normalized_delta = float(np.clip(self.ranker_switch_delta / 2.0, 0.0, 1.0))
        support = (0.35 + 0.65 * normalized_delta) * path_support
        if self.ranker_margin >= self.identity_ranker_min_margin:
            support += 0.25
        if self.motion_corroborated:
            support += 0.35
        if self.ranker_current_rank >= 3:
            support += 0.15
        evidence = self.ranker_evidence_by_id.get(best_id, 0.0) + support
        self.ranker_evidence_by_id[best_id] = evidence
        self.ranker_candidate_id = best_id
        self.ranker_evidence = float(evidence)
        self.ranker_candidate_streak = int(round(evidence))

        recovery = bool(
            current_track is None
            or current_track.lost_frames > _STATE_AWARE_RECOVERY_COAST
        )
        recent_rollback = bool(
            self.switch_event_model is not None
            and self.recent_ranker_switch_previous_id == best_id
            and self.recent_ranker_switch_age
            <= _SWITCH_EVENT_ROLLBACK_WINDOW
        )
        current_suspicious = bool(
            recovery
            or self.ranker_disagreement_streak >= 2
            or self.ranker_current_rank >= 3
            or (
                current_track is not None
                and self._track_looks_like_background(current_track)
            )
        )
        # A short, position-confident coast is not by itself evidence that the
        # identity is wrong. Low-margin challengers must wait through that
        # grace period; visible-but-suspicious identities can still recover
        # quickly. Motion corroboration already contributes to ``support`` and
        # therefore is not also applied as a second threshold discount.
        short_coast = bool(
            current_track is not None
            and current_track.predicted_only
            and not recovery
        )
        threshold = (
            1.8
            if recovery
            else 4.2
            if short_coast
            else 2.6
            if current_suspicious
            else 4.2
        )
        if recent_rollback:
            threshold = min(threshold, _SWITCH_EVENT_ROLLBACK_EVIDENCE)
        legacy_qualifies = bool(
            self.ranker_margin >= self.identity_ranker_min_margin
            and candidate.visible_streak >= 5
            and candidate.motion_reliability >= 0.10
            and (
                current_track is None
                or not self.multi_hypothesis_identity_enabled
                or self._identity_paths_allow_switch(current_track, candidate)
            )
        )
        if legacy_qualifies:
            if self.state_aware_legacy_candidate_id == best_id:
                self.state_aware_legacy_candidate_streak += 1
            else:
                self.state_aware_legacy_candidate_id = int(best_id)
                self.state_aware_legacy_candidate_streak = 1
        else:
            self.state_aware_legacy_candidate_id = None
            self.state_aware_legacy_candidate_streak = 0
        legacy_votes_needed = (
            _RANKER_RECOVERY_VOTES
            if current_track is None
            or current_track.lost_frames > _REAL_COAST_GRACE
            or current_track.predicted_only
            else _RANKER_SWITCH_VOTES
        )
        legacy_approved = bool(
            legacy_qualifies
            and self.state_aware_legacy_candidate_streak >= legacy_votes_needed
        )
        event.update(
            {
                "path_support": float(path_support),
                "ranker_evidence_after": float(evidence),
                "recovery": int(recovery),
                "short_coast": int(short_coast),
                "current_suspicious": int(current_suspicious),
                "threshold": float(threshold),
                "proposed": int(evidence >= threshold),
                "legacy_qualifies": int(legacy_qualifies),
                "legacy_streak": int(self.state_aware_legacy_candidate_streak),
                "legacy_approved": int(legacy_approved),
                "recent_rollback": int(recent_rollback),
            }
        )
        if self.switch_event_model is not None:
            self.switch_event_probability = float(
                self.switch_event_model.predict_probability(event)
            )
            model_approved = bool(
                evidence >= threshold
                and self.switch_event_probability
                >= self.switch_event_model.min_probability
                and (
                    recovery
                    or self.ranker_switch_delta
                    >= _SWITCH_EVENT_VISIBLE_MIN_DELTA
                )
            )
            # The learned gate can approve an earlier State-aware recovery,
            # but vetoing it must not remove a switch that the conservative
            # production rule would eventually have made.
            self.switch_event_approved = bool(model_approved or legacy_approved)
            event["switch_event_probability"] = self.switch_event_probability
            event["switch_event_model_approved"] = int(model_approved)
            event["switch_event_approved"] = int(self.switch_event_approved)
        else:
            self.switch_event_approved = bool(evidence >= threshold)
        self.switch_event = event
        if self.switch_event_approved:
            self.state_aware_legacy_candidate_id = None
            self.state_aware_legacy_candidate_streak = 0
            return candidate
        return None

    def _update_identity_confidence(
        self,
        target: Optional[_ShapeTrack],
        *,
        position_actionable: bool,
    ) -> None:
        """Update the experimental identity safety state.

        This is deliberately independent from Kalman position confidence.  A
        visible, stable BG track can be position-confident while its REAL
        identity is unsafe.
        """
        previous_safe = self.identity_safe
        if target is None:
            self.identity_confidence = 0.0
            self.identity_state = "uninitialized" if not self.white_seen else "lost"
            self.identity_safe = False
            self.hold_reason = "no_target"
            return
        if self.white_active:
            self.identity_confidence = 1.0
            self.identity_state = "seed"
            self.identity_safe = bool(position_actionable)
            self.hold_reason = None if self.identity_safe else "position_uncertain"
            return
        if not self.white_seen:
            self.identity_confidence = 0.0
            self.identity_state = "uninitialized"
            self.identity_safe = False
            self.hold_reason = "no_white_seed"
            return

        if self.identity_switched:
            # One observation frame is enough to reject an immediately bad
            # hand-off without suppressing several known-good pointer frames.
            self.identity_probation_frames = max(self.identity_probation_frames, 1)
        if target.predicted_only:
            self.identity_confidence = float(
                np.clip(0.55 * math.exp(-0.25 * target.lost_frames), 0.0, 1.0)
            )
            # Short coast intervals are common even on stable clips. Preserve
            # the previously established identity for a bounded interval; the
            # existing Kalman uncertainty gate remains independently active.
            self.identity_safe = bool(
                position_actionable
                and previous_safe
                and target.lost_frames <= _IDENTITY_SAFETY_COAST_GRACE
            )
            self.identity_state = "coasting" if self.identity_safe else "lost"
            self.hold_reason = None if self.identity_safe else "target_predicted"
            return

        if self.identity_probation_frames > 0:
            self.identity_probation_frames -= 1
            self.identity_confidence = 0.55
            self.identity_state = "probation"
            self.identity_safe = False
            self.hold_reason = "switch_probation"
            return

        if self.ranker_top_id is None:
            self.identity_ranker_unavailable_frames += 1
            self.identity_confidence = 0.45
            self.identity_state = "suspect"
            self.identity_safe = bool(
                position_actionable
                and previous_safe
                and self.identity_ranker_unavailable_frames <= 3
            )
            self.hold_reason = (
                None if self.identity_safe else "ranker_unavailable"
            )
            return
        self.identity_ranker_unavailable_frames = 0

        if self.ranker_top_id == target.id:
            gap_term = 0.16 * math.tanh(max(0.0, self.ranker_margin) / 2.0)
            streak_term = 0.14 * min(1.0, self.identity_match_streak / 4.0)
            self.identity_confidence = float(
                np.clip(0.58 + gap_term + streak_term, 0.0, 1.0)
            )
            self.identity_state = "locked"
            enter_safe = self.identity_match_streak >= 1
            keep_safe = previous_safe and self.identity_confidence >= 0.55
            self.identity_safe = bool(position_actionable and (enter_safe or keep_safe))
            self.hold_reason = None if self.identity_safe else "identity_warming"
            return

        self.identity_confidence = float(
            np.clip(
                0.70
                - 0.12 * min(self.identity_mismatch_streak, 4)
                - 0.08 * math.tanh(max(0.0, self.ranker_switch_delta) / 2.0),
                0.05,
                0.60,
            )
        )
        self.identity_state = "suspect"
        self.identity_safe = bool(
            position_actionable
            and previous_safe
            and self.identity_mismatch_streak < 2
            and self.identity_confidence >= 0.45
        )
        self.hold_reason = None if self.identity_safe else "ranker_disagrees"

    def _assign_roles(self, gray: Optional[np.ndarray] = None) -> bool:
        """Label REAL from four-cue scores after white fade."""
        scores = self._score_real_candidates()
        if self.multi_hypothesis_identity_enabled:
            self._advance_identity_hypotheses(scores, commit=False)
        switched = False
        current_track = (
            self.tracks.get(self.target_id) if self.target_id is not None else None
        )
        if self.stale_recovery_appearance_protect_frames > 0:
            self.stale_recovery_appearance_protect_frames -= 1

        def _apply_roles() -> None:
            for track in self.tracks.values():
                if track.id == self.target_id:
                    track.role = "real"
                elif track.lost_frames <= 4:
                    track.role = "bg"
                else:
                    track.role = "unknown"

        if not scores:
            self.contamination_streak = 0
            self.contamination_candidate = None
            self.challenger_evidence.clear()
            self.shadow_candidate_id = None
            self.shadow_candidate_votes.clear()
            self.identity_unvalidated_candidate_id = None
            self.identity_unvalidated_candidate_streak = 0
            self.identity_unvalidated_center = None
            _apply_roles()
            return False

        viable = {
            track_id: score
            for track_id, score in scores.items()
            if track_id in self.tracks and not self._leaving_frame(self.tracks[track_id])
        }
        ranking = viable or scores
        best_id = max(ranking, key=ranking.get)
        if best_id == self.target_id:
            self.identity_unvalidated_candidate_id = None
            self.identity_unvalidated_candidate_streak = 0
            self.identity_unvalidated_center = None
        if self.target_id is None:
            self.target_id = best_id
            switched = True
            self.contamination_streak = 0
            self.contamination_candidate = None
            self.challenger_evidence.clear()
            self.shadow_candidate_id = None
            self.shadow_candidate_votes.clear()
            _apply_roles()
            return switched

        ranker_candidate = self._ranker_switch_candidate(current_track, gray)
        if ranker_candidate is not None:
            recovery_protects_current = False
            if (
                self.stale_recovery_appearance_protect_frames > 0
                and current_track is not None
                and current_track.lost_frames == 0
                and not current_track.predicted_only
            ):
                current_appearance_error = self._appearance_distance(
                    self.target_appearance_anchor,
                    current_track.appearance,
                )
                challenger_appearance_error = self._appearance_distance(
                    self.target_appearance_anchor,
                    ranker_candidate.appearance,
                )
                recovery_protects_current = (
                    current_appearance_error <= 0.07
                    and challenger_appearance_error
                    >= current_appearance_error + 0.015
                )
            if not recovery_protects_current:
                previous_target_id = self.target_id
                is_recent_rollback = bool(
                    ranker_candidate.id
                    == self.recent_ranker_switch_previous_id
                    and self.recent_ranker_switch_age
                    <= _SWITCH_EVENT_ROLLBACK_WINDOW
                )
                self.target_id = ranker_candidate.id
                self.last_outlier_real_xy = ranker_candidate.state[:2]
                self.ranker_candidate_id = None
                self.ranker_candidate_streak = 0
                self.ranker_switched = True
                self.identity_switch_cooldown = _IDENTITY_SWITCH_COOLDOWN
                self.contamination_streak = 0
                self.contamination_candidate = None
                self.challenger_evidence.clear()
                self.shadow_candidate_id = None
                self.shadow_candidate_votes.clear()
                if self.switch_event_model is None:
                    self.recent_ranker_switch_previous_id = None
                    self.recent_ranker_switch_age = 0
                elif is_recent_rollback:
                    self.recent_ranker_switch_previous_id = None
                    self.recent_ranker_switch_age = 0
                else:
                    self.recent_ranker_switch_previous_id = previous_target_id
                    self.recent_ranker_switch_age = 0
                _apply_roles()
                return True
            self.ranker_candidate_id = None
            self.ranker_candidate_streak = 0

        healthy = (
            current_track is not None
            and current_track.lost_frames <= _REAL_COAST_GRACE
            and self.target_id in scores
        )
        if not healthy:
            # Coast on the same id; only adopt best when the old track is gone.
            if current_track is None and best_id != self.target_id:
                self.target_id = best_id
                switched = True
            elif (
                self.stale_coast_recovery_enabled
                and current_track is not None
                and current_track.lost_frames > _REAL_COAST_GRACE
            ):
                candidate = self.tracks.get(best_id)
                distance = (
                    float("inf")
                    if candidate is None
                    else math.hypot(
                        candidate.state[0] - current_track.state[0],
                        candidate.state[1] - current_track.state[1],
                    )
                )
                appearance_error = (
                    float("inf")
                    if candidate is None
                    else self._appearance_distance(
                        self.target_appearance_anchor,
                        candidate.appearance,
                    )
                )
                qualifies = (
                    candidate is not None
                    and candidate.lost_frames == 0
                    and not candidate.predicted_only
                    and candidate.visible_streak >= 5
                    and candidate.motion_reliability >= 0.15
                    and not self._leaving_frame(candidate)
                    and distance
                    <= 140.0 + 8.0 * min(current_track.lost_frames, 6)
                    and appearance_error
                    <= _STALE_RECOVERY_MAX_APPEARANCE_ERROR
                )
                if qualifies:
                    if self.stale_recovery_candidate_id == best_id:
                        self.stale_recovery_streak += 1
                    else:
                        self.stale_recovery_candidate_id = best_id
                        self.stale_recovery_streak = 1
                    if self.stale_recovery_streak >= 3:
                        self.target_id = best_id
                        self.last_outlier_real_xy = candidate.state[:2]
                        self.stale_recovery_committed = True
                        self.stale_recovery_hold_frames = (
                            _STALE_RECOVERY_HOLD_FRAMES
                        )
                        self.stale_recovery_appearance_protect_frames = (
                            _STALE_RECOVERY_APPEARANCE_PROTECT_FRAMES
                        )
                        switched = True
                else:
                    self.stale_recovery_candidate_id = None
                    self.stale_recovery_streak = 0
            self.contamination_streak = 0
            self.contamination_candidate = None
            self.challenger_evidence.clear()
            self.shadow_candidate_id = None
            self.shadow_candidate_votes.clear()
            if switched:
                self.stale_recovery_candidate_id = None
                self.stale_recovery_streak = 0
            _apply_roles()
            return switched

        if self.stale_recovery_hold_frames > 0:
            self.stale_recovery_hold_frames -= 1
            self.contamination_streak = 0
            self.contamination_candidate = None
            self.challenger_evidence.clear()
            self.identity_unvalidated_candidate_id = None
            self.identity_unvalidated_candidate_streak = 0
            self.identity_unvalidated_center = None
            _apply_roles()
            return False

        shadow_margin = scores.get(best_id, -1e9) - scores.get(
            self.target_id, -1e9
        )
        if (
            self.shadow_switch_enabled
            and best_id != self.target_id
            and shadow_margin < 45.0
            and self.tracks.get(best_id) is not None
            and self.tracks[best_id].visible_streak >= 30
        ):
            challenger = self.tracks.get(best_id)
            margin = shadow_margin
            identity_continuity_support = (
                challenger is not None
                and challenger.visible_streak >= 3
                and challenger.motion_reliability >= 0.05
                and challenger.appearance_score
                >= current_track.appearance_score - 0.75
            )
            motion_layout_support = (
                challenger is not None
                and (
                    challenger.rotation_score >= current_track.rotation_score + 1.5
                    or challenger.direction_score
                    >= current_track.direction_score + 12.0
                    or challenger.speed_score
                    >= current_track.speed_score + 1.5
                    or challenger.rigidity_score
                    >= current_track.rigidity_score + 0.8
                    or challenger.translation_residual
                    >= current_track.translation_residual + 4.0
                )
            )
            qualifies = (
                best_id != self.target_id
                and challenger is not None
                and challenger.lost_frames == 0
                and not challenger.predicted_only
                and not self._leaving_frame(challenger)
                and (
                    not self.multi_hypothesis_identity_enabled
                    or self._identity_paths_allow_switch(
                        current_track, challenger
                    )
                )
                and margin >= (
                    _REAL_SWITCH_MARGIN
                    + (
                        8.0
                        if current_track.lost_frames == 0
                        and not current_track.predicted_only
                        else 0.0
                    )
                )
                and identity_continuity_support
                and motion_layout_support
            )
            if best_id != self.shadow_candidate_id:
                self.shadow_candidate_id = best_id if qualifies else None
                self.shadow_candidate_votes = [qualifies] if qualifies else []
            elif self.shadow_candidate_id is not None:
                self.shadow_candidate_votes.append(qualifies)
                self.shadow_candidate_votes = self.shadow_candidate_votes[
                    -_SHADOW_SWITCH_WINDOW:
                ]
            if (
                self.shadow_candidate_id is not None
                and len(self.shadow_candidate_votes) >= _SHADOW_SWITCH_VOTES
                and sum(self.shadow_candidate_votes) >= _SHADOW_SWITCH_VOTES
            ):
                committed = self.tracks[self.shadow_candidate_id]
                self.target_id = committed.id
                self.last_outlier_real_xy = committed.state[:2]
                self.shadow_candidate_id = None
                self.shadow_candidate_votes.clear()
                self.challenger_evidence.clear()
                switched = True
            elif current_track.lost_frames == 0:
                self.last_outlier_real_xy = current_track.state[:2]
            self.contamination_streak = 0
            self.contamination_candidate = None
            _apply_roles()
            return switched

        if self.shadow_switch_enabled:
            self.shadow_candidate_id = None
            self.shadow_candidate_votes.clear()

        self.challenger_evidence = {
            track_id: evidence * _CHALLENGER_EVIDENCE_DECAY
            for track_id, evidence in self.challenger_evidence.items()
            if track_id in scores
            and evidence * _CHALLENGER_EVIDENCE_DECAY >= 0.1
        }
        challenger = self.tracks.get(best_id)
        recovery_protects_current = False
        if (
            self.stale_recovery_appearance_protect_frames > 0
            and current_track is not None
            and challenger is not None
            and current_track.lost_frames == 0
            and not current_track.predicted_only
        ):
            current_appearance_error = self._appearance_distance(
                self.target_appearance_anchor,
                current_track.appearance,
            )
            challenger_appearance_error = self._appearance_distance(
                self.target_appearance_anchor,
                challenger.appearance,
            )
            recovery_protects_current = (
                current_appearance_error <= 0.07
                and challenger_appearance_error
                >= current_appearance_error + 0.015
            )
        challenger_ok = (
            challenger is not None
            and challenger.lost_frames == 0
            and not challenger.predicted_only
            and best_id in scores
            and not recovery_protects_current
            # A shape already sliding off the panel cannot be the REAL one the
            # cursor has to follow, and would strand the label once it exits.
            and not self._leaving_frame(challenger)
            and (
                not self.multi_hypothesis_identity_enabled
                or self._identity_paths_allow_switch(
                    current_track, challenger
                )
            )
        )
        if best_id != self.target_id and challenger_ok:
            margin = scores[best_id] - scores.get(self.target_id, -1e9)
            # Current matched REAL needs a clearer four-cue lead to flip.
            needed = _REAL_SWITCH_MARGIN
            if current_track.lost_frames == 0 and not current_track.predicted_only:
                needed += 8.0
            if margin >= needed:
                if self.contamination_candidate == best_id:
                    self.contamination_streak += 1
                else:
                    self.contamination_candidate = best_id
                    self.contamination_streak = 1
                evidence = self.challenger_evidence.get(best_id, 0.0) + 1.0
                self.challenger_evidence[best_id] = evidence
                current_looks_like_bg = self._track_looks_like_background(
                    current_track
                )
                evidence_needed = (
                    _WEAK_REAL_EVIDENCE_NEEDED
                    if current_looks_like_bg
                    else _CHALLENGER_EVIDENCE_NEEDED
                )
                if (
                    self.contamination_streak >= 3
                    or evidence >= evidence_needed
                ):
                    self.target_id = best_id
                    switched = True
                    self.contamination_streak = 0
                    self.contamination_candidate = None
                    self.challenger_evidence.clear()
                    self.last_outlier_real_xy = challenger.state[:2]
            else:
                self.contamination_streak = 0
                self.contamination_candidate = None
        else:
            self.contamination_streak = 0
            self.contamination_candidate = None
            if current_track.lost_frames == 0:
                self.last_outlier_real_xy = current_track.state[:2]

        _apply_roles()
        return switched

    def _seed_real_from_bright(self, bright: ShapeDetection) -> None:
        if not self.tracks:
            return
        # Once seeded, freeze REAL for the whole white phase.  Bright flicker /
        # UI chrome must not steal the label.
        if self.target_id is not None and self.target_id in self.tracks:
            current = self.tracks[self.target_id]
            if (
                self.target_appearance_anchor is None
                and current.appearance is not None
            ):
                self.target_appearance_anchor = current.appearance.copy()
            # White phase: keep refreshing the geometric anchor at the seeded REAL.
            self.last_outlier_real_xy = current.state[:2]
            for track in self.tracks.values():
                track.role = "real" if track.id == self.target_id else (
                    "bg" if track.lost_frames <= 4 else "unknown"
                )
            return

        # Opening target sits near panel center; prefer that when distances tie.
        centers = [track.state[:2] for track in self.tracks.values()]
        panel_center = (
            float(np.mean([c[0] for c in centers])),
            float(np.mean([c[1] for c in centers])),
        )

        def _seed_key(track: _ShapeTrack) -> tuple[float, float]:
            bright_dist = math.hypot(
                track.state[0] - bright.center[0],
                track.state[1] - bright.center[1],
            )
            center_dist = math.hypot(
                track.state[0] - panel_center[0],
                track.state[1] - panel_center[1],
            )
            return (bright_dist, center_dist)

        nearest = min(self.tracks.values(), key=_seed_key)
        if math.hypot(
            nearest.state[0] - bright.center[0],
            nearest.state[1] - bright.center[1],
        ) <= max(40.0, bright.radius * 0.9):
            self.target_id = nearest.id
            if (
                self.target_appearance_anchor is None
                and nearest.appearance is not None
            ):
                self.target_appearance_anchor = nearest.appearance.copy()
            for track in self.tracks.values():
                track.role = "real" if track.id == nearest.id else (
                    "bg" if track.lost_frames <= 4 else "unknown"
                )

    def update(
        self,
        frame_bgr: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> LieDetectorTrackingResult:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("frame_bgr must be a non-empty BGR image")
        self.frame_size = (int(frame_bgr.shape[1]), int(frame_bgr.shape[0]))
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        cleaned, _ = self._remove_cursor(frame_bgr)
        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(cleaned, cv2.COLOR_BGR2HSV)
        self.identity_switched = False
        self.stale_recovery_committed = False
        self.ranker_switched = False
        self.flow_observation_count = 0
        self.flow_association_count = 0
        self.flow_coast_count = 0
        self.switch_event = None
        self.switch_event_probability = 0.0
        self.switch_event_approved = False
        if self.recent_ranker_switch_previous_id is not None:
            self.recent_ranker_switch_age += 1
            if (
                self.recent_ranker_switch_age
                > _SWITCH_EVENT_ROLLBACK_WINDOW
            ):
                self.recent_ranker_switch_previous_id = None
                self.recent_ranker_switch_age = 0

        bright = self._initial_bright_shape(gray, hsv)
        bright_valid = (
            bright is not None and bright.score * 255.0 >= self.acquire_brightness
        )
        allow_bright_seed = (
            bright_valid
            and bright is not None
            and not self.white_phase_completed
        )
        if allow_bright_seed and bright is not None:
            self.white_seen = True
            self.white_active = True
            self.white_absent_streak = 0
            if self.target_kind is None:
                aspect = bright.bbox[2] / max(1.0, bright.bbox[3])
                self.target_kind = (
                    "circle"
                    if bright.circularity >= 0.72 and 0.75 <= aspect <= 1.33
                    else "contour"
                )
                if bright.contour is not None:
                    self.target_contour = bright.contour.copy()
                self.target_radius = float(bright.radius)
        else:
            self.white_active = False
            if self.white_seen and not self.white_phase_completed:
                self.white_absent_streak += 1
                # Opening highlights can flicker for several frames. Require a
                # sustained absence before later bright UI is treated as DONE
                # chrome rather than another seed observation.
                if self.white_absent_streak >= 30:
                    self.white_phase_completed = True

        detections: list[ShapeDetection] = []
        if self.classical_candidates:
            detections = (
                self._circle_candidates(gray)
                if self.target_kind in (None, "circle")
                else self._contour_candidates(gray, self.target_contour)
            )
        if self.candidate_detector is not None:
            learned = self.candidate_detector.detect(cleaned, self.target_radius)
            if self.classical_candidates:
                detections = self._fuse_learned_candidates(detections, learned)
            else:
                detections = list(learned)
            yolo_radii = [
                float(item.observed_radius or item.radius)
                for item in learned
                if (item.observed_radius or item.radius) is not None
            ]
            if yolo_radii:
                median_radius = float(np.median(np.asarray(yolo_radii)))
                if self.target_radius is None:
                    self.target_radius = median_radius
                else:
                    self.target_radius = 0.65 * float(self.target_radius) + (
                        0.35 * median_radius
                    )
        if allow_bright_seed and bright is not None:
            nearby = any(
                math.hypot(
                    detection.center[0] - bright.center[0],
                    detection.center[1] - bright.center[1],
                )
                <= max(20.0, 0.55 * float(bright.radius))
                for detection in detections
            )
            if not nearby:
                detections.append(bright)
        detections = self._limit_patterns(detections)

        flow_observations: dict[int, FlowTrackObservation] = {}
        if self.preassociation_flow is not None:
            flow_observations = self.preassociation_flow.observe(gray)
            self.flow_observation_count = len(flow_observations)
            for track in self.tracks.values():
                track.set_flow_observation(flow_observations.get(track.id))

        rotation_edges = self._rotation_edge_map(gray)
        for detection in detections:
            detection.rotation_descriptor = self._rotation_descriptor(
                rotation_edges, detection
            )
            detection.appearance = self._appearance_descriptor(gray, detection)

        self._annotate_collective_motion(detections)
        reassociation_committed = self._associate(
            detections, timestamp, flow_observations
        )
        self._annotate_track_rotation_residuals()
        self._update_background_certificates(
            detections, bright, allow_bright_seed
        )
        if self.switch_motion_history is not None:
            self.switch_motion_history.update(
                self._constellation_views(), gray_frame=gray
            )

        if allow_bright_seed and bright is not None:
            # White highlight only seeds / reaffirms REAL; do not reclassify yet.
            self._seed_real_from_bright(bright)
        elif self.white_seen:
            # After fade: REAL = geometric outlier among the constellation.
            self.identity_switched = (
                self._assign_roles(gray) or reassociation_committed
            )
        if self.preassociation_flow is not None:
            flow_tracks: Iterable[_ShapeTrack]
            if (
                self.preassociation_flow_shadow_enabled
                or self.flow_association_enabled
            ):
                flow_tracks = self.tracks.values()
            else:
                target_for_flow = (
                    self.tracks.get(self.target_id)
                    if self.target_id is not None
                    else None
                )
                flow_tracks = () if target_for_flow is None else (target_for_flow,)
            self.preassociation_flow.commit(gray, flow_tracks)

        target = self.tracks.get(self.target_id) if self.target_id is not None else None
        if (
            target is not None
            and target.predicted_only
            and self._outside_frame(target.state[:2])
        ):
            # Never report a coasting prediction that has left the panel.
            target = None
        constellation = self._constellation_views()
        debug_tracks = [
            (track.id, (track.state[0], track.state[1]), track.state[2], track.lost_frames)
            for track in self.tracks.values()
        ]

        if target is None:
            self._update_identity_confidence(
                None, position_actionable=False
            )
            return LieDetectorTrackingResult(
                acquired=False,
                target_id=self.target_id,
                center=None,
                radius=None,
                confidence=0.0,
                predicted_only=True,
                lost_frames=0,
                detections=detections,
                tracks=debug_tracks,
                hypothesis_count=len(self.target_hypotheses),
                collective_delta=(
                    float(self.collective_delta[0]),
                    float(self.collective_delta[1]),
                ),
                white_active=self.white_active,
                identity_window_active=self.shadow_candidate_id is not None,
                identity_switched=self.identity_switched,
                actionable=False,
                bg_registry_state=self.bg_registry_state,
                bg_certified_count=self.bg_certified_count,
                stale_recovery_committed=self.stale_recovery_committed,
                ranker_candidate_id=self.ranker_candidate_id,
                ranker_margin=self.ranker_margin,
                ranker_switched=self.ranker_switched,
                ranker_top_id=self.ranker_top_id,
                ranker_current_rank=self.ranker_current_rank,
                ranker_current_score=self.ranker_current_score,
                ranker_switch_delta=self.ranker_switch_delta,
                ranker_evidence=self.ranker_evidence,
                motion_corroborated=self.motion_corroborated,
                identity_confidence=self.identity_confidence,
                identity_state=self.identity_state,
                identity_safe=self.identity_safe,
                hold_reason=self.hold_reason,
                position_actionable=False,
                switch_event=self.switch_event,
                switch_event_probability=self.switch_event_probability,
                switch_event_approved=self.switch_event_approved,
                flow_observation_count=self.flow_observation_count,
                flow_association_count=self.flow_association_count,
                flow_coast_count=self.flow_coast_count,
                constellation=constellation,
            )

        x, y, radius = target.state
        if not target.predicted_only and target.last_detection is not None:
            x, y = target.last_detection.center
            radius = float(
                target.last_detection.observed_radius or target.last_detection.radius
            )
        confidence = min(1.0, target.hits / 5.0) * math.exp(-0.15 * target.lost_frames)
        uncertainty = target.position_uncertainty
        position_actionable = (
            not target.predicted_only
            or 4.0 * uncertainty <= max(8.0, float(radius))
        )
        self._update_identity_confidence(
            target, position_actionable=position_actionable
        )
        actionable = bool(
            position_actionable
            and (not self.identity_safety_enabled or self.identity_safe)
        )
        return LieDetectorTrackingResult(
            acquired=True,
            target_id=target.id,
            center=(float(x), float(y)),
            radius=float(radius),
            confidence=float(confidence),
            predicted_only=target.predicted_only,
            lost_frames=target.lost_frames,
            detections=detections,
            tracks=debug_tracks,
            hypothesis_count=len(self.target_hypotheses),
            collective_delta=(
                float(self.collective_delta[0]),
                float(self.collective_delta[1]),
            ),
            collective_rotation_degrees=float(target.rotation_residual),
            recovery_active=target.predicted_only and not actionable,
            white_active=self.white_active,
            identity_window_active=self.shadow_candidate_id is not None,
            identity_switched=self.identity_switched,
            actionable=actionable,
            position_uncertainty_px=uncertainty,
            bg_registry_state=self.bg_registry_state,
            bg_certified_count=self.bg_certified_count,
            stale_recovery_committed=self.stale_recovery_committed,
            ranker_candidate_id=self.ranker_candidate_id,
            ranker_margin=self.ranker_margin,
            ranker_switched=self.ranker_switched,
            ranker_top_id=self.ranker_top_id,
            ranker_current_rank=self.ranker_current_rank,
            ranker_current_score=self.ranker_current_score,
            ranker_switch_delta=self.ranker_switch_delta,
            ranker_evidence=self.ranker_evidence,
            motion_corroborated=self.motion_corroborated,
            identity_confidence=self.identity_confidence,
            identity_state=self.identity_state,
            identity_safe=self.identity_safe,
            hold_reason=self.hold_reason,
            position_actionable=position_actionable,
            switch_event=self.switch_event,
            switch_event_probability=self.switch_event_probability,
            switch_event_approved=self.switch_event_approved,
            flow_observation_count=self.flow_observation_count,
            flow_association_count=self.flow_association_count,
            flow_coast_count=self.flow_coast_count,
            constellation=constellation,
        )
