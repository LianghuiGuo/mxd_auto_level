"""Conservative frame-to-frame provenance for known background shapes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Iterable, Optional

import cv2
import numpy as np


@dataclass
class _Certificate:
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    history: deque[int] = field(default_factory=lambda: deque(maxlen=20))
    rooted: bool = True
    age: int = 1
    missed: int = 0
    bad: int = 0


@dataclass(frozen=True)
class CertificateResult:
    state: str
    certified_indices: frozenset[int]
    mature_count: int


class LieBackgroundCertificates:
    """Carry white-phase BG identities forward without stale-map matching."""

    ROOT_WINDOW = 20
    ROOT_SUPPORT = 12
    ROOT_GATE = 12.0
    FADE_CONFIRM = 2
    MATCH_GATE = 7.0
    MISS_EXPANSION = 3.0
    AMBIGUITY_RATIO = 2.5
    CROSSING_GATE = 45.0
    MAX_MISSED = 2
    MATURITY = 10
    BORDER_MATURITY = 20
    BORDER_MAX_AGE = 90
    MAX_EMITTED = 4
    FLOW_SEARCH_GATE = 20.0
    FLOW_INLIER_GATE = 3.0
    FLOW_RESIDUAL_GATE = 4.0
    BORDER_BIRTH_BAND = 35.0
    REAL_BIRTH_KEEP_OUT = 60.0
    RECOVERY_FRAMES = 5

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.state = "dormant"
        self.certificates: list[_Certificate] = []
        self.white_frames = 0
        self.fade_frames = 0
        self.healthy_frames = 0
        self.fade_observations: list[np.ndarray] = []
        self.frame_size: Optional[tuple[int, int]] = None

    @staticmethod
    def _points(
        centers: Iterable[tuple[float, float]],
    ) -> np.ndarray:
        return np.asarray(list(centers), dtype=np.float64).reshape(-1, 2)

    @staticmethod
    def _exclude_real(
        points: np.ndarray,
        real_center: tuple[float, float],
        real_radius: float,
    ) -> Optional[np.ndarray]:
        if len(points) < 2:
            return None
        distances = np.linalg.norm(points - np.asarray(real_center), axis=1)
        keep_out = max(30.0, 1.1 * float(real_radius))
        if float(np.min(distances)) > keep_out:
            return None
        return points[distances > keep_out]

    def _root(self, background: np.ndarray) -> None:
        self.white_frames += 1
        for certificate in self.certificates:
            certificate.history.append(0)
        candidates: list[tuple[float, int, int]] = []
        for slot_index, certificate in enumerate(self.certificates):
            predicted = certificate.position + certificate.velocity
            distances = np.linalg.norm(background - predicted, axis=1)
            for detection_index, distance in enumerate(distances):
                if distance <= self.ROOT_GATE:
                    candidates.append((float(distance), slot_index, detection_index))
        used_slots: set[int] = set()
        used_detections: set[int] = set()
        for _, slot_index, detection_index in sorted(candidates):
            if slot_index in used_slots or detection_index in used_detections:
                continue
            certificate = self.certificates[slot_index]
            incoming = background[detection_index]
            certificate.velocity = 0.5 * certificate.velocity + 0.5 * (
                incoming - certificate.position
            )
            certificate.position = incoming.copy()
            certificate.history[-1] = 1
            certificate.age += 1
            used_slots.add(slot_index)
            used_detections.add(detection_index)
        for index, point in enumerate(background):
            if index in used_detections:
                continue
            history = deque([1], maxlen=self.ROOT_WINDOW)
            self.certificates.append(
                _Certificate(position=point.copy(), history=history)
            )
        # Slots which vanished for the complete rolling window cannot
        # contribute to finalization. Dropping them also bounds root work when
        # YOLO emits transient false positives.
        self.certificates = [
            certificate
            for certificate in self.certificates
            if not (
                len(certificate.history) == self.ROOT_WINDOW
                and not any(certificate.history)
            )
        ]

    def _finalize(self) -> bool:
        def trailing_misses(certificate: _Certificate) -> int:
            misses = 0
            for observed in reversed(certificate.history):
                if observed:
                    break
                misses += 1
            return misses

        self.certificates = [
            certificate
            for certificate in self.certificates
            if len(certificate.history) >= self.ROOT_SUPPORT
            and sum(certificate.history) >= self.ROOT_SUPPORT
        ]
        if len(self.certificates) < 6:
            self.state = "disabled"
            self.certificates.clear()
            return False
        for certificate in self.certificates:
            certificate.position += (
                trailing_misses(certificate) * certificate.velocity
            )
            certificate.age = self.MATURITY
            certificate.missed = 0
            certificate.bad = 0
        self.state = "armed"
        self.healthy_frames = self.RECOVERY_FRAMES
        return True

    @classmethod
    def _similarity(
        cls, source: np.ndarray, destination: np.ndarray
    ) -> Optional[tuple[np.ndarray, np.ndarray, int]]:
        if len(source) < 4:
            return None
        affine, mask = cv2.estimateAffinePartial2D(
            source.astype(np.float32).reshape(-1, 1, 2),
            destination.astype(np.float32).reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=cls.FLOW_INLIER_GATE,
            maxIters=100,
            confidence=0.99,
            refineIters=6,
        )
        if affine is None:
            return None
        matrix = np.asarray(affine[:, :2], dtype=np.float64)
        translation = np.asarray(affine[:, 2], dtype=np.float64)
        scale = math.hypot(float(matrix[0, 0]), float(matrix[0, 1]))
        if not 0.97 <= scale <= 1.03:
            return None
        angle = abs(math.degrees(math.atan2(-matrix[0, 1], matrix[0, 0])))
        if angle > 2.0 or float(np.linalg.norm(translation)) > 20.0:
            return None
        inliers = (
            int(np.count_nonzero(mask))
            if mask is not None
            else len(source)
        )
        if inliers < 4:
            return None
        return matrix, translation, inliers

    def _flow_prediction(self, observed: np.ndarray) -> tuple[list[np.ndarray], int]:
        velocity_predictions = [
            certificate.position + certificate.velocity
            for certificate in self.certificates
        ]
        source: list[np.ndarray] = []
        destination: list[np.ndarray] = []
        candidates = sorted(
            (
                (float(np.linalg.norm(point - detection)), slot_index, detection_index)
                for slot_index, point in enumerate(velocity_predictions)
                for detection_index, detection in enumerate(observed)
                if np.linalg.norm(point - detection) <= self.FLOW_SEARCH_GATE
            ),
            key=lambda item: item[0],
        )
        used_slots: set[int] = set()
        used_detections: set[int] = set()
        for _, slot_index, detection_index in candidates:
            if slot_index in used_slots or detection_index in used_detections:
                continue
            # Fit the complete previous-frame -> current-frame transform. The
            # velocity prediction is used only to establish tentative pairs.
            source.append(self.certificates[slot_index].position)
            destination.append(observed[detection_index])
            used_slots.add(slot_index)
            used_detections.add(detection_index)
        transform = self._similarity(
            np.asarray(source), np.asarray(destination)
        )
        if transform is None:
            return velocity_predictions, 0
        matrix, translation, inliers = transform
        transformed = [
            certificate.position @ matrix.T + translation
            for certificate in self.certificates
        ]
        return transformed, inliers

    def _add_border_births(
        self,
        observed: np.ndarray,
        used: set[int],
        *,
        real_center: Optional[tuple[float, float]],
    ) -> None:
        if self.frame_size is None:
            return
        width, height = self.frame_size
        existing = [certificate.position for certificate in self.certificates]
        for detection_index, point in enumerate(observed):
            if detection_index in used:
                continue
            border_distance = min(
                float(point[0]),
                float(point[1]),
                float(width - point[0]),
                float(height - point[1]),
            )
            if border_distance > self.BORDER_BIRTH_BAND:
                continue
            if (
                real_center is not None
                and np.linalg.norm(point - np.asarray(real_center))
                <= self.REAL_BIRTH_KEEP_OUT
            ):
                continue
            if existing and min(
                float(np.linalg.norm(point - position))
                for position in existing
            ) <= 2.0 * self.MATCH_GATE:
                continue
            self.certificates.append(
                _Certificate(position=point.copy(), rooted=False)
            )
            existing.append(point)

    def _is_mature(self, certificate: _Certificate) -> bool:
        required_age = (
            self.MATURITY if certificate.rooted else self.BORDER_MATURITY
        )
        return certificate.age >= required_age

    def _advance(
        self,
        observed: np.ndarray,
        *,
        real_center: Optional[tuple[float, float]] = None,
        allow_births: bool = True,
    ) -> CertificateResult:
        if len(observed) < 5:
            survivors: list[_Certificate] = []
            for certificate in self.certificates:
                certificate.position += certificate.velocity
                certificate.missed += 1
                if certificate.missed <= self.MAX_MISSED:
                    survivors.append(certificate)
            self.certificates = survivors
            self.state = "suspended"
            self.healthy_frames = 0
            return CertificateResult(self.state, frozenset(), 0)
        predicted, _flow_inliers = self._flow_prediction(observed)
        assignments: list[tuple[int, int, float]] = []
        revoked: set[int] = set()
        used: set[int] = set()
        order = sorted(
            range(len(self.certificates)),
            key=lambda index: self.certificates[index].age,
            reverse=True,
        )
        for slot_index in order:
            certificate = self.certificates[slot_index]
            distances = np.linalg.norm(observed - predicted[slot_index], axis=1)
            ranked = np.argsort(distances)
            first = int(ranked[0])
            nearest = float(distances[first])
            second = float(distances[int(ranked[1])]) if len(ranked) > 1 else 1e9
            gate = self.MATCH_GATE + self.MISS_EXPANSION * certificate.missed
            if nearest <= gate and second < max(
                self.CROSSING_GATE,
                self.AMBIGUITY_RATIO * nearest,
            ):
                revoked.add(slot_index)
                continue
            if nearest <= gate and first not in used:
                assignments.append((slot_index, first, nearest))
                used.add(first)

        assigned_slots = {item[0] for item in assignments}
        for slot_index, detection_index, residual in assignments:
            certificate = self.certificates[slot_index]
            incoming = observed[detection_index]
            certificate.velocity = 0.5 * certificate.velocity + 0.5 * (
                incoming - certificate.position
            )
            certificate.position = incoming.copy()
            certificate.age += 1
            certificate.missed = 0
            certificate.bad = (
                certificate.bad + 1
                if residual > self.FLOW_RESIDUAL_GATE
                else 0
            )
            if (
                certificate.bad >= 3
                or (
                    not certificate.rooted
                    and certificate.age > self.BORDER_MAX_AGE
                )
            ):
                revoked.add(slot_index)
        for slot_index, certificate in enumerate(self.certificates):
            if slot_index in assigned_slots or slot_index in revoked:
                continue
            certificate.position = predicted[slot_index]
            certificate.missed += 1
            if certificate.missed > self.MAX_MISSED:
                revoked.add(slot_index)
        certified_candidates = [
            (
                0 if self.certificates[slot_index].rooted else 1,
                residual,
                -self.certificates[slot_index].age,
                detection_index,
            )
            for slot_index, detection_index, residual in assignments
            if slot_index not in revoked
            and self._is_mature(self.certificates[slot_index])
            and self.certificates[slot_index].bad == 0
            and residual <= self.FLOW_RESIDUAL_GATE
        ]
        certified = frozenset(
            item[3]
            for item in sorted(certified_candidates)[: self.MAX_EMITTED]
        )
        self.certificates = [
            certificate
            for index, certificate in enumerate(self.certificates)
            if index not in revoked
        ]
        if allow_births:
            self._add_border_births(
                observed,
                used,
                real_center=real_center,
            )
        mature_count = len(certified)
        healthy = (
            mature_count >= 3
            and len(assignments) >= 3
        )
        if healthy:
            self.healthy_frames += 1
            if (
                self.state == "suspended"
                and self.healthy_frames >= self.RECOVERY_FRAMES
            ):
                self.state = "armed"
        else:
            self.healthy_frames = 0
            self.state = "suspended"
        emitted = certified if self.state == "armed" else frozenset()
        return CertificateResult(self.state, emitted, mature_count)

    def update(
        self,
        centers: Iterable[tuple[float, float]],
        *,
        white_active: bool,
        real_center: Optional[tuple[float, float]] = None,
        real_radius: float = 0.0,
        frame_size: Optional[tuple[int, int]] = None,
    ) -> CertificateResult:
        points = self._points(centers)
        if frame_size is not None:
            normalized_size = (int(frame_size[0]), int(frame_size[1]))
            if (
                self.frame_size is not None
                and normalized_size != self.frame_size
                and self.state != "dormant"
            ):
                self.state = "disabled"
                self.certificates.clear()
                return CertificateResult(self.state, frozenset(), 0)
            self.frame_size = normalized_size
        if self.state == "disabled":
            return CertificateResult(self.state, frozenset(), 0)
        if white_active:
            self.fade_frames = 0
            self.fade_observations.clear()
            # A highlight can flicker back during the opening fade. Once the
            # root has been finalized, keep carrying its provenance rather
            # than throwing it away and attempting to learn from a handful of
            # late white frames.
            if self.state in ("armed", "suspended"):
                return self._advance(
                    points,
                    real_center=real_center,
                    allow_births=False,
                )
            if real_center is None:
                return CertificateResult(self.state, frozenset(), 0)
            background = self._exclude_real(points, real_center, real_radius)
            if background is None or len(background) < 6:
                return CertificateResult(self.state, frozenset(), 0)
            self.state = "rooting"
            self._root(background)
            return CertificateResult(self.state, frozenset(), 0)
        if self.state == "rooting":
            self.fade_frames += 1
            self.fade_observations.append(points.copy())
            if self.fade_frames >= self.FADE_CONFIRM:
                if not self._finalize():
                    return CertificateResult(self.state, frozenset(), 0)
                result = CertificateResult(self.state, frozenset(), 0)
                # Do not leave finalized slots several moving frames behind.
                # Replaying the short fade buffer preserves frame-to-frame
                # provenance without matching against a stale root snapshot.
                for observation in self.fade_observations:
                    result = self._advance(
                        observation,
                        real_center=real_center,
                    )
                self.fade_observations.clear()
                return result
            return CertificateResult(self.state, frozenset(), 0)
        if self.state in ("armed", "suspended"):
            return self._advance(points, real_center=real_center)
        return CertificateResult(self.state, frozenset(), 0)
