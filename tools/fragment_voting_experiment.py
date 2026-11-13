"""Run synthetic experiments or authorized offline fragment-tracker evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

SCENARIOS = (
    "clean",
    "distractors",
    "partial_occlusion",
    "boundary",
    "invisible",
    "circle",
)
MOTION_SCENARIOS = ("weak_motion", "deforming_background")


def rotate_points(points: np.ndarray, angle_degrees: float) -> np.ndarray:
    angle = math.radians(angle_degrees)
    matrix = np.array(
        ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))),
        dtype=np.float32,
    )
    return np.asarray(points, dtype=np.float32) @ matrix.T


@dataclass(frozen=True)
class Fragment:
    fragment_id: int
    points: np.ndarray
    normals: np.ndarray
    offset: np.ndarray


@dataclass(frozen=True)
class Vote:
    fragment_id: int
    center: tuple[float, float]
    score: float
    matched_point: tuple[float, float] | None = None
    rotated_offset: tuple[float, float] | None = None
    motion_support: float = 0.0
    rotation_support: float = 0.0
    valid_support: float = 1.0
    raw_score: float | None = None
    peak_margin: float = 0.0
    fragment_weight: float = 1.0


@dataclass(frozen=True)
class VotePeak:
    center: tuple[float, float]
    votes: int
    quality: float
    evidence: float


def aggregate_votes(
    votes: list[Vote], fragment_count: int, tolerance: float = 3.0
) -> list[VotePeak]:
    peaks = []
    for seed in votes:
        unique: dict[int, Vote] = {}
        for vote in votes:
            if math.dist(seed.center, vote.center) > tolerance:
                continue
            previous = unique.get(vote.fragment_id)
            if previous is None or vote.score > previous.score:
                unique[vote.fragment_id] = vote
        selected = list(unique.values())
        weights = np.array([vote.score for vote in selected], dtype=np.float64)
        center = np.average([vote.center for vote in selected], axis=0, weights=weights)
        peak = VotePeak(
            tuple(float(value) for value in center),
            len(selected),
            float(weights.mean()),
            float(weights.sum() / fragment_count),
        )
        peaks.append(peak)
    peaks.sort(key=lambda peak: peak.evidence, reverse=True)
    distinct = []
    for peak in peaks:
        if all(math.dist(peak.center, other.center) > tolerance for other in distinct):
            distinct.append(peak)
    return distinct


@dataclass(frozen=True)
class TrackingResult:
    status: str
    center: tuple[float, float] | None
    predicted_center: tuple[float, float]
    rotation_degrees: float
    angular_velocity_dps: float
    votes: int
    fragment_count: int
    quality: float
    margin: float
    confidence: float
    seconds_without_observation: float
    search_radius: float
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)


class FragmentVotingTracker:
    CHANNELS = 8

    def __init__(
        self,
        *,
        grid_size: int = 3,
        coast_seconds: float = 0.45,
        angle_step: float = 3.0,
        angle_window: float = 12.0,
        match_threshold: float = 0.66,
        min_margin: float = 0.045,
    ) -> None:
        parameters = (
            coast_seconds,
            angle_step,
            angle_window,
            match_threshold,
            min_margin,
        )
        if (
            not isinstance(grid_size, int)
            or grid_size < 2
            or not all(math.isfinite(value) for value in parameters)
            or coast_seconds < 0
            or angle_step <= 0
            or angle_window < 0
        ):
            raise ValueError("invalid grid, coast budget or angular search limits")
        if not 0 < match_threshold <= 1 or not 0 <= min_margin <= 1:
            raise ValueError("match threshold and margin must be between zero and one")
        self.grid_size = grid_size
        self.coast_seconds = coast_seconds
        self.angle_step = angle_step
        self.angle_window = angle_window
        self.match_threshold = match_threshold
        self.min_margin = min_margin
        self.reset()

    def reset(self) -> None:
        self.fragments: list[Fragment] = []
        self.center: np.ndarray | None = None
        self.velocity = np.zeros(2, dtype=np.float64)
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.radius = 0.0
        self.last_timestamp: float | None = None
        self.last_observed_at = 0.0
        self.last_observed_center: np.ndarray | None = None
        self.angle_history: deque[tuple[float, float]] = deque(maxlen=12)
        self.rotationally_symmetric = False

    @staticmethod
    def _gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0.6)
        dx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        return dx, dy, cv2.magnitude(dx, dy)

    def initialize(
        self,
        template_mask: np.ndarray,
        center: tuple[float, float],
        timestamp: float = 0.0,
    ) -> None:
        if template_mask.ndim != 2 or min(template_mask.shape) < 8:
            raise ValueError(
                "template_mask must be a non-empty two-dimensional silhouette"
            )
        if (
            np.asarray(center).shape != (2,)
            or not np.isfinite(center).all()
            or not math.isfinite(timestamp)
        ):
            raise ValueError(
                "initial center must contain two finite coordinates and timestamp must be finite"
            )
        if (
            not np.isfinite(template_mask).all()
            or np.any(template_mask[[0, -1], :])
            or np.any(template_mask[:, [0, -1]])
        ):
            raise ValueError(
                "template silhouette must be finite and surrounded by a zero border"
            )
        mask = np.where(template_mask > 0, 255, 0).astype(np.uint8)
        dx, dy, magnitude = self._gradients(mask)
        rows, cols = np.nonzero(magnitude > 80.0)
        if len(rows) < 24:
            raise ValueError("template must contain a visible boundary")
        origin = np.array(((mask.shape[1] - 1) / 2, (mask.shape[0] - 1) / 2))
        points = np.column_stack((cols, rows)).astype(np.float32) - origin
        normals = np.column_stack((dx[rows, cols], dy[rows, cols]))
        normals /= np.maximum(magnitude[rows, cols, None], 1e-6)
        cell_x = np.minimum(cols * self.grid_size // mask.shape[1], self.grid_size - 1)
        cell_y = np.minimum(rows * self.grid_size // mask.shape[0], self.grid_size - 1)
        fragments = []
        for cell in range(self.grid_size**2):
            selection = cell_y * self.grid_size + cell_x == cell
            if np.count_nonzero(selection) < 12:
                continue
            local_points = points[selection]
            offset = local_points.mean(axis=0)
            fragments.append(
                Fragment(cell, local_points - offset, normals[selection], offset)
            )
        if len(fragments) < 3:
            raise ValueError(
                "template must expose at least three independent boundary fragments"
            )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
        radial = np.linalg.norm(contour - origin, axis=1)
        self.reset()
        self.fragments = fragments
        self.rotationally_symmetric = (
            float(radial.std() / max(radial.mean(), 1.0)) < 0.035
        )
        self.radius = float(np.max(np.linalg.norm(points, axis=1)))
        self.center = np.asarray(center, dtype=np.float64).copy()
        self.last_observed_center = self.center.copy()
        self.last_timestamp = float(timestamp)
        self.last_observed_at = float(timestamp)
        self.angle_history.append((float(timestamp), 0.0))

    @classmethod
    def _features(cls, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dx, dy, magnitude = cls._gradients(gray)
        angles = np.arctan2(dy, dx)
        bins = (
            np.rint(angles * cls.CHANNELS / (2 * math.pi)).astype(np.int32)
            % cls.CHANNELS
        )
        channels = []
        for index in range(cls.CHANNELS):
            edge = ((magnitude > 80.0) & (bins == index)).astype(np.float32)
            channels.append(cv2.dilate(edge, np.ones((3, 3), dtype=np.uint8)))
        return np.stack(channels, axis=2)

    @classmethod
    def _kernel(cls, fragment: Fragment, angle: float) -> tuple[np.ndarray, np.ndarray]:
        offset = rotate_points(fragment.offset[None, :], angle)[0]
        points = rotate_points(fragment.points, angle) + offset
        pixels = np.rint(points).astype(np.int32)
        low = pixels.min(axis=0)
        pixels -= low
        size = pixels.max(axis=0) + 1
        normals = rotate_points(fragment.normals, angle)
        bins = (
            np.rint(
                np.arctan2(normals[:, 1], normals[:, 0]) * cls.CHANNELS / (2 * math.pi)
            ).astype(np.int32)
            % cls.CHANNELS
        )
        kernel = np.zeros((int(size[1]), int(size[0]), cls.CHANNELS), dtype=np.float32)
        kernel[pixels[:, 1], pixels[:, 0], bins] = 1.0
        kernel /= kernel.sum()
        return kernel, low

    def _fragment_votes(
        self, features: np.ndarray, predicted: np.ndarray, radius: float, angle: float
    ) -> list[Vote]:
        votes = []
        height, width = features.shape[:2]
        center_low = np.floor(predicted - radius).astype(np.int32)
        center_high = np.ceil(predicted + radius).astype(np.int32)
        for fragment in self.fragments:
            kernel, offset = self._kernel(fragment, angle)
            kh, kw = kernel.shape[:2]
            low = center_low + offset
            high = center_high + offset + (kw, kh)
            crop = np.zeros(
                (int(high[1] - low[1]), int(high[0] - low[0]), self.CHANNELS),
                dtype=np.float32,
            )
            x0, y0 = max(0, int(low[0])), max(0, int(low[1]))
            x1, y1 = min(width, int(high[0])), min(height, int(high[1]))
            if x1 <= x0 or y1 <= y0:
                continue
            crop[y0 - low[1] : y1 - low[1], x0 - low[0] : x1 - low[0]] = features[
                y0:y1, x0:x1
            ]
            scores = cv2.matchTemplate(crop, kernel, cv2.TM_CCORR)
            ys, xs = np.indices(scores.shape)
            center_x, center_y = xs + center_low[0], ys + center_low[1]
            valid = (
                (center_x >= 0)
                & (center_x < width)
                & (center_y >= 0)
                & (center_y < height)
                & (
                    (center_x - predicted[0]) ** 2 + (center_y - predicted[1]) ** 2
                    <= radius**2
                )
            )
            scores[~valid] = -1.0
            for _ in range(2):
                _, quality, _, location = cv2.minMaxLoc(scores)
                if quality < self.match_threshold:
                    break
                center = center_low + np.asarray(location)
                votes.append(
                    Vote(
                        fragment.fragment_id,
                        tuple(float(v) for v in center),
                        min(1.0, float(quality)),
                    )
                )
                cv2.circle(scores, location, 6, -1.0, -1)
        return votes

    def update(self, frame: np.ndarray, timestamp: float) -> TrackingResult:
        if self.center is None or self.last_timestamp is None:
            raise RuntimeError("initialize the tracker before update")
        if (
            frame.ndim != 3
            or frame.shape[2] != 3
            or not frame.size
            or frame.dtype != np.uint8
        ):
            raise ValueError("frame must be a non-empty uint8 BGR image")
        if not math.isfinite(timestamp) or timestamp <= self.last_timestamp:
            raise ValueError("timestamps must be finite and strictly increasing")
        dt = float(timestamp - self.last_timestamp)
        predicted = self.center + self.velocity * dt
        predicted_angle = self.angle + self.angular_velocity * dt
        gap = float(timestamp - self.last_observed_at)
        search_radius = min(
            2.0 * self.radius,
            0.75 * self.radius + np.linalg.norm(self.velocity) * dt + 12 * gap,
        )
        features = self._features(frame)
        if self.rotationally_symmetric:
            angles = [0.0]
        else:
            window = min(30.0, self.angle_window + 20.0 * gap)
            steps = math.ceil(window / self.angle_step)
            angles = predicted_angle + np.arange(-steps, steps + 1) * self.angle_step
        candidates = []
        for angle in angles:
            peaks = aggregate_votes(
                self._fragment_votes(features, predicted, search_radius, float(angle)),
                len(self.fragments),
            )
            for peak in peaks:
                distance = math.dist(peak.center, predicted)
                prior = 0.025 * math.exp(-0.5 * (distance / max(self.radius, 1.0)) ** 2)
                angular_prior = 0.005 * min(
                    1.0, abs(float(angle) - predicted_angle) / 30.0
                )
                candidates.append(
                    (peak.evidence + prior - angular_prior, peak, float(angle))
                )
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        best = candidates[0] if candidates else None
        peak = best[1] if best else None
        rivals = [
            candidate[1].evidence
            for candidate in candidates
            if peak is not None and math.dist(candidate[1].center, peak.center) > 7.0
        ]
        margin = peak.evidence - max(rivals, default=0.0) if peak else 0.0
        accepted = bool(
            peak
            and peak.votes >= 3
            and peak.evidence >= 0.38
            and margin >= self.min_margin
        )
        quality = peak.quality if peak else 0.0
        votes = peak.votes if peak else 0
        confidence = 0.0
        if accepted:
            measured = np.asarray(peak.center)
            observed_dt = float(timestamp - self.last_observed_at)
            measured_velocity = (measured - self.last_observed_center) / observed_dt
            blend = 1.0 - math.exp(-observed_dt / 0.12)
            self.velocity = (1.0 - blend) * self.velocity + blend * measured_velocity
            self.center = 0.9 * measured + 0.1 * predicted
            self.angle = best[2]
            self.angle_history.append((float(timestamp), self.angle))
            recent = [(t, a) for t, a in self.angle_history if timestamp - t <= 0.5]
            if not self.rotationally_symmetric and len(recent) >= 3:
                times = np.array([t for t, _ in recent])
                values = np.array([a for _, a in recent])
                times -= times.mean()
                self.angular_velocity = float(
                    np.clip(
                        np.dot(times, values - values.mean()) / np.dot(times, times),
                        -180.0,
                        180.0,
                    )
                )
            self.last_observed_at = float(timestamp)
            self.last_observed_center = measured.copy()
            status = "observed"
            confidence = float(
                np.clip(peak.evidence * min(1.0, margin / 0.15), 0.0, 1.0)
            )
        else:
            self.center = predicted
            self.angle = predicted_angle
            status = "predicted" if gap <= self.coast_seconds else "lost"
        self.last_timestamp = float(timestamp)
        return TrackingResult(
            status=status,
            center=None
            if status == "lost"
            else tuple(float(value) for value in self.center),
            predicted_center=tuple(float(value) for value in predicted),
            rotation_degrees=self.angle,
            angular_velocity_dps=self.angular_velocity,
            votes=votes,
            fragment_count=len(self.fragments),
            quality=quality,
            margin=max(0.0, float(margin)),
            confidence=confidence,
            seconds_without_observation=float(timestamp - self.last_observed_at),
            search_radius=float(search_radius),
        )


VARIANTS = (
    "baseline",
    "p1",
    "p2",
    "p3",
    "optimized",
    "local_vote",
    "template_fixed",
    "search_fixed",
    "revised",
    "original_voting",
    "original_p0",
    "original_p1",
    "original_p2",
    "original_p3",
    "original_guarded",
    "motion_local",
    "motion_feedback",
    "motion_revised",
)


class BackgroundMotion:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.previous_gray = None
        self.previous_valid = None
        self.previous_center = None
        self.affine = None

    def observe(
        self, gray: np.ndarray, valid: np.ndarray, center: np.ndarray, radius: float
    ) -> tuple[np.ndarray, dict]:
        residual = np.zeros(gray.shape, dtype=np.float32)
        self.affine = None
        diagnostics = {
            "background_confidence": 0.0,
            "background_inliers": 0,
            "background_fit_error": 0.0,
            "background_status": "unavailable",
        }
        if self.previous_gray is None or self.previous_gray.shape != gray.shape:
            return residual, diagnostics
        mask = self.previous_valid.astype(np.uint8) * 255
        cv2.circle(
            mask,
            tuple(round(float(v)) for v in self.previous_center),
            round(1.6 * radius),
            0,
            -1,
        )
        mask[:55] = 0
        mask[-8:] = 0
        mask[:, :8] = 0
        mask[:, -8:] = 0
        points = cv2.goodFeaturesToTrack(
            self.previous_gray,
            maxCorners=160,
            qualityLevel=0.03,
            minDistance=12,
            mask=mask,
        )
        if points is None or len(points) < 12:
            diagnostics["background_status"] = "too_few_points"
            return residual, diagnostics
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray, gray, points, None, winSize=(21, 21), maxLevel=3
        )
        if moved is None:
            return residual, diagnostics
        returned, backward, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.previous_gray, moved, None, winSize=(21, 21), maxLevel=3
        )
        if returned is None:
            return residual, diagnostics
        source, target = points.reshape(-1, 2), moved.reshape(-1, 2)
        good = (status.ravel() > 0) & (backward.ravel() > 0)
        good &= np.isfinite(target).all(axis=1)
        good &= np.linalg.norm(returned.reshape(-1, 2) - source, axis=1) < 1.5
        ix = np.clip(np.rint(target[:, 0]).astype(np.int32), 0, gray.shape[1] - 1)
        iy = np.clip(np.rint(target[:, 1]).astype(np.int32), 0, gray.shape[0] - 1)
        good &= valid[iy, ix]
        good &= np.linalg.norm(target - center, axis=1) > 1.6 * radius
        if np.count_nonzero(good) < 12:
            diagnostics["background_status"] = "inconsistent_flow"
            return residual, diagnostics
        source, target = source[good], target[good]
        affine, inliers = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
            maxIters=300,
            confidence=0.99,
            refineIters=8,
        )
        if affine is None:
            return residual, diagnostics
        selected = inliers.ravel() > 0
        count = np.count_nonzero(selected)
        fitted = source @ affine[:, :2].T + affine[:, 2]
        error = float(
            np.median(np.linalg.norm(fitted[selected] - target[selected], axis=1))
        )
        cells = np.floor(
            source[selected] / np.array((gray.shape[1] / 4, gray.shape[0] / 3))
        ).astype(np.int32)
        coverage = len(np.unique(cells, axis=0)) / 12
        confidence = float(selected.mean() * coverage * math.exp(-error / 2))
        scale = math.sqrt(abs(float(np.linalg.det(affine[:, :2]))))
        diagnostics.update(
            background_confidence=confidence,
            background_inliers=int(count),
            background_fit_error=error,
        )
        if count < 10 or coverage < 0.25 or error > 1.8 or not 0.9 < scale < 1.1:
            diagnostics["background_status"] = "rejected_fit"
            diagnostics["background_confidence"] = 0.0
            return residual, diagnostics
        aligned = cv2.warpAffine(
            self.previous_gray, affine, (gray.shape[1], gray.shape[0])
        )
        aligned_valid = (
            cv2.warpAffine(
                self.previous_valid.astype(np.uint8),
                affine,
                (gray.shape[1], gray.shape[0]),
                flags=cv2.INTER_NEAREST,
            )
            > 0
        )
        usable = valid & aligned_valid
        difference = np.abs(gray.astype(np.float32) - aligned.astype(np.float32))
        noise = float(np.median(difference[usable])) if np.any(usable) else 0.0
        residual = np.clip((difference - noise - 2) / 22, 0, 1)
        residual[~usable] = 0
        diagnostics["background_status"] = "accepted"
        self.affine = affine.copy()
        return cv2.GaussianBlur(residual, (5, 5), 0.8), diagnostics

    def commit(self, gray: np.ndarray, valid: np.ndarray, center: np.ndarray) -> None:
        self.previous_gray = gray.copy()
        self.previous_valid = valid.copy()
        self.previous_center = center.copy()


def visible_template(
    frame: np.ndarray,
    valid: np.ndarray,
    expected: np.ndarray | None = None,
    radius: float | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = ((gray >= 210) & (hsv[:, :, 1] <= 75) & valid).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        if not 700 <= area <= 40000 or not 35 <= max(width, height) <= 240:
            continue
        moments = cv2.moments(contour)
        point = np.array(
            (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        )
        if expected is None:
            if not (
                0.35 * gray.shape[1] <= point[0] <= 0.65 * gray.shape[1]
                and 0.3 * gray.shape[0] <= point[1] <= 0.78 * gray.shape[0]
            ):
                continue
        elif np.linalg.norm(point - expected) > max(12.0, 0.5 * radius):
            continue
        component = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(component, [contour], -1, 255, -1)
        pixels = (component > 0) & valid
        if not np.any(pixels) or float(gray[pixels].mean()) < 220:
            continue
        extent = max(
            point[0] - x, x + width - point[0], point[1] - y, y + height - point[1]
        )
        side = 2 * math.ceil(extent + 7) + 1
        mask = cv2.getRectSubPix(
            component, (side, side), tuple(float(v) for v in point)
        )
        local_valid = (
            cv2.getRectSubPix(
                valid.astype(np.float32), (side, side), tuple(float(v) for v in point)
            )
            >= 0.99
        )
        mask = (mask >= 127).astype(np.uint8) * 255
        score = float(gray[pixels].mean()) - (
            0 if expected is None else np.linalg.norm(point - expected)
        )
        candidates.append((score, mask, local_valid, tuple(float(v) for v in point)))
    if not candidates:
        return None
    _, mask, local_valid, point = max(candidates, key=lambda item: item[0])
    return mask, local_valid, point


class OptimizedFragmentTracker:
    ORIENTATION_BINS = 8

    def __init__(self, variant: str = "optimized") -> None:
        if variant not in VARIANTS[1:]:
            raise ValueError(
                "optimized tracker variant must be p1, p2, p3 or optimized"
            )
        self.variant = variant
        self.stage = min(4, VARIANTS.index(variant))
        self.local_voting = variant in VARIANTS[5:]
        self.fix_template = variant in VARIANTS[6:]
        self.fix_search = variant in VARIANTS[7:]
        self.temporal_recovery = variant == "revised"
        self.coast_seconds = 0.45
        self.reset()

    def reset(self) -> None:
        self.model = None
        self.model_valid = None
        self.model_sum = None
        self.model_count = None
        self.points = np.empty((0, 2), np.float32)
        self.normals = np.empty((0, 2), np.float32)
        self.groups = np.empty(0, np.int32)
        self.point_weights = np.empty(0, np.float32)
        self.fragment_count = 0
        self.center = None
        self.velocity = np.zeros(2, np.float64)
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.radius = 0.0
        self.period = 360.0
        self.last_timestamp = None
        self.last_observed_at = 0.0
        self.last_observed_center = None
        self.last_white_at = 0.0
        self.white_samples = 0
        self.angle_history = deque(maxlen=20)
        self.pose_hypotheses = [0.0]
        self.pose_uncertainty = 12.0
        self.pending_recovery = None
        self.pending_since = None
        self.pending_last_at = None
        self.pending_angle = None
        self.pending_groups = None
        self.pending_frames = 0
        self.pending_dynamic = 0.0
        self.pending_anchor = None
        self.pending_anchor_angle = None
        self.pending_bg_anchor = None
        self.pending_bg_angle = None
        self.coverage_angles = set()
        self.template_ready = False
        self.recovery_count = 0
        self.cache = {}
        self.motion = BackgroundMotion()
        self.dynamic_distances = None
        self.current_background = {}
        self.last_score_group_motion = None
        self.last_score_full_quality = None

    def initialize(
        self,
        template: np.ndarray,
        center: tuple[float, float],
        timestamp: float = 0.0,
        template_valid: np.ndarray | None = None,
    ) -> None:
        checker = FragmentVotingTracker()
        checker.initialize(template, center, timestamp)
        if template_valid is not None and template_valid.shape != template.shape:
            raise ValueError("template validity mask must match silhouette")
        self.reset()
        self.model = (template > 0).astype(np.uint8) * 255
        self.model_valid = (
            np.ones(template.shape, bool)
            if template_valid is None
            else template_valid.copy().astype(bool)
        )
        self.model_sum = (self.model > 0).astype(np.float32) * self.model_valid
        self.model_count = self.model_valid.astype(np.float32)
        self.center = np.array(center, dtype=np.float64)
        self.last_observed_center = self.center.copy()
        self.last_timestamp = float(timestamp)
        self.last_observed_at = self.last_white_at = float(timestamp)
        self.white_samples = 1
        self.angle_history.append((float(timestamp), 0.0))
        self._build_geometry()
        self.period = self._symmetry_period()
        self.coverage_angles.add(0)

    def _build_geometry(self) -> None:
        contours, _ = cv2.findContours(
            self.model, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        count = min(160 if self.stage < 4 else 96, len(contour))
        indices = np.linspace(0, len(contour), count, endpoint=False).astype(np.int32)
        origin = np.array(
            ((self.model.shape[1] - 1) / 2, (self.model.shape[0] - 1) / 2)
        )
        points = contour[indices] - origin
        before, after = (
            contour[(indices - 3) % len(contour)],
            contour[(indices + 3) % len(contour)],
        )
        tangent = after - before
        normals = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-5)
        boundary_valid = (
            cv2.erode(self.model_valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        )
        valid = boundary_valid[
            contour[indices, 1].astype(np.int32), contour[indices, 0].astype(np.int32)
        ]
        left, right = contour[indices] - before, after - contour[indices]
        cosine = np.sum(left * right, axis=1) / np.maximum(
            np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-5
        )
        weights = 1.0 + (1.0 - cosine) if self.stage >= 3 else np.ones(count)
        groups = np.arange(count) * 8 // count
        self.points, self.normals = points[valid].astype(np.float32), normals[valid]
        self.groups, self.point_weights = (
            groups[valid],
            weights[valid].astype(np.float32),
        )
        self.fragment_count = len(np.unique(self.groups))
        self.radius = float(np.max(np.linalg.norm(points, axis=1)))
        self.cache.clear()

    def _symmetry_period(self) -> float:
        if self.fix_template:
            return self._valid_symmetry_period()
        radial = np.linalg.norm(self.points, axis=1)
        if len(radial) and radial.std() / max(radial.mean(), 1.0) < 0.035:
            return 0.0
        origin = ((self.model.shape[1] - 1) / 2, (self.model.shape[0] - 1) / 2)
        for order in range(8, 1, -1):
            affine = cv2.getRotationMatrix2D(origin, 360.0 / order, 1.0)
            rotated = (
                cv2.warpAffine(
                    self.model, affine, (self.model.shape[1], self.model.shape[0])
                )
                > 127
            )
            original = self.model > 0
            if (
                np.count_nonzero(rotated & original)
                / max(1, np.count_nonzero(rotated | original))
                > 0.9
            ):
                return 360.0 / order
        return 360.0

    def _valid_symmetry_period(self) -> float:
        if len(self.points) >= 24:
            radial = np.linalg.norm(self.points, axis=1)
            if radial.std() / max(radial.mean(), 1.0) < 0.035:
                return 0.0
        origin = ((self.model.shape[1] - 1) / 2, (self.model.shape[0] - 1) / 2)
        original = self.model > 0
        trusted = (
            cv2.erode(self.model_valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        )
        for order in range(8, 1, -1):
            angle = 360.0 / order
            accepted = True
            for multiplier in range(1, order):
                affine = cv2.getRotationMatrix2D(origin, angle * multiplier, 1.0)
                rotated = (
                    cv2.warpAffine(
                        self.model, affine, (self.model.shape[1], self.model.shape[0])
                    )
                    > 127
                )
                rotated_valid = (
                    cv2.warpAffine(
                        trusted.astype(np.uint8),
                        affine,
                        (self.model.shape[1], self.model.shape[0]),
                        flags=cv2.INTER_NEAREST,
                    )
                    > 0
                )
                usable = trusted & rotated_valid
                union = (original | rotated) & usable
                coverage = np.count_nonzero(union) / max(
                    1, np.count_nonzero(original | rotated)
                )
                overlap = np.count_nonzero(original & rotated & usable) / max(
                    1, np.count_nonzero(union)
                )
                if coverage < 0.55 or overlap < 0.90:
                    accepted = False
                    break
            if accepted:
                return angle
        return 360.0

    def _pose_delta(self, first: float, second: float) -> float:
        if self.period == 0:
            return 0.0
        return (first - second + self.period / 2) % self.period - self.period / 2

    def _rotated(self, angle: float) -> tuple[np.ndarray, np.ndarray]:
        discrete = round(angle / 2.0) * 2.0 if self.stage >= 4 else float(angle)
        key = round(discrete % 360.0, 4)
        if self.stage >= 4 and key in self.cache:
            return self.cache[key]
        points = rotate_points(self.points, discrete)
        normals = rotate_points(self.normals, discrete)
        bins = (
            np.rint(
                np.arctan2(normals[:, 1], normals[:, 0])
                * self.ORIENTATION_BINS
                / math.pi
            ).astype(np.int32)
            % self.ORIENTATION_BINS
        )
        if self.stage >= 4:
            self.cache[key] = (points, bins)
        return points, bins

    def _edge_maps(
        self, gray: np.ndarray, valid: np.ndarray, residual: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        dx, dy, magnitude = FragmentVotingTracker._gradients(gray)
        local = cv2.GaussianBlur(magnitude, (15, 15), 3.0)
        normalized = magnitude / (local + 8.0)
        edges = (magnitude > 8.0) & (normalized > 0.8) & valid
        nms = magnitude >= cv2.dilate(magnitude, np.ones((3, 3), np.uint8)) * 0.80
        edges &= nms
        directions = (
            np.rint(np.arctan2(dy, dx) * self.ORIENTATION_BINS / math.pi).astype(
                np.int32
            )
            % self.ORIENTATION_BINS
        )
        maps = []
        dynamic_maps = []
        for index in range(self.ORIENTATION_BINS):
            distance = np.minimum(
                (directions - index) % self.ORIENTATION_BINS,
                (index - directions) % self.ORIENTATION_BINS,
            )
            selected = edges & (distance <= 1)
            if np.any(selected):
                maps.append(
                    cv2.distanceTransform((~selected).astype(np.uint8), cv2.DIST_L2, 3)
                )
            else:
                maps.append(np.full(gray.shape, 20.0, np.float32))
            if self.temporal_recovery and residual is not None:
                dynamic_edge = selected & (residual > 0.08)
                dynamic_maps.append(
                    cv2.distanceTransform(
                        (~dynamic_edge).astype(np.uint8), cv2.DIST_L2, 3
                    )
                    if np.any(dynamic_edge)
                    else np.full(gray.shape, 20.0, np.float32)
                )
        self.dynamic_distances = np.stack(dynamic_maps) if dynamic_maps else None
        return np.stack(maps), magnitude

    def _score(
        self,
        centers: np.ndarray,
        angle: float,
        distances: np.ndarray,
        valid: np.ndarray,
        residual: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        points, bins = self._rotated(angle)
        pixels = np.rint(centers[:, None, :] + points[None, :, :]).astype(np.int32)
        height, width = valid.shape
        inside = (
            (pixels[:, :, 0] >= 0)
            & (pixels[:, :, 0] < width)
            & (pixels[:, :, 1] >= 0)
            & (pixels[:, :, 1] < height)
        )
        xs = np.clip(pixels[:, :, 0], 0, width - 1)
        ys = np.clip(pixels[:, :, 1], 0, height - 1)
        usable = inside & valid[ys, xs]
        support = np.exp(-0.5 * (distances[bins[None, :], ys, xs] / 1.8) ** 2)
        support *= usable
        weights = usable * self.point_weights[None, :]
        quality = np.sum(support * self.point_weights, axis=1) / np.maximum(
            weights.sum(axis=1), 1.0
        )
        dynamic = np.sum(residual[ys, xs] * weights, axis=1) / np.maximum(
            weights.sum(axis=1), 1.0
        )
        group_quality = []
        group_motion = []
        for group in np.unique(self.groups):
            selected = self.groups == group
            denominator = weights[:, selected].sum(axis=1)
            enough = usable[:, selected].sum(axis=1) >= max(
                3, 0.3 * np.count_nonzero(selected)
            )
            q = (support[:, selected] * self.point_weights[selected]).sum(
                axis=1
            ) / np.maximum(denominator, 1.0)
            group_quality.append(np.where(enough, q, 0.0))
            motion_values = residual[ys[:, selected], xs[:, selected]]
            if self.dynamic_distances is not None:
                edge_motion = np.exp(
                    -0.5
                    * (
                        self.dynamic_distances[
                            bins[None, selected], ys[:, selected], xs[:, selected]
                        ]
                        / 2.5
                    )
                    ** 2
                )
                motion_values = motion_values * edge_motion
            motion_q = np.sum(
                motion_values * weights[:, selected], axis=1
            ) / np.maximum(denominator, 1.0)
            group_motion.append(np.where(enough, motion_q, 0.0))
        group_quality = np.stack(group_quality, axis=1)
        group_motion = np.stack(group_motion, axis=1)
        visibility = usable.mean(axis=1)
        self.last_score_full_quality = quality.copy()
        self.last_score_group_motion = group_motion
        if self.local_voting:
            strong = group_quality >= 0.72
            count = strong.sum(axis=1)
            selected_mean = np.sum(group_quality * strong, axis=1) / np.maximum(
                count, 1
            )
            evidence_mass = count / max(1, self.fragment_count)
            quality = 0.75 * selected_mean + 0.25 * evidence_mass
            quality[count < 3] = 0.0
            if self.temporal_recovery:
                dynamic = np.sum(group_motion * strong, axis=1) / np.maximum(count, 1)
        quality[visibility < 0.25] = 0
        return quality, dynamic, group_quality, visibility

    def _centers(
        self, predicted: np.ndarray, radius: float, residual: np.ndarray
    ) -> np.ndarray:
        step = 4 if self.stage >= 4 else 2
        reach = math.ceil(radius / step)
        offsets = np.arange(-reach, reach + 1) * step
        xs, ys = np.meshgrid(offsets, offsets)
        centers = np.column_stack((xs.ravel(), ys.ravel())) + predicted
        centers = centers[np.linalg.norm(centers - predicted, axis=1) <= radius]
        centers = centers[
            (centers[:, 0] >= 0)
            & (centers[:, 0] < residual.shape[1])
            & (centers[:, 1] >= 0)
            & (centers[:, 1] < residual.shape[0])
        ]
        if self.fix_search and len(centers) > 320:
            distances = np.linalg.norm(centers - predicted, axis=1)
            near = np.argsort(distances)[:96]
            order = np.lexsort((centers[:, 0], centers[:, 1]))
            spatial = order[np.linspace(0, len(order) - 1, 160).astype(np.int32)]
            small = cv2.resize(
                residual,
                (max(1, residual.shape[1] // 4), max(1, residual.shape[0] // 4)),
                interpolation=cv2.INTER_AREA,
            )
            broad = cv2.GaussianBlur(small, (0, 0), max(1, self.radius / 12))
            pixels = np.clip(
                np.rint(centers / 4).astype(np.int32),
                (0, 0),
                (broad.shape[1] - 1, broad.shape[0] - 1),
            )
            motion = np.argsort(broad[pixels[:, 1], pixels[:, 0]])[-64:]
            chosen = np.unique(np.concatenate((near, spatial, motion)))
            centers = centers[chosen]
        elif self.stage >= 4 and len(centers) > 320:
            small = cv2.resize(
                residual,
                (max(1, residual.shape[1] // 4), max(1, residual.shape[0] // 4)),
                interpolation=cv2.INTER_AREA,
            )
            broad = cv2.GaussianBlur(small, (0, 0), max(1, self.radius / 12))
            pixels = np.rint(centers / 4).astype(np.int32)
            pixels[:, 0] = np.clip(pixels[:, 0], 0, broad.shape[1] - 1)
            pixels[:, 1] = np.clip(pixels[:, 1], 0, broad.shape[0] - 1)
            merit = broad[pixels[:, 1], pixels[:, 0]] + 0.10 * np.exp(
                -np.linalg.norm(centers - predicted, axis=1) / self.radius
            )
            centers = centers[np.argsort(merit)[-320:]]
        return centers.astype(np.float32)

    def _white_pose(
        self, observation: tuple, predicted_angle: float
    ) -> tuple[float, float, float]:
        mask, valid, _ = observation
        origin = ((mask.shape[1] - 1) / 2, (mask.shape[0] - 1) / 2)
        target_origin = np.array(
            ((self.model.shape[1] - 1) / 2, (self.model.shape[0] - 1) / 2)
        )
        candidates = []
        angles = [0.0] if self.period == 0 else predicted_angle + np.arange(-24, 25, 2)
        for angle in angles:
            affine = cv2.getRotationMatrix2D(origin, float(angle), 1.0)
            affine[:, 2] += target_origin - origin
            size = (self.model.shape[1], self.model.shape[0])
            rotated = cv2.warpAffine(mask, affine, size) > 127
            usable = (
                cv2.warpAffine(
                    valid.astype(np.uint8), affine, size, flags=cv2.INTER_NEAREST
                )
                > 0
            ) & self.model_valid
            original = self.model > 0
            union = np.count_nonzero((original | rotated) & usable)
            score = np.count_nonzero(original & rotated & usable) / max(1, union)
            candidates.append((score, float(angle)))
        candidates.sort(reverse=True)
        best = candidates[0]
        rival = max(
            (
                score
                for score, angle in candidates
                if abs(self._pose_delta(angle, best[1])) > 8
            ),
            default=0.0,
        )
        return best[1], best[0], best[0] - rival

    def _align_white_center(self, observation: tuple, angle: float) -> tuple:
        mask, valid, center = observation
        origin = np.array(((mask.shape[1] - 1) / 2, (mask.shape[0] - 1) / 2))
        model_origin = np.array(
            ((self.model.shape[1] - 1) / 2, (self.model.shape[0] - 1) / 2)
        )
        rotation = rotate_points(np.eye(2, dtype=np.float32), angle).T
        best = (-1.0, np.zeros(2))
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                shift = np.array((dx, dy))
                affine = np.column_stack(
                    (rotation, origin + shift - rotation @ model_origin)
                )
                size = (mask.shape[1], mask.shape[0])
                expected = cv2.warpAffine(self.model, affine, size) > 127
                trusted = (
                    cv2.warpAffine(
                        self.model_valid.astype(np.uint8),
                        affine,
                        size,
                        flags=cv2.INTER_NEAREST,
                    )
                    > 0
                )
                usable = valid & trusted
                actual = mask > 0
                union = np.count_nonzero((expected | actual) & usable)
                score = np.count_nonzero(expected & actual & usable) / max(
                    1, union
                ) - 0.0001 * np.linalg.norm(shift)
                if score > best[0]:
                    best = (score, shift)
        shift = best[1]
        affine = np.array(((1.0, 0.0, -shift[0]), (0.0, 1.0, -shift[1])))
        size = (mask.shape[1], mask.shape[0])
        aligned_mask = cv2.warpAffine(mask, affine, size)
        aligned_valid = (
            cv2.warpAffine(
                valid.astype(np.uint8), affine, size, flags=cv2.INTER_NEAREST
            )
            > 0
        )
        point = np.asarray(center) + shift
        return aligned_mask, aligned_valid, tuple(float(v) for v in point)

    def _fuse_white(self, observation: tuple, angle: float) -> None:
        mask, valid, _ = observation
        old_origin = ((mask.shape[1] - 1) / 2, (mask.shape[0] - 1) / 2)
        new_origin = np.array(
            ((self.model.shape[1] - 1) / 2, (self.model.shape[0] - 1) / 2)
        )
        affine = cv2.getRotationMatrix2D(old_origin, angle, 1.0)
        affine[:, 2] += new_origin - old_origin
        size = (self.model.shape[1], self.model.shape[0])
        aligned = cv2.warpAffine((mask > 0).astype(np.float32), affine, size)
        usable = (
            cv2.warpAffine(
                valid.astype(np.uint8), affine, size, flags=cv2.INTER_NEAREST
            )
            > 0
        )
        overlap = ((aligned > 0.5) & (self.model > 0) & usable).sum()
        union = (((aligned > 0.5) | (self.model > 0)) & usable).sum()
        if union == 0 or overlap / union < 0.70:
            return
        self.model_sum += aligned * usable
        self.model_count += usable
        self.model = (self.model_sum / np.maximum(self.model_count, 1) > 0.5).astype(
            np.uint8
        ) * 255
        self.model_valid = self.model_count > 0
        self.white_samples += 1
        self._build_geometry()
        if self.fix_template:
            new_period = self._symmetry_period()
            if new_period != self.period:
                self.period = new_period
                self.pose_hypotheses = [self.angle]
                self.angle_history.clear()
                self.angle_history.append((self.last_timestamp, self.angle))
            self.coverage_angles.add(round((angle % 360) / 15) % 24)
            known = np.mean(self.model_valid)
            required = 1 if self.period == 0 else max(3, math.ceil(self.period / 45))
            self.template_ready = known > 0.95 and len(self.coverage_angles) >= required

    def _reset_recovery(self) -> None:
        self.pending_recovery = self.pending_since = self.pending_last_at = None
        self.pending_angle = self.pending_groups = None
        self.pending_frames = 0
        self.pending_dynamic = 0.0
        self.pending_anchor = self.pending_anchor_angle = self.pending_bg_anchor = None
        self.pending_bg_angle = None

    def _recovery_evidence(
        self,
        center: np.ndarray,
        angle: float,
        groups: np.ndarray,
        dynamic: float,
        quality: float,
        white: bool,
        timestamp: float,
    ) -> bool:
        supported = groups >= 0.72
        gap = None if self.pending_last_at is None else timestamp - self.pending_last_at
        consistent = self.pending_recovery is not None and 0 < gap <= 0.16
        if consistent:
            distance = np.linalg.norm(center - self.pending_recovery)
            overlap = np.count_nonzero(supported & self.pending_groups) / max(
                1, np.count_nonzero(supported | self.pending_groups)
            )
            pose_error = abs(
                self._pose_delta(
                    angle, self.pending_angle + self.angular_velocity * gap
                )
            )
            consistent = (
                distance <= max(8.0, 180 * gap)
                and overlap >= 0.5
                and pose_error <= 12.0
            )
        if not consistent:
            self._reset_recovery()
            self.pending_since = timestamp
            self.pending_anchor = center.copy()
            self.pending_anchor_angle = angle
            self.pending_bg_anchor = center.copy()
            self.pending_bg_angle = angle
        self.pending_recovery = center.copy()
        self.pending_last_at = timestamp
        self.pending_angle = angle
        self.pending_groups = supported.copy()
        self.pending_frames += 1
        self.pending_dynamic += dynamic
        if quality < 0.78 or np.count_nonzero(supported) < 4:
            self._reset_recovery()
            return False
        duration = timestamp - self.pending_since
        reference = (
            self.pending_bg_anchor
            if self.current_background.get("background_status") == "accepted"
            else self.pending_anchor
        )
        displacement = np.linalg.norm(center - reference)
        reference_angle = (
            self.pending_bg_angle
            if self.current_background.get("background_status") == "accepted"
            else self.pending_anchor_angle
        )
        rotation = abs(self._pose_delta(angle, reference_angle))
        background_available = (
            self.current_background.get("background_status") == "accepted"
        )
        if background_available:
            foreground_motion = (
                max(displacement, rotation * math.pi / 180 * self.radius) >= 1.5
            )
            temporal = (
                foreground_motion and self.pending_dynamic / self.pending_frames >= 0.06
            )
        else:
            temporal = quality >= 0.84
        return self.pending_frames >= 3 and duration >= 0.10 and (white or temporal)

    def update(
        self,
        frame: np.ndarray,
        timestamp: float,
        invalid_mask: np.ndarray | None = None,
    ) -> TrackingResult:
        if self.center is None:
            raise RuntimeError("initialize the tracker before update")
        if (
            frame.ndim != 3
            or frame.shape[2] != 3
            or not frame.size
            or frame.dtype != np.uint8
        ):
            raise ValueError("frame must be a non-empty uint8 BGR image")
        if not math.isfinite(timestamp) or timestamp <= self.last_timestamp:
            raise ValueError("timestamps must be finite and strictly increasing")
        if invalid_mask is not None and invalid_mask.shape != frame.shape[:2]:
            raise ValueError("invalid mask must match input frame")
        started = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        valid = (
            np.ones(gray.shape, bool)
            if invalid_mask is None
            else ~invalid_mask.astype(bool)
        )
        valid[:5] = valid[-5:] = False
        valid[:, :5] = valid[:, -5:] = False
        dt = float(timestamp - self.last_timestamp)
        gap = float(timestamp - self.last_observed_at)
        if self.stage >= 3 and gap > self.coast_seconds:
            self.velocity *= math.exp(-dt / 0.25)
        predicted = self.center + self.velocity * dt
        predicted_angle = self.angle + self.angular_velocity * dt
        uncertainty = min(2.5 * self.radius, 8 + 25 * gap)
        search_radius = min(
            2.5 * self.radius, max(12.0, 0.40 * self.radius) + uncertainty
        )
        residual, background = self.motion.observe(gray, valid, predicted, self.radius)
        self.current_background = background
        if (
            self.temporal_recovery
            and self.pending_bg_anchor is not None
            and self.motion.affine is not None
        ):
            self.pending_bg_anchor = (
                self.motion.affine[:, :2] @ self.pending_bg_anchor
                + self.motion.affine[:, 2]
            )
            self.pending_bg_angle += math.degrees(
                math.atan2(self.motion.affine[1, 0], self.motion.affine[0, 0])
            )
        flow_at = time.perf_counter()
        distances, magnitude = self._edge_maps(
            gray, valid, residual if self.temporal_recovery else None
        )
        feature_at = time.perf_counter()
        white = None
        if self.stage >= 2 and timestamp - self.last_white_at < 0.4:
            white = visible_template(frame, valid, predicted, self.radius)
            if white is not None:
                self.last_white_at = timestamp
        centers = self._centers(predicted, search_radius, residual)
        if white is not None:
            centers = np.vstack((centers, np.array(white[2])))
        if self.period == 0:
            angles = [0.0]
        else:
            window = min(30.0, max(10.0, self.pose_uncertainty) + 12 * gap)
            step = 4.0 if self.stage >= 4 else 3.0
            angles = (
                predicted_angle
                + np.arange(-math.ceil(window / step), math.ceil(window / step) + 1)
                * step
            )
            if self.stage >= 2:
                angles = np.unique(
                    np.concatenate(
                        (
                            angles,
                            [
                                angle + self.angular_velocity * dt
                                for angle in self.pose_hypotheses
                            ],
                        )
                    )
                )
        candidates = []
        for angle in angles:
            if not len(centers) or not len(self.points):
                break
            quality, dynamic, group_quality, visibility = self._score(
                centers, float(angle), distances, valid, residual
            )
            motion_weight = (
                0.16 if background["background_status"] == "accepted" else 0.0
            )
            prior = 0.02 * np.exp(
                -np.linalg.norm(centers - predicted, axis=1) / max(self.radius, 1)
            )
            scores = quality + motion_weight * dynamic + prior
            for index in np.argsort(scores)[-4:]:
                candidates.append(
                    (
                        float(scores[index]),
                        centers[index],
                        float(angle),
                        float(quality[index]),
                        float(dynamic[index]),
                        group_quality[index],
                        float(visibility[index]),
                    )
                )
        candidates.sort(key=lambda item: item[0], reverse=True)
        if self.stage >= 4 and candidates:
            refined = []
            xs, ys = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
            offsets = np.column_stack((xs.ravel(), ys.ravel()))
            for item in candidates[:3]:
                local_centers = item[1] + offsets
                quality, dynamic, groups, visibility = self._score(
                    local_centers, item[2], distances, valid, residual
                )
                merit = (
                    quality
                    + (0.16 if background["background_status"] == "accepted" else 0)
                    * dynamic
                )
                index = int(np.argmax(merit))
                refined.append(
                    (
                        float(merit[index]),
                        local_centers[index],
                        item[2],
                        float(quality[index]),
                        float(dynamic[index]),
                        groups[index],
                        float(visibility[index]),
                    )
                )
            candidates = sorted(
                refined + candidates, key=lambda item: item[0], reverse=True
            )
        best = candidates[0] if candidates else None
        margin = 0.0
        votes = 0
        reason = "no_candidates"
        accepted = False
        if best is not None:
            rival = max(
                (
                    item[0]
                    for item in candidates
                    if np.linalg.norm(item[1] - best[1]) > 7
                ),
                default=0.0,
            )
            margin = best[0] - rival
            cutoff = 0.72 if self.local_voting else 0.68
            votes = int(np.count_nonzero(best[5] >= cutoff))
            required = 3 if self.stage < 3 else max(3, math.ceil(0.4 * len(best[5])))
            spread = np.flatnonzero(best[5] >= cutoff)
            distributed = len(spread) >= 3 and (spread.max() - spread.min() >= 2)
            minimum_quality = 0.74 if self.local_voting else 0.70
            accepted = (
                best[3] >= minimum_quality and votes >= required and margin >= 0.025
            )
            if self.stage >= 3:
                accepted &= distributed and best[6] >= 0.35
            if self.local_voting:
                accepted &= np.mean(best[5]) >= 0.35
            reason = (
                "weak_shape"
                if best[3] < minimum_quality
                else "insufficient_fragments"
                if votes < required
                else "ambiguous_center"
                if margin < 0.025
                else "poor_visibility"
            )
            if magnitude[valid].max(initial=0) < 8:
                accepted, reason = False, "no_edges"
        white_used = False
        white_pose_confidence = 0.0
        if white is not None:
            white_angle, white_quality, white_pose_confidence = self._white_pose(
                white, predicted_angle
            )
            if self.fix_template:
                white = self._align_white_center(white, white_angle)
                white_angle, white_quality, white_pose_confidence = self._white_pose(
                    white, predicted_angle
                )
            if white_quality >= 0.78:
                best = (
                    white_quality,
                    np.array(white[2]),
                    white_angle,
                    white_quality,
                    0.0,
                    np.ones(self.fragment_count),
                    1.0,
                )
                votes = self.fragment_count
                accepted, white_used, reason = True, True, "white_contour"
                margin = max(margin, white_pose_confidence)
        recovered = False
        background_recovery_margin = 0.0
        if accepted and self.temporal_recovery and gap > self.coast_seconds:
            if (
                self.pending_bg_anchor is not None
                and background["background_status"] == "accepted"
                and not white_used
            ):
                bg_q, bg_dynamic, _, _ = self._score(
                    np.array([self.pending_bg_anchor], np.float32),
                    self.pending_bg_angle,
                    distances,
                    valid,
                    residual,
                )
                background_recovery_margin = best[0] - float(
                    bg_q[0] + 0.16 * bg_dynamic[0]
                )
                if background_recovery_margin < 0.025:
                    accepted, reason = False, "background_explanation"
                    self._reset_recovery()
            if accepted and not self._recovery_evidence(
                best[1], best[2], best[5], best[4], best[3], white_used, timestamp
            ):
                accepted, reason = False, "recovery_pending"
            elif accepted:
                recovered = True
                self.recovery_count += 1
                self._reset_recovery()
        elif accepted and self.stage >= 3 and gap > self.coast_seconds:
            if self.pending_recovery is None or np.linalg.norm(
                best[1] - self.pending_recovery
            ) > max(12, 0.4 * self.radius):
                self.pending_recovery, self.pending_since = best[1].copy(), timestamp
                accepted, reason = False, "recovery_pending"
            elif timestamp - self.pending_since < 0.06:
                self.pending_recovery = best[1].copy()
                accepted, reason = False, "recovery_pending"
            else:
                recovered = True
                self.recovery_count += 1
                self.pending_recovery = self.pending_since = None
        elif not accepted:
            if self.temporal_recovery:
                self._reset_recovery()
            else:
                self.pending_recovery = self.pending_since = None
        if accepted:
            measured = best[1].astype(np.float64)
            observation_dt = timestamp - self.last_observed_at
            velocity = (measured - self.last_observed_center) / observation_dt
            speed = np.linalg.norm(velocity)
            if speed > 350:
                velocity *= 350 / speed
            blend = 1 - math.exp(-observation_dt / 0.10)
            self.velocity = (1 - blend) * self.velocity + blend * velocity
            self.center = measured if recovered else 0.9 * measured + 0.1 * predicted
            delta = self._pose_delta(best[2], predicted_angle)
            self.angle = predicted_angle + delta
            if self.stage >= 2:
                self.pose_hypotheses = (
                    [self.angle] if white_used else [item[2] for item in candidates[:3]]
                )
                pose_rivals = [
                    item
                    for item in candidates
                    if abs(self._pose_delta(item[2], self.angle)) > 8
                ]
                self.pose_uncertainty = (
                    (6.0 if white_pose_confidence > 0.02 or self.period == 0 else 18.0)
                    if white_used
                    else (
                        6.0
                        if not pose_rivals or best[0] - pose_rivals[0][0] > 0.03
                        else 18.0
                    )
                )
            self.angle_history.append((timestamp, self.angle))
            history = [
                (t, angle) for t, angle in self.angle_history if timestamp - t < 0.5
            ]
            if (
                self.period != 0
                and len(history) >= 4
                and (self.stage < 2 or white is not None or self.pose_uncertainty <= 6)
            ):
                times = np.array([t for t, _ in history])
                values = np.array([a for _, a in history])
                times -= times.mean()
                rate = float(
                    np.dot(times, values - values.mean()) / np.dot(times, times)
                )
                self.angular_velocity = float(np.clip(rate, -180, 180))
            self.last_observed_at = timestamp
            self.last_observed_center = measured.copy()
            learning = (
                (not self.template_ready)
                if self.fix_template
                else self.white_samples < 30
            )
            if white_used and learning and self.pose_uncertainty <= 6:
                self._fuse_white(white, self.angle)
            status, reason = "observed", "white_contour" if white_used else "accepted"
        else:
            self.center, self.angle = predicted, predicted_angle
            status = "predicted" if gap <= self.coast_seconds else "lost"
        inside = (
            0 <= self.center[0] < gray.shape[1] and 0 <= self.center[1] < gray.shape[0]
        )
        if not inside:
            status, reason = "lost", "outside_frame"
        self.last_timestamp = timestamp
        self.motion.commit(gray, valid, self.center)
        finished = time.perf_counter()
        diagnostics = {
            **background,
            "reject_reason": reason,
            "effective_fragments": votes,
            "pose_uncertainty_degrees": self.pose_uncertainty,
            "candidate_count": len(centers),
            "actual_search_radius": float(
                np.max(np.linalg.norm(centers - predicted, axis=1))
            )
            if len(centers)
            else 0.0,
            "symmetry_period_degrees": self.period,
            "template_ready": int(self.template_ready),
            "template_pose_bins": len(self.coverage_angles),
            "recovery_pending_frames": self.pending_frames,
            "background_recovery_margin": background_recovery_margin,
            "white_samples": self.white_samples,
            "position_uncertainty_px": uncertainty,
            "recovered": int(recovered),
            "recovery_count": self.recovery_count,
            "motion_support": 0.0 if best is None else best[4],
            "valid_boundary_fraction": 0.0 if best is None else best[6],
            "background_ms": (flow_at - started) * 1000,
            "features_ms": (feature_at - flow_at) * 1000,
            "matching_ms": (finished - feature_at) * 1000,
        }
        return TrackingResult(
            status,
            None if status == "lost" else tuple(float(v) for v in self.center),
            tuple(float(v) for v in predicted),
            self.angle,
            self.angular_velocity,
            votes,
            self.fragment_count,
            0.0 if best is None else best[3],
            max(0.0, margin),
            0.0 if status != "observed" else min(1.0, best[3]),
            timestamp - self.last_observed_at,
            search_radius,
            diagnostics,
        )


@dataclass(frozen=True)
class SquareFragment:
    fragment_id: int
    points: np.ndarray
    normals: np.ndarray
    offset: np.ndarray
    strength: float


class OriginalVotingTracker:
    def __init__(self, *, grid_size: int = 3) -> None:
        if grid_size not in (3, 4):
            raise ValueError("original voting uses a 3x3 or 4x4 square grid")
        self.grid_size = grid_size
        self.coast_seconds = 0.45
        self.reset()

    def reset(self) -> None:
        self.geometry = OptimizedFragmentTracker("revised")
        self.fragments: list[SquareFragment] = []
        self.cache = {}
        self.center = None
        self.velocity = np.zeros(2, np.float64)
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.last_timestamp = None
        self.last_observed_at = 0.0
        self.last_observed_center = None
        self.last_white_at = 0.0
        self.angle_history = deque(maxlen=20)
        self.pending = None
        self.recovery_count = 0
        self.motion = BackgroundMotion()
        self.match_records = []
        self.winning_records = []
        self.vote_peaks = []
        self.output_trace = None
        self.last_vote_trace = None
        self.selected_angle = None
        self.flow_previous = None
        self.flow_previous_valid = None

    @property
    def radius(self) -> float:
        return self.geometry.radius

    @property
    def period(self) -> float:
        return self.geometry.period

    @property
    def fragment_count(self) -> int:
        return len(self.fragments)

    def initialize(
        self,
        template: np.ndarray,
        center: tuple[float, float],
        timestamp: float = 0.0,
        template_valid: np.ndarray | None = None,
    ) -> None:
        self.reset()
        self.geometry.initialize(template, center, timestamp, template_valid)
        self._build_fragments()
        if self.fragment_count < 3:
            self.reset()
            raise ValueError(
                "opening template has fewer than three independent valid boundary fragments"
            )
        self.center = np.asarray(center, np.float64).copy()
        self.last_observed_center = self.center.copy()
        self.last_timestamp = self.last_observed_at = self.last_white_at = float(
            timestamp
        )
        self.angle_history.append((float(timestamp), 0.0))
        self.output_trace = {
            "source": "explicit_initialization",
            "center": list(center),
        }

    def _build_fragments(self) -> None:
        model = self.geometry.model
        dx, dy, magnitude = FragmentVotingTracker._gradients(model)
        trusted = (
            cv2.erode(
                self.geometry.model_valid.astype(np.uint8), np.ones((5, 5), np.uint8)
            )
            > 0
        )
        rows, cols = np.nonzero((magnitude >= 80) & trusted)
        origin = np.array(((model.shape[1] - 1) / 2, (model.shape[0] - 1) / 2))
        side = math.ceil(max(model.shape) / self.grid_size)
        fragments = []
        for grid_y in range(self.grid_size):
            for grid_x in range(self.grid_size):
                selected = (cols // side == grid_x) & (rows // side == grid_y)
                count = np.count_nonzero(selected)
                if count < 10:
                    continue
                local_cols, local_rows = cols[selected], rows[selected]
                point_positions = np.column_stack((local_cols, local_rows)).astype(
                    np.float32
                )
                normals = np.column_stack(
                    (dx[local_rows, local_cols], dy[local_rows, local_cols])
                )
                normals /= np.maximum(magnitude[local_rows, local_cols, None], 1e-6)
                offset = (
                    np.array(
                        (grid_x * side + (side - 1) / 2, grid_y * side + (side - 1) / 2)
                    )
                    - origin
                )
                spread = np.std(normals, axis=0).sum()
                indices = np.linspace(0, count - 1, min(64, count)).astype(np.int32)
                fragments.append(
                    SquareFragment(
                        grid_y * self.grid_size + grid_x,
                        point_positions[indices] - origin - offset,
                        normals[indices],
                        offset.astype(np.float32),
                        float(np.clip(0.6 + spread / 3, 0.6, 1.0)),
                    )
                )
        self.fragments = fragments
        self.cache.clear()

    def _rotated_fragment(
        self, fragment: SquareFragment, angle: float
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (fragment.fragment_id, round(angle / 2) % 180)
        if key in self.cache:
            return self.cache[key]
        rotation = key[1] * 2.0
        points = np.rint(rotate_points(fragment.points, rotation)).astype(np.int32)
        half = max(2, int(np.max(np.abs(points)))) + 1
        kernel = np.zeros((8, 2 * half + 1, 2 * half + 1), np.float32)
        normals = rotate_points(fragment.normals, rotation)
        bins = (
            np.rint(np.arctan2(normals[:, 1], normals[:, 0]) * 8 / math.pi).astype(
                np.int32
            )
            % 8
        )
        kernel[bins, points[:, 1] + half, points[:, 0] + half] = 1.0
        kernel /= kernel.sum()
        offset = rotate_points(fragment.offset[None, :], rotation)[0]
        self.cache[key] = kernel, offset
        return kernel, offset

    def _rotation_map(
        self, gray: np.ndarray, valid: np.ndarray
    ) -> tuple[np.ndarray, float]:
        rotation = np.zeros(gray.shape, np.float32)
        if (
            self.flow_previous is None
            or self.flow_previous.shape != gray.shape
            or self.motion.affine is None
        ):
            return rotation, 0.0
        mask = self.flow_previous_valid.astype(np.uint8) * 255
        mask[:55] = 0
        points = cv2.goodFeaturesToTrack(
            self.flow_previous,
            maxCorners=100,
            qualityLevel=0.025,
            minDistance=6,
            mask=mask,
        )
        if points is None or len(points) < 6:
            return rotation, 0.0
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            self.flow_previous, gray, points, None, winSize=(15, 15), maxLevel=2
        )
        if moved is None:
            return rotation, 0.0
        returned, backward, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.flow_previous, moved, None, winSize=(15, 15), maxLevel=2
        )
        if returned is None:
            return rotation, 0.0
        source, target = points.reshape(-1, 2), moved.reshape(-1, 2)
        good = (status.ravel() > 0) & (backward.ravel() > 0)
        good &= np.isfinite(target).all(axis=1)
        good &= np.linalg.norm(returned.reshape(-1, 2) - source, axis=1) < 1.5
        expected = source @ self.motion.affine[:, :2].T + self.motion.affine[:, 2]
        relative = target - expected
        for index in np.flatnonzero(good):
            neighbors = good & (
                np.linalg.norm(source - source[index], axis=1)
                < max(12.0, self.radius * 0.65)
            )
            if np.count_nonzero(neighbors) < 5:
                continue
            local = source[neighbors] - source[index]
            fit, _, _, _ = np.linalg.lstsq(
                np.column_stack((local, np.ones(len(local)))),
                relative[neighbors],
                rcond=None,
            )
            curl = abs(float(fit[0, 1] - fit[1, 0]))
            if curl < 0.015:
                continue
            point = tuple(round(float(value)) for value in target[index])
            if (
                not 0 <= point[0] < gray.shape[1]
                or not 0 <= point[1] < gray.shape[0]
                or not valid[point[1], point[0]]
            ):
                continue
            cv2.circle(
                rotation,
                point,
                max(4, round(self.radius * 0.25)),
                min(1.0, curl / 0.08),
                -1,
            )
        rotation[~valid] = 0
        return cv2.GaussianBlur(rotation, (5, 5), 0.8), min(
            1.0, float(np.count_nonzero(good)) / 40
        )

    def _response_maps(
        self,
        fragment: SquareFragment,
        angle: float,
        search_center: np.ndarray,
        search_radius: float,
        features: np.ndarray,
        valid: np.ndarray,
        motion: np.ndarray,
        rotation: np.ndarray,
        *,
        trim: bool = False,
    ) -> tuple | None:
        kernel, offset = self._rotated_fragment(fragment, angle)
        size = kernel.shape[1]
        half = size // 2
        expected_point = search_center + offset
        low = np.floor(expected_point - search_radius - half).astype(np.int32)
        high = np.ceil(expected_point + search_radius + half + 1).astype(np.int32)
        reference = np.array((half, half))
        if trim:
            rows, cols = np.nonzero(kernel.sum(axis=0))
            minimum = np.array((cols.min(), rows.min()))
            maximum = np.array((cols.max() + 1, rows.max() + 1))
            kernel = kernel[:, minimum[1] : maximum[1], minimum[0] : maximum[0]]
            low += minimum
            high -= size - maximum
            reference -= minimum
        height, width = valid.shape
        crop_shape = (int(high[1] - low[1]), int(high[0] - low[0]))
        cropped_valid = np.zeros(crop_shape, np.float32)
        cropped_motion = np.zeros(crop_shape, np.float32)
        cropped_rotation = np.zeros(crop_shape, np.float32)
        x0, y0 = max(0, int(low[0])), max(0, int(low[1]))
        x1, y1 = min(width, int(high[0])), min(height, int(high[1]))
        if x1 <= x0 or y1 <= y0:
            return None
        destination = (slice(y0 - low[1], y1 - low[1]), slice(x0 - low[0], x1 - low[0]))
        cropped_valid[destination] = valid[y0:y1, x0:x1]
        cropped_motion[destination] = motion[y0:y1, x0:x1]
        cropped_rotation[destination] = rotation[y0:y1, x0:x1]
        footprint = kernel.sum(axis=0)
        response_key = None
        if trim:
            response_key = (
                tuple(low),
                tuple(high),
                footprint.shape,
                footprint.tobytes(),
            )
        reuse = self._cached_footprint(response_key)
        sparse = trim
        out_shape = (
            crop_shape[0] - kernel.shape[1] + 1,
            crop_shape[1] - kernel.shape[2] + 1,
        )
        if sparse:
            shape_response = np.zeros(out_shape, np.float32)
            coverage = np.zeros(out_shape, np.float32) if reuse is None else reuse[0]
            motion_response = (
                np.zeros(out_shape, np.float32) if reuse is None else reuse[1]
            )
            rotation_response = (
                np.zeros(out_shape, np.float32) if reuse is None else reuse[2]
            )
            orientation_crop = np.zeros((8, *crop_shape), np.float32)
            orientation_crop[:, destination[0], destination[1]] = features[
                :, y0:y1, x0:x1
            ]
            channels, rows, cols = np.nonzero(kernel)
            for channel, row, col in zip(channels, rows, cols):
                weight = kernel[channel, row, col]
                region = (
                    slice(row, row + out_shape[0]),
                    slice(col, col + out_shape[1]),
                )
                shape_response += orientation_crop[(channel, *region)] * weight
                if reuse is None:
                    coverage += cropped_valid[region] * weight
                    motion_response += cropped_motion[region] * weight
                    rotation_response += cropped_rotation[region] * weight
            shape_response /= np.maximum(coverage, 1e-6)
            if reuse is None:
                motion_response /= np.maximum(coverage, 1e-6)
                rotation_response /= np.maximum(coverage, 1e-6)
                self._store_footprint(
                    response_key, (coverage, motion_response, rotation_response)
                )
        elif reuse is None:
            coverage = cv2.matchTemplate(cropped_valid, footprint, cv2.TM_CCORR)
        else:
            coverage = reuse[0]
        if not sparse:
            shape_response = np.zeros(coverage.shape, np.float32)
        for orientation in () if sparse else range(8):
            if not np.any(kernel[orientation]):
                continue
            crop = np.zeros(crop_shape, np.float32)
            crop[destination] = features[orientation, y0:y1, x0:x1]
            shape_response += cv2.matchTemplate(crop, kernel[orientation], cv2.TM_CCORR)
        if not sparse:
            shape_response /= np.maximum(coverage, 1e-6)
        if not sparse and reuse is None:
            motion_response = cv2.matchTemplate(
                cropped_motion, footprint, cv2.TM_CCORR
            ) / np.maximum(coverage, 1e-6)
            rotation_response = cv2.matchTemplate(
                cropped_rotation, footprint, cv2.TM_CCORR
            ) / np.maximum(coverage, 1e-6)
            self._store_footprint(
                response_key, (coverage, motion_response, rotation_response)
            )
        elif not sparse:
            motion_response, rotation_response = reuse[1], reuse[2]
        yy, xx = np.indices(shape_response.shape)
        match_x, match_y = xx + low[0] + reference[0], yy + low[1] + reference[1]
        center_x, center_y = match_x - offset[0], match_y - offset[1]
        in_gate = (center_x - search_center[0]) ** 2 + (
            center_y - search_center[1]
        ) ** 2 <= search_radius**2
        in_gate &= (
            (center_x >= 0) & (center_x < width) & (center_y >= 0) & (center_y < height)
        )
        return (
            shape_response,
            motion_response,
            rotation_response,
            coverage,
            in_gate,
            match_x,
            match_y,
            offset,
        )

    def _fragment_matches(
        self,
        fragment: SquareFragment,
        angle: float,
        search_center: np.ndarray,
        search_radius: float,
        features: np.ndarray,
        valid: np.ndarray,
        motion: np.ndarray,
        rotation: np.ndarray,
        motion_available: bool,
    ) -> list[Vote]:
        response = self._response_maps(
            fragment,
            angle,
            search_center,
            search_radius,
            features,
            valid,
            motion,
            rotation,
        )
        if response is None:
            return []
        (
            shape_response,
            motion_response,
            rotation_response,
            coverage,
            in_gate,
            match_x,
            match_y,
            offset,
        ) = response
        allowed = in_gate & (coverage >= 0.45) & (shape_response >= 0.72)
        if motion_available and self.last_timestamp - self.last_white_at >= 0.4:
            anomaly = (motion_response >= 0.04) | (rotation_response >= 0.03)
            if self.period == 0 and not np.any(anomaly & in_gate):
                allowed &= in_gate
            else:
                allowed &= anomaly
        objective = shape_response + (
            0.10 * motion_response + 0.05 * rotation_response if motion_available else 0
        )
        objective[~allowed] = -1
        votes = []
        for _ in range(3):
            _, peak, _, location = cv2.minMaxLoc(objective)
            if peak < 0:
                break
            x, y = location
            matched = np.array((match_x[y, x], match_y[y, x]), np.float64)
            center = matched - offset
            quality = min(1.0, float(shape_response[y, x]))
            votes.append(
                Vote(
                    fragment.fragment_id,
                    tuple(float(v) for v in center),
                    quality,
                    tuple(float(v) for v in matched),
                    tuple(float(v) for v in offset),
                    float(np.clip(motion_response[y, x], 0, 1)),
                    float(np.clip(rotation_response[y, x], 0, 1)),
                    float(np.clip(coverage[y, x], 0, 1)),
                )
            )
            cv2.circle(objective, location, 5, -1, -1)
        return votes

    @staticmethod
    def _supporting_votes(
        votes: list[Vote], peak: VotePeak, tolerance: float
    ) -> list[Vote]:
        selected = {}
        for vote in votes:
            if math.dist(vote.center, peak.center) > tolerance:
                continue
            if (
                vote.fragment_id not in selected
                or vote.score > selected[vote.fragment_id].score
            ):
                selected[vote.fragment_id] = vote
        return list(selected.values())

    def _cached_footprint(self, key):
        return None

    def _store_footprint(self, key, value) -> None:
        pass

    def _search_prediction(
        self, predicted: np.ndarray, predicted_angle: float, dt: float, timestamp: float
    ) -> tuple[np.ndarray, float]:
        if self.pending is not None and timestamp - self.pending["last_at"] <= 0.16:
            return self.pending["center"] + self.pending["velocity"] * dt, self.pending[
                "angle"
            ] + self.angular_velocity * dt
        return predicted.copy(), predicted_angle

    def _prepare_frame(
        self,
        frame: np.ndarray,
        valid: np.ndarray,
        features: np.ndarray,
        residual: np.ndarray,
        rotation: np.ndarray,
        timestamp: float,
    ) -> None:
        pass

    def _extra_diagnostics(self) -> dict:
        return {}

    def _vote_weight(self, vote: Vote) -> float:
        return vote.score

    def _associated_peaks(
        self, votes: list[Vote], tolerance: float
    ) -> list[tuple[VotePeak, list[Vote]]]:
        result = []
        for peak in aggregate_votes(votes, self.fragment_count, tolerance)[:3]:
            sources = self._supporting_votes(votes, peak, tolerance * 1.5)
            if not sources:
                continue
            weights = [self._vote_weight(vote) for vote in sources]
            center = np.average(
                [vote.center for vote in sources], axis=0, weights=weights
            )
            result.append(
                (
                    VotePeak(
                        tuple(float(value) for value in center),
                        len(sources),
                        float(np.mean([vote.score for vote in sources])),
                        float(sum(weights) / self.fragment_count),
                    ),
                    sources,
                )
            )
        return result

    def _submission_gate(
        self, best, predicted, predicted_angle, gap, dt, timestamp, background
    ) -> tuple[bool, str]:
        return True, "accepted_votes"

    def _trusted_angle(self, best, recovered: bool) -> bool:
        return True

    def _accept_peak(
        self, peak: VotePeak, supporting: list[Vote], margin: float
    ) -> tuple[bool, str]:
        if len(supporting) < 3:
            return False, "insufficient_independent_fragments"
        if peak.quality < 0.78:
            return False, "weak_fragment_matches"
        if margin < 0.035:
            return False, "ambiguous_vote_peak"
        offsets = np.array([vote.rotated_offset for vote in supporting])
        if (
            len(offsets) < 3
            or np.max(np.linalg.norm(offsets - offsets.mean(axis=0), axis=1))
            < 0.2 * self.radius
        ):
            return False, "clustered_fragment_sources"
        return True, "accepted_votes"

    def update(
        self,
        frame: np.ndarray,
        timestamp: float,
        invalid_mask: np.ndarray | None = None,
    ) -> TrackingResult:
        if self.center is None:
            raise RuntimeError("initialize before update")
        if (
            frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or not frame.size
        ):
            raise ValueError("frame must be a non-empty uint8 BGR image")
        if not math.isfinite(timestamp) or timestamp <= self.last_timestamp:
            raise ValueError("timestamps must be finite and strictly increasing")
        if invalid_mask is not None and invalid_mask.shape != frame.shape[:2]:
            raise ValueError("invalid mask must match frame")
        started = time.perf_counter()
        self.match_records, self.winning_records, self.vote_peaks = [], [], []
        self.selected_angle = None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        valid = (
            np.ones(gray.shape, bool)
            if invalid_mask is None
            else ~invalid_mask.astype(bool)
        )
        valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        valid[:5] = valid[-5:] = False
        valid[:, :5] = valid[:, -5:] = False
        dt = timestamp - self.last_timestamp
        gap = timestamp - self.last_observed_at
        if gap > self.coast_seconds:
            self.velocity *= math.exp(-dt / 0.25)
        predicted = self.center + self.velocity * dt
        predicted_angle = self.angle + self.angular_velocity * dt
        search_center, predicted_angle = self._search_prediction(
            predicted, predicted_angle, dt, timestamp
        )
        search_radius = min(2.0 * self.radius, max(10.0, 0.5 * self.radius) + 20 * gap)
        residual, background = self.motion.observe(
            gray, valid, search_center, self.radius
        )
        rotation, rotation_confidence = self._rotation_map(gray, valid)
        flow_at = time.perf_counter()
        distances, _ = self.geometry._edge_maps(gray, valid)
        features = np.exp(-0.5 * (distances / 1.8) ** 2).astype(np.float32)
        features[:, ~valid] = 0
        self._prepare_frame(frame, valid, features, residual, rotation, timestamp)
        feature_at = time.perf_counter()
        angles = (
            [0.0] if self.period == 0 else predicted_angle + np.arange(-12.0, 12.1, 4.0)
        )
        if gap > self.coast_seconds and self.period != 0:
            angles = predicted_angle + np.arange(-24.0, 24.1, 6.0)
        tolerance = max(3.0, min(5.0, 0.07 * self.radius))
        candidates = []
        all_votes = []
        for angle in angles:
            votes = []
            for fragment in self.fragments:
                votes.extend(
                    self._fragment_matches(
                        fragment,
                        float(angle),
                        search_center,
                        search_radius,
                        features,
                        valid,
                        residual,
                        rotation,
                        background["background_status"] == "accepted",
                    )
                )
            all_votes.extend((float(angle), vote) for vote in votes)
            for peak, sources in self._associated_peaks(votes, tolerance):
                dynamic = float(np.mean([vote.motion_support for vote in sources]))
                spin = float(np.mean([vote.rotation_support for vote in sources]))
                score = peak.evidence + 0.08 * dynamic + 0.04 * spin
                score -= 0.02 * min(
                    1.0,
                    abs(self.geometry._pose_delta(float(angle), predicted_angle))
                    / 24.0,
                )
                candidates.append((score, peak, float(angle), sources, dynamic, spin))
                self.vote_peaks.append(
                    {
                        "angle": float(angle),
                        "center": list(peak.center),
                        "votes": peak.votes,
                        "quality": peak.quality,
                        "evidence": peak.evidence,
                    }
                )
        candidates.sort(key=lambda item: item[0], reverse=True)
        best = candidates[0] if candidates else None
        margin, count, quality, dynamic, spin = 0.0, 0, 0.0, 0.0, 0.0
        reason = "no_fragment_matches"
        accepted, recovered = False, False
        if best is not None:
            rivals = [
                item[0]
                for item in candidates
                if math.dist(item[1].center, best[1].center) > tolerance * 2
            ]
            margin = best[0] - max(rivals, default=0.0)
            accepted, reason = self._accept_peak(best[1], best[3], margin)
            count, quality, dynamic, spin = (
                best[1].votes,
                best[1].quality,
                best[4],
                best[5],
            )
            self.selected_angle = best[2]
            self.winning_records = [
                self._vote_record(best[2], vote) for vote in best[3]
            ]
        if (
            accepted
            and gap > self.coast_seconds
            and background["background_status"] == "accepted"
            and (count < 4 or quality < 0.82)
        ):
            accepted, reason = False, "weak_recovery_votes"
        if accepted and gap > self.coast_seconds:
            point = np.asarray(best[1].center)
            continuous = (
                self.pending is not None and timestamp - self.pending["last_at"] <= 0.16
            )
            if continuous:
                continuous &= np.linalg.norm(
                    point - (self.pending["center"] + self.pending["velocity"] * dt)
                ) <= max(10.0, self.radius * 0.35)
                continuous &= (
                    abs(
                        self.geometry._pose_delta(
                            best[2], self.pending["angle"] + self.angular_velocity * dt
                        )
                    )
                    <= 12.0
                )
                current_ids = {vote.fragment_id for vote in best[3]}
                overlap = len(current_ids & self.pending["ids"]) / max(
                    1, len(current_ids | self.pending["ids"])
                )
                continuous &= overlap >= 0.4
            if not continuous:
                self.pending = {
                    "center": point.copy(),
                    "velocity": np.zeros(2),
                    "angle": best[2],
                    "ids": {vote.fragment_id for vote in best[3]},
                    "since": timestamp,
                    "last_at": timestamp,
                    "frames": 1,
                }
            else:
                self.pending["velocity"] = np.clip(
                    (point - self.pending["center"]) / dt, -250.0, 250.0
                )
                self.pending.update(
                    center=point.copy(),
                    angle=best[2],
                    last_at=timestamp,
                    ids={vote.fragment_id for vote in best[3]},
                    frames=self.pending["frames"] + 1,
                )
            if self.pending["frames"] < 3 or timestamp - self.pending["since"] < 0.08:
                accepted, reason = False, "vote_recovery_pending"
            elif (
                background["background_status"] == "accepted"
                and dynamic < 0.04
                and spin < 0.05
            ):
                accepted, reason = False, "recovery_without_relative_motion"
            else:
                recovered = True
        elif (
            not accepted
            and self.pending is not None
            and timestamp - self.pending["last_at"] > 0.16
        ):
            self.pending = None
        if accepted:
            accepted, reason = self._submission_gate(
                best, predicted, predicted_angle, gap, dt, timestamp, background
            )
        if not accepted:
            recovered = False
        if accepted:
            if recovered:
                self.recovery_count += 1
                self.pending = None
            measured = np.asarray(best[1].center, np.float64)
            observed_dt = timestamp - self.last_observed_at
            velocity = (measured - self.last_observed_center) / observed_dt
            speed = np.linalg.norm(velocity)
            if speed > 350:
                velocity *= 350 / speed
            blend = 1 - math.exp(-observed_dt / 0.10)
            self.velocity = (1 - blend) * self.velocity + blend * velocity
            self.center = measured if recovered else 0.9 * measured + 0.1 * predicted
            self.angle = predicted_angle + self.geometry._pose_delta(
                best[2], predicted_angle
            )
            self.last_observed_at = timestamp
            self.last_observed_center = measured.copy()
            self.output_trace = {
                "source": "fragment_votes",
                "timestamp": timestamp,
                "measured_center": list(best[1].center),
                "prediction": predicted.tolist(),
                "smoothed_center": self.center.tolist(),
                "recovered": recovered,
                "angle": best[2],
                "fragment_ids": [vote.fragment_id for vote in best[3]],
            }
            self.last_vote_trace = self.output_trace.copy()
            white = None
            if timestamp - self.last_white_at < 0.4:
                white = visible_template(frame, valid, measured, self.radius)
            white_trusted = False
            if white is not None:
                white = self.geometry._align_white_center(white, self.angle)
                white_angle, white_quality, certainty = self.geometry._white_pose(
                    white, self.angle
                )
                if white_quality >= 0.78:
                    white_trusted = True
                    self.last_white_at = timestamp
                    self.angle = predicted_angle + self.geometry._pose_delta(
                        white_angle, predicted_angle
                    )
                    self.geometry.angle, self.geometry.last_timestamp = (
                        self.angle,
                        timestamp,
                    )
                    if not self.geometry.template_ready and certainty >= 0.02:
                        self.geometry._fuse_white(white, self.angle)
                        self._build_fragments()
            trusted_angle = white_trusted or self._trusted_angle(best, recovered)
            if trusted_angle:
                self.angle_history.append((timestamp, self.angle))
            history = [(t, a) for t, a in self.angle_history if timestamp - t < 0.5]
            if trusted_angle and self.period != 0 and len(history) >= 4:
                times = np.array([t for t, _ in history])
                values = np.array([a for _, a in history])
                times -= times.mean()
                self.angular_velocity = float(
                    np.clip(
                        np.dot(times, values - values.mean()) / np.dot(times, times),
                        -180.0,
                        180.0,
                    )
                )
            status = "observed"
        else:
            self.center, self.angle = predicted, predicted_angle
            status = "predicted" if gap <= self.coast_seconds else "lost"
            if self.output_trace is not None:
                self.output_trace = {
                    "source": "prediction",
                    "timestamp": timestamp,
                    "last_observation_at": self.last_observed_at,
                    "last_observation": self.last_vote_trace,
                    "prediction": self.center.tolist(),
                }
        if (
            not 0 <= self.center[0] < frame.shape[1]
            or not 0 <= self.center[1] < frame.shape[0]
        ):
            status, reason = "lost", "outside_frame"
        self.match_records = [
            self._vote_record(angle, vote) for angle, vote in all_votes
        ]
        self.last_timestamp = timestamp
        self.motion.commit(gray, valid, self.center)
        self.flow_previous, self.flow_previous_valid = gray.copy(), valid.copy()
        finished = time.perf_counter()
        diagnostics = {
            **background,
            "reject_reason": reason,
            "effective_fragments": count,
            "candidate_count": len(all_votes),
            "white_samples": self.geometry.white_samples,
            "symmetry_period_degrees": self.period,
            "template_ready": int(self.geometry.template_ready),
            "template_pose_bins": len(self.geometry.coverage_angles),
            "recovered": int(recovered),
            "motion_support": dynamic,
            "rotation_support": spin,
            "rotation_flow_confidence": rotation_confidence,
            "valid_boundary_fraction": float(
                np.mean([record["valid_support"] for record in self.winning_records])
            )
            if self.winning_records
            else 0.0,
            "fragment_matches": len(all_votes),
            "vote_peak_count": len(self.vote_peaks),
            "winning_fragment_ids": ",".join(
                str(record["fragment_id"]) for record in self.winning_records
            ),
            "recovery_pending_frames": 0
            if self.pending is None
            else self.pending["frames"],
            "background_ms": (flow_at - started) * 1000,
            "features_ms": (feature_at - flow_at) * 1000,
            "matching_ms": (finished - feature_at) * 1000,
            **self._extra_diagnostics(),
        }
        return TrackingResult(
            status,
            None if status == "lost" else tuple(float(v) for v in self.center),
            tuple(float(v) for v in predicted),
            self.angle,
            self.angular_velocity,
            count,
            self.fragment_count,
            quality,
            max(0.0, margin),
            min(1.0, quality) if status == "observed" else 0.0,
            timestamp - self.last_observed_at,
            search_radius,
            diagnostics,
        )

    @staticmethod
    def _vote_record(angle: float, vote: Vote) -> dict:
        return {
            "angle": angle,
            "fragment_id": vote.fragment_id,
            "matched_point": list(vote.matched_point),
            "rotated_offset": list(vote.rotated_offset),
            "center": list(vote.center),
            "quality": vote.score,
            "raw_quality": vote.score if vote.raw_score is None else vote.raw_score,
            "peak_margin": vote.peak_margin,
            "fragment_weight": vote.fragment_weight,
            "motion_support": vote.motion_support,
            "rotation_support": vote.rotation_support,
            "valid_support": vote.valid_support,
        }

    def draw_votes(self, frame: np.ndarray) -> None:
        for record in self.winning_records:
            point = tuple(round(v) for v in record["matched_point"])
            center = tuple(round(v) for v in record["center"])
            cv2.circle(frame, point, 3, (255, 150, 80), -1)
            cv2.line(frame, point, center, (200, 100, 255), 1)
            cv2.putText(
                frame,
                str(record["fragment_id"]),
                (point[0] + 3, point[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 200, 80),
                1,
            )
        if self.winning_records:
            votes = np.array([record["center"] for record in self.winning_records])
            peak = tuple(round(float(v)) for v in votes.mean(axis=0))
            cv2.circle(frame, peak, 7, (180, 80, 255), 1)


GUARDED_VARIANTS = (
    "original_p0",
    "original_p1",
    "original_p2",
    "original_p3",
    "original_guarded",
)


class GuardedVotingTracker(OriginalVotingTracker):
    def __init__(
        self, variant: str = "original_guarded", *, grid_size: int = 3
    ) -> None:
        if variant not in GUARDED_VARIANTS:
            raise ValueError("invalid guarded voting variant")
        self.variant = variant
        self.guard_stage = GUARDED_VARIANTS.index(variant)
        super().__init__(grid_size=grid_size)

    def reset(self) -> None:
        super().reset()
        self.fragment_calibration = {}
        self.calibration_samples = 0
        self.transition = None
        self.last_committed_ids = set()
        self.last_committed_angle = 0.0
        self.innovation = 0.0
        self.position_uncertainty = 3.0
        self.current_valid = None
        self.current_features = None
        self.current_residual = None
        self.current_rotation = None
        self.local_motion_quality = 0.0
        self.local_spin = 0.0
        self.local_expansion = 0.0
        self.local_shear = 0.0
        self.response_cache = {}
        self.calibration_at = -math.inf
        self.submissions_delayed = 0
        self.source_rms = 0.0

    def _search_prediction(self, predicted, predicted_angle, dt, timestamp):
        center, angle = super()._search_prediction(
            predicted, predicted_angle, dt, timestamp
        )
        if (
            self.transition is not None
            and timestamp - self.transition["last_at"] <= 0.18
        ):
            elapsed = timestamp - self.transition["last_at"]
            return self.transition["center"] + self.transition[
                "velocity"
            ] * elapsed, self.transition["angle"] + self.transition[
                "angular_velocity"
            ] * elapsed
        return center, angle

    def _prepare_frame(
        self, frame, valid, features, residual, rotation, timestamp
    ) -> None:
        self.current_valid, self.current_features = valid, features
        self.current_residual, self.current_rotation = residual, rotation
        self.response_cache.clear()
        self.innovation = 0.0
        gap = timestamp - self.last_observed_at
        self.position_uncertainty = min(0.4 * self.radius, 3.0 + 14 * gap)
        if (
            timestamp - self.last_white_at > 0.12
            or timestamp - self.calibration_at < 0.18
        ):
            return
        self.calibration_at = timestamp
        height, width = valid.shape
        background_valid = valid.copy().astype(np.uint8)
        cv2.circle(
            background_valid,
            tuple(round(float(value)) for value in self.center),
            round(1.8 * self.radius),
            0,
            -1,
        )
        background_valid[:55] = 0
        background_valid = background_valid > 0
        positions = np.array(
            (
                (0.2 * width, 0.35 * height),
                (0.8 * width, 0.35 * height),
                (0.2 * width, 0.75 * height),
                (0.8 * width, 0.75 * height),
            )
        )
        for fragment in self.fragments:
            samples = []
            for point in positions:
                if np.linalg.norm(point - self.center) < 1.8 * self.radius:
                    continue
                response = self._response_maps(
                    fragment,
                    0.0,
                    point,
                    9.0,
                    features,
                    background_valid,
                    np.zeros(valid.shape, np.float32),
                    np.zeros(valid.shape, np.float32),
                    trim=self.guard_stage >= 4,
                )
                if response is None:
                    continue
                shape, _, _, coverage, gate, _, _, _ = response
                usable = gate & (coverage >= 0.95)
                if np.any(usable):
                    samples.extend(shape[usable].ravel()[::4].tolist())
            if not samples:
                continue
            threshold = float(np.percentile(samples, 95))
            old = self.fragment_calibration.get(fragment.fragment_id)
            self.fragment_calibration[fragment.fragment_id] = (
                threshold if old is None else 0.8 * old + 0.2 * threshold
            )
        self.calibration_samples += 1
        self.response_cache.clear()

    def _fragment_matches(
        self,
        fragment,
        angle,
        search_center,
        search_radius,
        features,
        valid,
        motion,
        rotation,
        motion_available,
    ):
        response = self._response_maps(
            fragment,
            angle,
            search_center,
            search_radius,
            features,
            valid,
            motion,
            rotation,
            trim=self.guard_stage >= 4,
        )
        if response is None:
            return []
        shape, dynamic, spin, coverage, gate, mx, my, offset = response
        allowed = gate & (coverage >= 0.45) & (shape >= 0.72)
        if motion_available and self.last_timestamp - self.last_white_at >= 0.4:
            anomaly = (dynamic >= 0.04) | (spin >= 0.03)
            if self.period != 0 or np.any(anomaly & gate):
                allowed &= anomaly
        objective = shape + (0.10 * dynamic + 0.05 * spin if motion_available else 0)
        objective = objective.copy()
        objective[~allowed] = -1
        matches = []
        for _ in range(3):
            _, peak, _, location = cv2.minMaxLoc(objective)
            if peak < 0:
                break
            x, y = location
            point = (float(mx[y, x]), float(my[y, x]))
            center = (point[0] - float(offset[0]), point[1] - float(offset[1]))
            copy = objective.copy()
            cv2.circle(copy, location, 7, -1, -1)
            second = float(cv2.minMaxLoc(copy)[1])
            uniqueness = max(0.0, float(peak) - max(0.0, second))
            raw = float(np.clip(shape[y, x], 0, 1))
            floor = self.fragment_calibration.get(fragment.fragment_id, 0.50)
            contrast = float(np.clip((raw - floor) / max(0.15, 1 - floor), 0, 1))
            weight = (
                fragment.strength
                * (0.35 + 0.65 * contrast)
                * (0.65 + 0.35 * min(1.0, uniqueness / 0.08))
            )
            matches.append(
                Vote(
                    fragment.fragment_id,
                    center,
                    raw,
                    point,
                    tuple(float(v) for v in offset),
                    float(np.clip(dynamic[y, x], 0, 1)),
                    float(np.clip(spin[y, x], 0, 1)),
                    float(np.clip(coverage[y, x], 0, 1)),
                    raw,
                    uniqueness,
                    float(weight),
                )
            )
            cv2.circle(objective, location, 5, -1, -1)
        return matches

    def _cached_footprint(self, key):
        return self.response_cache.get(key) if key is not None else None

    def _store_footprint(self, key, value) -> None:
        if key is not None:
            self.response_cache[key] = value

    def _vote_weight(self, vote):
        return max(0.05, vote.score * vote.fragment_weight)

    def _associated_peaks(self, votes, tolerance):
        if self.guard_stage < 2:
            return super()._associated_peaks(votes, tolerance)
        candidates = []
        for seed in votes:
            sources = {}
            for vote in votes:
                if math.dist(vote.center, seed.center) > tolerance:
                    continue
                old = sources.get(vote.fragment_id)
                if old is None or self._vote_weight(vote) > self._vote_weight(old):
                    sources[vote.fragment_id] = vote
            values = list(sources.values())
            values.sort(key=self._vote_weight, reverse=True)
            unique_edges = []
            for vote in values:
                if vote.matched_point is not None and any(
                    other.matched_point is not None
                    and math.dist(vote.matched_point, other.matched_point) < 5.0
                    for other in unique_edges
                ):
                    continue
                unique_edges.append(vote)
            values = unique_edges
            if not values:
                continue
            weights = [self._vote_weight(v) for v in values]
            center = np.average([v.center for v in values], axis=0, weights=weights)
            values = [v for v in values if math.dist(v.center, center) <= tolerance]
            if not values:
                continue
            weights = [self._vote_weight(v) for v in values]
            center = np.average([v.center for v in values], axis=0, weights=weights)
            peak = VotePeak(
                tuple(float(v) for v in center),
                len(values),
                float(np.mean([v.score for v in values])),
                float(sum(weights) / self.fragment_count),
            )
            candidates.append((peak, values))
        candidates.sort(key=lambda item: item[0].evidence, reverse=True)
        distinct = []
        for item in candidates:
            if all(
                math.dist(item[0].center, other[0].center) > tolerance
                for other in distinct
            ):
                distinct.append(item)
        return distinct[:3]

    def _accept_peak(self, peak, sources, margin):
        accepted, reason = super()._accept_peak(peak, sources, margin)
        if not accepted:
            return accepted, reason
        effective = sum(v.fragment_weight for v in sources)
        if effective < 1.3:
            return False, "low_fragment_distinctiveness"
        if self.guard_stage >= 2:
            centers = np.array([v.center for v in sources])
            weights = np.array([self._vote_weight(v) for v in sources])
            mean = np.average(centers, axis=0, weights=weights)
            self.source_rms = float(
                np.sqrt(
                    np.average(np.sum((centers - mean) ** 2, axis=1), weights=weights)
                )
            )
            if self.source_rms > max(2.5, 0.06 * self.radius):
                return False, "inconsistent_vote_geometry"
            offsets = np.array([v.rotated_offset for v in sources])
            singular = np.linalg.svd(offsets - offsets.mean(axis=0), compute_uv=False)
            if singular[-1] / max(singular[0], 1e-6) < 0.10:
                return False, "collinear_fragment_sources"
        return True, "accepted_votes"

    def _submission_gate(
        self, best, predicted, predicted_angle, gap, dt, timestamp, background
    ):
        sources = best[3]
        if (
            self.transition is not None
            and timestamp - self.transition["last_at"] > 0.18
        ):
            self.transition = None
        if self.guard_stage < 1:
            self.last_committed_ids = {v.fragment_id for v in sources}
            return True, "accepted_votes"
        measured = np.array(best[1].center)
        self.innovation = float(np.linalg.norm(measured - predicted))
        ids = {v.fragment_id for v in sources}
        overlap = (
            len(ids & self.last_committed_ids)
            / max(1, len(ids | self.last_committed_ids))
            if self.last_committed_ids
            else 1.0
        )
        pose_error = abs(self.geometry._pose_delta(best[2], predicted_angle))
        gate = max(
            6.0,
            min(
                0.3 * self.radius,
                self.position_uncertainty + np.linalg.norm(self.velocity) * dt * 0.2,
            ),
        )
        suspicious = (
            self.innovation > gate
            or pose_error > 10
            or (overlap < 0.35 and self.innovation > 4)
        )
        if self.transition is None and not suspicious:
            self.last_committed_ids = ids
            self.last_committed_angle = best[2]
            return True, "accepted_votes"
        old = self.transition
        consistent = old is not None and 0 < timestamp - old["last_at"] <= 0.18
        if consistent:
            elapsed = timestamp - old["last_at"]
            tolerance = max(6.0, 0.15 * self.radius)
            if old["frames"] == 1:
                consistent &= (
                    np.linalg.norm(measured - old["center"])
                    <= 350 * elapsed + tolerance
                )
                consistent &= (
                    abs(self.geometry._pose_delta(best[2], old["angle"]))
                    <= 180 * elapsed + 10
                )
            else:
                consistent &= (
                    np.linalg.norm(
                        measured - (old["center"] + old["velocity"] * elapsed)
                    )
                    <= tolerance
                )
                consistent &= (
                    abs(
                        self.geometry._pose_delta(
                            best[2], old["angle"] + old["angular_velocity"] * elapsed
                        )
                    )
                    <= 10
                )
            consistent &= len(ids & old["ids"]) / max(1, len(ids | old["ids"])) >= 0.4
        if not consistent:
            self.transition = {
                "center": measured.copy(),
                "velocity": np.zeros(2),
                "angle": best[2],
                "angular_velocity": self.angular_velocity,
                "ids": ids,
                "since": timestamp,
                "last_at": timestamp,
                "frames": 1,
            }
        else:
            elapsed = timestamp - old["last_at"]
            speed = (measured - old["center"]) / elapsed
            speed *= min(1.0, 350 / max(float(np.linalg.norm(speed)), 1e-6))
            angular_speed = float(
                np.clip(
                    self.geometry._pose_delta(best[2], old["angle"]) / elapsed,
                    -180,
                    180,
                )
            )
            blend = 1.0 if old["frames"] == 1 else 1 - math.exp(-elapsed / 0.10)
            self.transition.update(
                center=measured.copy(),
                velocity=(1 - blend) * old["velocity"] + blend * speed,
                angle=best[2],
                angular_velocity=(1 - blend) * old["angular_velocity"]
                + blend * angular_speed,
                ids=ids,
                last_at=timestamp,
                frames=old["frames"] + 1,
            )
        if self.transition["frames"] < 3 or timestamp - self.transition["since"] < 0.06:
            self.submissions_delayed += 1
            return False, "trajectory_submission_pending"
        if (
            self.guard_stage >= 3
            and timestamp - self.last_white_at > 0.4
            and background.get("background_status") == "accepted"
            and (
                self.local_motion_quality < 0.25
                or (best[4] < 0.10 and best[5] < 0.05)
                or abs(self.local_expansion) + self.local_shear
                > max(0.04, 2 * abs(self.local_spin))
            )
        ):
            return False, "unreliable_transition_motion"
        self.last_committed_ids = ids
        self.last_committed_angle = best[2]
        self.transition = None
        return True, "confirmed_trajectory_submission"

    def _trusted_angle(self, best, recovered):
        if self.guard_stage < 3:
            return True
        return (
            len(best[3]) >= 5
            and best[1].quality >= 0.85
            and self.local_motion_quality >= 0.35
            and not recovered
        )

    def _rotation_map(self, gray, valid):
        if self.guard_stage < 3:
            return super()._rotation_map(gray, valid)
        field = np.zeros(gray.shape, np.float32)
        self.local_motion_quality = self.local_spin = self.local_expansion = (
            self.local_shear
        ) = 0.0
        if self.flow_previous is None or self.motion.affine is None:
            return field, 0.0
        source_mask = self.flow_previous_valid.astype(np.uint8) * 255
        yy, xx = np.indices(gray.shape)
        source_mask[
            np.hypot(xx - self.center[0], yy - self.center[1]) > 1.4 * self.radius
        ] = 0
        source_mask[:55] = 0
        points = cv2.goodFeaturesToTrack(
            self.flow_previous,
            maxCorners=70,
            qualityLevel=0.02,
            minDistance=5,
            mask=source_mask,
        )
        if points is None or len(points) < 8:
            return field, 0.0
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            self.flow_previous, gray, points, None, winSize=(15, 15), maxLevel=2
        )
        if moved is None:
            return field, 0.0
        returned, backward, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.flow_previous, moved, None, winSize=(15, 15), maxLevel=2
        )
        if returned is None:
            return field, 0.0
        source, target = points.reshape(-1, 2), moved.reshape(-1, 2)
        selected = (
            (status.ravel() > 0)
            & (backward.ravel() > 0)
            & np.isfinite(target).all(axis=1)
        )
        selected &= np.linalg.norm(returned.reshape(-1, 2) - source, axis=1) < 1.2
        ix = np.clip(np.rint(target[:, 0]).astype(np.int32), 0, gray.shape[1] - 1)
        iy = np.clip(np.rint(target[:, 1]).astype(np.int32), 0, gray.shape[0] - 1)
        selected &= valid[iy, ix]
        source, target = source[selected], target[selected]
        if len(source) < 8:
            return field, 0.0
        affine, inliers = cv2.estimateAffine2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=200
        )
        if affine is None:
            return field, 0.0
        inlier = inliers.ravel() > 0
        fit = source @ affine[:, :2].T + affine[:, 2]
        error = float(np.median(np.linalg.norm(fit[inlier] - target[inlier], axis=1)))
        spread = np.ptp(source[inlier], axis=0)
        coverage = float(min(1.0, np.prod(spread) / max(1.0, self.radius**2)))
        reliability = float(np.mean(inlier) * coverage * math.exp(-error / 1.5))
        relative = affine[:, :2] @ np.linalg.inv(self.motion.affine[:, :2]) - np.eye(2)
        self.local_spin = float((relative[1, 0] - relative[0, 1]) * 0.5)
        self.local_expansion = float(np.trace(relative) * 0.5)
        self.local_shear = float(
            math.hypot(
                (relative[0, 0] - relative[1, 1]) * 0.5,
                (relative[0, 1] + relative[1, 0]) * 0.5,
            )
        )
        self.local_motion_quality = reliability
        if reliability >= 0.25 and abs(self.local_spin) > max(
            0.01, 1.2 * self.local_shear
        ):
            intensity = float(min(1.0, abs(self.local_spin) / 0.06) * reliability)
            for point in target[inlier]:
                cv2.circle(
                    field,
                    tuple(round(float(v)) for v in point),
                    max(4, round(0.2 * self.radius)),
                    intensity,
                    -1,
                )
        field[~valid] = 0
        return cv2.GaussianBlur(field, (5, 5), 0.8), reliability

    def _extra_diagnostics(self):
        return {
            "fragment_calibration_samples": self.calibration_samples,
            "trajectory_innovation_px": self.innovation,
            "position_uncertainty_px": self.position_uncertainty,
            "trajectory_pending_frames": 0
            if self.transition is None
            else self.transition["frames"],
            "trajectory_delayed_submissions": self.submissions_delayed,
            "vote_geometry_rms": self.source_rms,
            "local_motion_reliability": self.local_motion_quality,
            "local_spin": self.local_spin,
            "local_expansion": self.local_expansion,
            "local_shear": self.local_shear,
        }


MOTION_VARIANTS = ("motion_local", "motion_feedback", "motion_revised")


class MotionFeedbackTracker:
    def __init__(self, variant: str = "motion_feedback") -> None:
        if variant not in MOTION_VARIANTS:
            raise ValueError("invalid motion feedback variant")
        self.variant = variant
        self.coast_seconds = 0.45
        self.range_factor = 1.3
        self.max_speed = 350.0
        self.max_acceleration = 1800.0
        self.reset()

    def reset(self) -> None:
        self.center = None
        self.image_center = None
        self.image_velocity = np.zeros(2, np.float64)
        self.radius = 0.0
        self.velocity = np.zeros(2, np.float64)
        self.angle = self.angular_velocity = 0.0
        self.circular = False
        self.last_timestamp = self.last_observed_at = 0.0
        self.last_white_at = -math.inf
        self.previous_gray = self.previous_valid = self.previous_white = None
        self.motion = BackgroundMotion()
        self.pending = None
        self.recovery_count = 0
        self.flow_records = []
        self.hypothesis_records = []
        self.tracked_points = np.empty((0, 2), np.float32)
        self.flow_persistent = np.empty(0, bool)
        self.local_background_affine = None
        self.local_background_error = 0.0
        self.output_trace = None
        self.last_observation_trace = None

    @property
    def fragment_count(self) -> int:
        return 0

    @staticmethod
    def _bounded(vector: np.ndarray, limit: float) -> np.ndarray:
        return vector * min(1.0, limit / max(float(np.linalg.norm(vector)), 1e-8))

    def initialize(
        self,
        template: np.ndarray,
        center: tuple[float, float],
        timestamp: float = 0.0,
        template_valid: np.ndarray | None = None,
        initial_frame: np.ndarray | None = None,
        invalid_mask: np.ndarray | None = None,
    ) -> None:
        if (
            template.ndim != 2
            or min(template.shape) < 8
            or not np.isfinite(center).all()
            or not math.isfinite(timestamp)
            or np.count_nonzero(template) < 20
        ):
            raise ValueError("invalid initial silhouette, center or timestamp")
        if template_valid is not None and template_valid.shape != template.shape:
            raise ValueError("template validity must match silhouette")
        self.reset()
        yy, xx = np.nonzero(template)
        origin = np.array(((template.shape[1] - 1) / 2, (template.shape[0] - 1) / 2))
        self.radius = float(np.max(np.hypot(xx - origin[0], yy - origin[1])))
        self.center = np.array(center, np.float64)
        self.image_center = self.center.copy()
        self.last_timestamp = self.last_observed_at = self.last_white_at = timestamp
        area = float(np.count_nonzero(template))
        self.circular = area / max(1.0, math.pi * self.radius**2) > 0.87
        self.output_trace = {
            "source": "explicit_initialization",
            "center": list(center),
        }
        if initial_frame is not None:
            self._validate_frame(initial_frame, invalid_mask)
            gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
            valid = self._valid_pixels(gray.shape, invalid_mask)
            white = self._white_mask(initial_frame, valid)
            yy, xx = np.indices(gray.shape)
            seed_mask = (
                valid
                & (
                    np.hypot(xx - self.center[0], yy - self.center[1])
                    <= self.radius + 3
                )
                & white
            )
            seeds = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=90,
                qualityLevel=0.008,
                minDistance=4,
                mask=seed_mask.astype(np.uint8) * 255,
            )
            self.tracked_points = (
                np.empty((0, 2), np.float32) if seeds is None else seeds.reshape(-1, 2)
            )
            self._commit_frame(gray, valid, white)

    @staticmethod
    def _validate_frame(frame, invalid_mask):
        if (
            frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or not frame.size
        ):
            raise ValueError("frame must be non-empty uint8 BGR")
        if invalid_mask is not None and invalid_mask.shape != frame.shape[:2]:
            raise ValueError("invalid mask must match frame")

    @staticmethod
    def _valid_pixels(shape, invalid_mask):
        valid = (
            np.ones(shape, np.uint8)
            if invalid_mask is None
            else (~invalid_mask.astype(bool)).astype(np.uint8)
        )
        valid = cv2.erode(valid, np.ones((3, 3), np.uint8)) > 0
        valid[:5] = valid[-5:] = False
        valid[:, :5] = valid[:, -5:] = False
        return valid

    @staticmethod
    def _white_mask(frame, valid):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        saturation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
        return (gray >= 210) & (saturation <= 75) & valid

    def _commit_frame(self, gray, valid, white):
        self.previous_gray, self.previous_valid = gray.copy(), valid.copy()
        self.previous_white = white.copy()
        self.motion.commit(gray, valid, self.image_center)

    def _local_flow(self, gray, valid, anchor):
        empty = np.empty((0, 2), np.float32)
        if self.previous_gray is None:
            return empty, empty, np.empty(0), "flow_unavailable"
        if self.previous_gray.shape != gray.shape:
            return empty, empty, np.empty(0), "frame_size_changed"
        yy, xx = np.indices(gray.shape)
        distance = np.hypot(xx - anchor[0], yy - anchor[1])
        mask = self.previous_valid & (distance <= self.range_factor * self.radius)
        mask[:55] = False
        if (
            self.previous_white is not None
            and np.count_nonzero(self.previous_white & mask) > 200
        ):
            band = (
                cv2.dilate(
                    (self.previous_white & mask).astype(np.uint8),
                    np.ones((5, 5), np.uint8),
                )
                > 0
            )
            mask &= band
        points = cv2.goodFeaturesToTrack(
            self.previous_gray,
            maxCorners=180,
            qualityLevel=0.008,
            minDistance=4,
            mask=mask.astype(np.uint8) * 255,
        )
        fresh = empty if points is None else points.reshape(-1, 2)
        persistent = self.tracked_points
        if len(persistent):
            keep = (
                np.linalg.norm(persistent - anchor, axis=1)
                <= self.range_factor * self.radius
            )
            px = np.clip(persistent[:, 0].astype(int), 0, gray.shape[1] - 1)
            py = np.clip(persistent[:, 1].astype(int), 0, gray.shape[0] - 1)
            persistent = persistent[keep & self.previous_valid[py, px]]
            if len(persistent) and len(fresh):
                fresh = fresh[
                    np.min(
                        np.linalg.norm(fresh[:, None] - persistent[None, :], axis=2),
                        axis=1,
                    )
                    > 3
                ]
        ring = (
            self.previous_valid
            & (distance >= 1.5 * self.radius)
            & (distance <= 2.3 * self.radius)
        )
        ring[:55] = False
        controls = cv2.goodFeaturesToTrack(
            self.previous_gray,
            maxCorners=90,
            qualityLevel=0.015,
            minDistance=7,
            mask=ring.astype(np.uint8) * 255,
        )
        controls = empty if controls is None else controls.reshape(-1, 2)
        local_count = len(persistent) + len(fresh)
        points = (
            np.vstack((persistent, fresh, controls))
            .astype(np.float32)
            .reshape(-1, 1, 2)
        )
        if not len(points):
            return empty, empty, np.empty(0), "too_few_local_points"
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray, gray, points, None, winSize=(15, 15), maxLevel=3
        )
        if moved is None:
            return empty, empty, np.empty(0), "local_flow_failed"
        returned, backward, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.previous_gray, moved, None, winSize=(15, 15), maxLevel=3
        )
        if returned is None:
            return empty, empty, np.empty(0), "local_flow_failed"
        source, target = points.reshape(-1, 2), moved.reshape(-1, 2)
        fb = np.linalg.norm(returned.reshape(-1, 2) - source, axis=1)
        inside = (
            np.isfinite(target).all(axis=1)
            & (target[:, 0] >= 0)
            & (target[:, 0] < gray.shape[1])
            & (target[:, 1] >= 0)
            & (target[:, 1] < gray.shape[0])
        )
        ix = np.clip(np.nan_to_num(target[:, 0]).astype(int), 0, gray.shape[1] - 1)
        iy = np.clip(np.nan_to_num(target[:, 1]).astype(int), 0, gray.shape[0] - 1)
        good = (
            (status.ravel() > 0)
            & (backward.ravel() > 0)
            & inside
            & valid[iy, ix]
            & (fb < 1.0)
        )
        control_good = good & (np.arange(len(good)) >= local_count)
        self.local_background_affine = None
        if np.count_nonzero(control_good) >= 10:
            matrix, inliers = cv2.estimateAffinePartial2D(
                source[control_good],
                target[control_good],
                method=cv2.RANSAC,
                ransacReprojThreshold=1.2,
                maxIters=150,
            )
            if matrix is not None and np.mean(inliers) >= 0.65:
                control_source = source[control_good][inliers.ravel() > 0]
                control_target = target[control_good][inliers.ravel() > 0]
                residual = np.linalg.norm(
                    control_source @ matrix[:, :2].T + matrix[:, 2] - control_target,
                    axis=1,
                )
                if (
                    np.median(residual) < 0.8
                    and np.max(np.ptp(control_source, axis=0)) > self.radius
                ):
                    self.local_background_affine = matrix
                    self.local_background_error = float(np.median(residual))
        local_good = good & (np.arange(len(good)) < local_count)
        self.flow_persistent = (np.arange(len(good)) < len(persistent))[local_good]
        return (
            source[local_good],
            target[local_good],
            fb[local_good],
            "local_flow_ready",
        )

    def _motion_hypotheses(self, source, target, anchor, dt, background_error):
        self.hypothesis_records = []
        if len(source) < 6 or self.motion.affine is None:
            return [], "insufficient_motion_evidence"
        background_affine = (
            self.motion.affine
            if self.local_background_affine is None
            else self.local_background_affine
        )
        bg = source @ background_affine[:, :2].T + background_affine[:, 2]
        relative = target - bg
        threshold = max(0.65, 1.5 * max(background_error, self.local_background_error))
        dynamic = np.linalg.norm(relative, axis=1) > threshold
        if np.count_nonzero(dynamic) < 6:
            return [], "background_motion_only"
        selected = np.flatnonzero(dynamic)
        order = selected[np.argsort(np.linalg.norm(source[selected] - anchor, axis=1))]
        seeds = order[np.linspace(0, len(order) - 1, min(12, len(order))).astype(int)]
        hypotheses = []
        for seed in seeds:
            neighborhood = dynamic & (
                np.linalg.norm(source - source[seed], axis=1)
                <= max(18.0, 0.85 * self.radius)
            )
            if np.count_nonzero(neighborhood) < 6:
                continue
            affine, _ = cv2.estimateAffinePartial2D(
                source[neighborhood],
                target[neighborhood],
                method=cv2.RANSAC,
                ransacReprojThreshold=1.2,
                maxIters=160,
                refineIters=5,
            )
            if affine is None:
                continue
            fitted = source @ affine[:, :2].T + affine[:, 2]
            errors = np.linalg.norm(fitted - target, axis=1)
            support = dynamic & (errors <= 1.4)
            count = np.count_nonzero(support)
            if count < 6:
                continue
            points = source[support]
            spread = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
            if (
                spread[-1] / max(spread[0], 1e-6) < 0.12
                or np.max(np.ptp(points, axis=0)) < 0.35 * self.radius
            ):
                continue
            full, _ = cv2.estimateAffine2D(
                source[support],
                target[support],
                method=cv2.RANSAC,
                ransacReprojThreshold=1.2,
                maxIters=100,
            )
            if full is None:
                continue
            relative_linear = full[:, :2] @ np.linalg.inv(background_affine[:, :2])
            scales = np.linalg.svd(relative_linear, compute_uv=False)
            expansion = float(abs(math.sqrt(abs(np.linalg.det(relative_linear))) - 1))
            shear = float(abs(scales[0] - scales[1]))
            measured = affine[:, :2] @ anchor + affine[:, 2]
            rotation = math.degrees(math.atan2(affine[1, 0], affine[0, 0]))
            speed = float(np.linalg.norm(measured - anchor) / dt)
            error = float(np.median(errors[support]))
            coverage = min(
                1.0, float(np.prod(np.ptp(points, axis=0))) / max(1.0, self.radius**2)
            )
            bg_error = float(np.median(np.linalg.norm(relative[support], axis=1)))
            score = (
                coverage
                * min(1.0, count / 18)
                * math.exp(-error)
                * min(1.0, bg_error / (2 * threshold))
            )
            reason = "accepted_local_motion"
            if expansion > 0.045 or shear > 0.06:
                reason = "nonrigid_local_motion"
            elif speed > self.max_speed + 30 or abs(rotation) / dt > 200:
                reason = "excessive_local_motion"
            elif np.linalg.norm(measured - anchor) > max(
                0.3 * self.radius, self.max_speed * dt + 2
            ):
                reason = "outside_feedback_range"
            elif score < 0.18:
                reason = "weak_local_motion"
            elif (
                len(self.flow_persistent) == len(source)
                and self.last_timestamp - self.last_white_at > 0.12
                and np.count_nonzero(support & self.flow_persistent) < 3
            ):
                reason = "lost_motion_point_identity"
            record = {
                "center": measured.tolist(),
                "affine_matrix": affine.tolist(),
                "reference_center": anchor.tolist(),
                "support_indices": np.flatnonzero(support).tolist(),
                "support_points": int(count),
                "score": float(score),
                "fit_error": error,
                "background_difference": bg_error,
                "expansion": expansion,
                "shear": shear,
                "rotation_degrees": rotation,
                "reason": reason,
            }
            self.hypothesis_records.append(record)
            if reason != "accepted_local_motion":
                continue
            if any(np.linalg.norm(measured - h["center"]) < 2.5 for h in hypotheses):
                continue
            hypotheses.append(
                {**record, "affine": affine, "center": measured, "coverage": coverage}
            )
        hypotheses.sort(key=lambda h: h["score"], reverse=True)
        return hypotheses[
            :3
        ], "no_rigid_local_motion" if not hypotheses else "accepted_local_motion"

    def _feedback(self, measured, predicted, dt):
        innovation = measured - predicted
        dead_zone = max(0.5, 0.015 * self.radius)
        correction = self._bounded(innovation, 0.20 * self.radius)
        if np.linalg.norm(correction) <= dead_zone:
            correction[:] = 0
        desired = self._bounded((measured - self.image_center) / dt, self.max_speed)
        if self.variant == "motion_local":
            return measured.copy(), desired, correction
        blend = 1 - math.exp(-dt / 0.06)
        delta = self._bounded(
            blend * (desired - self.velocity), self.max_acceleration * dt
        )
        velocity = self._bounded(self.velocity + delta, self.max_speed)
        inertial = self.center + velocity * dt
        adjustment = self._bounded(
            (1 - math.exp(-dt / 0.08)) * correction, self.max_acceleration * dt**2
        )
        return inertial + adjustment, velocity, adjustment

    def update(self, frame, timestamp, invalid_mask=None):
        if self.center is None:
            raise RuntimeError("initialize before update")
        self._validate_frame(frame, invalid_mask)
        if not math.isfinite(timestamp) or timestamp <= self.last_timestamp:
            raise ValueError("timestamps must be finite and increasing")
        started = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        valid = self._valid_pixels(gray.shape, invalid_mask)
        white_pixels = self._white_mask(frame, valid)
        dt, gap = timestamp - self.last_timestamp, timestamp - self.last_observed_at
        self.flow_records, self.hypothesis_records = [], []
        self.local_background_affine = None
        self.local_background_error = 0.0
        self.flow_persistent = np.empty(0, bool)
        if gap > self.coast_seconds:
            self.velocity *= math.exp(-dt / 0.20)
            self.image_velocity *= math.exp(-dt / 0.20)
        predicted = self.center + self.velocity * dt
        image_prediction = self.image_center + self.image_velocity * dt
        anchor = self.image_center.copy()
        if self.pending is not None and timestamp - self.pending["last_at"] <= 0.18:
            anchor = self.pending["center"] + self.pending["velocity"] * (
                self.last_timestamp - self.pending["last_at"]
            )
        _, background = self.motion.observe(gray, valid, image_prediction, self.radius)
        bg_at = time.perf_counter()
        source, target, fb, reason = self._local_flow(gray, valid, anchor)
        white = (
            visible_template(frame, valid, image_prediction, self.radius)
            if timestamp - self.last_white_at <= 0.4
            else None
        )
        flow_at = time.perf_counter()
        best = None
        if white is not None:
            measured = np.array(white[2])
            if np.linalg.norm(measured - predicted) <= max(
                8.0, 0.35 * self.radius, self.max_speed * dt + 2
            ):
                best = {
                    "center": measured,
                    "score": 1.0,
                    "rotation_degrees": 0.0,
                    "coverage": 1.0,
                    "support_indices": [],
                    "support_points": 0,
                }
                reason = "visible_white_observation"
                self.last_white_at = timestamp
        if (
            best is not None
            and np.linalg.norm(best["center"] - predicted)
            > self.range_factor * self.radius
        ):
            best, reason = None, "outside_feedback_range"
        if best is None and background["background_status"] == "accepted":
            hypotheses, reason = self._motion_hypotheses(
                source, target, anchor, dt, background["background_fit_error"]
            )
            if hypotheses:
                best = hypotheses[0]
                if (
                    len(hypotheses) > 1
                    and hypotheses[0]["score"] - hypotheses[1]["score"] < 0.08
                ):
                    best, reason = None, "ambiguous_local_motion"
                elif (
                    np.linalg.norm(best["center"] - predicted)
                    > self.range_factor * self.radius
                ):
                    best, reason = None, "outside_feedback_range"
        elif best is None:
            reason = "untrusted_background_motion"
        chosen = set() if best is None else set(best["support_indices"])
        if best is not None and reason == "visible_white_observation":
            if len(target):
                tx, ty = (
                    np.clip(target[:, 0].astype(int), 0, gray.shape[1] - 1),
                    np.clip(target[:, 1].astype(int), 0, gray.shape[0] - 1),
                )
                chosen = set(np.flatnonzero(white_pixels[ty, tx]).tolist())
            self.tracked_points = target[sorted(chosen)].copy()
        elif chosen and len(self.flow_persistent) == len(target):
            retained = np.array(sorted(chosen), dtype=int)
            if timestamp - self.last_white_at <= 0.12:
                self.tracked_points = target[retained].copy()
            else:
                self.tracked_points = target[
                    retained[self.flow_persistent[retained]]
                ].copy()
        elif len(self.flow_persistent) == len(target):
            self.tracked_points = target[self.flow_persistent].copy()
        for i, (p, q) in enumerate(zip(source, target)):
            self.flow_records.append(
                {
                    "source": p.tolist(),
                    "target": q.tolist(),
                    "fb_error": float(fb[i]),
                    "selected": i in chosen,
                }
            )
        accepted = best is not None
        recovered = False
        if accepted and gap > self.coast_seconds:
            measured = best["center"]
            previous = self.pending
            continuous = (
                previous is not None and 0 < timestamp - previous["last_at"] <= 0.18
            )
            if continuous:
                elapsed = timestamp - previous["last_at"]
                tolerance = max(4.0, 0.10 * self.radius)
                if previous["frames"] == 1:
                    continuous = (
                        np.linalg.norm(measured - previous["center"])
                        <= self.max_speed * elapsed + tolerance
                    )
                else:
                    continuous = (
                        np.linalg.norm(
                            measured
                            - previous["center"]
                            - previous["velocity"] * elapsed
                        )
                        <= tolerance
                    )
            if not continuous:
                self.pending = {
                    "center": measured.copy(),
                    "velocity": np.zeros(2),
                    "frames": 1,
                    "since": timestamp,
                    "last_at": timestamp,
                }
            else:
                velocity = self._bounded(
                    (measured - previous["center"]) / (timestamp - previous["last_at"]),
                    self.max_speed,
                )
                self.pending.update(
                    center=measured.copy(),
                    velocity=velocity,
                    frames=previous["frames"] + 1,
                    last_at=timestamp,
                )
            accepted = (
                self.pending["frames"] >= 3
                and timestamp - self.pending["since"] >= 0.06
            )
            recovered = accepted
            if not accepted:
                reason = "local_recovery_pending"
        elif self.pending is not None and timestamp - self.pending["last_at"] > 0.18:
            self.pending = None
        feedback = np.zeros(2)
        if accepted:
            previous_image_center = self.image_center.copy()
            center, velocity, feedback = self._feedback(best["center"], predicted, dt)
            self.center, self.velocity = center, velocity
            self.image_velocity = self._bounded(
                (best["center"] - self.image_center) / dt, self.max_speed
            )
            self.image_center = best["center"].copy()
            self.last_observed_at = timestamp
            if not self.circular and reason != "visible_white_observation":
                self.angular_velocity = best["rotation_degrees"] / dt
                self.angle += best["rotation_degrees"]
            if recovered:
                self.recovery_count += 1
                self.pending = None
            self.output_trace = {
                "source": "white_observation"
                if reason == "visible_white_observation"
                else "local_motion",
                "timestamp": timestamp,
                "measured_center": best["center"].tolist(),
                "previous_image_center": previous_image_center.tolist(),
                "predicted_center": predicted.tolist(),
                "smoothed_center": self.center.tolist(),
                "feedback": feedback.tolist(),
                "support_indices": best["support_indices"],
                "affine_matrix": best.get("affine_matrix"),
                "reference_center": best.get("reference_center"),
                "recovered": recovered,
            }
            self.last_observation_trace = self.output_trace.copy()
            status = "observed"
        else:
            self.center = predicted
            self.image_center = image_prediction
            status = "predicted" if gap <= self.coast_seconds else "lost"
            self.output_trace = {
                "source": "prediction",
                "timestamp": timestamp,
                "last_observation_at": self.last_observed_at,
                "last_observation": self.last_observation_trace,
                "prediction": self.center.tolist(),
            }
        if not (
            0 <= self.center[0] < gray.shape[1] and 0 <= self.center[1] < gray.shape[0]
        ):
            status, reason = "lost", "outside_frame"
        if status == "lost" and recovered:
            self.recovery_count -= 1
            recovered = False
        self.last_timestamp = timestamp
        self._commit_frame(gray, valid, white_pixels)
        finished = time.perf_counter()
        diagnostics = {
            **background,
            "reject_reason": reason,
            "flow_points": len(source),
            "flow_support_points": 0 if best is None else best["support_points"],
            "flow_hypotheses": len(self.hypothesis_records),
            "persistent_flow_points": int(np.count_nonzero(self.flow_persistent)),
            "local_background_used": int(self.local_background_affine is not None),
            "flow_fit_error": 0.0 if best is None else best.get("fit_error", 0.0),
            "flow_background_difference": 0.0
            if best is None
            else best.get("background_difference", 0.0),
            "feedback_dx": float(feedback[0]),
            "feedback_dy": float(feedback[1]),
            "velocity_x": float(self.velocity[0]),
            "velocity_y": float(self.velocity[1]),
            "motion_support": 0.0 if best is None else best["score"],
            "recovered": int(recovered),
            "actual_search_radius": self.range_factor * self.radius,
            "symmetry_period_degrees": 0.0 if self.circular else 360.0,
            "recovery_pending_frames": 0
            if self.pending is None
            else self.pending["frames"],
            "background_ms": (bg_at - started) * 1000,
            "local_flow_ms": (flow_at - bg_at) * 1000,
            "feedback_ms": (finished - flow_at) * 1000,
        }
        return TrackingResult(
            status,
            None if status == "lost" else tuple(self.center),
            tuple(predicted),
            self.angle,
            self.angular_velocity,
            0,
            0,
            0.0 if best is None else best["score"],
            0.0,
            0.0 if not accepted else best["score"],
            timestamp - self.last_observed_at,
            self.range_factor * self.radius,
            diagnostics,
        )

    def draw_motion(self, frame):
        for record in self.flow_records:
            p, q = (
                tuple(round(float(v)) for v in record[key])
                for key in ("source", "target")
            )
            cv2.arrowedLine(
                frame,
                p,
                q,
                (0, 210, 255) if record["selected"] else (150, 110, 70),
                1,
                tipLength=0.25,
            )
        if self.center is not None:
            point = tuple(round(float(v)) for v in self.center)
            cv2.circle(
                frame, point, round(self.range_factor * self.radius), (190, 120, 90), 1
            )
            cv2.circle(
                frame, point, max(1, round(0.015 * self.radius)), (190, 120, 90), 1
            )


class RevisedMotionFeedbackTracker(MotionFeedbackTracker):
    def __init__(self) -> None:
        super().__init__("motion_revised")

    def reset(self) -> None:
        super().reset()
        self.candidate_points = np.empty((0, 2), np.float32)
        self.point_ids = np.empty(0, int)
        self.point_since = np.empty(0)
        self.candidate_ids = np.empty(0, int)
        self.candidate_since = np.empty(0)
        self.candidate_frames = np.empty(0, int)
        self.next_point_id = 0
        self.flow_metadata = []
        self.flow_timestamp = 0.0
        self.flow_anchor = None
        self.white_area = 0.0
        self.white_extents = np.zeros(2)
        self.centroid_bbox_offset = np.zeros(2)
        self.white_records = []
        self.rejection_counts = {}
        self.position_uncertainty = 3.0
        self.last_measured = None
        self.last_measured_at = 0.0
        self.point_promotions = 0
        self.flow_loss_details = {}

    def initialize(
        self,
        template,
        center,
        timestamp=0.0,
        template_valid=None,
        initial_frame=None,
        invalid_mask=None,
    ):
        super().initialize(
            template, center, timestamp, template_valid, initial_frame, invalid_mask
        )
        yy, xx = np.nonzero(template)
        self.white_area = float(len(xx))
        self.white_extents = np.array((np.ptp(xx) + 1, np.ptp(yy) + 1), float)
        bbox = np.array(((xx.min() + xx.max()) / 2, (yy.min() + yy.max()) / 2))
        self.centroid_bbox_offset = np.array((xx.mean(), yy.mean())) - bbox
        self.flow_timestamp = self.last_measured_at = timestamp
        self.flow_anchor = self.image_center.copy()
        self.last_measured = self.image_center.copy()
        self.point_ids = np.arange(len(self.tracked_points))
        self.point_since = np.full(len(self.tracked_points), timestamp)
        self.next_point_id = len(self.point_ids)

    def _white_observation(self, frame, valid, predicted, timestamp):
        self.white_records = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        white = self._white_mask(frame, valid)
        yy, xx = np.indices(gray.shape)
        distance = np.hypot(xx - predicted[0], yy - predicted[1])
        search_radius = min(
            1.3 * self.radius,
            max(0.65 * self.radius, self.position_uncertainty + 0.35 * self.radius),
        )
        region = distance <= search_radius + self.radius
        grouping = white & region
        bridge = (~valid) & (distance <= 1.05 * self.radius)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (grouping | bridge).astype(np.uint8), 8
        )
        candidates = []
        for label in range(1, count):
            bx, by, bw, bh, area = stats[label]
            if area < 0.25 * self.white_area:
                continue
            view = np.s_[by : by + bh, bx : bx + bw]
            patch = (labels[view] == label) & white[view] & region[view]
            ys, xs = np.nonzero(patch)
            if len(xs) < 0.25 * self.white_area:
                continue
            ys, xs = ys + by, xs + bx
            extent = np.array((np.ptp(xs) + 1, np.ptp(ys) + 1), float)
            bbox = np.array(((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2))
            centroid = np.array((xs.mean(), ys.mean()))
            invalid_fraction = float(
                np.mean(~valid[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1])
            )
            partial = invalid_fraction > 0.02 or len(xs) < 0.72 * self.white_area
            center = bbox + self.centroid_bbox_offset if partial else centroid
            quality = float(min(1.0, len(xs) / self.white_area))
            reason = "white_accepted"
            if len(xs) > 1.25 * self.white_area or np.max(extent) > 2.5 * self.radius:
                reason = "white_size_mismatch"
            elif np.min(extent) < 0.5 * np.min(self.white_extents):
                reason = "white_insufficient_extent"
            elif (
                xs.min() <= 5
                or xs.max() >= gray.shape[1] - 6
                or ys.min() <= 5
                or ys.max() >= gray.shape[0] - 6
            ):
                reason = "white_boundary_truncated"
            elif np.linalg.norm(center - predicted) > search_radius:
                reason = "white_position_mismatch"
            elif partial and quality < 0.40:
                reason = "white_occlusion_uncertain"
            record = {
                "center": center.tolist(),
                "area": len(xs),
                "quality": quality,
                "partial": partial,
                "invalid_fraction": invalid_fraction,
                "reason": reason,
            }
            self.white_records.append(record)
            if reason == "white_accepted":
                pixels = np.zeros(gray.shape, bool)
                pixels[ys, xs] = True
                score = quality - 0.2 * np.linalg.norm(center - predicted) / max(
                    self.radius, 1
                )
                candidates.append(
                    (
                        score,
                        {
                            "center": center,
                            "score": min(0.98, 0.65 + 0.33 * quality),
                            "support_indices": [],
                            "support_points": 0,
                            "rotation_degrees": 0.0,
                            "pixels": pixels,
                            "partial": partial,
                            "kind": "white_observation",
                            "uncertainty": max(
                                2.0, (0.12 if partial else 0.04) * self.radius
                            ),
                        },
                    )
                )
        candidates.sort(key=lambda item: item[0], reverse=True)
        if not candidates:
            return (
                None,
                "white_not_found"
                if not self.white_records
                else self.white_records[0]["reason"],
            )
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.12:
            return None, "white_ambiguous"
        return candidates[0][1], "white_accepted"

    def _reseed_white(self, gray, valid, best, timestamp):
        mask = best["pixels"] & valid
        seeds = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=100,
            qualityLevel=0.008,
            minDistance=4,
            mask=mask.astype(np.uint8) * 255,
        )
        points = np.empty((0, 2), np.float32) if seeds is None else seeds.reshape(-1, 2)
        self.tracked_points = points
        self.point_ids = np.arange(self.next_point_id, self.next_point_id + len(points))
        self.next_point_id += len(points)
        self.point_since = np.full(len(points), timestamp)
        self.candidate_points = np.empty((0, 2), np.float32)
        self.candidate_ids = np.empty(0, int)
        self.candidate_since = np.empty(0)
        self.candidate_frames = np.empty(0, int)

    def _local_flow(self, gray, valid, anchor):
        trusted = self.tracked_points.copy()
        previous = np.vstack((trusted, self.candidate_points))
        old_ids = np.concatenate((self.point_ids, self.candidate_ids))
        old_since = np.concatenate((self.point_since, self.candidate_since))
        old_frames = np.concatenate((np.full(len(trusted), 3), self.candidate_frames))
        self.tracked_points = previous
        original_valid = self.previous_valid
        if original_valid is not None:
            self.previous_valid = (
                cv2.erode(original_valid.astype(np.uint8), np.ones((15, 15), np.uint8))
                > 0
            )
        patch_valid = cv2.erode(valid.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        try:
            source, target, fb, reason = super()._local_flow(gray, patch_valid, anchor)
        finally:
            self.previous_valid = original_valid
            self.tracked_points = trusted
        self.flow_metadata = []
        trusted_flags = []
        for point in source:
            nearest = (
                int(np.argmin(np.linalg.norm(previous - point, axis=1)))
                if len(previous)
                else -1
            )
            linked = nearest >= 0 and np.linalg.norm(previous[nearest] - point) < 0.75
            if linked:
                meta = {
                    "point_id": int(old_ids[nearest]),
                    "since": float(old_since[nearest]),
                    "frames": int(old_frames[nearest]),
                    "trusted": nearest < len(trusted),
                }
            else:
                meta = {
                    "point_id": self.next_point_id,
                    "since": self.flow_timestamp,
                    "frames": 0,
                    "trusted": False,
                }
                self.next_point_id += 1
            self.flow_metadata.append(meta)
            trusted_flags.append(meta["trusted"])
        self.flow_persistent = np.array(trusted_flags, bool)
        if len(trusted) and original_valid is not None:
            tx = np.clip(trusted[:, 0].astype(int), 0, gray.shape[1] - 1)
            ty = np.clip(trusted[:, 1].astype(int), 0, gray.shape[0] - 1)
            outside = np.linalg.norm(trusted - anchor, axis=1) > 1.3 * self.radius
            masked = ~self.previous_valid[ty, tx] | ~patch_valid[ty, tx]
            retained = int(np.count_nonzero(self.flow_persistent))
            self.flow_loss_details = {
                "trusted_input": len(trusted),
                "trusted_retained": retained,
                "outside_local_range": int(np.count_nonzero(outside)),
                "invalid_patch_at_previous_location": int(
                    np.count_nonzero(masked & ~outside)
                ),
                "other_lk_or_target_rejection": max(
                    0, len(trusted) - retained - int(np.count_nonzero(outside | masked))
                ),
            }
        return source, target, fb, reason

    def _motion_hypotheses(self, source, target, anchor, dt, background_error):
        self.hypothesis_records = []
        self.rejection_counts = {}

        def reject(reason):
            self.rejection_counts[reason] = self.rejection_counts.get(reason, 0) + 1

        if len(source) < 6:
            return [], "flow_support_insufficient"
        if self.motion.affine is None:
            return [], "background_untrusted"
        bg = (
            self.motion.affine
            if self.local_background_affine is None
            else self.local_background_affine
        )
        expected = source @ bg[:, :2].T + bg[:, 2]
        bg_residual = np.linalg.norm(target - expected, axis=1)
        noise = max(0.30, 1.5 * max(background_error, self.local_background_error))
        threshold = noise + 4 * dt
        ids = np.flatnonzero(self.flow_persistent)
        order = np.concatenate((ids, np.flatnonzero(~self.flow_persistent)))
        seeds = order[np.linspace(0, len(order) - 1, min(12, len(order))).astype(int)]
        if len(ids):
            seeds = np.unique(np.concatenate((ids[:8], seeds)))[:16]
        candidates = []
        for seed in seeds:
            group = np.linalg.norm(source - source[seed], axis=1) <= max(
                18.0, 0.85 * self.radius
            )
            if np.count_nonzero(group) < 6:
                reject("flow_cluster_insufficient")
                continue
            matrix, _ = cv2.estimateAffinePartial2D(
                source[group],
                target[group],
                method=cv2.RANSAC,
                ransacReprojThreshold=1.0,
                maxIters=160,
                refineIters=5,
            )
            if matrix is None:
                reject("flow_fit_failed")
                continue
            error = np.linalg.norm(
                source @ matrix[:, :2].T + matrix[:, 2] - target, axis=1
            )
            support = error < 1.1
            n = int(np.count_nonzero(support))
            if n < 6:
                reject("flow_inliers_insufficient")
                continue
            positions = source[support]
            singular = np.linalg.svd(
                positions - positions.mean(axis=0), compute_uv=False
            )
            if (
                singular[-1] / max(singular[0], 1e-6) < 0.12
                or np.max(np.ptp(positions, axis=0)) < 0.35 * self.radius
            ):
                reject("flow_spatial_degenerate")
                continue
            full, _ = cv2.estimateAffine2D(
                source[support],
                target[support],
                method=cv2.RANSAC,
                ransacReprojThreshold=1.0,
                maxIters=100,
            )
            if full is None:
                reject("flow_full_fit_failed")
                continue
            scales = np.linalg.svd(
                full[:, :2] @ np.linalg.inv(bg[:, :2]), compute_uv=False
            )
            expansion = abs(
                math.sqrt(abs(np.linalg.det(full[:, :2] @ np.linalg.inv(bg[:, :2]))))
                - 1
            )
            shear = abs(scales[0] - scales[1])
            measured = matrix[:, :2] @ anchor + matrix[:, 2]
            rotation = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
            contrast = float(np.median(bg_residual[support]))
            fit_error = float(np.median(error[support]))
            trusted_count = int(np.count_nonzero(support & self.flow_persistent))
            coverage = min(
                1.0, np.prod(np.ptp(positions, axis=0)) / max(1.0, self.radius**2)
            )
            quality = float(
                coverage
                * min(1.0, n / 18)
                * math.exp(-fit_error)
                * min(1.0, contrast / max(threshold * 2, 1e-6))
            )
            reason = "accepted_local_motion"
            if expansion > 0.012 + 1.1 * dt or shear > 0.02 + 1.6 * dt:
                reason = "flow_nonrigid"
            elif (
                np.linalg.norm(measured - anchor) / dt > self.max_speed + 30
                or abs(rotation) / dt > 200
            ):
                reason = "flow_speed_excessive"
            elif contrast <= threshold:
                reason = (
                    "motion_unobservable_hold"
                    if trusted_count >= 6
                    else "background_explanation"
                )
            elif quality < 0.20:
                reason = "flow_quality_low"
            elif trusted_count < 3 and (
                n < 10 or quality < 0.45 or contrast < max(0.9, 2 * threshold)
            ):
                reason = "recovery_identity_weak"
            record = {
                "center": measured.tolist(),
                "reference_center": anchor.tolist(),
                "affine_matrix": matrix.tolist(),
                "support_indices": np.flatnonzero(support).tolist(),
                "support_points": n,
                "trusted_support": trusted_count,
                "score": quality,
                "fit_error": fit_error,
                "background_difference": contrast,
                "motion_threshold": threshold,
                "expansion": float(expansion),
                "shear": float(shear),
                "rotation_degrees": rotation,
                "reason": reason,
            }
            self.hypothesis_records.append(record)
            if reason != "accepted_local_motion":
                reject(reason)
                continue
            if all(np.linalg.norm(measured - c["center"]) > 2.5 for c in candidates):
                candidates.append(
                    {
                        **record,
                        "center": measured,
                        "kind": "local_motion",
                        "needs_confirmation": trusted_count < 3,
                    }
                )
        candidates.sort(key=lambda c: c["score"], reverse=True)
        if candidates:
            return candidates[:3], "accepted_local_motion"
        reason = (
            "motion_unobservable_hold"
            if self.rejection_counts.get("motion_unobservable_hold")
            else max(
                self.rejection_counts,
                key=self.rejection_counts.get,
                default="flow_fit_failed",
            )
        )
        return [], reason

    def _update_points(self, target, best, timestamp, accepted):
        trusted_points, trusted_ids, trusted_since = [], [], []
        candidate_points, candidate_ids, candidate_since, candidate_frames = (
            [],
            [],
            [],
            [],
        )
        selected = set() if best is None else set(best["support_indices"])
        self.point_promotions = 0
        can_promote = (
            accepted and best is not None and not best.get("needs_confirmation", False)
        )
        for index, (point, metadata) in enumerate(zip(target, self.flow_metadata)):
            if metadata["trusted"] and (index in selected or not accepted):
                trusted_points.append(point)
                trusted_ids.append(metadata["point_id"])
                trusted_since.append(metadata["since"])
            elif index in selected:
                frames = metadata["frames"] + 1
                if (
                    can_promote
                    and frames >= 3
                    and timestamp - metadata["since"] >= 0.06
                ):
                    trusted_points.append(point)
                    trusted_ids.append(metadata["point_id"])
                    trusted_since.append(metadata["since"])
                    self.point_promotions += 1
                else:
                    candidate_points.append(point)
                    candidate_ids.append(metadata["point_id"])
                    candidate_since.append(metadata["since"])
                    candidate_frames.append(frames)
        self.tracked_points = np.array(trusted_points[:100], np.float32).reshape(-1, 2)
        self.point_ids = np.array(trusted_ids[:100], int)
        self.point_since = np.array(trusted_since[:100])
        self.candidate_points = np.array(candidate_points[:80], np.float32).reshape(
            -1, 2
        )
        self.candidate_ids = np.array(candidate_ids[:80], int)
        self.candidate_since = np.array(candidate_since[:80])
        self.candidate_frames = np.array(candidate_frames[:80], int)

    def _confirm(self, best, timestamp):
        previous = self.pending
        point = best["center"]
        ids = {self.flow_metadata[i]["point_id"] for i in best["support_indices"]}
        continuous = (
            previous is not None
            and 0 < timestamp - previous["last_at"] <= 0.18
            and previous["kind"] == best["kind"]
        )
        if continuous:
            elapsed = timestamp - previous["last_at"]
            limit = (
                self.max_speed * elapsed + 4
                if previous["frames"] == 1
                else max(4.0, 0.12 * self.radius)
            )
            continuous = (
                np.linalg.norm(
                    point - previous["center"] - previous["velocity"] * elapsed
                )
                <= limit
            )
            if best["kind"] == "local_motion":
                continuous &= (
                    len(ids & previous["ids"]) / max(1, len(ids | previous["ids"]))
                    >= 0.4
                )
        if not continuous:
            self.pending = {
                "center": point.copy(),
                "velocity": np.zeros(2),
                "last_at": timestamp,
                "since": timestamp,
                "frames": 1,
                "ids": ids,
                "kind": best["kind"],
            }
        else:
            velocity = self._bounded(
                (point - previous["center"]) / (timestamp - previous["last_at"]),
                self.max_speed,
            )
            self.pending.update(
                center=point.copy(),
                velocity=velocity,
                last_at=timestamp,
                frames=previous["frames"] + 1,
                ids=ids,
            )
        return self.pending["frames"] >= 3 and timestamp - self.pending["since"] >= 0.08

    def update(self, frame, timestamp, invalid_mask=None):
        if self.center is None:
            raise RuntimeError("initialize before update")
        self._validate_frame(frame, invalid_mask)
        if not math.isfinite(timestamp) or timestamp <= self.last_timestamp:
            raise ValueError("timestamps must be finite and increasing")
        started = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        valid = self._valid_pixels(gray.shape, invalid_mask)
        dt = timestamp - self.last_timestamp
        gap = timestamp - self.last_observed_at
        if gap > self.coast_seconds:
            self.velocity *= math.exp(-dt / 0.2)
            self.image_velocity *= math.exp(-dt / 0.2)
        predicted = self.center + self.velocity * dt
        image_prediction = self.image_center + self.image_velocity * dt
        self.position_uncertainty = min(0.9 * self.radius, 3 + 20 * gap)
        search_prediction = image_prediction.copy()
        if self.pending is not None and timestamp - self.pending["last_at"] <= 0.18:
            search_prediction = self.pending["center"] + self.pending["velocity"] * (
                timestamp - self.pending["last_at"]
            )
        self.flow_records = []
        self.hypothesis_records = []
        self.white_records = []
        self.rejection_counts = {}
        self.flow_loss_details = {}
        self.point_promotions = 0
        white, white_reason = self._white_observation(
            frame, valid, search_prediction, timestamp
        )
        white_at = time.perf_counter()
        best = white
        reason = "visible_white_observation" if white is not None else white_reason
        source = np.empty((0, 2), np.float32)
        target = source.copy()
        fb = np.empty(0)
        self.flow_persistent = np.empty(0, bool)
        self.flow_metadata = []
        background = {
            "background_status": "not_evaluated",
            "background_fit_error": 0.0,
            "background_confidence": 0.0,
            "background_inliers": 0,
        }
        flow_elapsed = timestamp - self.flow_timestamp
        sampled = flow_elapsed >= 1 / 35
        bg_at = flow_at = white_at
        if best is None and sampled:
            self.local_background_affine = None
            self.local_background_error = 0.0
            _, background = self.motion.observe(
                gray, valid, image_prediction, self.radius
            )
            bg_at = time.perf_counter()
            source, target, fb, flow_reason = self._local_flow(
                gray, valid, self.flow_anchor
            )
            flow_at = time.perf_counter()
            if flow_reason != "local_flow_ready":
                candidates, reason = [], flow_reason
            else:
                candidates, reason = self._motion_hypotheses(
                    source,
                    target,
                    self.flow_anchor,
                    flow_elapsed,
                    background["background_fit_error"],
                )
            if candidates:
                best = candidates[0]
                if (
                    len(candidates) > 1
                    and candidates[0]["score"] - candidates[1]["score"] < 0.08
                ):
                    best = None
                    reason = "motion_hypotheses_ambiguous"
        elif best is None:
            reason = "motion_accumulating"
        accepted = best is not None
        recovered = False
        if best is not None:
            partial_uncertain = best.get("partial", False) and (
                timestamp - self.last_white_at > 0.2
                or np.linalg.norm(best["center"] - image_prediction) > 0.2 * self.radius
            )
            needs_confirmation = (
                gap > self.coast_seconds
                or best.get("needs_confirmation", False)
                or partial_uncertain
            )
            if needs_confirmation:
                accepted = self._confirm(best, timestamp)
                reason = "recovery_confirmed" if accepted else "recovery_pending"
                recovered = accepted and gap > self.coast_seconds
            if np.linalg.norm(best["center"] - image_prediction) > 1.3 * self.radius:
                accepted = False
                recovered = False
                reason = "candidate_outside_local_range"
        elif self.pending is not None and timestamp - self.pending["last_at"] > 0.18:
            self.pending = None
        if best is None and reason == "motion_unobservable_hold":
            self.image_velocity *= math.exp(-dt / 0.06)
            self.velocity *= math.exp(-dt / 0.06)
            predicted = self.center + self.velocity * dt
            image_prediction = self.image_center + self.image_velocity * dt
        feedback = np.zeros(2)
        if (
            accepted
            and best["kind"] == "local_motion"
            and not recovered
            and (
                best["score"] < 0.20
                or (
                    best.get("trusted_support", 0) < 3
                    and not best.get("needs_confirmation", False)
                )
            )
        ):
            accepted = False
            reason = "motion_identity_support_low"
        if accepted:
            old_measured = self.last_measured.copy()
            observed_dt = timestamp - self.last_measured_at
            measured = best["center"]
            if recovered:
                self.center = measured.copy()
                self.image_center = measured.copy()
                self.velocity = np.zeros(2)
                self.image_velocity = np.zeros(2)
            else:
                measured_velocity = self._bounded(
                    (measured - old_measured) / observed_dt, self.max_speed
                )
                blend = 1 - math.exp(-dt / 0.06)
                velocity_delta = self._bounded(
                    blend * (measured_velocity - self.velocity),
                    self.max_acceleration * dt,
                )
                self.velocity = self._bounded(
                    self.velocity + velocity_delta, self.max_speed
                )
                correction = measured - predicted
                if np.linalg.norm(correction) <= max(0.5, 0.015 * self.radius):
                    correction[:] = 0
                feedback = self._bounded(
                    (1 - math.exp(-dt / 0.08)) * correction,
                    min(0.20 * self.radius, self.max_acceleration * dt**2),
                )
                self.center += self.velocity * dt + feedback
                self.image_velocity = measured_velocity
                self.image_center = measured.copy()
            self.last_measured = measured.copy()
            self.last_measured_at = timestamp
            self.last_observed_at = timestamp
            if best["kind"] == "white_observation":
                self.last_white_at = timestamp
                self._reseed_white(gray, valid, best, timestamp)
            else:
                self._update_points(target, best, timestamp, accepted)
                if best.get("needs_confirmation", False):
                    self.tracked_points = target[best["support_indices"]].copy()
                    self.point_ids = np.array(
                        [
                            self.flow_metadata[i]["point_id"]
                            for i in best["support_indices"]
                        ],
                        int,
                    )
                    self.point_since = np.full(len(self.point_ids), timestamp)
                    self.candidate_points = np.empty((0, 2), np.float32)
                    self.candidate_ids = np.empty(0, int)
                    self.candidate_since = np.empty(0)
                    self.candidate_frames = np.empty(0, int)
                if not self.circular:
                    self.angle += best["rotation_degrees"]
                    self.angular_velocity = best["rotation_degrees"] / flow_elapsed
            if recovered:
                self.recovery_count += 1
            self.pending = None
            self.output_trace = {
                "source": best["kind"],
                "timestamp": timestamp,
                "measured_center": measured.tolist(),
                "smoothed_center": self.center.tolist(),
                "predicted_center": predicted.tolist(),
                "feedback": feedback.tolist(),
                "reference_center": best.get("reference_center"),
                "affine_matrix": best.get("affine_matrix"),
                "support_indices": best["support_indices"],
                "recovered": recovered,
            }
            self.last_observation_trace = self.output_trace.copy()
            status = "observed"
        else:
            self.center = predicted
            self.image_center = image_prediction
            if sampled and white is None:
                self._update_points(target, best, timestamp, False)
            status = "predicted" if gap <= self.coast_seconds else "lost"
            self.output_trace = {
                "source": "prediction",
                "timestamp": timestamp,
                "last_observation_at": self.last_observed_at,
                "last_observation": self.last_observation_trace,
                "prediction": self.center.tolist(),
            }
        if (sampled and white is None) or (
            accepted and best["kind"] == "white_observation"
        ):
            self._commit_frame(gray, valid, self._white_mask(frame, valid))
            self.flow_timestamp = timestamp
            self.flow_anchor = (
                self.image_center.copy()
                if self.pending is None
                else self.pending["center"].copy()
            )
            self.motion.previous_center = self.flow_anchor.copy()
        selected = set() if best is None else set(best["support_indices"])
        for i, (p, q) in enumerate(zip(source, target)):
            self.flow_records.append(
                {
                    "source": p.tolist(),
                    "target": q.tolist(),
                    "fb_error": float(fb[i]),
                    "selected": i in selected,
                    **self.flow_metadata[i],
                }
            )
        if not (
            0 <= self.center[0] < gray.shape[1] and 0 <= self.center[1] < gray.shape[0]
        ):
            status = "lost"
            reason = "outside_frame"
        self.last_timestamp = timestamp
        finished = time.perf_counter()
        diagnostics = {
            **background,
            "reject_reason": reason,
            "white_reject_reason": white_reason,
            "motion_reject_details": json.dumps(self.rejection_counts, sort_keys=True),
            "flow_loss_details": json.dumps(self.flow_loss_details, sort_keys=True),
            "flow_points": len(source),
            "flow_support_points": 0 if best is None else best["support_points"],
            "persistent_flow_points": int(np.count_nonzero(self.flow_persistent)),
            "candidate_flow_points": len(self.candidate_points),
            "point_promotions": self.point_promotions,
            "flow_hypotheses": len(self.hypothesis_records),
            "local_background_used": int(self.local_background_affine is not None),
            "flow_fit_error": 0.0 if best is None else best.get("fit_error", 0.0),
            "flow_background_difference": 0.0
            if best is None
            else best.get("background_difference", 0.0),
            "flow_interval_seconds": flow_elapsed,
            "white_candidates": len(self.white_records),
            "position_uncertainty_px": self.position_uncertainty,
            "feedback_dx": float(feedback[0]),
            "feedback_dy": float(feedback[1]),
            "velocity_x": float(self.velocity[0]),
            "velocity_y": float(self.velocity[1]),
            "motion_support": 0.0 if best is None else best["score"],
            "recovered": int(recovered),
            "recovery_pending_frames": 0
            if self.pending is None
            else self.pending["frames"],
            "actual_search_radius": 1.3 * self.radius,
            "background_ms": (bg_at - white_at) * 1000,
            "local_flow_ms": (flow_at - bg_at) * 1000,
            "white_ms": (white_at - started) * 1000,
            "feedback_ms": (finished - flow_at) * 1000,
        }
        return TrackingResult(
            status,
            None if status == "lost" else tuple(self.center),
            tuple(predicted),
            self.angle,
            self.angular_velocity,
            0,
            0,
            0.0 if best is None else best["score"],
            0.0,
            best["score"] if accepted else 0.0,
            timestamp - self.last_observed_at,
            1.3 * self.radius,
            diagnostics,
        )


def make_tracker(variant: str):
    if variant == "baseline":
        return FragmentVotingTracker()
    if variant == "original_voting":
        return OriginalVotingTracker()
    if variant in GUARDED_VARIANTS:
        return GuardedVotingTracker(variant)
    if variant == "motion_revised":
        return RevisedMotionFeedbackTracker()
    if variant in MOTION_VARIANTS:
        return MotionFeedbackTracker(variant)
    return OptimizedFragmentTracker(variant)


def load_annotations(path: Path | None) -> dict[tuple[str, int], dict]:
    if path is None:
        return {}
    annotations = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"video", "frame", "x", "y"} <= set(reader.fieldnames or []):
            raise ValueError("annotations require video,frame,x,y columns")
        for row in reader:
            key = (row["video"], int(row["frame"]))
            if key[1] < 0 or key in annotations:
                raise ValueError(
                    "annotation frames must be non-negative and unique per video"
                )
            point = (float(row["x"]), float(row["y"]))
            if not np.isfinite(point).all():
                raise ValueError("annotation coordinates must be finite")
            annotations[key] = {
                "center": point,
                "phase": row.get("phase", ""),
                "visibility": row.get("visibility", ""),
            }
    return annotations


def make_template(shape: str = "star", radius: int = 30) -> np.ndarray:
    if shape not in ("star", "irregular", "circle") or radius < 10:
        raise ValueError("use star, irregular or circle with radius at least 10")
    side = 2 * (radius + 5) + 1
    center = side // 2
    mask = np.zeros((side, side), dtype=np.uint8)
    if shape == "circle":
        cv2.circle(mask, (center, center), radius, 255, -1)
    else:
        if shape == "star":
            angles = np.arange(10) * math.pi / 5 - math.pi / 2
            radii = np.where(np.arange(10) % 2 == 0, radius, 0.46 * radius)
        else:
            angles = np.arange(9) * 2 * math.pi / 9
            radii = radius * np.array(
                (1.0, 0.61, 0.83, 0.52, 0.94, 0.72, 0.46, 0.89, 0.63)
            )
        vertices = (
            np.column_stack((np.cos(angles) * radii, np.sin(angles) * radii)) + center
        )
        cv2.fillPoly(mask, [np.rint(vertices).astype(np.int32)], 255)
    return mask


def placed_mask(
    template: np.ndarray,
    size: tuple[int, int],
    center: tuple[float, float],
    angle: float,
) -> np.ndarray:
    radians = math.radians(angle)
    matrix = np.array(
        (
            (math.cos(radians), -math.sin(radians)),
            (math.sin(radians), math.cos(radians)),
        ),
        dtype=np.float64,
    )
    origin = np.array(((template.shape[1] - 1) / 2, (template.shape[0] - 1) / 2))
    affine = np.column_stack((matrix, np.asarray(center) - matrix @ origin))
    return cv2.warpAffine(template, affine, size, flags=cv2.INTER_NEAREST)


@dataclass(frozen=True)
class SyntheticSample:
    frame: np.ndarray
    timestamp: float
    truth_center: tuple[float, float]
    truth_rotation: float
    visibility: float


class SyntheticSequence:
    def __init__(
        self,
        scenario: str,
        seed: int,
        *,
        fps: float = 24.0,
        seconds: float = 3.0,
        size: tuple[int, int] = (480, 320),
    ) -> None:
        if (
            scenario not in SCENARIOS + MOTION_SCENARIOS
            or not math.isfinite(fps)
            or not math.isfinite(seconds)
            or fps <= 0
            or seconds <= 0
            or min(size) < 160
        ):
            raise ValueError("invalid synthetic scenario, duration, rate or frame size")
        self.scenario = scenario
        self.seed = seed
        self.fps = float(fps)
        self.seconds = float(seconds)
        self.size = size
        self.template = make_template("circle" if scenario == "circle" else "irregular")
        self.initial_center = (0.30 * size[0], 0.48 * size[1])
        rng = np.random.default_rng(seed)
        coarse = rng.uniform(48, 98, size=(16, 24)).astype(np.float32)
        texture = cv2.resize(coarse, size, interpolation=cv2.INTER_CUBIC)
        texture += rng.normal(0, 1.2, size=(size[1], size[0]))
        self.background = np.repeat(
            np.clip(texture, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2
        )

    def sample(self, timestamp: float) -> SyntheticSample:
        width, height = self.size
        t = float(timestamp)
        progress = t / self.seconds
        cx = self.initial_center[0] + 42 * t + 12 * math.sin(1.7 * t)
        cy = self.initial_center[1] + 27 * math.sin(1.3 * t)
        if self.scenario == "boundary":
            cx = self.initial_center[0] + (
                width - 7 - self.initial_center[0]
            ) * math.sin(math.pi * progress)
            cy = self.initial_center[1] + 15 * math.sin(2 * math.pi * progress)
        angle = 48 * t
        background_motion = np.array(
            ((1.0, 0.0, 7 * t), (0.0, 1.0, 3 * t)), dtype=np.float32
        )
        frame = cv2.warpAffine(
            self.background,
            background_motion,
            self.size,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        if self.scenario == "deforming_background":
            yy, xx = np.indices((height, width), dtype=np.float32)
            region = np.exp(
                -((xx - 0.56 * width) ** 2 + (yy - 0.5 * height) ** 2) / (2 * 65**2)
            )
            phase = math.sin(3 * t)
            frame = cv2.remap(
                frame,
                xx + 10 * phase * region,
                yy + 6 * phase * region,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
        if self.scenario == "distractors":
            for center, pose in (
                ((0.52 * width + 7 * t, 0.55 * height + 3 * t), 10.0),
                ((0.72 * width + 7 * t, 0.35 * height + 3 * t), 65.0),
            ):
                mask = placed_mask(self.template, self.size, center, pose)
                frame[mask > 0] = (224, 224, 224)
        mask = placed_mask(self.template, self.size, (cx, cy), angle)
        visible = mask > 0
        full_area = int(np.count_nonzero(self.template))
        if self.scenario == "invisible" and 0.34 <= progress <= 0.68:
            visible[:] = False
        if self.scenario in MOTION_SCENARIOS and t >= 0.45:
            frame[visible] = np.clip(
                frame[visible].astype(np.int16) + 16, 0, 255
            ).astype(np.uint8)
            boundary = (mask > 0) & (cv2.erode(mask, np.ones((3, 3), np.uint8)) == 0)
            frame[boundary] = (120, 120, 120)
        else:
            frame[visible] = (232, 232, 232)
        if self.scenario == "partial_occlusion" and 0.30 <= progress <= 0.63:
            x0, x1 = int(cx - 8), min(width, int(cx + 39))
            y0, y1 = max(0, int(cy - 39)), min(height, int(cy + 39))
            frame[y0:y1, max(0, x0) : x1] = (88, 88, 88)
            visible[y0:y1, max(0, x0) : x1] = False
        return SyntheticSample(
            frame,
            t,
            (float(cx), float(cy)),
            angle,
            float(np.count_nonzero(visible) / full_area),
        )

    def __iter__(self):
        for index in range(round(self.seconds * self.fps) + 1):
            yield self.sample(index / self.fps)


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot summarize an empty run")
    observed = [row for row in rows if row["status"] == "observed"]
    errors = [row["error_px"] for row in rows if row["error_px"] is not None]
    observed_errors = [row["error_px"] for row in observed]
    longest_lost = longest_unobserved = lost_run = unobserved_run = 0.0
    previous_time = 0.0
    for row in rows:
        dt = row["timestamp"] - previous_time
        previous_time = row["timestamp"]
        lost_run = lost_run + dt if row["status"] == "lost" else 0.0
        unobserved_run = unobserved_run + dt if row["status"] != "observed" else 0.0
        longest_lost = max(longest_lost, lost_run)
        longest_unobserved = max(longest_unobserved, unobserved_run)
    times = [row["processing_ms"] for row in rows]
    return {
        "frames": len(rows),
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in ("observed", "predicted", "lost")
        },
        "prediction_coverage": len(errors) / len(rows),
        "observation_coverage": len(observed) / len(rows),
        "visible_observation_coverage": (
            sum(row["status"] == "observed" and row["visibility"] > 0 for row in rows)
            / max(1, sum(row["visibility"] > 0 for row in rows))
        ),
        "false_observations_when_invisible": sum(
            row["status"] == "observed" and row["visibility"] == 0 for row in rows
        ),
        "within_5px_all_frames": sum(error <= 5 for error in errors) / len(rows),
        "within_10px_all_frames": sum(error <= 10 for error in errors) / len(rows),
        "mean_error_px": float(np.mean(errors)) if errors else None,
        "p90_error_px": float(np.percentile(errors, 90)) if errors else None,
        "mean_observed_error_px": float(np.mean(observed_errors))
        if observed_errors
        else None,
        "longest_lost_seconds": longest_lost,
        "longest_without_observation_seconds": longest_unobserved,
        "mean_processing_ms": float(np.mean(times)),
        "p90_processing_ms": float(np.percentile(times, 90)),
    }


def draw_overlay(
    sample: SyntheticSample, result: TrackingResult, error: float | None
) -> np.ndarray:
    frame = sample.frame.copy()
    truth = tuple(round(value) for value in sample.truth_center)
    cv2.drawMarker(frame, truth, (80, 210, 80), cv2.MARKER_TILTED_CROSS, 16, 1)
    colors = {
        "initialized": (200, 210, 255),
        "observed": (0, 210, 255),
        "predicted": (0, 130, 255),
        "lost": (80, 80, 255),
    }
    color = colors[result.status]
    if result.center is not None:
        center = tuple(round(value) for value in result.center)
        cv2.drawMarker(frame, center, color, cv2.MARKER_CROSS, 20, 2)
        cv2.circle(frame, center, 34, color, 1)
    error_text = "n/a" if error is None else f"{error:.1f}px"
    lines = (
        f"{result.status}  t={sample.timestamp:.2f}s  error={error_text}",
        f"votes={result.votes}/{result.fragment_count} quality={result.quality:.2f} margin={result.margin:.2f}",
        "green=truth (evaluation only) | yellow/orange=estimate",
    )
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, 67), (20, 20, 20), -1)
    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (8, 17 + index * 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color if index == 0 else (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return frame


def run_experiment(
    scenario: str,
    seed: int,
    output_dir: Path,
    *,
    fps: float = 24.0,
    seconds: float = 3.0,
    save_video: bool = True,
    variant: str = "baseline",
) -> tuple[list[dict], dict]:
    sequence = SyntheticSequence(scenario, seed, fps=fps, seconds=seconds)
    tracker = make_tracker(variant)
    if variant in MOTION_VARIANTS:
        tracker.initialize(
            sequence.template,
            sequence.initial_center,
            0.0,
            initial_frame=sequence.sample(0.0).frame,
        )
    else:
        tracker.initialize(sequence.template, sequence.initial_center, 0.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"{scenario}_seed_{seed}"
    writer = None
    video_path = output_dir / f"{name}.mp4"
    if save_video:
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, sequence.size
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"unable to create {video_path}")
    rows = []
    try:
        for index, sample in enumerate(sequence):
            if index == 0:
                if writer is not None:
                    initial = TrackingResult(
                        "initialized",
                        sequence.initial_center,
                        sequence.initial_center,
                        0.0,
                        0.0,
                        0,
                        len(tracker.fragments)
                        if variant == "baseline"
                        else tracker.fragment_count,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                    )
                    writer.write(draw_overlay(sample, initial, 0.0))
                continue
            started = time.perf_counter()
            result = tracker.update(sample.frame, sample.timestamp)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            error = (
                None
                if result.center is None
                else math.dist(result.center, sample.truth_center)
            )
            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "frame": index,
                    "timestamp": sample.timestamp,
                    "status": result.status,
                    "center_x": None if result.center is None else result.center[0],
                    "center_y": None if result.center is None else result.center[1],
                    "truth_x": sample.truth_center[0],
                    "truth_y": sample.truth_center[1],
                    "truth_rotation": sample.truth_rotation,
                    "rotation_degrees": result.rotation_degrees,
                    "angular_velocity_dps": result.angular_velocity_dps,
                    "visibility": sample.visibility,
                    "votes": result.votes,
                    "fragment_count": result.fragment_count,
                    "quality": result.quality,
                    "margin": result.margin,
                    "confidence": result.confidence,
                    "error_px": error,
                    "seconds_without_observation": result.seconds_without_observation,
                    "processing_ms": elapsed_ms,
                    **(result.diagnostics if variant in MOTION_VARIANTS else {}),
                }
            )
            if writer is not None:
                canvas = draw_overlay(sample, result, error)
                if variant in MOTION_VARIANTS:
                    tracker.draw_motion(canvas)
                    cv2.rectangle(
                        canvas, (0, 24), (canvas.shape[1] - 1, 45), (20, 20, 20), -1
                    )
                    cv2.putText(
                        canvas,
                        f"flow={result.diagnostics['flow_support_points']}/{result.diagnostics['flow_points']} | {result.diagnostics['reject_reason']}",
                        (8, 38),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (235, 235, 235),
                        1,
                    )
                writer.write(canvas)
    finally:
        if writer is not None:
            writer.release()
    summary = {
        "scenario": scenario,
        "seed": seed,
        "variant": variant,
        "fps": fps,
        "seconds": seconds,
        "video": str(video_path) if save_video else None,
        **summarize(rows),
    }
    return rows, summary


def opening_template(
    frame: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float]] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = ((gray >= 210) & (hsv[:, :, 1] <= 75)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    height, width = mask.shape
    candidates = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if not (700 <= area <= 40000 and 35 <= w <= 240 and 35 <= h <= 240):
            continue
        center = (x + (w - 1) / 2, y + (h - 1) / 2)
        if not (
            0.35 * width <= center[0] <= 0.65 * width
            and 0.30 * height <= center[1] <= 0.78 * height
        ):
            continue
        if not 0.5 <= w / h <= 2.0:
            continue
        component = np.zeros_like(mask)
        cv2.drawContours(component, [contour], -1, 255, -1)
        brightness = cv2.mean(gray, mask=component)[0]
        if brightness < 220:
            continue
        side = max(w, h) + 12
        if side % 2 == 0:
            side += 1
        template = cv2.getRectSubPix(component, (side, side), center)
        template = (template >= 127).astype(np.uint8) * 255
        candidates.append((brightness, template, center))
    if not candidates:
        return None
    _, template, center = max(candidates, key=lambda item: item[0])
    return template, center


def evaluation_diagnostics(rows: list[dict]) -> dict:
    active = [row for row in rows if row.get("evaluation_active", False)]
    tracking = [
        row for row in rows if row["status"] not in ("uninitialized", "initialized")
    ]
    times = [row["processing_ms"] for row in tracking]
    reasons = {}
    for row in active:
        reason = row.get("reject_reason", row["status"])
        reasons[reason] = reasons.get(reason, 0) + 1
    annotated = [row for row in active if row.get("truth_x") is not None]
    errors = [
        row["truth_error_px"]
        for row in annotated
        if row.get("truth_error_px") is not None
    ]
    phases = {}
    for phase in sorted({row.get("annotation_phase", "") for row in annotated} - {""}):
        subset = [row for row in annotated if row["annotation_phase"] == phase]
        phase_errors = [
            row["truth_error_px"]
            for row in subset
            if row.get("truth_error_px") is not None
        ]
        phases[phase] = {
            "frames": len(subset),
            "coverage": len(phase_errors) / len(subset),
            "mean_error_px": float(np.mean(phase_errors)) if phase_errors else None,
        }
    recoveries = [row for row in active if row.get("recovered", 0)]
    return {
        "p95_tracking_ms": float(np.percentile(times, 95)) if times else None,
        "over_source_frame_budget_ratio": sum(
            row["processing_ms"] > row.get("frame_budget_ms", math.inf)
            for row in tracking
        )
        / len(tracking)
        if tracking
        else None,
        "reject_reason_counts": reasons,
        "recovery_events": len(recoveries),
        "recoveries_far_from_cursor_proxy": sum(
            row.get("proxy_error_px") is not None and row["proxy_error_px"] > 100
            for row in recoveries
        ),
        "independent_truth_frames": len(annotated),
        "independent_truth_coverage": len(errors) / len(annotated)
        if annotated
        else None,
        "mean_independent_truth_error_px": float(np.mean(errors)) if errors else None,
        "annotation_phase_metrics": phases,
        "mean_stage_ms": {
            key: float(np.mean([row.get(key, 0.0) for row in tracking]))
            if tracking
            else None
            for key in (
                "preprocess_ms",
                "background_ms",
                "features_ms",
                "matching_ms",
                "local_flow_ms",
                "feedback_ms",
                "white_ms",
            )
        },
    }


def real_metrics(rows: list[dict], fps: float) -> dict:
    active = [row for row in rows if row["evaluation_active"]]
    proxy = [row for row in active if row["proxy_x"] is not None]
    errors = [
        row["proxy_error_px"] for row in proxy if row["proxy_error_px"] is not None
    ]
    statuses = ("uninitialized", "initialized", "observed", "predicted", "lost")
    unavailable_run = longest_unavailable = 0
    for row in proxy:
        if row["center_x"] is None:
            unavailable_run += 1
            longest_unavailable = max(longest_unavailable, unavailable_run)
        else:
            unavailable_run = 0
    times = [
        row["processing_ms"]
        for row in rows
        if row["status"] not in ("uninitialized", "initialized")
    ]
    result = {
        "source_frames": len(rows),
        "active_frames": len(active),
        "cursor_proxy_frames": len(proxy),
        "scored_prediction_frames": len(errors),
        "active_status_counts": {
            status: sum(row["status"] == status for row in active)
            for status in statuses
        },
        "proxy_prediction_coverage": len(errors) / len(proxy) if proxy else None,
        "proxy_observation_coverage": sum(row["status"] == "observed" for row in proxy)
        / len(proxy)
        if proxy
        else None,
        "mean_cursor_proxy_error_px": float(np.mean(errors)) if errors else None,
        "p90_cursor_proxy_error_px": float(np.percentile(errors, 90))
        if errors
        else None,
        "max_cursor_proxy_error_px": max(errors) if errors else None,
        "longest_missing_proxy_frames": longest_unavailable,
        "longest_missing_proxy_seconds": longest_unavailable / fps,
        "mean_tracking_ms": float(np.mean(times)) if times else None,
        "p90_tracking_ms": float(np.percentile(times, 90)) if times else None,
    }
    for threshold in (10, 20, 40):
        result[f"within_{threshold}px_proxy_all_frames"] = (
            sum(error <= threshold for error in errors) / len(proxy) if proxy else None
        )
    result.update(evaluation_diagnostics(rows))
    unavailable = current_run = 0.0
    previous = None
    for row in active:
        timestamp = row.get("timestamp")
        dt = (
            (timestamp - previous)
            if timestamp is not None and previous is not None
            else 1 / fps
        )
        previous = timestamp
        current_run = current_run + dt if row["center_x"] is None else 0.0
        unavailable = max(unavailable, current_run)
    result["longest_unavailable_source_seconds"] = unavailable
    return result


def aggregate_recording_metrics(rows: list[dict], videos: list[dict]) -> dict:
    active = [row for row in rows if row["evaluation_active"]]
    proxy = [row for row in active if row["proxy_x"] is not None]
    errors = [
        row["proxy_error_px"] for row in proxy if row["proxy_error_px"] is not None
    ]
    tracking = [
        row["processing_ms"]
        for row in rows
        if row["status"] not in ("uninitialized", "initialized")
    ]
    metrics = {
        "selected_videos": len(videos),
        "completed_videos": sum("source_frames" in video for video in videos),
        "initialization_failures": [
            video["video"]
            for video in videos
            if video.get("initialization") == "failed"
        ],
        "roi_failures": [
            video["video"]
            for video in videos
            if video.get("status") == "roi_unavailable"
        ],
        "source_frames": len(rows),
        "active_frames": len(active),
        "cursor_proxy_frames": len(proxy),
        "scored_prediction_frames": len(errors),
        "active_status_counts": {
            status: sum(row["status"] == status for row in active)
            for status in (
                "uninitialized",
                "initialized",
                "observed",
                "predicted",
                "lost",
            )
        },
        "proxy_prediction_coverage": len(errors) / len(proxy) if proxy else None,
        "proxy_observation_coverage": sum(row["status"] == "observed" for row in proxy)
        / len(proxy)
        if proxy
        else None,
        "mean_cursor_proxy_error_px": float(np.mean(errors)) if errors else None,
        "p90_cursor_proxy_error_px": float(np.percentile(errors, 90))
        if errors
        else None,
        "mean_tracking_ms": float(np.mean(tracking)) if tracking else None,
        "p90_tracking_ms": float(np.percentile(tracking, 90)) if tracking else None,
    }
    for threshold in (10, 20, 40):
        metrics[f"within_{threshold}px_proxy_all_frames"] = (
            sum(error <= threshold for error in errors) / len(proxy) if proxy else None
        )
    metrics.update(evaluation_diagnostics(rows))
    metrics["max_unavailable_source_seconds"] = max(
        (video.get("longest_unavailable_source_seconds", 0.0) for video in videos),
        default=0.0,
    )
    return metrics


def recording_signature(
    video: Path, spec: dict, variant: str, annotations: Path | None, save_video: bool
) -> dict:
    stat = video.stat()
    return {
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "input_path": str(video.resolve()),
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "roi": spec["roi"],
        "evaluation_window": [spec["active_start"], spec["active_end"]],
        "variant": variant,
        "save_video": save_video,
        "annotations_sha256": None
        if annotations is None
        else hashlib.sha256(annotations.read_bytes()).hexdigest(),
    }


def read_frame_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    integers = {
        "frame",
        "votes",
        "background_inliers",
        "effective_fragments",
        "candidate_count",
        "template_ready",
        "template_pose_bins",
        "recovery_pending_frames",
        "white_samples",
        "recovered",
        "fragment_matches",
        "vote_peak_count",
        "fragment_calibration_samples",
        "trajectory_pending_frames",
        "trajectory_delayed_submissions",
        "flow_points",
        "flow_support_points",
        "flow_hypotheses",
        "persistent_flow_points",
        "local_background_used",
        "candidate_flow_points",
        "point_promotions",
        "white_candidates",
    }
    strings = {
        "video",
        "variant",
        "status",
        "annotation_phase",
        "annotation_visibility",
        "reject_reason",
        "background_status",
        "winning_fragment_ids",
        "white_reject_reason",
        "motion_reject_details",
        "flow_loss_details",
    }
    for row in rows:
        for key, value in row.items():
            if key in strings:
                continue
            if key == "evaluation_active":
                row[key] = value == "True"
            elif value == "":
                row[key] = None
            elif key in integers:
                row[key] = int(value)
            else:
                row[key] = float(value)
    return rows


def resumable_recording(
    output_dir: Path, video: Path, signature: dict
) -> tuple[dict, list[dict]] | None:
    marker = output_dir / f"{video.stem}_complete.json"
    if not marker.is_file():
        return None
    summary = json.loads(marker.read_text(encoding="utf-8"))
    if summary.get("signature") != signature:
        raise ValueError(
            f"completed run inputs or implementation changed for {video.name}; use a new output directory"
        )
    frames_path = output_dir / f"{video.stem}_frames.csv"
    if not frames_path.is_file():
        return None
    rows = read_frame_rows(frames_path)
    if len(rows) != summary["source_frames"] or any(
        row["frame"] != index for index, row in enumerate(rows)
    ):
        return None
    if signature["save_video"]:
        capture = cv2.VideoCapture(summary["output_video"])
        count = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            count += 1
        capture.release()
        if count != len(rows):
            return None
    if (
        signature["variant"] == "original_voting"
        or signature["variant"] in GUARDED_VARIANTS
    ):
        evidence_path = output_dir / f"{video.stem}_votes.jsonl"
        if not evidence_path.is_file():
            return None
        with evidence_path.open(encoding="utf-8") as handle:
            count = sum(1 for _ in handle)
        if count != len(rows):
            return None
    if signature["variant"] in MOTION_VARIANTS:
        evidence_path = output_dir / f"{video.stem}_motion.jsonl"
        if not evidence_path.is_file():
            return None
        with evidence_path.open(encoding="utf-8") as handle:
            if sum(1 for _ in handle) != len(rows):
                return None
    return summary, rows


def evaluate_recordings(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from ml.detect_lie_panels import detect_recording
    from src.engine.LieDetectorTracker import LieDetectorTracker, green_cursor_mask
    from tools.lie_detector_replay import cursor_ground_truth

    annotations = load_annotations(args.annotations)
    specs = {
        entry["filename"]: entry
        for entry in json.loads(args.recording_config.read_text(encoding="utf-8"))[
            "recordings"
        ]
    }
    videos = sorted(args.recordings_dir.rglob("测谎录屏*.mp4"))
    if args.recording:
        videos = [path for path in videos if path.name in args.recording]
        missing = set(args.recording) - {path.name for path in videos}
        if missing:
            raise ValueError(f"recordings not found: {sorted(missing)}")
    if not videos:
        raise ValueError("no recordings selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries, all_rows = [], []
    for video in videos:
        spec = specs.get(video.name) or detect_recording(video, probe_count=30)
        if spec is None:
            summary = {"video": video.name, "status": "roi_unavailable"}
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            continue
        signature = recording_signature(
            video, spec, args.variant, args.annotations, not args.no_video
        )
        resumed = (
            resumable_recording(args.output_dir, video, signature)
            if getattr(args, "resume", False)
            else None
        )
        if resumed is not None:
            summaries.append(resumed[0])
            all_rows.extend(resumed[1])
            print(
                json.dumps({"video": video.name, "resumed": True}, ensure_ascii=False),
                flush=True,
            )
            continue
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"unable to open {video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        x, y, width, height = (int(value) for value in spec["roi"])
        active_start, active_end = (
            float(spec["active_start"]),
            float(spec["active_end"]),
        )
        tracker = make_tracker(args.variant)
        initialized = False
        seed_at = None
        seed_errors = []
        writer = None
        output_suffix = "motion" if args.variant in MOTION_VARIANTS else "fragment"
        output_video = args.output_dir / f"{video.stem}_{output_suffix}.mp4"
        if not args.no_video:
            writer = cv2.VideoWriter(
                str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                cap.release()
                writer.release()
                raise RuntimeError(f"unable to create {output_video}")
        rows = []
        index = 0
        evidence_suffix = "motion" if args.variant in MOTION_VARIANTS else "votes"
        evidence_handle = (
            (args.output_dir / f"{video.stem}_{evidence_suffix}.jsonl").open(
                "w", encoding="utf-8"
            )
            if args.variant == "original_voting"
            or args.variant in GUARDED_VARIANTS
            or args.variant in MOTION_VARIANTS
            else None
        )
        try:
            while True:
                ok, full_frame = cap.read()
                if not ok:
                    break
                if y + height > full_frame.shape[0] or x + width > full_frame.shape[1]:
                    raise ValueError(f"ROI exceeds frame for {video.name}")
                frame = full_frame[y : y + height, x : x + width]
                timestamp = index / fps
                proxy = cursor_ground_truth(frame)
                started = time.perf_counter()
                invalid = None
                if args.variant == "baseline":
                    cleaned, _ = LieDetectorTracker._remove_cursor(frame)
                else:
                    invalid = (
                        cv2.dilate(green_cursor_mask(frame), np.ones((9, 9), np.uint8))
                        > 0
                    )
                    cleaned = frame
                preprocess_ms = (time.perf_counter() - started) * 1000
                diagnostics = {}
                status = "uninitialized"
                center = None
                votes = 0
                confidence = 0.0
                if not initialized and timestamp <= min(5.0, active_end):
                    seed = (
                        opening_template(cleaned)
                        if args.variant == "baseline"
                        else visible_template(frame, ~invalid)
                    )
                    if seed is not None:
                        template, point = (
                            (seed[0], seed[1])
                            if args.variant == "baseline"
                            else (seed[0], seed[2])
                        )
                        try:
                            if args.variant == "baseline":
                                tracker.initialize(template, point, timestamp)
                            elif args.variant in MOTION_VARIANTS:
                                tracker.initialize(
                                    template,
                                    point,
                                    timestamp,
                                    template_valid=seed[1],
                                    initial_frame=frame,
                                    invalid_mask=invalid,
                                )
                            else:
                                tracker.initialize(
                                    template, point, timestamp, template_valid=seed[1]
                                )
                        except ValueError as exc:
                            seed_errors.append(str(exc))
                        else:
                            initialized = True
                            seed_at = timestamp
                            status = "initialized"
                            center = point
                            cv2.imwrite(
                                str(args.output_dir / f"{video.stem}_template.png"),
                                template,
                            )
                            cv2.imwrite(
                                str(args.output_dir / f"{video.stem}_seed.jpg"), frame
                            )
                elif initialized:
                    result = (
                        tracker.update(cleaned, timestamp)
                        if args.variant == "baseline"
                        else tracker.update(frame, timestamp, invalid)
                    )
                    diagnostics = result.diagnostics
                    status, center, votes, confidence = (
                        result.status,
                        result.center,
                        result.votes,
                        result.confidence,
                    )
                elapsed = (time.perf_counter() - started) * 1000
                error = (
                    math.dist(center, proxy)
                    if center is not None and proxy is not None
                    else None
                )
                annotation = annotations.get((video.name, index))
                truth = None if annotation is None else annotation["center"]
                truth_error = (
                    None
                    if truth is None or center is None
                    else math.dist(center, truth)
                )
                row = {
                    "video": video.name,
                    "variant": args.variant,
                    "frame": index,
                    "timestamp": timestamp,
                    "evaluation_active": active_start <= timestamp <= active_end,
                    "status": status,
                    "center_x": None if center is None else center[0],
                    "center_y": None if center is None else center[1],
                    "proxy_x": None if proxy is None else proxy[0],
                    "proxy_y": None if proxy is None else proxy[1],
                    "proxy_error_px": error,
                    "votes": votes,
                    "confidence": confidence,
                    "processing_ms": elapsed,
                    "frame_budget_ms": 1000 / fps,
                    "preprocess_ms": preprocess_ms,
                    "truth_x": None if truth is None else truth[0],
                    "truth_y": None if truth is None else truth[1],
                    "truth_error_px": truth_error,
                    "annotation_phase": ""
                    if annotation is None
                    else annotation["phase"],
                    "annotation_visibility": ""
                    if annotation is None
                    else annotation["visibility"],
                    "reject_reason": diagnostics.get("reject_reason", status),
                    "background_status": diagnostics.get(
                        "background_status", "not_applicable"
                    ),
                    "background_confidence": diagnostics.get(
                        "background_confidence", 0.0
                    ),
                    "background_inliers": diagnostics.get("background_inliers", 0),
                    "background_fit_error": diagnostics.get(
                        "background_fit_error", 0.0
                    ),
                    "effective_fragments": diagnostics.get(
                        "effective_fragments", votes
                    ),
                    "pose_uncertainty_degrees": diagnostics.get(
                        "pose_uncertainty_degrees", 0.0
                    ),
                    "candidate_count": diagnostics.get("candidate_count", 0),
                    "actual_search_radius": diagnostics.get(
                        "actual_search_radius", 0.0
                    ),
                    "symmetry_period_degrees": diagnostics.get(
                        "symmetry_period_degrees", 0.0
                    ),
                    "template_ready": diagnostics.get("template_ready", 0),
                    "template_pose_bins": diagnostics.get("template_pose_bins", 0),
                    "recovery_pending_frames": diagnostics.get(
                        "recovery_pending_frames", 0
                    ),
                    "background_recovery_margin": diagnostics.get(
                        "background_recovery_margin", 0.0
                    ),
                    "white_samples": diagnostics.get(
                        "white_samples", 1 if initialized else 0
                    ),
                    "position_uncertainty_px": diagnostics.get(
                        "position_uncertainty_px", 0.0
                    ),
                    "recovered": diagnostics.get("recovered", 0),
                    "motion_support": diagnostics.get("motion_support", 0.0),
                    "valid_boundary_fraction": diagnostics.get(
                        "valid_boundary_fraction", 0.0
                    ),
                    "background_ms": diagnostics.get("background_ms", 0.0),
                    "features_ms": diagnostics.get("features_ms", 0.0),
                    "matching_ms": diagnostics.get("matching_ms", 0.0),
                    "search_radius": 0.0
                    if status in ("uninitialized", "initialized")
                    else result.search_radius,
                    "fragment_matches": diagnostics.get("fragment_matches", 0),
                    "vote_peak_count": diagnostics.get("vote_peak_count", 0),
                    "winning_fragment_ids": diagnostics.get("winning_fragment_ids", ""),
                    "rotation_support": diagnostics.get("rotation_support", 0.0),
                    "rotation_flow_confidence": diagnostics.get(
                        "rotation_flow_confidence", 0.0
                    ),
                    "fragment_calibration_samples": diagnostics.get(
                        "fragment_calibration_samples", 0
                    ),
                    "trajectory_innovation_px": diagnostics.get(
                        "trajectory_innovation_px", 0.0
                    ),
                    "trajectory_pending_frames": diagnostics.get(
                        "trajectory_pending_frames", 0
                    ),
                    "trajectory_delayed_submissions": diagnostics.get(
                        "trajectory_delayed_submissions", 0
                    ),
                    "vote_geometry_rms": diagnostics.get("vote_geometry_rms", 0.0),
                    "local_motion_reliability": diagnostics.get(
                        "local_motion_reliability", 0.0
                    ),
                    "local_spin": diagnostics.get("local_spin", 0.0),
                    "local_expansion": diagnostics.get("local_expansion", 0.0),
                    "local_shear": diagnostics.get("local_shear", 0.0),
                }
                if args.variant in MOTION_VARIANTS:
                    for key in (
                        "flow_points",
                        "flow_support_points",
                        "flow_hypotheses",
                        "persistent_flow_points",
                        "local_background_used",
                        "flow_fit_error",
                        "flow_background_difference",
                        "feedback_dx",
                        "feedback_dy",
                        "velocity_x",
                        "velocity_y",
                        "local_flow_ms",
                        "feedback_ms",
                    ):
                        row[key] = diagnostics.get(key, 0)
                if args.variant == "motion_revised":
                    for key in (
                        "candidate_flow_points",
                        "point_promotions",
                        "flow_interval_seconds",
                        "white_candidates",
                        "white_ms",
                    ):
                        row[key] = diagnostics.get(key, 0)
                    row["white_reject_reason"] = diagnostics.get(
                        "white_reject_reason", "not_evaluated"
                    )
                    row["motion_reject_details"] = diagnostics.get(
                        "motion_reject_details", "{}"
                    )
                    row["flow_loss_details"] = diagnostics.get(
                        "flow_loss_details", "{}"
                    )
                rows.append(row)
                if evidence_handle is not None:
                    evidence_handle.write(
                        json.dumps(
                            {
                                "frame": index,
                                "timestamp": timestamp,
                                "status": status,
                                **(
                                    {
                                        "flow_points": tracker.flow_records,
                                        "motion_hypotheses": tracker.hypothesis_records,
                                        **(
                                            {
                                                "white_candidates": tracker.white_records,
                                                "motion_rejections": tracker.rejection_counts,
                                                "flow_loss_details": tracker.flow_loss_details,
                                            }
                                            if args.variant == "motion_revised"
                                            else {}
                                        ),
                                    }
                                    if args.variant in MOTION_VARIANTS
                                    else {
                                        "matches": tracker.match_records,
                                        "vote_peaks": tracker.vote_peaks,
                                        "winning_matches": tracker.winning_records,
                                    }
                                ),
                                "output_trace": tracker.output_trace,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                if writer is not None:
                    canvas = frame.copy()
                    if (
                        args.variant == "original_voting"
                        or args.variant in GUARDED_VARIANTS
                    ):
                        tracker.draw_votes(canvas)
                    elif args.variant in MOTION_VARIANTS:
                        tracker.draw_motion(canvas)
                    if center is not None:
                        color = (
                            (0, 210, 255)
                            if status in ("observed", "initialized")
                            else (0, 130, 255)
                        )
                        cv2.drawMarker(
                            canvas,
                            tuple(round(value) for value in center),
                            color,
                            cv2.MARKER_CROSS,
                            24,
                            2,
                        )
                    if proxy is not None:
                        cv2.circle(
                            canvas,
                            tuple(round(value) for value in proxy),
                            18,
                            (255, 200, 80),
                            1,
                        )
                    cv2.rectangle(canvas, (0, 0), (width - 1, 55), (15, 15, 15), -1)
                    error_text = "n/a" if error is None else f"{error:.1f}px"
                    support_text = (
                        f"flow={row['flow_support_points']}/{row['flow_points']}"
                        if args.variant in MOTION_VARIANTS
                        else f"votes={votes}"
                    )
                    cv2.putText(
                        canvas,
                        f"{status} t={timestamp:.2f}s {support_text} cursor-proxy error={error_text}",
                        (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (240, 240, 240),
                        1,
                    )
                    cv2.putText(
                        canvas,
                        f"{args.variant} | {row['reject_reason']} | bg={row['background_confidence']:.2f} | cursor proxy is not truth",
                        (8, 43),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (240, 240, 240),
                        1,
                    )
                    writer.write(canvas)
                index += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if evidence_handle is not None:
                evidence_handle.close()
        summary = {
            "video": video.name,
            "variant": args.variant,
            "input_path": str(video.resolve()),
            "fps": fps,
            "roi": spec["roi"],
            "roi_source": "configured" if video.name in specs else "auto_chrome",
            "evaluation_window": [active_start, active_end],
            "window_source": "configured"
            if video.name in specs
            else "auto_chrome_estimate",
            "seed_timestamp": seed_at,
            "initialization": "opening_white_silhouette" if initialized else "failed",
            "seed_errors": sorted(set(seed_errors)),
            "output_video": None if args.no_video else str(output_video),
            "signature": signature,
            **real_metrics(rows, fps),
        }
        summaries.append(summary)
        all_rows.extend(rows)
        with (args.output_dir / f"{video.stem}_frames.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer_csv.writeheader()
            writer_csv.writerows(rows)
        (args.output_dir / f"{video.stem}_complete.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        report = {
            "scope": "authorized local offline evaluation; no game integration or pointer control",
            "tracker": args.variant,
            "truth": "cursor proxy scored separately from optional independent annotations; neither enters tracker",
            "annotation_coordinates": "panel-local pixels; frame index is zero-based source frame",
            "input": "cursor-inpainted baseline"
            if args.variant == "baseline"
            else "raw panel with cursor invalid mask; no inpainted evidence",
            "parameters_frozen": True,
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "videos": summaries,
        }
        report["aggregate"] = aggregate_recording_metrics(all_rows, summaries)
        (args.output_dir / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if all_rows:
        with (args.output_dir / "frames.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer_csv = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer_csv.writeheader()
            writer_csv.writerows(all_rows)
    report = {
        "scope": "authorized local offline evaluation; no game integration or pointer control",
        "tracker": args.variant,
        "truth": "cursor proxy and optional independent truth are evaluation-only",
        "parameters_frozen": True,
        "evaluation_complete": len(summaries) == len(videos),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "videos": summaries,
        "aggregate": aggregate_recording_metrics(all_rows, summaries),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 17, 29])
    parser.add_argument(
        "--scenarios", choices=SCENARIOS + MOTION_SCENARIOS, nargs="+", default=None
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS, default="baseline")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="verify completed recording signatures and resume matching inputs only",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="independent panel-local target CSV: video,frame,x,y; optional phase,visibility",
    )
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=None,
        help="authorized offline recordings directory; recursively evaluates all 测谎录屏*.mp4",
    )
    parser.add_argument(
        "--recording",
        action="append",
        default=[],
        help="optional exact recording filename; repeatable",
    )
    parser.add_argument(
        "--recording-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "ml/lie_videos_config.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.scenarios is None:
        args.scenarios = list(
            SCENARIOS + (MOTION_SCENARIOS if args.variant in MOTION_VARIANTS else ())
        )
    if args.output_dir is None:
        folder = (
            (
                "fragment_voting_real_eval"
                if args.variant == "baseline"
                else f"fragment_voting_real_{args.variant}"
            )
            if args.recordings_dir is not None
            else (
                "fragment_voting_experiment"
                if args.variant == "baseline"
                else f"fragment_voting_synthetic_{args.variant}"
            )
        )
        if args.variant in MOTION_VARIANTS:
            folder = (
                "motion_feedback_real"
                if args.recordings_dir is not None
                else "motion_feedback_synthetic"
            )
        if args.variant == "original_guarded":
            folder = (
                "fragment_voting_original_guarded"
                if args.recordings_dir is not None
                else "fragment_voting_guarded_synthetic"
            )
        if args.variant == "original_voting":
            folder = (
                "fragment_voting_original_voting"
                if args.recordings_dir is not None
                else "fragment_voting_original_synthetic"
            )
        args.output_dir = Path(__file__).resolve().parents[1] / "log" / folder
    if args.recordings_dir is not None:
        return evaluate_recordings(args)
    if args.recording:
        parser.error("--recording requires --recordings-dir")
    if (
        not math.isfinite(args.fps)
        or not math.isfinite(args.seconds)
        or args.fps <= 0
        or args.seconds <= 0
    ):
        parser.error("fps and seconds must be finite and positive")
    if round(args.fps * args.seconds) < 1:
        parser.error("duration must contain at least one update frame")
    all_rows, runs = [], []
    for seed in dict.fromkeys(args.seeds):
        for scenario in dict.fromkeys(args.scenarios):
            rows, summary = run_experiment(
                scenario,
                seed,
                args.output_dir,
                fps=args.fps,
                seconds=args.seconds,
                save_video=not args.no_video,
                variant=args.variant,
            )
            all_rows.extend(rows)
            runs.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    with (args.output_dir / "frames.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    report = {
        "scope": "synthetic visible-edge experiment; no external video, game integration or pointer control",
        "initialization": "explicit silhouette and center at t=0; truth is evaluation-only after initialization",
        "evaluation": "all update frames, including invisible frames; initialization excluded",
        "runs": runs,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
