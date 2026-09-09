"""Runtime gate for the in-game lie-detector mouse-follow challenge.

Three responsibilities, in order:

1. Decide whether the challenge panel is really on screen.  ``detect_frame``
   from :mod:`ml.detect_lie_panels` only *estimates the ROI* and happily
   anchors on the chat bar of an ordinary gameplay frame, so panel presence is
   established by matching the unique ``谎言探测仪`` title text instead.
2. While the panel is up, run the tracker and expose a pointer target, but
   only when the tracker reports the estimate as actionable.
3. After the panel closes, the NPC posts a success dialog that blocks the
   character until its ``确认`` button is clicked, so keep automation paused
   and expose that button until the dialog is gone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from ml.detect_lie_panels import detect_frame
from src.engine.LieDetectorTracker import (
    LieDetectorTracker,
    LieDetectorTrackingResult,
)
from src.engine.LieShapeYoloDetector import LieShapeYoloDetector
from src.utils.logger import logger

# The bundled title template was cut from a 1280x720 capture, so the expected
# scale for any other frame follows that reference.
_TEMPLATE_REFERENCE_SIZE = (1280, 720)
_TEMPLATE_SCALE_TRIALS = (0.92, 1.0, 1.08)
# Title text lives in the upper part of the panel; skip the rest of the frame.
_TITLE_SEARCH_HEIGHT_RATIO = 0.55

PHASE_IDLE = "idle"
PHASE_PANEL = "panel"
PHASE_CONFIRM = "confirm"


@dataclass
class LieDetectorRuntimeResult:
    engaged: bool
    phase: str = PHASE_IDLE
    panel_roi: Optional[tuple[int, int, int, int]] = None
    tracking: Optional[LieDetectorTrackingResult] = None
    target_frame: Optional[tuple[float, float]] = None
    confirm_click: Optional[tuple[float, float]] = None


class LiePanelGate:
    """Match the panel's title text to prove the challenge is on screen.

    This runs on every gameplay frame, so the common "no panel" case must stay
    cheap: only the scale implied by the frame size is tried each frame, and
    the ±8% variants are swept periodically (and cached once one wins).
    """

    def __init__(
        self,
        template_path: Path,
        threshold: float,
        *,
        sweep_interval: int = 8,
    ) -> None:
        template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if template is None:
            raise FileNotFoundError(
                f"lie-detector title template not found: {template_path}"
            )
        self.template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.threshold = float(threshold)
        self.sweep_interval = max(1, int(sweep_interval))
        self._cached_scale: Optional[tuple[float, float]] = None
        self._cached_for_size: Optional[tuple[int, int]] = None
        self._calls = 0

    def _candidate_scales(
        self, frame_size: tuple[int, int]
    ) -> list[tuple[float, float]]:
        base_x = frame_size[0] / _TEMPLATE_REFERENCE_SIZE[0]
        base_y = frame_size[1] / _TEMPLATE_REFERENCE_SIZE[1]
        if self._cached_for_size != frame_size:
            self._cached_scale = None
            self._cached_for_size = frame_size
        primary = self._cached_scale or (base_x, base_y)
        if self._calls % self.sweep_interval:
            return [primary]
        variants = [
            (base_x * trial, base_y * trial)
            for trial in _TEMPLATE_SCALE_TRIALS
        ]
        return [primary] + [scale for scale in variants if scale != primary]

    def score(self, frame_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        region = gray[
            : max(1, int(_TITLE_SEARCH_HEIGHT_RATIO * gray.shape[0])), :
        ]
        height, width = self.template.shape
        frame_size = (frame_bgr.shape[1], frame_bgr.shape[0])
        scales = self._candidate_scales(frame_size)
        self._calls += 1

        best = -1.0
        best_scale = None
        for scale_x, scale_y in scales:
            resized = cv2.resize(
                self.template,
                (
                    max(4, int(round(width * scale_x))),
                    max(4, int(round(height * scale_y))),
                ),
            )
            if (
                resized.shape[0] >= region.shape[0]
                or resized.shape[1] >= region.shape[1]
            ):
                continue
            matched = cv2.matchTemplate(region, resized, cv2.TM_CCOEFF_NORMED)
            peak = float(cv2.minMaxLoc(matched)[1])
            if peak > best:
                best = peak
                best_scale = (scale_x, scale_y)
            if peak >= self.threshold:
                break
        if best >= self.threshold and best_scale is not None:
            self._cached_scale = best_scale
        return best

    def present(self, frame_bgr: np.ndarray) -> bool:
        return self.score(frame_bgr) >= self.threshold


def _orange_mask(hsv: np.ndarray) -> np.ndarray:
    return (
        (hsv[:, :, 0] >= 5) & (hsv[:, :, 0] <= 25)
        & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 120)
    ).astype(np.uint8)


def _dialog_candidates(
    hsv: np.ndarray, frame_width: int, frame_height: int
) -> list[tuple[int, int, int]]:
    """Return ``(x, y, width)`` of plausible NPC message boxes."""

    blue = (
        (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 112)
        & (hsv[:, :, 1] >= 40) & (hsv[:, :, 1] <= 160)
        & (hsv[:, :, 2] >= 150)
    ).astype(np.uint8) * 255
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    contours, _ = cv2.findContours(
        blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates: list[tuple[float, tuple[int, int, int]]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        # The box never touches a frame edge, unlike the minimap and the
        # bottom chat strip.
        if x <= 0 or y <= 0 or x + width >= frame_width:
            continue
        if not 0.10 * frame_width <= width <= 0.45 * frame_width:
            continue
        if not 0.05 * frame_height <= height <= 0.35 * frame_height:
            continue
        # Nearby characters and skill effects can merge into the mask, so the
        # aspect/fill gates stay loose and the orange button is the real proof.
        if not 1.6 <= width / max(height, 1) <= 3.6:
            continue
        if cv2.contourArea(contour) / max(width * height, 1) < 0.65:
            continue
        if not 0.30 <= (x + 0.5 * width) / frame_width <= 0.70:
            continue
        candidates.append((width * height, (x, y, width)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [box for _score, box in candidates]


def find_success_confirm(
    frame_bgr: np.ndarray,
) -> Optional[tuple[float, float]]:
    """Locate the ``确认`` button of the lie-detector success dialog.

    The button is derived from the blue message box rather than found by
    contour search, because the sandy terrain behind the dialog carries the
    same orange tones.  Only the box's top edge and width are used (its lower
    edge drifts when characters or effects merge into the blue mask), and the
    prediction is then snapped onto real button pixels.
    """

    frame_height, frame_width = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    orange = _orange_mask(hsv)

    for x, y, width in _dialog_candidates(hsv, frame_width, frame_height):
        # Ratios measured against the dialog width, which is stable across the
        # 720p/1080p/engine-scale captures.
        predicted = (x + 0.918 * width, y + 0.455 * width)
        reach = max(5, int(round(0.06 * width)))
        cx, cy = int(round(predicted[0])), int(round(predicted[1]))
        x0, y0 = max(0, cx - reach), max(0, cy - reach)
        window = orange[y0:cy + reach, x0:cx + reach]
        if window.size == 0 or float(window.mean()) < 0.20:
            continue
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            cv2.morphologyEx(window, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        )
        best_index = None
        best_area = 0
        for index in range(1, count):
            area = stats[index, cv2.CC_STAT_AREA]
            if area > best_area:
                best_area = area
                best_index = index
        if best_index is None or best_area < 0.10 * window.size:
            continue
        return (
            float(x0 + centroids[best_index][0]),
            float(y0 + centroids[best_index][1]),
        )
    return None


class LieDetectorRuntime:
    """Own the panel gate, tracker lifecycle, and success-dialog handling."""

    def __init__(
        self,
        config: dict,
        *,
        panel_gate: Optional[LiePanelGate] = None,
        roi_detector: Callable[
            [np.ndarray], Optional[tuple[int, int, int, int]]
        ] = detect_frame,
        confirm_detector: Callable[
            [np.ndarray], Optional[tuple[float, float]]
        ] = find_success_confirm,
        tracker: Optional[LieDetectorTracker] = None,
    ) -> None:
        self.config = config
        self.roi_detector = roi_detector
        self.confirm_detector = confirm_detector
        self.confirm_frames = max(1, int(config.get("panel_confirm_frames", 2)))
        self.miss_frames = max(1, int(config.get("panel_miss_frames", 3)))
        self.confirm_wait_seconds = float(
            config.get("confirm_wait_seconds", 12.0)
        )
        self.confirm_click_interval = float(
            config.get("confirm_click_interval", 0.8)
        )
        self.confirm_max_clicks = max(1, int(config.get("confirm_max_clicks", 6)))
        self._panel_gate = panel_gate
        self._tracker = tracker
        self._reset_state()

    # -- lifecycle ---------------------------------------------------------
    def warm_up(self) -> None:
        """Load template and model weights before the challenge appears."""
        self._ensure_panel_gate()
        self._ensure_tracker()

    def reset(self) -> None:
        self._reset_tracker()
        self._reset_state()

    def _reset_state(self) -> None:
        self.phase = PHASE_IDLE
        self._seen_streak = 0
        self._miss_streak = 0
        self._roi: Optional[tuple[int, int, int, int]] = None
        self._confirm_deadline = 0.0
        self._confirm_clicks = 0
        # Sentinel so the very first sighting of the dialog clicks at once
        # instead of waiting out one click interval.
        self._last_confirm_click_at = float("-inf")

    def _reset_tracker(self) -> None:
        if self._tracker is not None:
            self._tracker.reset()
            detector = getattr(self._tracker, "candidate_detector", None)
            if detector is not None and hasattr(detector, "reset"):
                detector.reset()

    def _ensure_panel_gate(self) -> LiePanelGate:
        if self._panel_gate is None:
            self._panel_gate = LiePanelGate(
                Path(
                    self.config.get(
                        "title_template", "misc/lie_detector_title_cn.png"
                    )
                ),
                float(self.config.get("title_match_threshold", 0.70)),
            )
        return self._panel_gate

    def _ensure_tracker(self) -> LieDetectorTracker:
        if self._tracker is None:
            detector = LieShapeYoloDetector(
                Path(self.config.get("model", "models/lie_shape_yolo_manual.pt")),
                confidence=float(self.config.get("confidence", 0.28)),
                image_size=int(self.config.get("image_size", 768)),
                inference_stride=int(self.config.get("inference_stride", 1)),
                device=self.config.get("device") or None,
            )
            self._tracker = LieDetectorTracker(
                candidate_detector=detector,
                multi_hypothesis_identity=bool(
                    self.config.get("multi_hypothesis_identity", True)
                ),
            )
        return self._tracker

    # -- per-frame ---------------------------------------------------------
    @staticmethod
    def _valid_roi(roi: tuple[int, int, int, int], frame: np.ndarray) -> bool:
        x, y, width, height = roi
        frame_height, frame_width = frame.shape[:2]
        return (
            width >= 120
            and height >= 100
            and x >= 0
            and y >= 0
            and x + width <= frame_width
            and y + height <= frame_height
        )

    def update(
        self, frame_bgr: np.ndarray, timestamp: float
    ) -> LieDetectorRuntimeResult:
        if self._ensure_panel_gate().present(frame_bgr):
            return self._update_panel(frame_bgr, timestamp)
        if self.phase == PHASE_PANEL:
            self._miss_streak += 1
            self._seen_streak = 0
            if self._miss_streak < self.miss_frames:
                return LieDetectorRuntimeResult(
                    engaged=True, phase=PHASE_PANEL, panel_roi=self._roi
                )
            self._enter_confirm_phase(timestamp)
        if self.phase == PHASE_CONFIRM:
            return self._update_confirm(frame_bgr, timestamp)
        return LieDetectorRuntimeResult(engaged=False, phase=PHASE_IDLE)

    def _update_panel(
        self, frame_bgr: np.ndarray, timestamp: float
    ) -> LieDetectorRuntimeResult:
        self._miss_streak = 0
        self._seen_streak += 1
        self.phase = PHASE_PANEL

        roi = self.roi_detector(frame_bgr)
        if roi is not None and self._valid_roi(roi, frame_bgr):
            self._roi = roi
        if self._roi is None:
            # Panel proven present but ROI not resolved yet: still pause the
            # bot, just don't touch the pointer.
            return LieDetectorRuntimeResult(engaged=True, phase=PHASE_PANEL)

        x, y, width, height = self._roi
        tracking = self._ensure_tracker().update(
            frame_bgr[y:y + height, x:x + width], timestamp
        )
        target = None
        if (
            self._seen_streak >= self.confirm_frames
            and tracking.actionable
            and tracking.center is not None
        ):
            target = (
                float(x) + tracking.center[0],
                float(y) + tracking.center[1],
            )
        return LieDetectorRuntimeResult(
            engaged=True,
            phase=PHASE_PANEL,
            panel_roi=self._roi,
            tracking=tracking,
            target_frame=target,
        )

    def _enter_confirm_phase(self, timestamp: float) -> None:
        self._reset_tracker()
        self.phase = PHASE_CONFIRM
        self._roi = None
        self._seen_streak = 0
        self._miss_streak = 0
        self._confirm_deadline = timestamp + self.confirm_wait_seconds
        self._confirm_clicks = 0
        self._last_confirm_click_at = float("-inf")
        logger.info(
            "[Lie Detector] Panel closed; waiting for the success dialog "
            "so its 确认 button can be clicked."
        )

    def _update_confirm(
        self, frame_bgr: np.ndarray, timestamp: float
    ) -> LieDetectorRuntimeResult:
        if timestamp >= self._confirm_deadline:
            if self._confirm_clicks == 0:
                logger.info(
                    "[Lie Detector] No success dialog appeared before the "
                    "timeout; resuming normal automation."
                )
            self.reset()
            return LieDetectorRuntimeResult(engaged=False, phase=PHASE_IDLE)

        button = self.confirm_detector(frame_bgr)
        if button is None:
            if self._confirm_clicks > 0:
                # The dialog we just clicked is gone.
                self.reset()
                return LieDetectorRuntimeResult(engaged=False, phase=PHASE_IDLE)
            return LieDetectorRuntimeResult(engaged=True, phase=PHASE_CONFIRM)

        click = None
        if (
            self._confirm_clicks < self.confirm_max_clicks
            and timestamp - self._last_confirm_click_at
            >= self.confirm_click_interval
        ):
            click = button
            self._confirm_clicks += 1
            self._last_confirm_click_at = timestamp
        return LieDetectorRuntimeResult(
            engaged=True,
            phase=PHASE_CONFIRM,
            confirm_click=click,
        )

    # -- debug -------------------------------------------------------------
    @staticmethod
    def draw_debug(frame: np.ndarray, result: LieDetectorRuntimeResult) -> None:
        if result.confirm_click is not None or result.phase == PHASE_CONFIRM:
            color = (0, 200, 255)
            cv2.putText(
                frame,
                "LIE: CONFIRM",
                (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )
            if result.confirm_click is not None:
                point = tuple(
                    int(round(value)) for value in result.confirm_click
                )
                cv2.circle(frame, point, 10, color, 2)
            return

        if result.panel_roi is None:
            return
        x, y, width, height = result.panel_roi
        tracking = result.tracking
        actionable = bool(tracking is not None and tracking.actionable)
        color = (0, 220, 0) if actionable else (0, 180, 255)
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            frame,
            "LIE: FOLLOW" if actionable else "LIE: HOLD",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )
        if result.target_frame is not None:
            target = tuple(int(round(value)) for value in result.target_frame)
            cv2.circle(frame, target, 8, color, 2)
