"""YOLO candidate detector used by the lie-detector tracker.

The detector only proposes shape boxes.  Target identity is still established
from the initial white highlight and maintained by :mod:`LieDetectorTracker`.
Keeping these responsibilities separate also makes it impossible for the
green evaluation cursor to leak into the detector input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.engine.LieDetectorTracker import ShapeDetection


class LieShapeYoloDetector:
    """Convert Ultralytics detections into tracker shape candidates."""

    def __init__(
        self,
        weights: str | Path,
        *,
        confidence: float = 0.01,
        iou: float = 0.55,
        image_size: int = 640,
        max_detections: int = 160,
        device: Optional[str] = None,
        inference_stride: int = 1,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "YOLO lie-shape detection requires the 'ultralytics' package"
            ) from exc

        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights}")
        self.model = YOLO(str(self.weights))
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.image_size = int(image_size)
        self.max_detections = int(max_detections)
        self.device = device
        self.inference_stride = max(1, int(inference_stride))
        self.frame_index = 0

    def reset(self) -> None:
        self.frame_index = 0

    def detect(
        self,
        frame_bgr: np.ndarray,
        reference_radius: Optional[float] = None,
    ) -> list[ShapeDetection]:
        """Detect plausible game shapes in a cursor-cleaned BGR frame."""

        run_inference = self.frame_index % self.inference_stride == 0
        self.frame_index += 1
        if not run_inference:
            return []

        kwargs = {
            "source": frame_bgr,
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.image_size,
            "max_det": self.max_detections,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        result = self.model.predict(**kwargs)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        height, width = frame_bgr.shape[:2]
        detections: list[ShapeDetection] = []
        for xyxy, confidence in zip(boxes, confidences):
            x1, y1, x2, y2 = (float(value) for value in xyxy)
            box_width = max(1.0, x2 - x1)
            box_height = max(1.0, y2 - y1)
            aspect = box_width / box_height
            observed_radius = 0.25 * (box_width + box_height)
            if not 0.30 <= aspect <= 3.30:
                continue
            if reference_radius is not None and not (
                0.48 * reference_radius
                <= observed_radius
                <= 1.75 * reference_radius
            ):
                continue
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            radius = float(reference_radius or observed_radius)
            fixed_size = max(2, int(round(2.0 * radius)))
            detections.append(
                ShapeDetection(
                    center=(cx, cy),
                    radius=radius,
                    bbox=(
                        max(0, min(width - 1, int(round(cx - radius)))),
                        max(0, min(height - 1, int(round(cy - radius)))),
                        fixed_size,
                        fixed_size,
                    ),
                    score=float(confidence),
                    source="yolo",
                    observed_radius=float(observed_radius),
                    yolo_confidence=float(confidence),
                )
            )
        return detections
