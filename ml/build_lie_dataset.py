#!/usr/bin/env python3
"""Build an exactly-labelled lie-detector YOLO dataset from recordings.

The recordings do not contain renderer ground truth, and contour detection on
the raw frames produces many false positives from the textured background.
Instead of turning those noisy detections into labels, this builder extracts:

* a robust temporal-median background from each recording; and
* the highlighted white target's actual silhouette.

It then renders rotated/faded copies of that silhouette on the recovered game
background.  Since the builder controls every placement, the YOLO boxes are
exact.  The output is a separate single-class dataset and never touches the
existing monster dataset.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random
import shutil

import cv2
import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parent


@dataclass(frozen=True)
class RecordingSpec:
    filename: str
    # Inner tan canvas in the full recording: x, y, width, height.
    roi: tuple[int, int, int, int]
    # Clean sub-area of the canvas.  This excludes subtitles/log windows.
    safe_crop: tuple[int, int, int, int]
    seed_time: float
    active_start: float
    active_end: float
    family: str


RECORDINGS = (
    RecordingSpec(
        "测谎视频.mp4",
        (418, 198, 1012, 676),
        (116, 0, 780, 520),
        0.0,
        0.3,
        13.0,
        "circle",
    ),
    RecordingSpec(
        "测谎视频2.mp4",
        (290, 109, 700, 464),
        (0, 0, 700, 464),
        8.5,
        8.5,
        24.5,
        "star",
    ),
    RecordingSpec(
        "测谎视频3.mp4",
        (474, 213, 982, 656),
        (0, 0, 982, 656),
        0.0,
        0.3,
        12.3,
        "irregular",
    ),
    RecordingSpec(
        "测谎视频4.mp4",
        (429, 150, 1050, 700),
        (0, 0, 900, 600),
        0.0,
        0.3,
        12.8,
        "rounded_polygon",
    ),
)


@dataclass
class ShapeAsset:
    spec: RecordingSpec
    background: np.ndarray
    texture_frames: list[np.ndarray]
    points: np.ndarray
    base_diameter: float
    min_count: int
    max_count: int


@dataclass(frozen=True)
class ProceduralShapeSpec:
    family: str
    kind: str
    diameter: float
    min_count: int
    max_count: int


# Together with the four recording-derived silhouettes, these make 15 shape
# families.  Square and diamond are intentionally not separate: once arbitrary
# rotation is enabled they are the same geometry.
PROCEDURAL_SHAPES = (
    ProceduralShapeSpec("triangle", "regular_3", 135.0, 7, 11),
    ProceduralShapeSpec("square", "regular_4", 145.0, 6, 10),
    ProceduralShapeSpec("rectangle", "rectangle", 175.0, 5, 9),
    ProceduralShapeSpec("pentagon", "regular_5", 145.0, 6, 10),
    ProceduralShapeSpec("hexagon", "regular_6", 145.0, 6, 10),
    ProceduralShapeSpec("octagon", "regular_8", 145.0, 6, 10),
    ProceduralShapeSpec("capsule", "capsule", 175.0, 5, 9),
    ProceduralShapeSpec("trapezoid", "trapezoid", 155.0, 6, 10),
    ProceduralShapeSpec("concave_arrow", "concave_arrow", 155.0, 6, 10),
    ProceduralShapeSpec("cross", "cross", 145.0, 6, 10),
    ProceduralShapeSpec("gear", "gear_8", 135.0, 7, 11),
)


# Ten additional asymmetric silhouettes.  Each seed defines a genuinely
# different radial geometry; the last five use Chaikin corner cutting to cover
# rounded organic outlines as well as sharp concave polygons.
IRREGULAR_SHAPES = (
    ProceduralShapeSpec("irregular_01", "irregular:101:5:0", 150.0, 6, 10),
    ProceduralShapeSpec("irregular_02", "irregular:211:6:0", 150.0, 6, 10),
    ProceduralShapeSpec("irregular_03", "irregular:307:7:0", 145.0, 6, 10),
    ProceduralShapeSpec("irregular_04", "irregular:401:8:0", 145.0, 6, 10),
    ProceduralShapeSpec("irregular_05", "irregular:503:9:0", 140.0, 7, 11),
    ProceduralShapeSpec("irregular_06", "irregular:601:6:2", 160.0, 6, 9),
    ProceduralShapeSpec("irregular_07", "irregular:701:7:2", 155.0, 6, 10),
    ProceduralShapeSpec("irregular_08", "irregular:809:8:2", 150.0, 6, 10),
    ProceduralShapeSpec("irregular_09", "irregular:907:10:1", 145.0, 6, 10),
    ProceduralShapeSpec("irregular_10", "irregular:1009:12:2", 145.0, 6, 10),
)

ALL_PROCEDURAL_SHAPES = PROCEDURAL_SHAPES + IRREGULAR_SHAPES


def read_frame(capture: cv2.VideoCapture, fps: float, timestamp: float) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(timestamp * fps)))
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Unable to decode frame at {timestamp:.3f}s")
    return frame


def crop_canvas(frame: np.ndarray, spec: RecordingSpec) -> np.ndarray:
    x, y, width, height = spec.roi
    if y + height > frame.shape[0] or x + width > frame.shape[1]:
        raise ValueError(f"ROI {spec.roi} exceeds frame {frame.shape[1]}x{frame.shape[0]}")
    return frame[y:y + height, x:x + width]


def green_cursor_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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


def remove_cursor(frame: np.ndarray) -> np.ndarray:
    mask = cv2.dilate(
        green_cursor_mask(frame),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    )
    if not np.any(mask):
        return frame.copy()
    return cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)


def highlighted_contour(canvas: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Extract the large, low-saturation white target shown during countdown."""

    clean = remove_cursor(canvas)
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(clean, cv2.COLOR_BGR2HSV)
    mask = np.where((gray >= 205) & (hsv[:, :, 1] <= 85), 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, np.ndarray, tuple[int, int, int, int]]] = []
    height, width = gray.shape
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(float(h), 1.0)
        if not (
            900 <= area <= 50000
            and 30 <= w <= min(280, width // 2)
            and 30 <= h <= min(280, height // 2)
            and 0.45 <= aspect <= 2.2
            and y >= int(0.18 * height)
        ):
            continue
        # Countdown glyphs are narrow and usually above the target.  Area and
        # squareness make the highlighted shape win on all four recordings.
        rank = area * math.exp(-0.7 * abs(math.log(max(aspect, 1e-3))))
        candidates.append((rank, contour, (x, y, w, h)))
    if not candidates:
        raise RuntimeError("Unable to find highlighted white target")
    _, contour, bbox = max(candidates, key=lambda item: item[0])
    return contour, bbox


def temporal_background(
    capture: cv2.VideoCapture,
    fps: float,
    spec: RecordingSpec,
    output_size: tuple[int, int],
    sample_count: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Recover a clean-ish canvas using a temporal median over moving shapes."""

    sx, sy, sw, sh = spec.safe_crop
    frames: list[np.ndarray] = []
    for timestamp in np.linspace(spec.active_start, spec.active_end, sample_count):
        canvas = crop_canvas(read_frame(capture, fps, float(timestamp)), spec)
        canvas = remove_cursor(canvas)
        safe = canvas[sy:sy + sh, sx:sx + sw]
        safe = cv2.resize(safe, output_size, interpolation=cv2.INTER_AREA)
        frames.append(safe)
    # The shape sprites move across the canvas, so fewer than half of the
    # observations at a pixel contain a sprite boundary.  Median removes them
    # much more reliably than averaging.
    background = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    return cv2.bilateralFilter(background, 5, 15, 15), frames


def load_asset(
    spec: RecordingSpec,
    output_size: tuple[int, int],
    background_samples: int,
) -> ShapeAsset:
    path = REPO / spec.filename
    if not path.exists():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    seed = crop_canvas(read_frame(capture, fps, spec.seed_time), spec)
    contour, bbox = highlighted_contour(seed)
    x, y, width, height = bbox
    center = np.array([x + width / 2.0, y + height / 2.0], dtype=np.float32)
    points = contour.reshape(-1, 2).astype(np.float32) - center

    _, _, safe_width, safe_height = spec.safe_crop
    out_width, out_height = output_size
    scale_x = out_width / float(safe_width)
    scale_y = out_height / float(safe_height)
    # The safe crops all have ~1.5 aspect ratio, so this is almost isotropic.
    points[:, 0] *= scale_x
    points[:, 1] *= scale_y
    diameter = max(width * scale_x, height * scale_y)
    background, texture_frames = temporal_background(
        capture,
        fps,
        spec,
        output_size,
        background_samples,
    )
    capture.release()
    count_ranges = {
        "circle": (7, 11),
        "star": (9, 14),
        "irregular": (5, 8),
        "rounded_polygon": (4, 7),
    }
    min_count, max_count = count_ranges[spec.family]
    return ShapeAsset(
        spec,
        background,
        texture_frames,
        points,
        float(diameter),
        min_count,
        max_count,
    )


def regular_polygon(vertex_count: int) -> np.ndarray:
    angles = np.linspace(-math.pi / 2.0, 3.0 * math.pi / 2.0, vertex_count, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)


def capsule_points(arc_samples: int = 18) -> np.ndarray:
    radius = 0.34
    half_straight = 0.50
    right_angles = np.linspace(-math.pi / 2.0, math.pi / 2.0, arc_samples)
    left_angles = np.linspace(math.pi / 2.0, 3.0 * math.pi / 2.0, arc_samples)
    right = np.column_stack(
        (half_straight + radius * np.cos(right_angles), radius * np.sin(right_angles))
    )
    left = np.column_stack(
        (-half_straight + radius * np.cos(left_angles), radius * np.sin(left_angles))
    )
    return np.vstack((right, left)).astype(np.float32)


def chaikin_smooth(points: np.ndarray, iterations: int) -> np.ndarray:
    result = points.astype(np.float32)
    for _ in range(iterations):
        following = np.roll(result, -1, axis=0)
        first = 0.75 * result + 0.25 * following
        second = 0.25 * result + 0.75 * following
        result = np.column_stack((first, second)).reshape(-1, 2)
    return result


def irregular_radial_points(seed: int, vertex_count: int, smoothing: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    step = 2.0 * math.pi / vertex_count
    angles = np.arange(vertex_count, dtype=np.float32) * step - math.pi / 2.0
    angles += rng.uniform(-0.22 * step, 0.22 * step, vertex_count)
    radii = rng.uniform(0.58, 1.0, vertex_count)
    # Force at least one pronounced inward notch.  This keeps the families
    # visibly irregular even after arbitrary rotation and scale augmentation.
    notch = seed % vertex_count
    radii[notch] *= rng.uniform(0.48, 0.68)
    points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles))).astype(
        np.float32
    )
    return chaikin_smooth(points, smoothing)


def procedural_points(kind: str, diameter: float) -> np.ndarray:
    if kind.startswith("regular_"):
        points = regular_polygon(int(kind.split("_", 1)[1]))
    elif kind.startswith("irregular:"):
        _, seed, vertex_count, smoothing = kind.split(":")
        points = irregular_radial_points(int(seed), int(vertex_count), int(smoothing))
    elif kind == "rectangle":
        points = np.array(
            [[-0.62, -0.36], [0.62, -0.36], [0.62, 0.36], [-0.62, 0.36]],
            dtype=np.float32,
        )
    elif kind == "capsule":
        points = capsule_points()
    elif kind == "trapezoid":
        points = np.array(
            [[-0.62, 0.43], [0.62, 0.43], [0.43, -0.43], [-0.35, -0.43]],
            dtype=np.float32,
        )
    elif kind == "concave_arrow":
        points = np.array(
            [
                [-0.62, -0.22],
                [0.02, -0.22],
                [0.02, -0.48],
                [0.64, 0.0],
                [0.02, 0.48],
                [0.02, 0.22],
                [-0.62, 0.22],
            ],
            dtype=np.float32,
        )
    elif kind == "cross":
        points = np.array(
            [
                [-0.18, -0.62],
                [0.18, -0.62],
                [0.18, -0.18],
                [0.62, -0.18],
                [0.62, 0.18],
                [0.18, 0.18],
                [0.18, 0.62],
                [-0.18, 0.62],
                [-0.18, 0.18],
                [-0.62, 0.18],
                [-0.62, -0.18],
                [-0.18, -0.18],
            ],
            dtype=np.float32,
        )
    elif kind == "gear_8":
        angles = np.linspace(-math.pi / 2.0, 3.0 * math.pi / 2.0, 16, endpoint=False)
        radii = np.where(np.arange(16) % 2 == 0, 1.0, 0.72)
        points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles))).astype(
            np.float32
        )
    else:
        raise ValueError(f"Unknown procedural shape kind: {kind}")

    points -= (np.min(points, axis=0) + np.max(points, axis=0)) / 2.0
    extent = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1e-6)
    return points * (diameter / extent)


def make_procedural_assets(recorded_assets: list[ShapeAsset]) -> list[ShapeAsset]:
    assets: list[ShapeAsset] = []
    for index, shape in enumerate(ALL_PROCEDURAL_SHAPES):
        source = recorded_assets[index % len(recorded_assets)]
        spec = replace(source.spec, family=shape.family)
        assets.append(
            ShapeAsset(
                spec=spec,
                background=source.background,
                texture_frames=source.texture_frames,
                points=procedural_points(shape.kind, shape.diameter),
                base_diameter=shape.diameter,
                min_count=shape.min_count,
                max_count=shape.max_count,
            )
        )
    return assets


def quilt_background(asset: ShapeAsset, rng: random.Random) -> np.ndarray:
    """Make a sharp background while breaking all pre-existing full shapes.

    A temporal median removes old shapes but also blurs the busy rock texture.
    Small overlapping patches from real frames restore that texture.  Patches
    are deliberately smaller than every target silhouette, so an original
    recording's full unlabelled boundary cannot survive the quilting step.
    """

    height, width = asset.background.shape[:2]
    patch_size = 96
    stride = 64
    axis = np.hanning(patch_size).astype(np.float32)
    feather = np.maximum(np.outer(axis, axis), 0.035)[:, :, None]
    total = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    frames = asset.texture_frames
    for y in range(-32, height, stride):
        for x in range(-32, width, stride):
            frame = frames[rng.randrange(len(frames))]
            source_x = rng.randint(0, max(0, width - patch_size))
            source_y = rng.randint(0, max(0, height - patch_size))
            patch = frame[
                source_y:source_y + patch_size,
                source_x:source_x + patch_size,
            ]
            if rng.random() < 0.5:
                patch = cv2.flip(patch, 1)
            if rng.random() < 0.5:
                patch = cv2.flip(patch, 0)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + patch_size), min(height, y + patch_size)
            px1, py1 = x1 - x, y1 - y
            px2, py2 = px1 + (x2 - x1), py1 + (y2 - y1)
            local_weight = feather[py1:py2, px1:px2]
            total[y1:y2, x1:x2] += patch[py1:py2, px1:px2] * local_weight
            weights[y1:y2, x1:x2] += local_weight
    quilt = total / np.maximum(weights, 1e-5)
    # Blend a little median background back in so seams do not become a model
    # shortcut, while retaining most high-frequency detail from real frames.
    return np.clip(0.86 * quilt + 0.14 * asset.background, 0, 255).astype(np.uint8)


def transformed_polygon(
    points: np.ndarray,
    center: tuple[float, float],
    scale: float,
    angle_degrees: float,
) -> np.ndarray:
    angle = math.radians(angle_degrees)
    matrix = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float32,
    )
    transformed = (points @ matrix.T) * scale
    transformed += np.asarray(center, dtype=np.float32)
    return np.rint(transformed).astype(np.int32)


def bbox_from_polygon(
    polygon: np.ndarray,
    image_width: int,
    image_height: int,
    padding: int = 5,
) -> tuple[int, int, int, int] | None:
    x1 = max(0, int(np.min(polygon[:, 0])) - padding)
    y1 = max(0, int(np.min(polygon[:, 1])) - padding)
    x2 = min(image_width, int(np.max(polygon[:, 0])) + padding + 1)
    y2 = min(image_height, int(np.max(polygon[:, 1])) + padding + 1)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    return x1, y1, x2, y2


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if not intersection:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / float(area_a + area_b - intersection)


def render_shape(
    image: np.ndarray,
    background: np.ndarray,
    polygon: np.ndarray,
    rng: random.Random,
    highlighted: bool,
) -> None:
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255, lineType=cv2.LINE_AA)
    inside = mask > 0

    if highlighted:
        image[inside] = (242, 244, 242)
    else:
        dx = rng.randint(-35, 35)
        dy = rng.randint(-35, 35)
        ys, xs = np.nonzero(inside)
        source_y = np.clip(ys + dy, 0, height - 1)
        source_x = np.clip(xs + dx, 0, width - 1)
        shifted = background[source_y, source_x].astype(np.float32)
        original = image[ys, xs].astype(np.float32)
        opacity = rng.uniform(0.58, 0.86)
        image[ys, xs] = np.clip(
            opacity * shifted + (1.0 - opacity) * original,
            0,
            255,
        ).astype(np.uint8)

    # The game uses an embossed double boundary: dark shadow outside and a
    # light inner ridge.  Small per-image variation avoids a single fixed RGB
    # shortcut while keeping the visual domain faithful.
    dark = (
        rng.randint(55, 78),
        rng.randint(91, 119),
        rng.randint(119, 151),
    )
    light = (
        rng.randint(135, 166),
        rng.randint(177, 207),
        rng.randint(201, 229),
    )
    shadow = polygon + np.array([2, 3], dtype=np.int32)
    cv2.polylines(image, [shadow], True, (45, 67, 83), 8, cv2.LINE_AA)
    cv2.polylines(image, [polygon], True, dark, 6, cv2.LINE_AA)
    cv2.polylines(image, [polygon], True, light, 2, cv2.LINE_AA)


def augment_image(image: np.ndarray, rng: random.Random) -> np.ndarray:
    result = image.astype(np.float32)
    gain = rng.uniform(0.90, 1.10)
    bias = rng.uniform(-7.0, 7.0)
    result = np.clip(result * gain + bias, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        result = cv2.GaussianBlur(result, (3, 3), rng.uniform(0.25, 0.75))
    return result


def render_sample(
    asset: ShapeAsset,
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], dict[str, object]]:
    background = quilt_background(asset, rng)
    height, width = background.shape[:2]
    image = background.copy()

    # Gentle background re-framing produces many domains without introducing
    # unrealistic mosaic seams.
    scale = rng.uniform(1.00, 1.08)
    scaled = cv2.resize(background, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    max_x = max(0, scaled.shape[1] - width)
    max_y = max(0, scaled.shape[0] - height)
    crop_x = rng.randint(0, max_x) if max_x else 0
    crop_y = rng.randint(0, max_y) if max_y else 0
    image = scaled[crop_y:crop_y + height, crop_x:crop_x + width].copy()
    render_background = image.copy()

    diameter = asset.base_diameter
    count = rng.randint(asset.min_count, asset.max_count)
    highlighted_index = rng.randrange(count) if rng.random() < 0.14 else -1

    boxes: list[tuple[int, int, int, int]] = []
    polygons: list[np.ndarray] = []
    attempts = 0
    while len(boxes) < count and attempts < count * 80:
        attempts += 1
        object_scale = rng.uniform(0.78, 1.12)
        radius = 0.55 * diameter * object_scale
        center = (
            rng.uniform(-0.18 * radius, width + 0.18 * radius),
            rng.uniform(-0.18 * radius, height + 0.18 * radius),
        )
        polygon = transformed_polygon(
            asset.points,
            center,
            object_scale,
            rng.uniform(0.0, 360.0),
        )
        box = bbox_from_polygon(polygon, width, height)
        if box is None:
            continue
        unclipped_width = max(1, int(np.max(polygon[:, 0]) - np.min(polygon[:, 0])))
        unclipped_height = max(1, int(np.max(polygon[:, 1]) - np.min(polygon[:, 1])))
        visibility = (box[2] - box[0]) * (box[3] - box[1]) / float(
            unclipped_width * unclipped_height
        )
        if visibility < 0.35 or any(box_iou(box, other) > 0.12 for other in boxes):
            continue
        boxes.append(box)
        polygons.append(polygon)

    for index, polygon in enumerate(polygons):
        render_shape(
            image,
            render_background,
            polygon,
            rng,
            highlighted=index == highlighted_index,
        )

    image = augment_image(image, rng)
    metadata = {
        "source_video": asset.spec.filename,
        "shape_family": asset.spec.family,
        "object_count": len(boxes),
        "highlighted_index": highlighted_index if highlighted_index < len(boxes) else -1,
    }
    return image, boxes, metadata


def yolo_lines(
    boxes: list[tuple[int, int, int, int]],
    width: int,
    height: int,
) -> str:
    rows = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2.0 / width
        cy = (y1 + y2) / 2.0 / height
        bw = (x2 - x1) / float(width)
        bh = (y2 - y1) / float(height)
        rows.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(rows) + ("\n" if rows else "")


def make_preview(
    samples: list[tuple[np.ndarray, list[tuple[int, int, int, int]], str]],
    output: Path,
) -> None:
    tiles = []
    for image, boxes, label in samples:
        tile = image.copy()
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(tile, (x1, y1), (x2, y2), (40, 230, 40), 2)
        cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4)
        cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
        tiles.append(cv2.resize(tile, (384, 256), interpolation=cv2.INTER_AREA))
    if not tiles:
        return
    grid = int(math.ceil(math.sqrt(len(tiles))))
    blank = np.zeros_like(tiles[0])
    tiles.extend([blank] * (grid * grid - len(tiles)))
    rows = [
        np.hstack(tiles[index:index + grid])
        for index in range(0, grid * grid, grid)
    ]
    cv2.imwrite(str(output), np.vstack(rows))


def reset_output(root: Path) -> None:
    # This script owns only ml/lie_dataset.  Never accept a broad or ambiguous
    # deletion target from the command line.
    expected = (HERE / "lie_dataset").resolve()
    if root.resolve() != expected:
        raise ValueError(f"Refusing to replace unexpected output directory: {root}")
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "assets").mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=2000, help="number of training images")
    parser.add_argument("--val", type=int, default=500, help="number of validation images")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--background-samples", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.train < 1 or args.val < 1 or args.width < 320 or args.height < 240:
        parser.error("train/val must be positive and output size at least 320x240")

    output = HERE / "lie_dataset"
    reset_output(output)
    recorded_assets = [
        load_asset(spec, (args.width, args.height), args.background_samples)
        for spec in RECORDINGS
    ]
    assets = recorded_assets + make_procedural_assets(recorded_assets)
    for asset in assets:
        cv2.imwrite(str(output / "assets" / f"background_{asset.spec.family}.jpg"), asset.background)
        mask = np.zeros((320, 320), dtype=np.uint8)
        display_points = asset.points.copy()
        max_extent = max(np.ptp(display_points[:, 0]), np.ptp(display_points[:, 1]), 1.0)
        display_points *= 230.0 / max_extent
        display_points += np.array([160.0, 160.0], dtype=np.float32)
        cv2.fillPoly(mask, [np.rint(display_points).astype(np.int32)], 255)
        cv2.imwrite(str(output / "assets" / f"shape_{asset.spec.family}.png"), mask)

    rng = random.Random(args.seed)
    manifest: list[dict[str, object]] = []
    preview_samples = []
    totals = {asset.spec.family: 0 for asset in assets}
    total_boxes = 0
    for split, count in (("train", args.train), ("val", args.val)):
        for index in range(count):
            # Round-robin guarantees exact source balance; shuffled rendering
            # parameters still make every image unique.
            asset = assets[index % len(assets)]
            image, boxes, metadata = render_sample(asset, rng)
            stem = f"lie_{split}_{index:05d}"
            image_path = output / "images" / split / f"{stem}.jpg"
            label_path = output / "labels" / split / f"{stem}.txt"
            if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise RuntimeError(f"Unable to write {image_path}")
            label_path.write_text(
                yolo_lines(boxes, args.width, args.height),
                encoding="utf-8",
            )
            record = {
                "split": split,
                "image": str(image_path.relative_to(HERE)),
                "label": str(label_path.relative_to(HERE)),
                **metadata,
            }
            manifest.append(record)
            totals[asset.spec.family] += 1
            total_boxes += len(boxes)
            if len(preview_samples) < len(assets):
                preview_samples.append((image, boxes, f"{asset.spec.family} n={len(boxes)}"))

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_preview(preview_samples, output / "preview.jpg")
    data_yaml = {
        "path": "lie_dataset",
        "train": "images/train",
        "val": "images/val",
        "names": {0: "lie_shape"},
    }
    (HERE / "data_lie.yaml").write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    summary = {
        "seed": args.seed,
        "image_size": [args.width, args.height],
        "train_images": args.train,
        "val_images": args.val,
        "total_boxes": total_boxes,
        "images_per_family": totals,
        "shape_family_count": len(assets),
        "procedural_shapes": [asdict(shape) for shape in ALL_PROCEDURAL_SHAPES],
        "sources": [asdict(spec) for spec in RECORDINGS],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Preview: {output / 'preview.jpg'}")
    print(
        "Train: python3 ml/train.py --data data_lie.yaml --name lie_shape "
        "--out models/lie_shape_yolo.pt"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
