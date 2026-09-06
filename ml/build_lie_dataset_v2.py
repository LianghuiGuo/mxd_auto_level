#!/usr/bin/env python3
"""Build a domain-aligned lie-shape dataset and short tracking sequences.

Compared with ``build_lie_dataset.py`` this generator deliberately models the
hard parts seen in real recordings:

* weak refractive boundaries instead of a clean, fixed RGB outline;
* one geometry and almost one size per mini-game;
* overlap, clipping, motion blur, compression and cursor-removal artefacts;
* empty/hard-negative frames;
* source-disjoint train/validation backgrounds;
* clip/track/target metadata for evaluating the downstream tracker.

The six files under ``ml/videos`` are intentionally never read here.  They can
remain a genuinely held-out real test set.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import shutil
import sys
from typing import Iterable

import cv2
import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_lie_dataset import (  # noqa: E402
    ALL_PROCEDURAL_SHAPES,
    RECORDINGS,
    ShapeAsset,
    bbox_from_polygon,
    box_iou,
    load_asset,
    make_procedural_assets,
    remove_cursor,
    transformed_polygon,
    yolo_lines,
)


@dataclass(frozen=True)
class Geometry:
    family: str
    points: np.ndarray
    base_diameter: float
    recorded: bool
    min_count: int
    max_count: int


@dataclass
class ObjectState:
    track_id: int
    center: np.ndarray
    velocity: np.ndarray
    scale: float
    angle: float
    angular_velocity: float
    is_target: bool


@dataclass(frozen=True)
class RenderProfile:
    refraction_px: float
    lens_strength: float
    texture_alpha: float
    inner_width: float
    outer_width: float
    inner_gain: float
    outer_gain: float


def _safe_output(path: Path) -> Path:
    resolved = path.resolve()
    ml_root = HERE.resolve()
    if resolved.parent != ml_root or not resolved.name.startswith("lie_dataset_v2"):
        raise ValueError(
            "output must be a direct child of ml/ named lie_dataset_v2*; "
            f"got {resolved}"
        )
    return resolved


def reset_output(path: Path) -> None:
    root = _safe_output(path)
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def build_geometries(recorded_assets: list[ShapeAsset]) -> list[Geometry]:
    recorded = [
        Geometry(
            family=asset.spec.family,
            points=asset.points.copy(),
            base_diameter=float(asset.base_diameter),
            recorded=True,
            min_count=asset.min_count,
            max_count=asset.max_count,
        )
        for asset in recorded_assets
    ]
    procedural_assets = make_procedural_assets(recorded_assets)
    procedural = [
        Geometry(
            family=asset.spec.family,
            points=asset.points.copy(),
            base_diameter=float(asset.base_diameter),
            recorded=False,
            min_count=asset.min_count,
            max_count=asset.max_count,
        )
        for asset in procedural_assets
    ]
    return recorded + procedural


def choose_geometry(
    geometries: list[Geometry],
    rng: random.Random,
    recorded_ratio: float,
) -> Geometry:
    recorded = [item for item in geometries if item.recorded]
    procedural = [item for item in geometries if not item.recorded]
    pool = recorded if rng.random() < recorded_ratio else procedural
    return pool[rng.randrange(len(pool))]


def clean_background(asset: ShapeAsset, rng: random.Random) -> np.ndarray:
    """Use one coherent median background; never quilt small unrelated tiles."""

    background = asset.background.copy()
    height, width = background.shape[:2]
    # Restore a little high-frequency detail lost by temporal median filtering.
    low = cv2.GaussianBlur(background, (0, 0), 1.15)
    amount = rng.uniform(0.35, 0.75)
    background = cv2.addWeighted(background, 1.0 + amount, low, -amount, 0)

    scale = rng.uniform(1.0, 1.055)
    scaled = cv2.resize(
        background, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )
    max_x = max(0, scaled.shape[1] - width)
    max_y = max(0, scaled.shape[0] - height)
    offset_x = rng.randint(0, max_x) if max_x else 0
    offset_y = rng.randint(0, max_y) if max_y else 0
    return scaled[offset_y:offset_y + height, offset_x:offset_x + width].copy()


def random_profile(rng: random.Random) -> RenderProfile:
    return RenderProfile(
        refraction_px=rng.uniform(2.0, 11.0),
        lens_strength=rng.uniform(-0.045, 0.055),
        texture_alpha=rng.uniform(0.48, 0.92),
        inner_width=rng.uniform(1.0, 3.5),
        outer_width=rng.uniform(1.5, 5.5),
        inner_gain=rng.uniform(1.03, 1.18),
        outer_gain=rng.uniform(0.68, 0.91),
    )


def _polygon_roi(
    polygon: np.ndarray,
    width: int,
    height: int,
    padding: int = 16,
) -> tuple[int, int, int, int] | None:
    return bbox_from_polygon(polygon, width, height, padding=padding)


def render_refractive_shape(
    image: np.ndarray,
    polygon: np.ndarray,
    profile: RenderProfile,
    highlight_alpha: float,
) -> None:
    """Render a weak background-dependent refraction and bevel in place."""

    height, width = image.shape[:2]
    roi = _polygon_roi(polygon, width, height)
    if roi is None:
        return
    x1, y1, x2, y2 = roi
    local_polygon = polygon - np.array([x1, y1], dtype=np.int32)
    roi_height, roi_width = y2 - y1, x2 - x1
    binary = np.zeros((roi_height, roi_width), dtype=np.uint8)
    cv2.fillPoly(binary, [local_polygon], 255, lineType=cv2.LINE_8)
    if not np.any(binary):
        return

    inside_distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform(255 - binary, cv2.DIST_L2, 5)
    signed_distance = inside_distance - outside_distance
    gradient_x = cv2.Sobel(signed_distance, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(signed_distance, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y) + 1e-4
    normal_x = gradient_x / magnitude
    normal_y = gradient_y / magnitude

    local_y, local_x = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    center = np.mean(polygon.astype(np.float32), axis=0)
    radius = max(
        float(np.ptp(polygon[:, 0])),
        float(np.ptp(polygon[:, 1])),
        1.0,
    ) * 0.5
    edge_falloff = np.exp(-np.abs(signed_distance) / max(5.0, 0.22 * radius))
    inside = (binary > 0).astype(np.float32)
    displacement = profile.refraction_px * edge_falloff
    map_x = (
        local_x
        + normal_x * displacement
        + inside * (local_x - center[0]) * profile.lens_strength
    )
    map_y = (
        local_y
        + normal_y * displacement
        + inside * (local_y - center[1]) * profile.lens_strength
    )
    warped = cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    ).astype(np.float32)
    destination = image[y1:y2, x1:x2].astype(np.float32)

    # Feather only the shape interior; the displacement is strongest near its
    # boundary while the centre remains close to the original background.
    feather = np.clip(inside_distance / 2.2, 0.0, 1.0) * inside
    alpha = (profile.texture_alpha * feather)[:, :, None]
    destination = destination * (1.0 - alpha) + warped * alpha

    outer = (
        (signed_distance <= 0.0)
        & (signed_distance >= -profile.outer_width)
    ).astype(np.float32)
    inner = (
        (signed_distance > 0.0)
        & (signed_distance <= profile.inner_width)
    ).astype(np.float32)
    # Directional bevel lighting avoids a fixed RGB outline shortcut.
    light_direction = np.clip(0.72 - 0.28 * normal_x - 0.32 * normal_y, 0.0, 1.0)
    outer_factor = 1.0 - outer * (1.0 - profile.outer_gain) * light_direction
    inner_factor = 1.0 + inner * (profile.inner_gain - 1.0) * (1.0 - light_direction)
    destination *= (outer_factor * inner_factor)[:, :, None]

    if highlight_alpha > 0.0:
        highlight = np.clip(feather * highlight_alpha, 0.0, 1.0)[:, :, None]
        white = np.full_like(destination, 246.0)
        destination = destination * (1.0 - highlight) + white * highlight

    image[y1:y2, x1:x2] = np.clip(destination, 0, 255).astype(np.uint8)


def _approx_box(
    geometry: Geometry,
    center: np.ndarray,
    scale: float,
    angle: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    polygon = transformed_polygon(
        geometry.points, tuple(float(value) for value in center), scale, angle
    )
    return bbox_from_polygon(polygon, width, height, padding=2)


def make_objects(
    geometry: Geometry,
    width: int,
    height: int,
    rng: random.Random,
) -> list[ObjectState]:
    minimum = max(3, int(round(geometry.min_count * 1.15)))
    maximum = min(20, max(minimum, int(round(geometry.max_count * 1.45))))
    count = rng.randint(minimum, maximum)
    global_scale = rng.uniform(0.74, 1.24)
    shared_direction = rng.uniform(0.0, 2.0 * math.pi)
    shared_speed = rng.uniform(1.0, 7.5)
    shared_velocity = shared_speed * np.array(
        [math.cos(shared_direction), math.sin(shared_direction)], dtype=np.float32
    )
    shared_angular_velocity = rng.uniform(-7.0, 7.0)

    objects: list[ObjectState] = []
    boxes: list[tuple[int, int, int, int]] = []
    radius = 0.55 * geometry.base_diameter * global_scale
    attempts = 0
    while len(objects) < count and attempts < count * 120:
        attempts += 1
        center = np.array(
            [
                rng.uniform(-0.28 * radius, width + 0.28 * radius),
                rng.uniform(-0.28 * radius, height + 0.28 * radius),
            ],
            dtype=np.float32,
        )
        angle = rng.uniform(0.0, 360.0)
        local_scale = global_scale * rng.uniform(0.97, 1.03)
        box = _approx_box(geometry, center, local_scale, angle, width, height)
        if box is None:
            continue
        polygon = transformed_polygon(
            geometry.points, tuple(center), local_scale, angle
        )
        full_w = max(1, int(np.ptp(polygon[:, 0])))
        full_h = max(1, int(np.ptp(polygon[:, 1])))
        visibility = (box[2] - box[0]) * (box[3] - box[1]) / float(full_w * full_h)
        if visibility < 0.25 or any(box_iou(box, other) > 0.45 for other in boxes):
            continue
        is_target = len(objects) == 0
        if is_target:
            target_direction = shared_direction + rng.uniform(0.8, 2.4)
            target_speed = rng.uniform(1.2, 8.5)
            velocity = target_speed * np.array(
                [math.cos(target_direction), math.sin(target_direction)],
                dtype=np.float32,
            )
            angular_velocity = shared_angular_velocity + rng.choice((-1.0, 1.0)) * rng.uniform(3.0, 12.0)
        else:
            velocity = shared_velocity.copy()
            angular_velocity = shared_angular_velocity
        objects.append(
            ObjectState(
                track_id=len(objects) + 1,
                center=center,
                velocity=velocity,
                scale=local_scale,
                angle=angle,
                angular_velocity=angular_velocity,
                is_target=is_target,
            )
        )
        boxes.append(box)
    return objects


def add_hard_negative_fragments(
    image: np.ndarray,
    rng: random.Random,
) -> None:
    """Add incomplete background-like edges that must not become detections."""

    height, width = image.shape[:2]
    for _ in range(rng.randint(2, 8)):
        center = (rng.randrange(width), rng.randrange(height))
        axes = (rng.randint(18, 80), rng.randint(12, 65))
        start = rng.randint(0, 280)
        end = min(360, start + rng.randint(18, 75))
        sample = image[center[1], center[0]].astype(int)
        delta = rng.randint(-35, 25)
        color = tuple(int(np.clip(value + delta, 0, 255)) for value in sample)
        cv2.ellipse(image, center, axes, rng.uniform(0, 180), start, end, color, rng.randint(1, 3), cv2.LINE_AA)


def add_cursor_then_remove(
    image: np.ndarray,
    objects: list[ObjectState],
    frame_number: int,
    rng: random.Random,
) -> np.ndarray:
    raw = image.copy()
    height, width = raw.shape[:2]
    if objects and rng.random() < 0.72:
        state = objects[0] if rng.random() < 0.70 else objects[rng.randrange(len(objects))]
        center_array = state.center + state.velocity * frame_number
        center = tuple(np.rint(center_array).astype(int))
    else:
        center = (rng.randrange(width), rng.randrange(height))
    radius = rng.randint(15, 20)
    cv2.circle(raw, center, radius + 2, (20, 70, 20), 3, cv2.LINE_AA)
    cv2.circle(raw, center, radius, (20, 255, 70), 3, cv2.LINE_AA)
    cv2.line(raw, (center[0] - radius, center[1]), (center[0] + radius, center[1]), (20, 255, 70), 2, cv2.LINE_AA)
    cv2.line(raw, (center[0], center[1] - radius), (center[0], center[1] + radius), (20, 255, 70), 2, cv2.LINE_AA)
    return remove_cursor(raw)


def degrade_frame(image: np.ndarray, rng: random.Random) -> np.ndarray:
    result = image
    if rng.random() < 0.35:
        length = rng.randint(2, 5)
        angle = rng.uniform(0.0, math.pi)
        kernel = np.zeros((length, length), dtype=np.float32)
        p1 = (
            int(round((length - 1) * (0.5 - 0.5 * math.cos(angle)))),
            int(round((length - 1) * (0.5 - 0.5 * math.sin(angle)))),
        )
        p2 = (length - 1 - p1[0], length - 1 - p1[1])
        cv2.line(kernel, p1, p2, 1.0, 1)
        kernel /= max(float(np.sum(kernel)), 1.0)
        result = cv2.filter2D(result, -1, kernel)
    if rng.random() < 0.45:
        scale = rng.uniform(0.78, 0.96)
        small = cv2.resize(result, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        result = cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    gain = rng.uniform(0.94, 1.06)
    bias = rng.uniform(-5.0, 5.0)
    result = np.clip(result.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    if rng.random() < 0.72:
        quality = rng.randint(68, 96)
        ok, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                result = decoded
    return result


def _target_highlight(
    phase: str,
    frame_number: int,
    clip_length: int,
) -> float:
    if phase == "highlight":
        return 0.92
    if phase == "fade":
        progress = frame_number / max(1, clip_length - 1)
        return float(0.88 * (1.0 - progress))
    return 0.0


def render_clip_frame(
    base_background: np.ndarray,
    geometry: Geometry,
    objects: list[ObjectState],
    profile: RenderProfile,
    phase: str,
    frame_number: int,
    clip_length: int,
    rng: random.Random,
    cursor_probability: float,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[dict[str, object]]]:
    image = base_background.copy()
    height, width = image.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    tracks: list[dict[str, object]] = []
    for state in objects:
        center = state.center + state.velocity * frame_number
        angle = state.angle + state.angular_velocity * frame_number
        polygon = transformed_polygon(
            geometry.points, tuple(float(value) for value in center), state.scale, angle
        )
        box = bbox_from_polygon(polygon, width, height, padding=3)
        if box is None:
            continue
        highlight = (
            _target_highlight(phase, frame_number, clip_length)
            if state.is_target
            else 0.0
        )
        render_refractive_shape(image, polygon, profile, highlight)
        boxes.append(box)
        tracks.append(
            {
                "track_id": state.track_id,
                "bbox_xyxy": list(box),
                "center": [float(center[0]), float(center[1])],
                "angle_degrees": float(angle % 360.0),
                "is_target": state.is_target,
            }
        )
    image = degrade_frame(image, rng)
    if rng.random() < cursor_probability:
        image = add_cursor_then_remove(image, objects, frame_number, rng)
    return image, boxes, tracks


def _split_backgrounds(
    assets: list[ShapeAsset],
    validation_source: str,
) -> tuple[list[ShapeAsset], list[ShapeAsset]]:
    validation = [asset for asset in assets if asset.spec.filename == validation_source]
    training = [asset for asset in assets if asset.spec.filename != validation_source]
    if not validation:
        choices = ", ".join(asset.spec.filename for asset in assets)
        raise ValueError(
            f"unknown validation source {validation_source!r}; choices: {choices}"
        )
    if not training:
        raise ValueError("at least one background source must remain for training")
    return training, validation


def generate_split(
    *,
    split: str,
    frame_count: int,
    clip_length: int,
    backgrounds: list[ShapeAsset],
    geometries: list[Geometry],
    output: Path,
    rng: random.Random,
    recorded_ratio: float,
    negative_ratio: float,
    cursor_probability: float,
) -> tuple[list[dict[str, object]], int, int]:
    manifest: list[dict[str, object]] = []
    total_boxes = 0
    negative_frames = 0
    index = 0
    clip_index = 0
    clip_count = int(math.ceil(frame_count / clip_length))
    negative_clip_count = int(round(clip_count * negative_ratio))
    if negative_ratio > 0.0 and clip_count:
        negative_clip_count = max(1, negative_clip_count)
    negative_flags = [True] * negative_clip_count + [False] * (
        clip_count - negative_clip_count
    )
    rng.shuffle(negative_flags)
    while index < frame_count:
        background_asset = backgrounds[clip_index % len(backgrounds)]
        geometry = choose_geometry(geometries, rng, recorded_ratio)
        background = clean_background(background_asset, rng)
        is_negative = negative_flags[clip_index]
        objects = [] if is_negative else make_objects(
            geometry, background.shape[1], background.shape[0], rng
        )
        profile = random_profile(rng)
        phase_roll = rng.random()
        phase = "active" if phase_roll < 0.70 else (
            "fade" if phase_roll < 0.90 else "highlight"
        )
        for frame_number in range(min(clip_length, frame_count - index)):
            if is_negative:
                image = background.copy()
                add_hard_negative_fragments(image, rng)
                image = degrade_frame(image, rng)
                if rng.random() < cursor_probability:
                    image = add_cursor_then_remove(image, [], frame_number, rng)
                boxes: list[tuple[int, int, int, int]] = []
                tracks: list[dict[str, object]] = []
                negative_frames += 1
            else:
                image, boxes, tracks = render_clip_frame(
                    background,
                    geometry,
                    objects,
                    profile,
                    phase,
                    frame_number,
                    clip_length,
                    rng,
                    cursor_probability,
                )
            stem = f"lie_v2_{split}_{index:06d}"
            image_path = output / "images" / split / f"{stem}.jpg"
            label_path = output / "labels" / split / f"{stem}.txt"
            if not cv2.imwrite(
                str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 96]
            ):
                raise RuntimeError(f"unable to write {image_path}")
            label_path.write_text(
                yolo_lines(boxes, image.shape[1], image.shape[0]), encoding="utf-8"
            )
            manifest.append(
                {
                    "split": split,
                    "image": str(image_path.relative_to(HERE)),
                    "label": str(label_path.relative_to(HERE)),
                    "source_video": background_asset.spec.filename,
                    "clip_id": f"{split}_{clip_index:06d}",
                    "frame_in_clip": frame_number,
                    "phase": "negative" if is_negative else phase,
                    "shape_family": None if is_negative else geometry.family,
                    "recorded_geometry": False if is_negative else geometry.recorded,
                    "objects": tracks,
                }
            )
            total_boxes += len(boxes)
            index += 1
        clip_index += 1
    return manifest, total_boxes, negative_frames


def _write_preview(output: Path, manifest: Iterable[dict[str, object]]) -> None:
    records = list(manifest)
    if not records:
        return
    # Prefer a mixture of phases and include a negative example when possible.
    selected: list[dict[str, object]] = []
    for phase in ("active", "fade", "highlight", "negative"):
        selected.extend([item for item in records if item["phase"] == phase][:3])
    selected = selected[:12]
    tiles: list[np.ndarray] = []
    for item in selected:
        image = cv2.imread(str(HERE / str(item["image"])))
        if image is None:
            continue
        for obj in item["objects"]:
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            color = (20, 20, 240) if obj["is_target"] else (40, 220, 40)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        label = f"{item['phase']} {item['shape_family']}"
        cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(cv2.resize(image, (350, 232), interpolation=cv2.INTER_AREA))
    if not tiles:
        return
    blank = np.zeros_like(tiles[0])
    tiles.extend([blank] * (12 - len(tiles)))
    rows = [np.hstack(tiles[index:index + 4]) for index in range(0, 12, 4)]
    cv2.imwrite(str(output / "preview.jpg"), np.vstack(rows))


def _validated_yolo_label_count(path: Path) -> int:
    count = 0
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields")
        class_id = int(fields[0])
        coordinates = [float(value) for value in fields[1:]]
        if class_id != 0 or any(not 0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(
                f"{path}:{line_number}: expected class 0 and normalized coordinates"
            )
        count += 1
    return count


def import_reviewed_real_dataset(
    source: Path | None,
    output: Path,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Append human-reviewed YOLO frames without treating pseudo-labels as truth.

    ``source`` must use ``images/{train,val}`` and ``labels/{train,val}``.
    Keeping this opt-in prevents the six held-out test recordings from being
    silently mixed into training.
    """

    counts = {"train_frames": 0, "val_frames": 0, "boxes": 0}
    if source is None:
        return [], counts
    source = source.resolve()
    manifest: list[dict[str, object]] = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    for split in ("train", "val"):
        image_dir = source / "images" / split
        label_dir = source / "labels" / split
        if not image_dir.exists():
            continue
        if not label_dir.is_dir():
            raise ValueError(f"reviewed label directory missing: {label_dir}")
        images = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        )
        for index, image_path in enumerate(images):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise ValueError(f"reviewed label missing: {label_path}")
            box_count = _validated_yolo_label_count(label_path)
            stem = f"lie_v2_reviewed_{split}_{index:06d}"
            destination_image = (
                output / "images" / split / f"{stem}{image_path.suffix.lower()}"
            )
            destination_label = output / "labels" / split / f"{stem}.txt"
            shutil.copy2(image_path, destination_image)
            shutil.copy2(label_path, destination_label)
            counts[f"{split}_frames"] += 1
            counts["boxes"] += box_count
            manifest.append(
                {
                    "split": split,
                    "image": str(destination_image.relative_to(HERE)),
                    "label": str(destination_label.relative_to(HERE)),
                    "source_video": None,
                    "source_type": "human_reviewed_real",
                    "clip_id": None,
                    "frame_in_clip": None,
                    "phase": "real",
                    "shape_family": None,
                    "recorded_geometry": True,
                    "objects": [],
                    "box_count": box_count,
                }
            )
    if not manifest:
        raise ValueError(
            f"no reviewed images found under {source}/images/{{train,val}}"
        )
    return manifest, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=8000)
    parser.add_argument("--val", type=int, default=1000)
    parser.add_argument("--width", type=int, default=700)
    parser.add_argument("--height", type=int, default=464)
    parser.add_argument("--clip-length", type=int, default=4)
    parser.add_argument("--background-samples", type=int, default=72)
    parser.add_argument("--recorded-ratio", type=float, default=0.68)
    parser.add_argument("--negative-ratio", type=float, default=0.16)
    parser.add_argument("--cursor-probability", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--validation-source",
        default=RECORDINGS[-1].filename,
        help="recording reserved exclusively for validation backgrounds",
    )
    parser.add_argument("--output", type=Path, default=HERE / "lie_dataset_v2")
    parser.add_argument(
        "--reviewed-real",
        type=Path,
        help=(
            "optional human-reviewed YOLO dataset with images/labels train/val; "
            "appended after synthetic generation"
        ),
    )
    args = parser.parse_args()
    if args.train < 1 or args.val < 1:
        parser.error("--train and --val must be positive")
    if args.width < 320 or args.height < 240:
        parser.error("output size must be at least 320x240")
    if args.clip_length < 1:
        parser.error("--clip-length must be positive")
    for name in ("recorded_ratio", "negative_ratio", "cursor_probability"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")

    output = _safe_output(args.output)
    reset_output(output)
    recorded_assets = [
        load_asset(
            spec,
            (args.width, args.height),
            args.background_samples,
        )
        for spec in RECORDINGS
    ]
    train_backgrounds, val_backgrounds = _split_backgrounds(
        recorded_assets, args.validation_source
    )
    geometries = build_geometries(recorded_assets)

    train_manifest, train_boxes, train_negatives = generate_split(
        split="train",
        frame_count=args.train,
        clip_length=args.clip_length,
        backgrounds=train_backgrounds,
        geometries=geometries,
        output=output,
        rng=random.Random(args.seed),
        recorded_ratio=args.recorded_ratio,
        negative_ratio=args.negative_ratio,
        cursor_probability=args.cursor_probability,
    )
    val_manifest, val_boxes, val_negatives = generate_split(
        split="val",
        frame_count=args.val,
        clip_length=args.clip_length,
        backgrounds=val_backgrounds,
        geometries=geometries,
        output=output,
        rng=random.Random(args.seed + 1_000_003),
        recorded_ratio=args.recorded_ratio,
        negative_ratio=args.negative_ratio,
        cursor_probability=args.cursor_probability,
    )
    reviewed_manifest, reviewed_counts = import_reviewed_real_dataset(
        args.reviewed_real, output
    )
    manifest = train_manifest + val_manifest + reviewed_manifest
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_preview(output, manifest)

    data_yaml = {
        "path": output.name,
        "train": "images/train",
        "val": "images/val",
        "names": {0: "lie_shape"},
    }
    data_path = (
        HERE / "data_lie_v2.yaml"
        if output.name == "lie_dataset_v2"
        else output / "data.yaml"
    )
    data_path.write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    summary = {
        "version": 2,
        "seed": args.seed,
        "image_size": [args.width, args.height],
        "clip_length": args.clip_length,
        "train_frames": args.train,
        "val_frames": args.val,
        "train_boxes": train_boxes,
        "val_boxes": val_boxes,
        "train_negative_frames": train_negatives,
        "val_negative_frames": val_negatives,
        "reviewed_real": reviewed_counts,
        "recorded_geometry_ratio": args.recorded_ratio,
        "cursor_probability": args.cursor_probability,
        "train_background_sources": [
            asset.spec.filename for asset in train_backgrounds
        ],
        "val_background_sources": [asset.spec.filename for asset in val_backgrounds],
        "held_out_real_test_directory": "ml/videos (never read by this generator)",
        "shape_families": [item.family for item in geometries],
        "procedural_specs": [item.family for item in ALL_PROCEDURAL_SHAPES],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Preview: {output / 'preview.jpg'}")
    print(
        "Train: python3 ml/train.py --lie-detector "
        f"--data {data_path.relative_to(HERE)} "
        "--name lie_shape_v2 --out models/lie_shape_yolo_v2.pt"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
