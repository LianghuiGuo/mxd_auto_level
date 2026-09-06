#!/usr/bin/env python3
"""Semi-automatic real-recording pre-labeller for the lie-detector mini-game.

This tool implements the "real semi-automatic annotation" plan:

* extract frames from the four root-level recordings at a phase-aware frame
  rate (dense during fade-out, sparse during the opaque highlight phase);
* remove the green in-game cursor so it can never become a label;
* propose candidate boxes primarily from a temporal-median background diff,
  optionally augmented by the existing low-confidence lie-shape YOLO model;
* snap every candidate to the per-game standard shape size recovered from the
  initial white highlight;
* enforce short-term trajectory continuity to fill single-frame misses and drop
  instantaneous false positives;
* emit YOLO pre-labels plus a CVAT/Label-Studio-importable task.

The output layout matches ``build_lie_dataset_v2.py --reviewed-real`` so that,
after a human review pass, the frames can be mixed back into the V2 synthetic
dataset without a parallel pipeline.

Recordings, their panel ROI/active windows and the train/val assignment are all
read from ``ml/lie_videos_config.json`` (produced by ``ml/detect_lie_panels.py``
and then human-reviewed).  Splitting is by whole recording so adjacent, highly
correlated frames never straddle train/val -- never split a single clip's frames
by time (e.g. first 80%% train / last 20%% val), which leaks near-identical
frames across the split.

    python3 ml/build_lie_real_dataset.py                       # config-driven
    python3 ml/build_lie_real_dataset.py --val-video 测谎录屏1.mp4 --val-video 测谎录屏7.mp4
    python3 ml/build_lie_real_dataset.py --model models/lie_shape_yolo.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import sys
from typing import Optional

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from build_lie_dataset import (  # noqa: E402
    RecordingSpec,
    box_iou,
    crop_canvas,
    highlighted_contour,
    read_frame,
    remove_cursor,
    yolo_lines,
)

DEFAULT_CONFIG = HERE / "lie_videos_config.json"


def load_specs(
    config_path: Path,
) -> tuple[Path, list[tuple[RecordingSpec, str]]]:
    """Read the per-recording panel config and return (videos_dir, [(spec, split)]).

    ``split`` is ``"val"`` when the filename is listed under ``"val"`` in the
    config, otherwise ``"train"``.  All recordings must carry a concrete ROI and
    active window (auto-detected then human-reviewed); ``family == "unknown"`` is
    allowed but a missing/zero ROI is rejected so we never crop garbage.
    """

    if not config_path.is_file():
        raise FileNotFoundError(
            f"panel config not found: {config_path}. Run ml/detect_lie_panels.py "
            "first, then review the ROIs."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    videos_dir = (REPO / config["videos_dir"]).resolve()
    val_names = set(config.get("val", []))
    specs: list[tuple[RecordingSpec, str]] = []
    for entry in config["recordings"]:
        roi = tuple(int(v) for v in entry["roi"])
        if len(roi) != 4 or roi[2] <= 0 or roi[3] <= 0:
            raise ValueError(f"invalid roi for {entry['filename']}: {entry['roi']}")
        safe = entry.get("safe_crop") or [0, 0, roi[2], roi[3]]
        spec = RecordingSpec(
            filename=entry["filename"],
            roi=roi,
            safe_crop=tuple(int(v) for v in safe),
            seed_time=float(entry.get("seed_time", 0.0)),
            active_start=float(entry.get("active_start", 0.0)),
            active_end=float(entry["active_end"]),
            family=entry.get("family", "unknown"),
        )
        split = "val" if spec.filename in val_names else "train"
        specs.append((spec, split))
    return videos_dir, specs


# --------------------------------------------------------------------------- #
# Frame sampling
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SamplingProfile:
    """Phase-aware sampling FPS relative to each recording's active window."""

    highlight_fps: float = 2.0
    fade_fps: float = 9.0
    motion_fps: float = 4.0
    # Fraction of the active window (after the seed) treated as the fade phase.
    fade_fraction: float = 0.22


@dataclass
class SampledFrame:
    timestamp: float
    phase: str
    canvas: np.ndarray  # cursor-cleaned, cropped to spec.roi


def _phase_at(spec: RecordingSpec, timestamp: float, profile: SamplingProfile) -> str:
    if timestamp <= spec.active_start:
        return "highlight"
    fade_end = spec.active_start + profile.fade_fraction * (
        spec.active_end - spec.active_start
    )
    if timestamp <= fade_end:
        return "fade"
    return "motion"


def _sample_timestamps(spec: RecordingSpec, profile: SamplingProfile) -> list[tuple[float, str]]:
    """Return (timestamp, phase) pairs covering highlight/fade/motion phases."""

    stops: list[tuple[float, str]] = []

    def _emit(start: float, end: float, fps: float, phase: str) -> None:
        if end <= start or fps <= 0:
            return
        count = max(1, int(round((end - start) * fps)))
        for value in np.linspace(start, end, count, endpoint=False):
            stops.append((float(value), phase))

    highlight_end = spec.active_start
    fade_end = spec.active_start + profile.fade_fraction * (
        spec.active_end - spec.active_start
    )
    # Highlight phase: from the very first frame up to activation.
    _emit(0.0, max(highlight_end, 1e-3), profile.highlight_fps, "highlight")
    _emit(highlight_end, fade_end, profile.fade_fps, "fade")
    _emit(fade_end, spec.active_end, profile.motion_fps, "motion")

    # Deduplicate near-identical timestamps (highlight/fade boundary overlap).
    stops.sort(key=lambda item: item[0])
    deduped: list[tuple[float, str]] = []
    for timestamp, phase in stops:
        if deduped and abs(timestamp - deduped[-1][0]) < 1e-3:
            continue
        deduped.append((timestamp, phase))
    return deduped


# --------------------------------------------------------------------------- #
# Candidate proposal
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in canvas coords
    score: float
    source: str  # "bgdiff" | "yolo" | "interp"


def _reference_size(canvas: np.ndarray) -> Optional[tuple[float, float]]:
    """Recover the per-game standard shape size from the white highlight."""

    try:
        _, (x, y, w, h) = highlighted_contour(canvas)
    except RuntimeError:
        return None
    return float(w), float(h)


def _nms(candidates: list[Candidate], iou_threshold: float) -> list[Candidate]:
    ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
    kept: list[Candidate] = []
    for candidate in ordered:
        if all(box_iou(candidate.bbox, other.bbox) < iou_threshold for other in kept):
            kept.append(candidate)
    return kept


def _snap_to_reference(
    bbox: tuple[int, int, int, int],
    reference: Optional[tuple[float, float]],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Fix candidate size to the standard shape size, then clip to the image.

    Off-edge objects keep their visible (clipped) box instead of an oversized
    one that would extend past the frame, which is what a reviewer expects.
    """

    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    if reference is not None:
        ref_w, ref_h = reference
    else:
        ref_w, ref_h = float(x2 - x1), float(y2 - y1)
    nx1 = int(round(cx - ref_w / 2.0))
    ny1 = int(round(cy - ref_h / 2.0))
    nx2 = int(round(cx + ref_w / 2.0))
    ny2 = int(round(cy + ref_h / 2.0))
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(frame_width, nx2)
    ny2 = min(frame_height, ny2)
    return nx1, ny1, nx2, ny2


def _background_diff_candidates(
    canvas: np.ndarray,
    background: np.ndarray,
    reference: Optional[tuple[float, float]],
    diff_threshold: int,
    min_area_ratio: float,
) -> list[Candidate]:
    """Foreground blobs from |frame - median background|, size-snapped."""

    resized_bg = background
    if background.shape[:2] != canvas.shape[:2]:
        resized_bg = cv2.resize(
            background, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_AREA
        )
    diff = cv2.absdiff(canvas, resized_bg)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, diff_threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    height, width = canvas.shape[:2]
    if reference is not None:
        ref_area = reference[0] * reference[1]
    else:
        ref_area = 0.12 * width * height
    min_area = max(200.0, min_area_ratio * ref_area)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[Candidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        bbox = _snap_to_reference((x, y, x + w, y + h), reference, width, height)
        if bbox[2] - bbox[0] < 12 or bbox[3] - bbox[1] < 12:
            continue
        # Score by how much of the snapped box is actually foreground.
        sub = mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        fill = float(np.count_nonzero(sub)) / max(1, sub.size)
        candidates.append(Candidate(bbox, 0.30 + 0.70 * fill, "bgdiff"))
    return candidates


def _yolo_candidates(
    detector,
    canvas: np.ndarray,
    reference: Optional[tuple[float, float]],
) -> list[Candidate]:
    if detector is None:
        return []
    height, width = canvas.shape[:2]
    detections = detector.detect(canvas)
    candidates: list[Candidate] = []
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        bbox = _snap_to_reference((x1, y1, x2, y2), reference, width, height)
        if bbox[2] - bbox[0] < 12 or bbox[3] - bbox[1] < 12:
            continue
        # Auxiliary evidence only; keep confidence modest so bgdiff wins ties.
        score = 0.25 + 0.5 * float(getattr(detection, "yolo_confidence", 0.0) or 0.0)
        candidates.append(Candidate(bbox, score, "yolo"))
    return candidates


# --------------------------------------------------------------------------- #
# Trajectory continuity across sampled frames
# --------------------------------------------------------------------------- #
@dataclass
class _Track:
    box: tuple[int, int, int, int]
    last_index: int
    hits: int = 1
    history: list[tuple[int, tuple[int, int, int, int]]] = field(default_factory=list)


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _apply_trajectory_continuity(
    per_frame: list[list[Candidate]],
    reference: Optional[tuple[float, float]],
    frame_size: tuple[int, int],
    match_distance: float,
    min_hits: int,
    max_gap: int,
) -> list[list[Candidate]]:
    """Link candidates across frames; interpolate gaps, drop lone detections.

    This does not need stable identities for YOLO training; it only needs each
    kept box, on each frame, to sit on a shape.  Tracks with too few hits are
    treated as instantaneous false positives and removed.
    """

    width, height = frame_size
    tracks: list[_Track] = []
    for index, candidates in enumerate(per_frame):
        unmatched = list(candidates)
        for track in tracks:
            if track.last_index == index:
                continue
            if not unmatched:
                break
            tx, ty = _bbox_center(track.box)
            best = None
            best_distance = match_distance
            for candidate in unmatched:
                cx, cy = _bbox_center(candidate.bbox)
                distance = float(np.hypot(cx - tx, cy - ty))
                if distance < best_distance:
                    best_distance = distance
                    best = candidate
            if best is not None:
                track.box = best.bbox
                track.last_index = index
                track.hits += 1
                track.history.append((index, best.bbox))
                unmatched.remove(best)
        for candidate in unmatched:
            track = _Track(candidate.bbox, index, 1, [(index, candidate.bbox)])
            tracks.append(track)

    # Rebuild per-frame boxes from confident tracks, interpolating short gaps.
    result: list[list[Candidate]] = [[] for _ in per_frame]
    for track in tracks:
        if track.hits < min_hits:
            continue
        history = sorted(track.history, key=lambda item: item[0])
        for (i0, b0), (i1, b1) in zip(history, history[1:]):
            result[i0].append(Candidate(b0, 1.0, "bgdiff"))
            gap = i1 - i0
            if 1 < gap <= max_gap + 1:
                for step in range(1, gap):
                    alpha = step / float(gap)
                    interp = tuple(
                        int(round(b0[k] * (1 - alpha) + b1[k] * alpha)) for k in range(4)
                    )
                    interp = _snap_to_reference(
                        interp, reference, width, height
                    )
                    result[i0 + step].append(Candidate(interp, 0.9, "interp"))
        last_index, last_box = history[-1]
        result[last_index].append(Candidate(last_box, 1.0, "bgdiff"))

    # Deduplicate any overlap introduced by interpolation.
    return [_nms(frame, 0.6) for frame in result]


# --------------------------------------------------------------------------- #
# Detector loading (optional YOLO auxiliary)
# --------------------------------------------------------------------------- #
def _load_detector(model_path: Optional[Path], conf: float):
    if model_path is None:
        return None
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {model_path}")
    from src.engine.LieShapeYoloDetector import LieShapeYoloDetector

    return LieShapeYoloDetector(model_path, confidence=conf)


# --------------------------------------------------------------------------- #
# Per-recording processing
# --------------------------------------------------------------------------- #
def process_recording(
    spec: RecordingSpec,
    *,
    videos_dir: Path,
    split: str,
    output: Path,
    profile: SamplingProfile,
    detector,
    background_samples: int,
    diff_threshold: int,
    min_area_ratio: float,
    nms_iou: float,
    match_distance: float,
    min_hits: int,
    max_gap: int,
) -> list[dict]:
    path = videos_dir / spec.filename
    if not path.exists():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)

    # Recover the standard shape size from the initial white highlight.
    seed_canvas = crop_canvas(read_frame(capture, fps, spec.seed_time), spec)
    reference = _reference_size(remove_cursor(seed_canvas))

    # Temporal-median background at native canvas size (roi, not safe_crop).
    background = _canvas_background(capture, fps, spec, background_samples)

    stops = _sample_timestamps(spec, profile)
    frames: list[SampledFrame] = []
    for timestamp, phase in stops:
        try:
            raw = read_frame(capture, fps, timestamp)
        except RuntimeError:
            continue
        canvas = remove_cursor(crop_canvas(raw, spec))
        frames.append(SampledFrame(timestamp, phase, canvas))
    capture.release()

    if detector is not None:
        detector.reset()

    per_frame: list[list[Candidate]] = []
    for frame in frames:
        candidates = _background_diff_candidates(
            frame.canvas, background, reference, diff_threshold, min_area_ratio
        )
        candidates += _yolo_candidates(detector, frame.canvas, reference)
        per_frame.append(_nms(candidates, nms_iou))

    height, width = frames[0].canvas.shape[:2] if frames else (0, 0)
    linked = _apply_trajectory_continuity(
        per_frame,
        reference,
        (width, height),
        match_distance,
        min_hits,
        max_gap,
    )

    records: list[dict] = []
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    stem_base = Path(spec.filename).stem
    for index, (frame, candidates) in enumerate(zip(frames, linked)):
        stem = f"{stem_base}_{index:05d}"
        image_path = image_dir / f"{stem}.jpg"
        cv2.imwrite(str(image_path), frame.canvas)
        boxes = [candidate.bbox for candidate in candidates]
        label_path = label_dir / f"{stem}.txt"
        label_path.write_text(
            yolo_lines(boxes, frame.canvas.shape[1], frame.canvas.shape[0]),
            encoding="utf-8",
        )
        records.append(
            {
                "video": spec.filename,
                "frame_index": index,
                "timestamp": round(frame.timestamp, 3),
                "phase": frame.phase,
                "split": split,
                "image": str(image_path.relative_to(output)),
                "label": str(label_path.relative_to(output)),
                "reference_size": (
                    [round(reference[0], 1), round(reference[1], 1)]
                    if reference is not None
                    else None
                ),
                "objects": [
                    {
                        "track_id": obj_index,
                        "bbox_xyxy": list(candidate.bbox),
                        "source": candidate.source,
                        "is_target": None,  # filled during human review
                    }
                    for obj_index, candidate in enumerate(candidates)
                ],
                "box_count": len(candidates),
            }
        )
    return records


def _canvas_background(
    capture: cv2.VideoCapture,
    fps: float,
    spec: RecordingSpec,
    sample_count: int,
) -> np.ndarray:
    """Temporal-median background over the roi crop (full canvas, no resize)."""

    frames: list[np.ndarray] = []
    for timestamp in np.linspace(spec.active_start, spec.active_end, sample_count):
        try:
            raw = read_frame(capture, fps, float(timestamp))
        except RuntimeError:
            continue  # timestamp at/after the last decodable frame
        frames.append(remove_cursor(crop_canvas(raw, spec)))
    if not frames:
        raise RuntimeError(
            f"no decodable frames in active window for {spec.filename}"
        )
    background = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    return cv2.bilateralFilter(background, 5, 15, 15)


# --------------------------------------------------------------------------- #
# CVAT export
# --------------------------------------------------------------------------- #
def _write_cvat_manifest(output: Path, records: list[dict]) -> None:
    """Emit a simple JSON task list Label Studio / CVAT importers can consume.

    We keep this format-neutral (image path + normalized-independent xyxy boxes)
    so it can be adapted to either tool without a heavy dependency.
    """

    tasks = []
    for record in records:
        tasks.append(
            {
                "image": record["image"],
                "video": record["video"],
                "phase": record["phase"],
                "annotations": [
                    {
                        "label": "lie_shape",
                        "bbox_xyxy": obj["bbox_xyxy"],
                        "source": obj["source"],
                    }
                    for obj in record["objects"]
                ],
            }
        )
    (output / "cvat_tasks.json").write_text(
        json.dumps({"tasks": tasks}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_preview(output: Path, records: list[dict]) -> None:
    tiles: list[np.ndarray] = []
    step = max(1, len(records) // 12)
    for record in records[::step][:12]:
        image = cv2.imread(str(output / record["image"]))
        if image is None:
            continue
        for obj in record["objects"]:
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (40, 220, 40), 1)
        label = f"{record['video']} {record['phase']}"
        cv2.putText(image, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(image, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(cv2.resize(image, (350, 232), interpolation=cv2.INTER_AREA))
    if not tiles:
        return
    blank = np.zeros_like(tiles[0])
    tiles.extend([blank] * ((-len(tiles)) % 4))
    rows = [np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(output / "preview.jpg"), np.vstack(rows))


def _safe_output(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.parent != HERE.resolve() or not resolved.name.startswith("lie_real"):
        raise ValueError(
            "output must be a direct child of ml/ named lie_real*; " f"got {resolved}"
        )
    return resolved


def reset_output(path: Path) -> None:
    root = _safe_output(path)
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="per-recording panel config (ROI/active/family + val list)",
    )
    parser.add_argument(
        "--val-video",
        action="append",
        default=None,
        help=(
            "recording reserved for validation (split is by whole video); "
            "repeatable. Overrides the 'val' list in the config when given."
        ),
    )
    parser.add_argument("--output", type=Path, default=HERE / "lie_real_dataset")
    parser.add_argument(
        "--model",
        type=Path,
        help="optional lie-shape YOLO weights used as an auxiliary proposal",
    )
    parser.add_argument("--yolo-conf", type=float, default=0.05)
    parser.add_argument("--background-samples", type=int, default=72)
    parser.add_argument("--diff-threshold", type=int, default=22)
    parser.add_argument("--min-area-ratio", type=float, default=0.18)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--match-distance", type=float, default=48.0)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--highlight-fps", type=float, default=2.0)
    parser.add_argument("--fade-fps", type=float, default=9.0)
    parser.add_argument("--motion-fps", type=float, default=4.0)
    args = parser.parse_args()

    videos_dir, specs = load_specs(args.config)
    known = {spec.filename for spec, _ in specs}
    if args.val_video:
        override = set(args.val_video)
        unknown = override - known
        if unknown:
            parser.error(f"--val-video not in config: {sorted(unknown)}")
        specs = [
            (spec, "val" if spec.filename in override else "train")
            for spec, _ in specs
        ]

    profile = SamplingProfile(
        highlight_fps=args.highlight_fps,
        fade_fps=args.fade_fps,
        motion_fps=args.motion_fps,
    )
    output = _safe_output(args.output)
    reset_output(output)
    detector = _load_detector(args.model, args.yolo_conf)

    all_records: list[dict] = []
    per_split: dict[str, list[dict]] = {"train": [], "val": []}
    for spec, split in specs:
        records = process_recording(
            spec,
            videos_dir=videos_dir,
            split=split,
            output=output,
            profile=profile,
            detector=detector,
            background_samples=args.background_samples,
            diff_threshold=args.diff_threshold,
            min_area_ratio=args.min_area_ratio,
            nms_iou=args.nms_iou,
            match_distance=args.match_distance,
            min_hits=args.min_hits,
            max_gap=args.max_gap,
        )
        per_split[split].extend(records)
        all_records.extend(records)

    # Per-split trajectory metadata for downstream tracker/DeepSORT evaluation.
    for split, records in per_split.items():
        (output / "metadata" / f"{split}_tracks.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    (output / "manifest.json").write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # Single-class list for label-studio-converter / CVAT YOLO import.
    (output / "classes.txt").write_text("lie_shape\n", encoding="utf-8")
    _write_cvat_manifest(output, all_records)
    _write_preview(output, all_records)

    val_videos = sorted({spec.filename for spec, split in specs if split == "val"})
    train_videos = sorted({spec.filename for spec, split in specs if split == "train"})
    summary = {
        "output": str(output),
        "config": str(args.config),
        "videos_dir": str(videos_dir),
        "train_videos": train_videos,
        "val_videos": val_videos,
        "train_frames": len(per_split["train"]),
        "val_frames": len(per_split["val"]),
        "train_boxes": sum(r["box_count"] for r in per_split["train"]),
        "val_boxes": sum(r["box_count"] for r in per_split["val"]),
        "used_yolo_auxiliary": args.model is not None,
        "note": (
            "Pre-labels are NOT ground truth. Review in CVAT/Label Studio, then "
            "mix in via: python3 ml/build_lie_dataset_v2.py --reviewed-real "
            f"{output}"
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Preview: {output / 'preview.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
