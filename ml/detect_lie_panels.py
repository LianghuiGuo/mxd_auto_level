#!/usr/bin/env python3
"""Auto-detect the lie-detector panel ROI from its UI chrome border.

The MapleStory lie-detector dialog has a distinctive frame:

* a pale title bar at the top (``谎言探测仪``);
* a pale instruction bar at the bottom
  (``移动鼠标跟随逐渐变透明的图形。``);
* the rocky pattern canvas sandwiched between them.

This tool anchors on those two solid chrome bars (much more reliable than
trying to segment the rock texture itself), then takes the rectangle between
them as the pattern ROI.

The older hand-labelled ``测谎视频*.mp4`` clips keep their existing ROIs unless
you pass ``--force``.  The ``测谎录屏*.mp4`` series is the primary auto target.

    python3 ml/detect_lie_panels.py
    python3 ml/detect_lie_panels.py --only 录屏
    python3 ml/detect_lie_panels.py --from-config   # render existing ROIs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DEFAULT_CONFIG = HERE / "lie_videos_config.json"


# --------------------------------------------------------------------------- #
# Chrome-bar detection
# --------------------------------------------------------------------------- #
def _pale_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Mask the pale gray/blue UI chrome used by the dialog bars."""

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return (
        (hsv[:, :, 1] <= 70) & (hsv[:, :, 2] >= 160) & (hsv[:, :, 2] <= 250)
    ).astype(np.uint8) * 255


def find_instruction_bar(frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return ``(x, y, w, h)`` of the bottom instruction bar, or None."""

    height, width = frame_bgr.shape[:2]
    mask = _pale_mask(frame_bgr)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (51, 5))
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_bar_h = max(12, int(0.015 * height))
    max_bar_h = max(80, int(0.10 * height))  # ~72 @720p, ~108 @1080p
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, bar_w, bar_h = cv2.boundingRect(contour)
        if bar_w < 0.22 * width or bar_w > 0.72 * width:
            continue
        if bar_h < min_bar_h or bar_h > max_bar_h:
            continue
        if bar_w / max(bar_h, 1) < 6.0:
            continue
        fill = float(cv2.contourArea(contour)) / max(float(bar_w * bar_h), 1.0)
        if fill < 0.40:
            continue
        # Instruction bar sits in the lower half of the panel / frame.
        if y < 0.22 * height or y > 0.92 * height:
            continue
        center_x = x + bar_w / 2.0
        score = (
            bar_w
            * bar_h
            * fill
            * (1.0 - 0.8 * abs(center_x / width - 0.5))
        )
        candidates.append((score, (x, y, bar_w, bar_h)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def find_title_bar(
    frame_bgr: np.ndarray,
    instruction: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    """Return ``(x, y, w, h)`` of the title chrome above the instruction bar."""

    ix, iy, iw, _ih = instruction
    height, width = frame_bgr.shape[:2]
    # Search a band above the instruction bar, roughly one panel-height tall.
    y0 = max(0, iy - int(1.20 * iw))
    y1 = max(y0 + 1, iy - 6)
    x0 = max(0, ix - 40)
    x1 = min(width, ix + iw + 40)
    band = frame_bgr[y0:y1, x0:x1]
    if band.size == 0:
        return None
    mask = _pale_mask(band)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 2))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 3))
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    expected_top = iy - int(0.85 * iw)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, bar_w, bar_h = cv2.boundingRect(contour)
        fx, fy = x0 + x, y0 + y
        # Title chrome may be shorter than the full panel width (tab-like) but
        # should still cover a large fraction of it.
        if bar_w < 0.40 * iw:
            continue
        max_title_h = max(55, int(0.06 * height))
        if bar_h < 6 or bar_h > max_title_h:
            continue
        score = float(bar_w) - 0.55 * abs(fy - max(expected_top, 0))
        candidates.append((score, (fx, fy, bar_w, bar_h)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def pattern_roi_from_chrome(
    instruction: tuple[int, int, int, int],
    title: tuple[int, int, int, int] | None,
    *,
    side_inset: int = 6,
    top_inset: int = 4,
    bottom_inset: int = 4,
) -> tuple[int, int, int, int] | None:
    """Pattern canvas = between title bottom and instruction top."""

    ix, iy, iw, _ih = instruction
    if title is not None:
        _tx, ty, _tw, th = title
        left = ix + side_inset
        top = ty + th + top_inset
        right = ix + iw - side_inset
        bottom = iy - bottom_inset
    else:
        # Fallback proportions when title chrome is missing on this frame.
        left = ix + side_inset
        right = ix + iw - side_inset
        height = int(0.68 * iw)
        bottom = iy - bottom_inset
        top = bottom - height
    width = right - left
    height = bottom - top
    if width < 120 or height < 100:
        return None
    return (int(left), int(top), int(width), int(height))


def detect_frame(frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    instruction = find_instruction_bar(frame_bgr)
    if instruction is None:
        return None
    title = find_title_bar(frame_bgr, instruction)
    return pattern_roi_from_chrome(instruction, title)


# --------------------------------------------------------------------------- #
# Per-recording aggregation
# --------------------------------------------------------------------------- #
def detect_recording(
    path: Path,
    *,
    probe_count: int = 20,
) -> dict | None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        capture.release()
        return None
    duration = frame_count / fps

    samples: list[tuple[float, tuple[int, int, int, int]]] = []
    for timestamp in np.linspace(0.2, max(duration - 0.2, 0.2), probe_count):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(timestamp * fps)))
        ok, frame = capture.read()
        if not ok:
            continue
        roi = detect_frame(frame)
        if roi is None:
            continue
        samples.append((float(timestamp), roi))
    capture.release()
    if len(samples) < 2:
        return None

    rois = np.asarray([roi for _, roi in samples], dtype=np.float64)
    # Reject outliers that diverge strongly from the median box.
    median = np.median(rois, axis=0)
    scale = np.maximum(median[2:], 1.0)
    deltas = np.abs(rois - median)
    keep = (
        (deltas[:, 0] < 0.08 * scale[0])
        & (deltas[:, 1] < 0.08 * scale[1])
        & (deltas[:, 2] < 0.12 * scale[0])
        & (deltas[:, 3] < 0.12 * scale[1])
    )
    if int(np.count_nonzero(keep)) >= 2:
        kept_samples = [sample for sample, flag in zip(samples, keep) if flag]
    else:
        kept_samples = samples
    kept = np.asarray([roi for _, roi in kept_samples], dtype=np.float64)
    x, y, width, height = (int(round(v)) for v in np.median(kept, axis=0))

    times = [timestamp for timestamp, _ in kept_samples]
    active_start = float(max(0.0, times[0] - 0.2))
    active_end = float(min(duration, times[-1] + 0.4))
    seed_time = float(times[max(0, len(times) // 4)])

    margin_x = max(4, int(0.02 * width))
    margin_y = max(4, int(0.02 * height))
    safe_crop = (
        margin_x,
        margin_y,
        width - 2 * margin_x,
        height - 2 * margin_y,
    )
    return {
        "filename": path.name,
        "roi": [x, y, width, height],
        "safe_crop": [safe_crop[0], safe_crop[1], safe_crop[2], safe_crop[3]],
        "seed_time": round(seed_time, 2),
        "active_start": round(active_start, 2),
        "active_end": round(active_end, 2),
        "family": "unknown",
        "auto_detected": True,
        "roi_probe_hits": len(kept_samples),
        "duration": round(duration, 2),
    }


# --------------------------------------------------------------------------- #
# Preview / config I/O
# --------------------------------------------------------------------------- #
def _write_preview(videos_dir: Path, specs: list[dict], output: Path) -> None:
    tiles: list[np.ndarray] = []
    for spec in specs:
        capture = cv2.VideoCapture(str(videos_dir / spec["filename"]))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(float(spec["seed_time"]) * fps)))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            continue
        x, y, width, height = (int(v) for v in spec["roi"])
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 3)
        split = spec.get("split", "")
        tag = f" [{split}]" if split else ""
        auto = " auto" if spec.get("auto_detected") else ""
        label = (
            f"{spec['filename']}{tag}{auto} "
            f"{spec['active_start']}-{spec['active_end']}s"
        )
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(
            frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1
        )
        tiles.append(cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA))
    if not tiles:
        return
    blank = np.zeros_like(tiles[0])
    tiles.extend([blank] * ((-len(tiles)) % 3))
    rows = [np.hstack(tiles[index : index + 3]) for index in range(0, len(tiles), 3)]
    cv2.imwrite(str(output), np.vstack(rows))


def _load_existing(config_path: Path) -> dict:
    if not config_path.is_file():
        return {"videos_dir": "ml/videos", "val": [], "recordings": []}
    return json.loads(config_path.read_text(encoding="utf-8"))


def _render_from_config(config_path: Path, videos_override: Path) -> int:
    if not config_path.is_file():
        print(f"config not found: {config_path}", file=sys.stderr)
        return 1
    config = json.loads(config_path.read_text(encoding="utf-8"))
    videos_dir = videos_override
    if videos_override == (HERE / "videos") and config.get("videos_dir"):
        videos_dir = (REPO / config["videos_dir"]).resolve()
    val_names = set(config.get("val", []))
    specs = []
    for entry in config["recordings"]:
        entry = dict(entry)
        entry["split"] = "val" if entry["filename"] in val_names else "train"
        specs.append(entry)
    preview = config_path.parent / "preview_panels_config.jpg"
    _write_preview(videos_dir, specs, preview)
    print(f"Rendered {len(specs)} ROIs from {config_path}")
    print(f"Preview: {preview}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=HERE / "videos")
    parser.add_argument("--output", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--probe-count", type=int, default=36)
    parser.add_argument(
        "--only",
        default="录屏",
        help=(
            "only auto-detect filenames containing this substring "
            "(default: 录屏). Pass empty string to try every mp4."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing config entries for matched videos",
    )
    parser.add_argument(
        "--from-config",
        action="store_true",
        help="do NOT auto-detect; render ROIs from the existing config",
    )
    args = parser.parse_args()

    if args.from_config:
        return _render_from_config(args.output, args.videos)

    videos = sorted(args.videos.glob("*.mp4"))
    if not videos:
        parser.error(f"no mp4 recordings found under {args.videos}")
    if args.only:
        videos = [path for path in videos if args.only in path.name]
        if not videos:
            parser.error(f"no mp4 matched --only={args.only!r} under {args.videos}")

    existing = _load_existing(args.output)
    by_name = {
        entry["filename"]: entry for entry in existing.get("recordings", [])
    }
    val_names = list(existing.get("val", []))

    detected = 0
    failed: list[str] = []
    for video in videos:
        previous = by_name.get(video.name, {})
        # Without --force, leave explicitly hand-tuned entries alone. Entries
        # already marked auto_detected (or brand-new filenames) are refreshed.
        if (
            previous
            and not args.force
            and not previous.get("auto_detected")
            and not args.only
        ):
            print(f"skip hand-tuned {video.name}")
            continue
        try:
            spec = detect_recording(video, probe_count=args.probe_count)
        except Exception as exc:  # pragma: no cover - IO / codec issues
            print(f"WARN: {video.name}: {exc}", file=sys.stderr)
            failed.append(video.name)
            continue
        if spec is None:
            print(f"WARN: could not detect panel in {video.name}", file=sys.stderr)
            failed.append(video.name)
            continue
        if previous.get("family") and previous["family"] != "unknown":
            spec["family"] = previous["family"]
        by_name[video.name] = spec
        detected += 1
        print(json.dumps(spec, ensure_ascii=False))

    recordings = sorted(by_name.values(), key=lambda item: item["filename"])
    config = {
        "videos_dir": str(args.videos.relative_to(REPO))
        if args.videos.is_relative_to(REPO)
        else str(args.videos),
        "note": (
            "测谎录屏* ROIs are auto-detected from the dialog chrome "
            "(title bar + instruction bar). 测谎视频* keep hand-tuned ROIs. "
            "REVIEW preview_panels.jpg / preview_panels_config.jpg and adjust "
            "roi/active_*/family before building datasets. A recording is 'val' "
            "iff its filename is listed in 'val'."
        ),
        "val": val_names,
        "recordings": recordings,
    }
    args.output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    preview = args.output.parent / "preview_panels.jpg"
    _write_preview(args.videos, recordings, preview)
    print(f"Config: {args.output}")
    print(f"Preview: {preview}")
    print(f"Auto-detected: {detected}; failed: {failed or 'none'}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
