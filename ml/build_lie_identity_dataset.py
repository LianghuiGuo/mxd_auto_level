#!/usr/bin/env python3
"""Build an automatically labelled trajectory-level REAL ranking dataset.

The green cursor is used only as an offline label source.  It is never exposed
as a model feature: LieDetectorTracker removes it before extracting detections
and all exported features come from tracker state.

Outputs per split:

* ``<split>.csv``: one flat LightGBM/XGBoost row per candidate track;
* ``<split>.jsonl``: one listwise query per frame, including short histories
  for a future GRU experiment;
* ``manifest.json``: feature schema, video split and rejection statistics.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Iterable, Sequence

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.LieDetectorTracker import (  # noqa: E402
    ConstellationTrackView,
    LieDetectorTracker,
)
from src.engine.LieIdentityRanker import (  # noqa: E402
    BASE_FEATURE_NAMES,
    FLAT_FEATURE_NAMES,
    HISTORY_FEATURE_NAMES,
    aggregate_history,
    extract_frame_features,
)
from src.engine.LieShapeYoloDetector import (  # noqa: E402
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_WEIGHTS,
    LieShapeYoloDetector,
)
from tools.lie_detector_replay import cursor_ground_truth  # noqa: E402


@dataclass(frozen=True)
class LabelDecision:
    track_id: int | None
    reason: str
    nearest_distance: float | None = None
    second_distance: float | None = None


def _visible_candidates(
    views: Iterable[ConstellationTrackView],
    min_visible_streak: int,
) -> list[ConstellationTrackView]:
    return [
        view
        for view in views
        if view.lost_frames == 0
        and not view.predicted_only
        and view.visible_streak >= min_visible_streak
    ]


def select_cursor_label(
    views: Sequence[ConstellationTrackView],
    cursor: tuple[float, float] | None,
    *,
    label_radius: float = 40.0,
    ambiguity_margin: float = 18.0,
    min_visible_streak: int = 2,
) -> LabelDecision:
    """Choose a clean REAL label or explain why the frame is rejected."""
    if cursor is None:
        return LabelDecision(None, "cursor_missing")
    candidates = _visible_candidates(views, min_visible_streak)
    if not candidates:
        return LabelDecision(None, "no_visible_track")
    ranked = sorted(
        (
            math.hypot(
                float(view.center[0]) - float(cursor[0]),
                float(view.center[1]) - float(cursor[1]),
            ),
            view.track_id,
        )
        for view in candidates
    )
    nearest_distance, nearest_id = ranked[0]
    second_distance = ranked[1][0] if len(ranked) > 1 else None
    if nearest_distance > label_radius:
        return LabelDecision(
            None,
            "no_nearby_track",
            nearest_distance,
            second_distance,
        )
    if (
        second_distance is not None
        and (
            second_distance <= label_radius
            or second_distance - nearest_distance < ambiguity_margin
        )
    ):
        return LabelDecision(
            None,
            "ambiguous_tracks",
            nearest_distance,
            second_distance,
        )
    return LabelDecision(
        nearest_id,
        "accepted",
        nearest_distance,
        second_distance,
    )


def _rounded_vector(
    feature: dict[str, float],
    names: Sequence[str],
) -> list[float]:
    return [round(float(feature[name]), 6) for name in names]


class _SplitWriter:
    def __init__(self, output: Path, split: str):
        self._json_handle = (output / f"{split}.jsonl").open(
            "w", encoding="utf-8"
        )
        self._csv_handle = (output / f"{split}.csv").open(
            "w", newline="", encoding="utf-8"
        )
        metadata = (
            "query_id",
            "video",
            "frame",
            "timestamp",
            "track_id",
            "label",
            "group_size",
            "label_distance",
            "label_margin",
        )
        self._csv_writer = csv.DictWriter(
            self._csv_handle,
            fieldnames=[*metadata, *FLAT_FEATURE_NAMES],
        )
        self._csv_writer.writeheader()

    def write(
        self,
        *,
        query_id: str,
        video: str,
        frame_index: int,
        timestamp: float,
        label_track_id: int,
        label_distance: float,
        label_margin: float | None,
        candidates: Sequence[ConstellationTrackView],
        histories: dict[int, deque[dict[str, float]]],
        history_size: int,
    ) -> None:
        payload_candidates = []
        for view in candidates:
            history = list(histories[view.track_id])
            flat = aggregate_history(history, history_size=history_size)
            is_real = int(view.track_id == label_track_id)
            row: dict[str, object] = {
                "query_id": query_id,
                "video": video,
                "frame": frame_index,
                "timestamp": f"{timestamp:.6f}",
                "track_id": view.track_id,
                "label": is_real,
                "group_size": len(candidates),
                "label_distance": f"{label_distance:.6f}",
                "label_margin": (
                    ""
                    if label_margin is None
                    else f"{label_margin:.6f}"
                ),
            }
            row.update(
                {name: f"{flat[name]:.8g}" for name in FLAT_FEATURE_NAMES}
            )
            self._csv_writer.writerow(row)
            payload_candidates.append(
                {
                    "track_id": view.track_id,
                    "label": is_real,
                    "features": _rounded_vector(flat, FLAT_FEATURE_NAMES),
                    "history": [
                        _rounded_vector(item, HISTORY_FEATURE_NAMES)
                        for item in history
                    ],
                }
            )
        payload = {
            "query_id": query_id,
            "video": video,
            "frame": frame_index,
            "timestamp": round(timestamp, 6),
            "label_track_id": label_track_id,
            "label_distance": round(label_distance, 6),
            "label_margin": (
                None if label_margin is None else round(label_margin, 6)
            ),
            "candidates": payload_candidates,
        }
        self._json_handle.write(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        )

    def close(self) -> None:
        self._json_handle.close()
        self._csv_handle.close()


def _selected_clip_numbers(value: str) -> set[int] | None:
    if value.strip().lower() == "all":
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def _clip_number(filename: str) -> int | None:
    stem = Path(filename).stem
    prefix = "测谎录屏"
    if not stem.startswith(prefix):
        return None
    suffix = stem[len(prefix):]
    return int(suffix) if suffix.isdigit() else None


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{path} is not empty; pass --overwrite to replace it"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_dataset(args: argparse.Namespace) -> dict[str, object]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected = _selected_clip_numbers(args.clips)
    val_videos = set(config.get("val", []))
    recordings = []
    for recording in config.get("recordings", []):
        number = _clip_number(recording["filename"])
        if selected is not None and number not in selected:
            continue
        video = args.videos / recording["filename"]
        if video.is_file():
            recordings.append((recording, video))
    if not recordings:
        raise ValueError("no matching recordings found")

    _prepare_output(args.output, args.overwrite)
    writers = {
        split: _SplitWriter(args.output, split)
        for split in ("train", "val")
    }
    total_rejections: Counter[str] = Counter()
    split_queries: Counter[str] = Counter()
    split_rows: Counter[str] = Counter()
    video_summaries: list[dict[str, object]] = []

    try:
        for recording, video_path in recordings:
            split = "val" if recording["filename"] in val_videos else "train"
            detector = None
            if not args.no_yolo:
                detector = LieShapeYoloDetector(
                    args.model,
                    confidence=args.yolo_conf,
                    image_size=args.yolo_imgsz,
                    inference_stride=args.yolo_stride,
                )
            tracker = LieDetectorTracker(
                candidate_detector=detector,
                multi_hypothesis_identity=True,
                stale_coast_recovery=True,
            )
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError(f"unable to open {video_path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            x, y, width, height = (int(value) for value in recording["roi"])
            active_start = float(recording.get("active_start", 0.0))
            active_end = float(recording.get("active_end", float("inf")))
            histories: dict[int, deque[dict[str, float]]] = defaultdict(
                lambda: deque(maxlen=args.history)
            )
            rejections: Counter[str] = Counter()
            query_count = 0
            row_count = 0
            catchup_streak = 0
            catchup_track_id: int | None = None
            caught_up = False
            saw_white = False
            white_faded = False
            frame_index = 0

            while True:
                ok, full_frame = capture.read()
                if not ok:
                    break
                timestamp = frame_index / fps
                if timestamp > active_end:
                    break
                if (
                    x < 0
                    or y < 0
                    or x + width > full_frame.shape[1]
                    or y + height > full_frame.shape[0]
                ):
                    raise ValueError(
                        f"ROI {recording['roi']} exceeds frame in {video_path}"
                    )
                frame = full_frame[y:y + height, x:x + width]
                cursor = cursor_ground_truth(frame)
                result = tracker.update(frame, timestamp)
                views = result.constellation
                features = extract_frame_features(
                    views,
                    target_id=result.target_id,
                    frame_width=width,
                    frame_height=height,
                )
                for track_id, feature in features.items():
                    histories[track_id].append(feature)

                if result.white_active:
                    saw_white = True
                elif saw_white:
                    white_faded = True

                if timestamp < active_start:
                    rejections["before_active_window"] += 1
                    frame_index += 1
                    continue
                catchup_decision = select_cursor_label(
                    views,
                    cursor,
                    label_radius=args.catchup_radius,
                    ambiguity_margin=args.ambiguity_margin,
                    min_visible_streak=args.min_visible_streak,
                )
                if catchup_decision.track_id is None:
                    catchup_track_id = None
                    catchup_streak = 0
                elif catchup_decision.track_id == catchup_track_id:
                    catchup_streak += 1
                    if catchup_streak >= args.catchup_frames:
                        caught_up = True
                else:
                    catchup_track_id = catchup_decision.track_id
                    catchup_streak = 1

                if not caught_up:
                    rejections["before_cursor_catchup"] += 1
                else:
                    candidates = _visible_candidates(
                        views, args.min_visible_streak
                    )
                    if len(candidates) < args.min_candidates:
                        rejections["too_few_candidates"] += 1
                    else:
                        decision = select_cursor_label(
                            views,
                            cursor,
                            label_radius=args.label_radius,
                            ambiguity_margin=args.ambiguity_margin,
                            min_visible_streak=args.min_visible_streak,
                        )
                        if decision.track_id is None:
                            rejections[decision.reason] += 1
                        else:
                            hard_frame = bool(
                                decision.track_id != result.target_id
                                or result.predicted_only
                                or result.identity_switched
                                or result.stale_recovery_committed
                            )
                            if (
                                frame_index % args.sample_stride != 0
                                and not hard_frame
                            ):
                                rejections["sample_stride"] += 1
                                frame_index += 1
                                continue
                            query_id = (
                                f"{Path(recording['filename']).stem}:"
                                f"{frame_index:06d}"
                            )
                            writers[split].write(
                                query_id=query_id,
                                video=recording["filename"],
                                frame_index=frame_index,
                                timestamp=timestamp,
                                label_track_id=decision.track_id,
                                label_distance=float(
                                    decision.nearest_distance or 0.0
                                ),
                                label_margin=(
                                    None
                                    if decision.second_distance is None
                                    else decision.second_distance
                                    - float(decision.nearest_distance or 0.0)
                                ),
                                candidates=candidates,
                                histories=histories,
                                history_size=args.history,
                            )
                            query_count += 1
                            row_count += len(candidates)
                frame_index += 1

            capture.release()
            total_rejections.update(rejections)
            split_queries[split] += query_count
            split_rows[split] += row_count
            video_summaries.append(
                {
                    "filename": recording["filename"],
                    "split": split,
                    "frames_read": frame_index,
                    "queries": query_count,
                    "candidate_rows": row_count,
                    "rejections": dict(sorted(rejections.items())),
                    "cursor_caught_up": caught_up,
                    "white_faded": white_faded,
                }
            )
            print(
                f"{recording['filename']}: {query_count} queries, "
                f"{row_count} candidates -> {split}",
                flush=True,
            )
    finally:
        for writer in writers.values():
            writer.close()

    manifest: dict[str, object] = {
        "format_version": 1,
        "task": "listwise_real_track_ranking",
        "base_feature_names": list(BASE_FEATURE_NAMES),
        "flat_feature_names": list(FLAT_FEATURE_NAMES),
        "history_feature_names": list(HISTORY_FEATURE_NAMES),
        "history_frames": args.history,
        "label_policy": {
            "source": "green_cursor_nearest_visible_track",
            "label_radius": args.label_radius,
            "ambiguity_margin": args.ambiguity_margin,
            "catchup_frames": args.catchup_frames,
            "catchup_radius": args.catchup_radius,
            "min_visible_streak": args.min_visible_streak,
            "min_candidates": args.min_candidates,
            "sample_stride": args.sample_stride,
            "hard_frames_bypass_stride": True,
            "requires_white_fade": False,
            "catchup_requires_same_track": True,
        },
        "detector": {
            "enabled": not args.no_yolo,
            "model": None if args.no_yolo else str(args.model),
            "confidence": args.yolo_conf,
            "image_size": args.yolo_imgsz,
            "stride": args.yolo_stride,
        },
        "splits": {
            split: {
                "queries": split_queries[split],
                "candidate_rows": split_rows[split],
            }
            for split in ("train", "val")
        },
        "total_rejections": dict(sorted(total_rejections.items())),
        "videos": video_summaries,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/lie_videos_config.json"),
    )
    parser.add_argument("--videos", type=Path, default=Path("ml/videos"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/lie_identity_dataset"),
    )
    parser.add_argument(
        "--clips",
        default="all",
        help="comma-separated 测谎录屏 numbers, or all",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--model", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--yolo-conf", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--yolo-imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--yolo-stride", type=int, default=1)
    parser.add_argument("--history", type=int, default=16)
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--label-radius", type=float, default=40.0)
    parser.add_argument("--ambiguity-margin", type=float, default=18.0)
    parser.add_argument("--catchup-frames", type=int, default=5)
    parser.add_argument("--catchup-radius", type=float, default=40.0)
    parser.add_argument("--min-visible-streak", type=int, default=2)
    parser.add_argument("--min-candidates", type=int, default=2)
    args = parser.parse_args()
    if args.history < 2:
        parser.error("--history must be >= 2")
    if args.sample_stride < 1:
        parser.error("--sample-stride must be >= 1")
    if args.min_candidates < 2:
        parser.error("--min-candidates must be >= 2")
    if not args.config.is_file():
        parser.error(f"config not found: {args.config}")
    if not args.no_yolo and not args.model.is_file():
        parser.error(f"YOLO weights not found: {args.model}")
    return args


def main() -> int:
    args = parse_args()
    manifest = build_dataset(args)
    split_summary = manifest["splits"]
    print(
        "wrote "
        f"{split_summary['train']['queries']} train and "
        f"{split_summary['val']['queries']} val queries to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
