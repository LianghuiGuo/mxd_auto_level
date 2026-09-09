"""Extract representative high-error REAL predictions from replay CSVs.

The output is intended for visual failure analysis.  Each selected ROI frame
contains the tracker prediction, cursor ground truth, their error vector, and
the key tracking state copied from the corresponding replay CSV row.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tools.lie_detector_replay import cursor_ground_truth


def load_recordings(config_path: Path) -> dict[str, dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        entry["filename"]: entry
        for entry in config.get("recordings", [])
    }


def parse_value(raw: str, cast: type) -> Any:
    return None if raw == "" else cast(raw)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key in ("frame", "target_id", "lost_frames", "detection_count",
                        "track_count", "identity_switched", "predicted_only",
                        "white_active"):
                row[key] = parse_value(raw[key], int)
            for key in ("timestamp", "pred_x", "pred_y", "gt_x", "gt_y",
                        "error_px", "collective_dx", "collective_dy",
                        "collective_rotation_deg"):
                row[key] = parse_value(raw[key], float)
            rows.append(row)
    return rows


def select_rows(
    rows: list[dict[str, Any]],
    count: int,
    error_threshold: float,
    min_gap_seconds: float,
) -> list[dict[str, Any]]:
    candidates = [
        row for row in rows
        if row["error_px"] is not None and row["error_px"] > error_threshold
    ]
    ranked = sorted(candidates, key=lambda row: row["error_px"], reverse=True)
    selected: list[dict[str, Any]] = []
    # Prefer distinct failure moments.  If a short clip cannot satisfy the
    # requested spacing, progressively relax it while still avoiding duplicates.
    for gap in (min_gap_seconds, min_gap_seconds / 2.0, 0.0):
        for row in ranked:
            if row in selected:
                continue
            if all(abs(row["timestamp"] - item["timestamp"]) >= gap for item in selected):
                selected.append(row)
                if len(selected) == count:
                    return sorted(selected, key=lambda item: item["timestamp"])
    return sorted(selected, key=lambda item: item["timestamp"])


def nearby_switch(rows: list[dict[str, Any]], index: int, radius: int = 15) -> bool:
    start = max(0, index - radius)
    return any(bool(rows[pos]["identity_switched"]) for pos in range(start, index + 1))


def select_missing_real_rows(
    rows: list[dict[str, Any]],
    count: int,
    active_start: float,
    active_end: float,
) -> list[dict[str, Any]]:
    candidates = [
        row for row in rows
        if active_start <= row["timestamp"] <= active_end
        and row["target_id"] is None
    ]
    if not candidates:
        return []
    targets = np.linspace(active_start, active_end, count + 2)[1:-1]
    selected: list[dict[str, Any]] = []
    for target in targets:
        row = min(
            (item for item in candidates if item not in selected),
            key=lambda item: abs(item["timestamp"] - target),
            default=None,
        )
        if row is not None:
            selected.append(row)
    return selected


def annotate(frame: np.ndarray, row: dict[str, Any], switch_nearby: bool) -> np.ndarray:
    image = frame.copy()
    gt = (int(round(row["gt_x"])), int(round(row["gt_y"])))
    cv2.circle(image, gt, 18, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.circle(image, gt, 4, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.putText(image, "GT", (gt[0] + 8, gt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2, cv2.LINE_AA)
    if row["pred_x"] is not None and row["pred_y"] is not None:
        pred = (int(round(row["pred_x"])), int(round(row["pred_y"])))
        radius = max(8, int(round(
            parse_value(str(row.get("radius", "")), float) or 20
        )))
        cv2.circle(image, pred, radius, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(image, pred, 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.arrowedLine(
            image, pred, gt, (0, 180, 255), 2, cv2.LINE_AA, tipLength=0.08
        )
        cv2.putText(image, "PRED", (pred[0] + 8, pred[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2, cv2.LINE_AA)

    error_text = (
        "NO REAL"
        if row["error_px"] is None
        else f"err={row['error_px']:.1f}px REAL={row['target_id']}"
    )
    line1 = (
        f"f={row['frame']} t={row['timestamp']:.2f}s "
        f"{error_text}"
    )
    line2 = (
        f"lost={row['lost_frames']} pred_only={row['predicted_only']} "
        f"det={row['detection_count']} tracks={row['track_count']} "
        f"switch15={'Y' if switch_nearby else 'N'}"
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 58), (16, 16, 16), -1)
    cv2.putText(image, line1, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.49,
                (220, 220, 220), 1, cv2.LINE_AA)
    return image


def contact_sheet(images: list[np.ndarray]) -> np.ndarray:
    target_width = 520
    resized = []
    for image in images:
        scale = target_width / image.shape[1]
        resized.append(cv2.resize(image, (target_width, int(image.shape[0] * scale))))
    height = max(image.shape[0] for image in resized)
    tiles = []
    for image in resized:
        if image.shape[0] < height:
            image = cv2.copyMakeBorder(
                image, 0, height - image.shape[0], 0, 0,
                cv2.BORDER_CONSTANT, value=(24, 24, 24),
            )
        tiles.append(image)
    rows = []
    for start in range(0, len(tiles), 2):
        pair = tiles[start:start + 2]
        if len(pair) == 1:
            pair.append(np.full_like(pair[0], 24))
        rows.append(cv2.hconcat(pair))
    return cv2.vconcat(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, default=Path("ml/videos"))
    parser.add_argument("--csv-dir", type=Path, default=Path("log/lie_vis"))
    parser.add_argument("--config", type=Path, default=Path("ml/lie_videos_config.json"))
    parser.add_argument("--output", type=Path, default=Path("log/lie_error_analysis"))
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=40.0)
    parser.add_argument("--min-gap", type=float, default=0.5)
    args = parser.parse_args()

    recordings = load_recordings(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "selection": {
            "count": args.count,
            "error_threshold_px": args.threshold,
            "preferred_min_gap_seconds": args.min_gap,
        },
        "videos": {},
    }

    for number in range(1, 12):
        label = f"录屏{number}"
        video_path = args.videos / f"测谎录屏{number}.mp4"
        csv_path = args.csv_dir / f"{label}_fourcue.csv"
        if not video_path.exists() or not csv_path.exists():
            report["videos"][label] = {"status": "missing input"}
            continue
        rows = load_rows(csv_path)
        selected = select_rows(rows, args.count, args.threshold, args.min_gap)
        recording = recordings[video_path.name]
        selection_kind = "high_error"
        if not selected:
            selected = select_missing_real_rows(
                rows,
                args.count,
                float(recording["active_start"]),
                float(recording["active_end"]),
            )
            selection_kind = "missing_real"
        if not selected:
            report["videos"][label] = {"status": "no failure samples"}
            continue
        roi = tuple(int(value) for value in recording["roi"])
        capture = cv2.VideoCapture(str(video_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        output_dir = args.output / label
        output_dir.mkdir(parents=True, exist_ok=True)
        images: list[np.ndarray] = []
        entries: list[dict[str, Any]] = []
        for rank, row in enumerate(selected, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
            ok, full_frame = capture.read()
            if not ok:
                continue
            x, y, width, height = roi
            frame = full_frame[y:y + height, x:x + width]
            if row["gt_x"] is None or row["gt_y"] is None:
                ground_truth = cursor_ground_truth(frame)
                if ground_truth is None:
                    continue
                row = dict(row)
                row["gt_x"], row["gt_y"] = ground_truth
            row_index = int(row["frame"])
            switch_nearby = nearby_switch(rows, row_index)
            image = annotate(frame, row, switch_nearby)
            error_suffix = (
                "no_real"
                if row["error_px"] is None
                else f"err{row['error_px']:.0f}"
            )
            filename = f"{rank:02d}_f{row['frame']}_{error_suffix}.jpg"
            cv2.imwrite(str(output_dir / filename), image)
            images.append(image)
            entries.append({
                "rank": rank,
                "file": str(output_dir / filename),
                "frame": row["frame"],
                "timestamp": row["timestamp"],
                "error_px": row["error_px"],
                "target_id": row["target_id"],
                "predicted_only": bool(row["predicted_only"]),
                "lost_frames": row["lost_frames"],
                "detection_count": row["detection_count"],
                "track_count": row["track_count"],
                "near_switch_previous_15_frames": switch_nearby,
                "white_active": bool(row["white_active"]),
                "selection_kind": selection_kind,
            })
        capture.release()
        if images:
            sheet_path = args.output / f"{label}_contact.jpg"
            cv2.imwrite(str(sheet_path), contact_sheet(images))
            report["videos"][label] = {
                "status": "ok",
                "selection_kind": selection_kind,
                "video_fps": fps,
                "contact_sheet": str(sheet_path),
                "frames": entries,
            }

    metadata_path = args.output / "selected_frames.json"
    metadata_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
