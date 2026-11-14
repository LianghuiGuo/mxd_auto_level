from __future__ import annotations

import atexit
from concurrent.futures import Future
import csv
import math
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np

from src.utils.logger import logger

logger = logger.for_category("lie_detector")


class _DiagnosticWriter:
    def __init__(self):
        self._jobs = queue.Queue(maxsize=4)
        self._thread = None
        self._lock = threading.Lock()

    def submit(self, function, *args):
        future = Future()
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="LieDiagnosticWriter", daemon=True
                )
                self._thread.start()
            try:
                self._jobs.put_nowait((function, args, future))
            except queue.Full:
                logger.warning("[Lie Detector] Diagnostic queue full; skipping save without blocking tracking.")
                return None
        return future

    def _run(self):
        while True:
            function, args, future = self._jobs.get()
            try:
                future.set_result(function(*args))
            except Exception as exc:
                logger.warning(f"[Lie Detector] Background diagnostic save failed: {exc}")
                future.set_exception(exc)
            finally:
                function = args = future = None
                self._jobs.task_done()

    def wait(self, timeout=2.0):
        deadline = time.monotonic() + timeout
        with self._jobs.all_tasks_done:
            while self._jobs.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._jobs.all_tasks_done.wait(remaining)
        return True


_writer = _DiagnosticWriter()
atexit.register(_writer.wait)


def wait_for_diagnostic_writes(timeout=2.0):
    return _writer.wait(timeout)


def _write_image(path, image):
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise OSError(f"Could not write {path}")


def _overlay(frame, record):
    image = frame.copy()
    if record["status"] == 0:
        cx, cy = record["center_x"], record["center_y"]
        x, y, width, height = (
            record["bbox_x"], record["bbox_y"], record["bbox_width"], record["bbox_height"]
        )
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.circle(image, (cx, cy), 6, (0, 0, 255), 2)
        cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    confidence = record["confidence"]
    label = (
        f"SDK frame={record['frame']} status={record['status']} "
        f"conf={confidence:.3f} sdk_ms={record['sdk_ms']}"
    )
    cv2.putText(image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return image


def _write_input_snapshot(frame):
    started = time.monotonic()
    output = Path("log")
    output.mkdir(exist_ok=True)
    _write_image(output / "lie_sdk_input_frame.png", frame)
    logger.info(
        "[Lie Detector] SDK input snapshot saved in background to "
        "log/lie_sdk_input_frame.png; coordinates are local to the SDK input image; "
        f"write_ms={(time.monotonic() - started) * 1000:.1f}."
    )


def save_input_snapshot(frame):
    return _writer.submit(_write_input_snapshot, frame.copy())


def _write_first_lock(frame, record):
    started = time.monotonic()
    output = Path("log")
    output.mkdir(exist_ok=True)
    _write_image(output / "lie_sdk_first_lock_frame.png", frame)
    _write_image(output / "lie_sdk_first_lock_overlay.png", _overlay(frame, record))
    logger.info(
        "[Lie Detector SDK] First-lock snapshots saved in background to "
        "log/lie_sdk_first_lock_{frame,overlay}.png; "
        f"frame={record['frame']}; captured_at={record['captured_at']}; "
        f"center=({record['center_x']},{record['center_y']}); "
        "green=SDK bbox, red=SDK center (not measured mouse position); "
        f"write_ms={(time.monotonic() - started) * 1000:.1f}."
    )


def save_first_lock_snapshot(frame, record):
    return _writer.submit(_write_first_lock, frame.copy(), dict(record))


def _write_recent_trace(records, frames, reason):
    started = time.monotonic()
    output = Path("log")
    output.mkdir(exist_ok=True)
    csv_path = output / "lie_sdk_recent_trace.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if frames:
        tile_width = min(960, max(frame.shape[1] for _, frame in frames))
        tile_height = max(round(frame.shape[0] * tile_width / frame.shape[1]) for _, frame in frames)
        sheet = np.zeros((math.ceil(len(frames) / 2) * (tile_height + 28), tile_width * 2, 3), np.uint8)
        by_frame = {record["frame"]: record for record in records}
        for index, (number, frame) in enumerate(frames):
            record = by_frame[number]
            image = _overlay(frame, record)
            resized_height = round(image.shape[0] * tile_width / image.shape[1])
            image = cv2.resize(image, (tile_width, resized_height), interpolation=cv2.INTER_AREA)
            x, y = index % 2 * tile_width, index // 2 * (tile_height + 28)
            sheet[y:y + resized_height, x:x + tile_width] = image
            cv2.putText(
                sheet, f"frame={number} captured_at={record['captured_at']}",
                (x + 8, y + tile_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
        _write_image(output / "lie_sdk_recent_overlay.png", sheet)
    logger.info(
        f"[Lie Detector SDK] Recent trace saved in background; reason={reason}; "
        f"records={len(records)}; sampled_frames={len(frames)}; "
        "files=log/lie_sdk_recent_trace.csv, log/lie_sdk_recent_overlay.png; "
        "last two seconds of SDK observations, images sampled at most 5 FPS; "
        "screen requests are not measured cursor positions; files overwrite the previous trace; "
        f"write_ms={(time.monotonic() - started) * 1000:.1f}."
    )


def save_recent_trace(records, frames, reason):
    if not records:
        return None
    return _writer.submit(_write_recent_trace, [dict(record) for record in records], list(frames), reason)
