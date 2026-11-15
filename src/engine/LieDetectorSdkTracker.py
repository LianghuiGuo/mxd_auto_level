from __future__ import annotations

from collections import deque
import ctypes
import datetime
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Optional
import uuid

import numpy as np

from src.engine.LieDetectorDiagnostics import (
    save_first_lock_snapshot,
    save_input_snapshot,
    save_recent_trace,
    save_session,
)
from src.engine.LieDetectorTracker import LieDetectorTrackingResult
from src.utils.logger import logger

logger = logger.for_category("lie_detector")

DEFAULT_SDK_DIRECTORY = "trtyoloLie_SDK_v220_x64/bin64"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _error_text(value: Optional[bytes]) -> str:
    return value.decode("gbk", errors="replace") if value else ""


class LieDetectorSdkTracker:
    def __init__(
        self,
        sdk_directory: str | Path = DEFAULT_SDK_DIRECTORY,
        *,
        force_cpu: bool = False,
        session_recording: bool = True,
        session_image_fps: float = 3.0,
        session_max_memory_mb: float = 128.0,
    ) -> None:
        self._lock = threading.Lock()
        self._handle = None
        self._lib = None
        self._dll_directory = None
        self._frame_size = None
        self.last_ms = 0
        self.backend = ""
        self.session_recording = bool(session_recording)
        self.session_image_fps = float(session_image_fps)
        memory_mb = float(session_max_memory_mb)
        if not math.isfinite(self.session_image_fps) or self.session_image_fps <= 0:
            raise ValueError("session_image_fps must be positive and finite")
        if not math.isfinite(memory_mb) or memory_mb <= 0:
            raise ValueError("session_max_memory_mb must be positive and finite")
        self._session_max_bytes = int(memory_mb * 1024 * 1024)
        self._clear_diagnostics()
        initialization_started = time.monotonic()

        if sys.platform != "win32" or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise RuntimeError("trtyolo SDK requires Windows and 64-bit Python")

        directory = Path(sdk_directory).expanduser()
        if not directory.is_absolute():
            directory = _PROJECT_ROOT / directory
        directory = directory.resolve()
        for name in ("trtyoloLie.dll", "best.onnx.enc", "auth.txt"):
            if not (directory / name).is_file():
                raise FileNotFoundError(f"trtyolo SDK file not found: {directory / name}")

        try:
            self._dll_directory = os.add_dll_directory(str(directory))
            try:
                self._lib = ctypes.WinDLL(str(directory / "trtyoloLie.dll"))
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot load trtyolo SDK from {directory}: {exc}. "
                    "Keep the complete bin64 directory and install the "
                    "Microsoft Visual C++ 2015-2022 x64 runtime."
                ) from exc
            self._bind_functions()
            version = self._lib.TT_Version()
            if version != 220:
                raise RuntimeError(f"Expected trtyolo SDK v220, got {version}")
            self._lib.TT_SetForceCpu(int(force_cpu))
            self._handle = self._lib.TTx_Create(None)
            if not self._handle:
                reason = _error_text(self._lib.TTx_LastCreateError())
                if not reason:
                    reason = _error_text(self._lib.TT_AuthError())
                machine_code = self._lib.TT_MachineCode() or b""
                raise RuntimeError(
                    f"trtyolo SDK initialization failed: {reason}; "
                    f"machine code: {machine_code.decode('ascii', errors='replace')}. "
                    "Check bin64/auth.txt, network access and machine-code authorization."
                )
            self.backend = (self._lib.TTx_Backend(self._handle) or b"").decode("ascii")
            logger.info(
                f"[Lie Detector] SDK v{version} ready; backend={self.backend}; "
                f"directory={directory}; force_cpu={force_cpu}; "
                f"initialization_ms={(time.monotonic() - initialization_started) * 1000:.1f}."
            )
        except Exception:
            self.close()
            raise

    def _clear_diagnostics(self) -> None:
        self._status_counts = [0, 0, 0]
        self._sdk_ms_sum = 0.0
        self._call_ms_sum = 0.0
        self._first_feed_at = None
        self._last_summary_at = None
        self._last_state = None
        self._last_transition_at = float("-inf")
        self._first_lock_frame = None
        self._notready_streak = 0
        self._recent_records = deque(maxlen=256)
        self._recent_frames = deque(maxlen=11)
        self._last_frame_sample_at = float("-inf")
        self._session_id = None
        self._session_records = []
        self._session_frames = []
        self._session_bytes = 0
        self._session_truncated = False
        self._session_records_truncated = False
        self._session_next_sample_at = None

    def _log_summary(self, reason: str, now: float) -> None:
        frames = sum(self._status_counts)
        if not frames:
            return
        elapsed = now - self._first_feed_at
        logger.info(
            f"[Lie Detector SDK] Summary reason={reason}; backend={self.backend}; "
            f"roi_wh={self._frame_size}; frames={frames}; "
            f"LOCKED={self._status_counts[0]} NOTREADY={self._status_counts[1]} "
            f"ERROR={self._status_counts[2]}; notready_streak={self._notready_streak}; "
            f"first_lock_frame={self._first_lock_frame}; elapsed_s={elapsed:.2f}; "
            f"avg_sdk_ms={self._sdk_ms_sum / frames:.1f}; "
            f"avg_call_ms={self._call_ms_sum / frames:.1f}."
        )
        self._last_summary_at = now

    def _record_diagnostics(
        self, status: int, out, width: int, height: int, stride: int,
        call_ms: float, now: float,
    ) -> None:
        if self._first_feed_at is None:
            self._first_feed_at = now - call_ms / 1000.0
            self._last_summary_at = self._first_feed_at
        self._status_counts[status if status in (0, 1) else 2] += 1
        self._sdk_ms_sum += self.last_ms
        self._call_ms_sum += call_ms
        self._notready_streak = self._notready_streak + 1 if status == 1 else 0
        frames = sum(self._status_counts)
        first_lock = status == 0 and self._first_lock_frame is None
        if first_lock:
            self._first_lock_frame = frames
        state = {0: "LOCKED", 1: "NOTREADY"}.get(status, "ERROR")
        if status == 0 and not (0 <= out[1] < width and 0 <= out[2] < height):
            state = "OUT_OF_BOUNDS"
        if first_lock or (
            state != self._last_state and now - self._last_transition_at >= 1.0
        ):
            message = (
                f"[Lie Detector SDK] State {self._last_state or 'INITIAL'}->{state}; "
                f"frame={frames}; status={status}; roi_wh=({width},{height}); "
                f"stride={stride}; fmt=BGR24; sdk_ms={self.last_ms}; "
                f"call_ms={call_ms:.1f}; elapsed_s={now - self._first_feed_at:.2f}; "
                f"out={list(out) if status == 0 else 'not valid'}."
            )
            if state in ("ERROR", "OUT_OF_BOUNDS"):
                logger.warning(message)
            else:
                logger.info(message)
            self._last_transition_at = now
            self._last_state = state
        if now - self._last_summary_at >= 5.0:
            self._log_summary("interval", now)

    def _record_recent_frame(self, frame, out, status, timestamp, completed_at, call_ms):
        valid = status == 0
        previous = self._recent_records[-1] if self._recent_records else None
        captured_at = float(timestamp) if timestamp is not None else completed_at
        record = {
            "frame": sum(self._status_counts),
            "captured_at": captured_at,
            "processed_at": completed_at,
            "source_gap_ms": (captured_at - previous["captured_at"]) * 1000 if previous else None,
            "status": status,
            "center_x": int(out[1]) if valid else None,
            "center_y": int(out[2]) if valid else None,
            "bbox_x": int(out[3]) if valid else None,
            "bbox_y": int(out[4]) if valid else None,
            "bbox_width": int(out[5]) if valid else None,
            "bbox_height": int(out[6]) if valid else None,
            "confidence": out[7] / 1000.0 if valid else 0.0,
            "sdk_ms": self.last_ms,
            "call_ms": call_ms,
            "frame_age_ms": (completed_at - captured_at) * 1000 if timestamp is not None else None,
            "backend": self.backend,
            "input_width": frame.shape[1],
            "input_height": frame.shape[0],
            "mouse_action": "no_callback",
            "requested_x": None,
            "requested_y": None,
            "mouse_call_ms": None,
            "action_completed_at": None,
            "action_frame_age_ms": None,
        }
        self._recent_records.append(record)
        cutoff = captured_at - 2.0
        while self._recent_records and self._recent_records[0]["captured_at"] < cutoff:
            self._recent_records.popleft()
        while self._recent_frames and self._recent_frames[0][1] < cutoff:
            self._recent_frames.popleft()
        sampled_frame = None
        if captured_at - self._last_frame_sample_at >= 0.2:
            sampled_frame = frame.copy()
            self._recent_frames.append((record["frame"], captured_at, sampled_frame))
            self._last_frame_sample_at = captured_at
        if self.session_recording:
            self._record_session_frame(frame, record, sampled_frame)
        return record

    def _record_session_frame(self, frame, record, sampled_frame):
        captured_at = record["captured_at"]
        if self._session_next_sample_at is None:
            self._session_next_sample_at = captured_at
            logger.info(
                f"[Lie Detector SDK] Session recording started; session={self._session_id}; "
                f"image_fps={self.session_image_fps}; max_image_memory_mb={self._session_max_bytes / 1048576:.1f}; "
                "saving after reset/stop; CSV records each SDK call."
            )
        if len(self._session_records) < 10000:
            self._session_records.append(record)
        elif not self._session_records_truncated:
            self._session_records_truncated = True
            logger.warning("[Lie Detector SDK] Session reached 10000 records; further session recording skipped.")
        if self._session_records_truncated or self._session_truncated:
            return
        if captured_at + 1e-9 < self._session_next_sample_at:
            return
        if self._session_bytes + frame.nbytes > self._session_max_bytes:
            self._session_truncated = True
            logger.warning(
                "[Lie Detector SDK] Session image memory limit reached; "
                "keeping existing samples and continuing CSV/SDK processing."
            )
            return
        if sampled_frame is None:
            sampled_frame = frame.copy()
        self._session_frames.append((record["frame"], sampled_frame))
        self._session_bytes += frame.nbytes
        period = 1.0 / self.session_image_fps
        start = self._session_records[0]["captured_at"]
        self._session_next_sample_at = start + (math.floor((captured_at - start) / period + 1e-9) + 1) * period

    def _save_session(self, reason):
        if not self._session_records:
            return
        metadata = {
            "session_id": self._session_id,
            "reason": reason,
            "image_fps": self.session_image_fps,
            "image_memory_bytes": self._session_bytes,
            "max_image_memory_bytes": self._session_max_bytes,
            "records": len(self._session_records),
            "sampled_frames": len(self._session_frames),
            "captured_start": self._session_records[0]["captured_at"],
            "captured_end": self._session_records[-1]["captured_at"],
            "backend": self.backend,
            "input_wh": list(self._frame_size),
            "truncated": self._session_truncated or self._session_records_truncated,
            "images_truncated": self._session_truncated,
            "records_truncated": self._session_records_truncated,
        }
        queued = save_session(self._session_records, self._session_frames, metadata)
        if queued is not None:
            logger.info(
                f"[Lie Detector SDK] Session save queued; directory=log/lie_sdk_sessions/{self._session_id}; "
                f"records={len(self._session_records)}; sampled_frames={len(self._session_frames)}."
            )
        self._session_records = []
        self._session_frames = []

    def record_mouse_action(self, timestamp, action, location, call_ms, completed_at):
        with self._lock:
            if not self._recent_records:
                return
            record = self._recent_records[-1]
            if timestamp is None or timestamp != record["captured_at"]:
                return
            record.update(
                mouse_action=action,
                requested_x=location[0] if location is not None else None,
                requested_y=location[1] if location is not None else None,
                mouse_call_ms=call_ms,
                action_completed_at=completed_at,
                action_frame_age_ms=(completed_at - timestamp) * 1000,
            )

    def _save_recent_trace(self, reason):
        if self._recent_records:
            first = self._recent_records[0]["frame"]
            frames = [(number, frame) for number, _timestamp, frame in self._recent_frames if number >= first]
            save_recent_trace(list(self._recent_records), frames, reason, self._session_id)

    def _bind_functions(self) -> None:
        integer = ctypes.c_int
        pointer = ctypes.c_void_p
        integers = ctypes.POINTER(integer)
        signatures = {
            "TT_Version": ([], integer),
            "TT_SetForceCpu": ([integer], integer),
            "TT_MachineCode": ([], ctypes.c_char_p),
            "TT_AuthError": ([], ctypes.c_char_p),
            "TTx_Create": ([ctypes.c_char_p], pointer),
            "TTx_LastCreateError": ([], ctypes.c_char_p),
            "TTx_Destroy": ([pointer], None),
            "TTx_Reset": ([pointer], integer),
            "TTx_Detect": (
                [pointer, pointer, integer, integer, integer, integer, integers],
                integer,
            ),
            "TTx_LastError": ([pointer], ctypes.c_char_p),
            "TTx_LastMs": ([pointer], integer),
            "TTx_Backend": ([pointer], ctypes.c_char_p),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self._lib, name)
            function.argtypes = arguments
            function.restype = result

    def update(
        self, frame_bgr: np.ndarray, timestamp: Optional[float] = None
    ) -> LieDetectorTrackingResult:
        if (
            not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.dtype != np.uint8
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
            or frame_bgr.size == 0
        ):
            raise ValueError("trtyolo SDK input must be a non-empty uint8 BGR image")
        frame = np.ascontiguousarray(frame_bgr)
        height, width = frame.shape[:2]
        with self._lock:
            if not self._handle:
                raise RuntimeError("trtyolo SDK tracker is closed")
            frame_size = (width, height)
            if self._frame_size is not None and self._frame_size != frame_size:
                logger.info(
                    f"[Lie Detector SDK] Input size changed: "
                    f"{self._frame_size}->{frame_size}; resetting tracker."
                )
                self._reset_locked()
            self._frame_size = frame_size
            out = (ctypes.c_int * 8)()
            if self._first_feed_at is None:
                self._session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8]
                logger.info(
                    f"[Lie Detector SDK] First input roi_wh=({width},{height}); "
                    f"stride={frame.strides[0]}; fmt=BGR24; backend={self.backend}; "
                    f"diagnostic_directory=log/lie_sdk_sessions/{self._session_id}."
                )
                save_input_snapshot(frame, self._session_id)
            call_started = time.monotonic()
            status = self._lib.TTx_Detect(
                self._handle,
                ctypes.c_void_p(frame.ctypes.data),
                width,
                height,
                frame.strides[0],
                0,
                out,
            )
            completed_at = time.monotonic()
            call_ms = (completed_at - call_started) * 1000.0
            self.last_ms = self._lib.TTx_LastMs(self._handle)
            first_lock = status == 0 and self._first_lock_frame is None
            self._record_diagnostics(
                status, out, width, height, frame.strides[0], call_ms, completed_at
            )
            record = self._record_recent_frame(frame, out, status, timestamp, completed_at, call_ms)
            if first_lock:
                save_first_lock_snapshot(frame, record, self._session_id)
            if status == 2:
                reason = _error_text(self._lib.TTx_LastError(self._handle))
                raise RuntimeError(
                    f"SDK status=2; frame={sum(self._status_counts)}; "
                    f"roi_wh=({width},{height}); stride={frame.strides[0]}; "
                    f"sdk_ms={self.last_ms}; error={reason}"
                )
            if status not in (0, 1):
                raise RuntimeError(f"trtyolo SDK returned invalid status {status}")

            locked = status == 0
            center = (float(out[1]), float(out[2])) if locked else None
            actionable = bool(
                locked
                and 0 <= out[1] < width
                and 0 <= out[2] < height
            )
            return LieDetectorTrackingResult(
                acquired=locked,
                target_id=None,
                center=center,
                radius=0.25 * (out[5] + out[6]) if locked else None,
                confidence=out[7] / 1000.0 if locked else 0.0,
                predicted_only=False,
                lost_frames=0,
                detections=[],
                tracks=[],
                actionable=actionable,
                position_actionable=actionable,
            )

    def _reset_locked(self) -> None:
        if self._lib.TTx_Reset(self._handle) == 2:
            raise RuntimeError(_error_text(self._lib.TTx_LastError(self._handle)))
        self._log_summary("reset", time.monotonic())
        self._save_recent_trace("reset")
        self._save_session("reset")
        self._clear_diagnostics()
        self._frame_size = None
        self.last_ms = 0

    def reset(self) -> None:
        with self._lock:
            if self._handle:
                self._reset_locked()

    def close(self) -> None:
        with self._lock:
            try:
                if self._handle:
                    self._log_summary("close", time.monotonic())
                    self._save_recent_trace("close")
                    self._save_session("close")
                    self._lib.TTx_Destroy(self._handle)
            finally:
                self._handle = None
                self._lib = None
                self._frame_size = None
                self._recent_records.clear()
                self._recent_frames.clear()
                self._session_records.clear()
                self._session_frames.clear()
                if self._dll_directory is not None:
                    self._dll_directory.close()
                    self._dll_directory = None
