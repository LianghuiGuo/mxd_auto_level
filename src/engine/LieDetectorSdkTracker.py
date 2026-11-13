from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

from src.engine.LieDetectorTracker import LieDetectorTrackingResult
from src.utils.logger import logger

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
    ) -> None:
        self._lock = threading.Lock()
        self._handle = None
        self._lib = None
        self._dll_directory = None
        self._frame_size = None
        self.last_ms = 0
        self.backend = ""
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

    def _save_first_lock_diagnostics(
        self, frame: np.ndarray, out, timestamp: Optional[float]
    ) -> None:
        output = Path("log")
        write_started = time.monotonic()
        try:
            output.mkdir(exist_ok=True)
            overlay = frame.copy()
            cx, cy, x, y, width, height = (int(value) for value in out[1:7])
            cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), 2)
            cv2.drawMarker(overlay, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(
                overlay,
                f"SDK LOCKED frame={self._first_lock_frame} "
                f"conf={out[7] / 1000.0:.3f} sdk_ms={self.last_ms}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )
            for name, image in (
                ("lie_sdk_first_lock_frame.png", frame),
                ("lie_sdk_first_lock_overlay.png", overlay),
            ):
                path = output / name
                if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise OSError(f"Could not write {path}")
            logger.info(
                "[Lie Detector SDK] First-lock snapshots saved to "
                "log/lie_sdk_first_lock_{frame,overlay}.png; "
                f"frame={self._first_lock_frame}; captured_at={timestamp}; "
                f"center=({cx},{cy}); bbox=({x},{y},{width},{height}); "
                "coordinates are SDK input pixels; "
                "green=SDK bbox, red=SDK center; "
                "the next challenge overwrites these files; "
                f"write_ms={(time.monotonic() - write_started) * 1000:.1f}."
            )
        except (OSError, cv2.error) as exc:
            logger.warning(f"[Lie Detector SDK] Could not save first-lock snapshots: {exc}")

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
                logger.info(
                    f"[Lie Detector SDK] First input roi_wh=({width},{height}); "
                    f"stride={frame.strides[0]}; fmt=BGR24; backend={self.backend}."
                )
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
            if first_lock:
                self._save_first_lock_diagnostics(frame, out, timestamp)
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
                    self._lib.TTx_Destroy(self._handle)
            finally:
                self._handle = None
                self._lib = None
                self._frame_size = None
                if self._dll_directory is not None:
                    self._dll_directory.close()
                    self._dll_directory = None
