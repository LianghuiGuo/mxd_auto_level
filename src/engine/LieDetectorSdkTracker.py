from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys
import threading
from typing import Optional

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
                f"directory={directory}."
            )
        except Exception:
            self.close()
            raise

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
                self._reset_locked()
            self._frame_size = frame_size
            out = (ctypes.c_int * 8)()
            status = self._lib.TTx_Detect(
                self._handle,
                ctypes.c_void_p(frame.ctypes.data),
                width,
                height,
                frame.strides[0],
                0,
                out,
            )
            self.last_ms = self._lib.TTx_LastMs(self._handle)
            if status == 2:
                raise RuntimeError(_error_text(self._lib.TTx_LastError(self._handle)))
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
                    self._lib.TTx_Destroy(self._handle)
            finally:
                self._handle = None
                self._lib = None
                self._frame_size = None
                if self._dll_directory is not None:
                    self._dll_directory.close()
                    self._dll_directory = None
