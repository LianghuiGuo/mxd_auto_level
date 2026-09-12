"""Non-blocking local alerts for confirmed lie-detector challenges."""

from __future__ import annotations

from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

from src.utils.logger import logger


class LocalSoundAlert:
    """Play a best-effort warning sound without blocking the bot loop.

    Only one playback worker may run at a time.  The runtime emits one event
    per confirmed challenge as the primary de-duplication mechanism; the
    in-flight guard here also protects callers from accidental duplicate
    submissions.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        repeat: int = 2,
        interval_seconds: float = 0.18,
        sound_file: str = "",
        play_once: Optional[Callable[[], None]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        try:
            parsed_repeat = int(repeat)
        except (TypeError, ValueError):
            parsed_repeat = 2
        try:
            parsed_interval = float(interval_seconds)
        except (TypeError, ValueError):
            parsed_interval = 0.18
        self.repeat = max(1, min(10, parsed_repeat))
        self.interval_seconds = max(0.0, min(5.0, parsed_interval))
        self.sound_file = Path(sound_file).expanduser() if sound_file else None
        self._injected_player = play_once
        self._lock = threading.Lock()
        self._playing = False
        self._worker: Optional[threading.Thread] = None

    def notify(self) -> bool:
        """Schedule one alert sequence and return whether it was accepted."""
        if not self.enabled:
            return False
        with self._lock:
            if self._playing:
                return False
            self._playing = True
            self._worker = threading.Thread(
                target=self._run,
                name="lie-sound-alert",
                daemon=True,
            )
            self._worker.start()
        return True

    def wait(self, timeout: Optional[float] = None) -> None:
        """Wait for current playback; intended for tests and shutdown tools."""
        with self._lock:
            worker = self._worker
        if worker is not None:
            worker.join(timeout)

    def _run(self) -> None:
        try:
            for index in range(self.repeat):
                self._play_once()
                if index + 1 < self.repeat and self.interval_seconds > 0.0:
                    time.sleep(self.interval_seconds)
            logger.info("[Lie Detector] Local sound alert played.")
        except Exception as exc:
            # Alert failures must never interrupt tracking or mouse control.
            logger.warning(f"[Lie Detector] Local sound alert failed: {exc}")
        finally:
            with self._lock:
                self._playing = False

    def _play_once(self) -> None:
        if self._injected_player is not None:
            self._injected_player()
            return

        system = platform.system()
        custom = self.sound_file
        if custom is not None and not custom.is_file():
            logger.warning(
                f"[Lie Detector] Alert sound file not found: {custom}; "
                "using the system warning sound."
            )
            custom = None

        if system == "Windows":
            import winsound

            if custom is not None:
                winsound.PlaySound(
                    str(custom),
                    winsound.SND_FILENAME | winsound.SND_NODEFAULT,
                )
            else:
                winsound.Beep(1050, 170)
                winsound.Beep(1450, 260)
            return

        if system == "Darwin":
            sound = custom or Path("/System/Library/Sounds/Sosumi.aiff")
            player = shutil.which("afplay")
            if player is not None and sound.is_file():
                subprocess.run(
                    [player, str(sound)],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15.0,
                )
                return

        if system == "Linux":
            if custom is not None:
                for executable in ("paplay", "aplay"):
                    player = shutil.which(executable)
                    if player is not None:
                        subprocess.run(
                            [player, str(custom)],
                            check=False,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=15.0,
                        )
                        return
            canberra = shutil.which("canberra-gtk-play")
            if canberra is not None:
                subprocess.run(
                    [canberra, "--id", "dialog-warning"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15.0,
                )
                return

        # Last-resort terminal bell for unsupported/headless environments.
        sys.stderr.write("\a")
        sys.stderr.flush()
