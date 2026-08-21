import time

from src.input.KeyBoardController import get_last_backend_info, press_key
from src.states.base_state import State
from src.utils.logger import logger


class RestingState(State):
    """Stop completely, sit with safe retries, then rest without blocking."""

    _SUSPEND_ACK_TIMEOUT_SECONDS = 2.0

    def on_enter(self):
        cfg = self.bot.cfg.get("chair_rest", {})
        self._duration_minutes = float(cfg.get("duration_minutes", 5.0))
        self._chair_key = str(cfg.get("key", "c") or "c").strip().lower()
        self._settle_seconds = float(cfg.get("settle_seconds", 1.0))
        self._sit_retry_count = int(cfg.get("sit_retry_count", 3))
        self._sit_retry_interval = float(cfg.get("sit_retry_interval", 0.5))

        # Clear the previous acknowledgement before publishing the suspend
        # request. The keyboard thread sets it after it has observed the
        # request and released any action that was already in flight.
        self.bot.kb._automation_suspend_keys_released = False
        self.bot.kb.automation_suspended = True
        self.bot.kb.set_command("none none none")
        self.bot.kb.release_all_key()

        self.bot.cmd_move_x = "none"
        self.bot.cmd_move_y = "none"
        self.bot.cmd_action = "none"
        self.bot._chair_rest_pending = False
        self.bot._chair_rest_rounds = 0
        self.bot._chair_rest_until = 0.0

        now = time.monotonic()
        self._phase = "waiting_for_suspend"
        self._suspend_requested_at = now
        self._settle_until = 0.0
        self._next_sit_attempt_at = 0.0
        self._sit_attempts = 0
        self._last_countdown_log = 0.0
        logger.info(
            "[Chair Rest] Stopping controls; waiting for the keyboard "
            "thread to confirm all held actions are released."
        )

    def on_exit(self):
        self.bot.kb.set_command("none none none")
        self.bot.kb.release_all_key()
        self.bot.kb.automation_suspended = False

        # Resting intentionally produces no attacks or movement. Reset both
        # watchdog clocks so resuming cannot immediately trigger an unstuck
        # action, channel change, or return-home action.
        now = time.time()
        self.bot.t_watch_dog = now
        self.bot.t_last_attack = now
        self.bot._chair_rest_until = 0.0
        self._phase = "finished"
        logger.info("[Chair Rest] Finished; resuming hunting.")

    def check_transitions(self):
        if self._phase == "resting" and \
                time.monotonic() >= self.bot._chair_rest_until:
            return "hunting"
        return None

    def _press_chair_once(self, now):
        self._sit_attempts += 1
        if not self.bot.is_disable_control:
            press_key(self._chair_key, 0.05)
            backend, _, failures = get_last_backend_info()
            logger.info(
                f"[Chair Rest] Chair-key attempt "
                f"{self._sit_attempts}/{self._sit_retry_count}: "
                f"key={self._chair_key!r} backend={backend} "
                f"consecutive_failures={failures}."
            )
        else:
            logger.info(
                f"[Chair Rest] Control-disabled chair-key attempt "
                f"{self._sit_attempts}/{self._sit_retry_count}: "
                f"would press key={self._chair_key!r}."
            )

        if self._sit_attempts >= self._sit_retry_count:
            self._phase = "resting"
            self.bot._chair_rest_until = (
                time.monotonic() + self._duration_minutes * 60.0
            )
            self._last_countdown_log = 0.0
            logger.info(
                f"[Chair Rest] Sitting attempts complete; resting for "
                f"{self._duration_minutes:g} minute(s)."
            )
        else:
            self._phase = "pressing_chair"
            self._next_sit_attempt_at = now + self._sit_retry_interval

    def on_frame(self):
        # Reassert an idle command in case another subsystem wrote a command
        # between frames. KeyboardController remains suspended as the primary
        # protection against movement, attacks, force-heal actions and buffs.
        self.bot.kb.set_command("none none none")
        self.bot.cmd_move_x = "none"
        self.bot.cmd_move_y = "none"
        self.bot.cmd_action = "none"

        now = time.monotonic()
        if self._phase == "waiting_for_suspend":
            acknowledged = bool(
                self.bot.kb._automation_suspend_keys_released
            )
            timed_out = (
                now - self._suspend_requested_at
                >= self._SUSPEND_ACK_TIMEOUT_SECONDS
            )
            if acknowledged or self.bot.is_disable_control or timed_out:
                if timed_out and not acknowledged:
                    logger.warning(
                        "[Chair Rest] Keyboard suspend acknowledgement timed "
                        "out; proceeding after the safety settle delay."
                    )
                self._phase = "settling"
                self._settle_until = now + self._settle_seconds
                logger.info(
                    f"[Chair Rest] Controls released; waiting "
                    f"{self._settle_seconds:g}s for the character to stop "
                    "and finish movement/landing animations."
                )
            return

        if self._phase == "settling":
            if now >= self._settle_until:
                self._press_chair_once(now)
            return

        if self._phase == "pressing_chair":
            if now >= self._next_sit_attempt_at:
                self._press_chair_once(now)
            return

        if self._phase == "resting":
            if now - self._last_countdown_log >= 30.0:
                self._last_countdown_log = now
                remaining = max(0.0, self.bot._chair_rest_until - now)
                logger.info(
                    f"[Chair Rest] Remaining: "
                    f"{remaining / 60.0:.1f} minute(s)."
                )
