import time

from src.input.KeyBoardController import press_key
from src.states.base_state import State
from src.utils.logger import logger


class RestingState(State):
    """Sit on a chair for a configured duration without blocking the engine."""

    def on_enter(self):
        cfg = self.bot.cfg.get("chair_rest", {})
        duration_minutes = float(cfg.get("duration_minutes", 5.0))
        chair_key = str(cfg.get("key", "c") or "c").strip().lower()

        # Suppress the keyboard-controller loop before releasing keys, so a
        # concurrent buff/attack iteration cannot immediately stand us up.
        self.bot.kb.automation_suspended = True
        self.bot.kb.set_command("none none none")
        self.bot.kb.release_all_key()

        self.bot.cmd_move_x = "none"
        self.bot.cmd_move_y = "none"
        self.bot.cmd_action = "none"
        self.bot._chair_rest_pending = False
        self.bot._chair_rest_rounds = 0
        self.bot._chair_rest_until = time.monotonic() + duration_minutes * 60.0
        self._last_countdown_log = 0.0

        # A chair key commonly toggles sitting, so it must be sent exactly
        # once. The timer is checked frame-by-frame; the engine never sleeps
        # for the whole rest duration and remains responsive to shutdown.
        if not self.bot.is_disable_control:
            press_key(chair_key, 0.05)
            logger.info(
                f"[Chair Rest] Started: pressed key={chair_key!r}; "
                f"resting for {duration_minutes:g} minute(s)."
            )
        else:
            logger.info(
                f"[Chair Rest] Started in control-disabled mode; would press "
                f"key={chair_key!r} and rest for {duration_minutes:g} minute(s)."
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
        logger.info("[Chair Rest] Finished; resuming hunting.")

    def check_transitions(self):
        if time.monotonic() >= self.bot._chair_rest_until:
            return "hunting"
        return None

    def on_frame(self):
        # Reassert an idle command in case another subsystem wrote a command
        # between frames. KeyboardController remains suspended as the primary
        # protection against movement, attacks, force-heal actions and buffs.
        self.bot.kb.set_command("none none none")
        self.bot.cmd_move_x = "none"
        self.bot.cmd_move_y = "none"
        self.bot.cmd_action = "none"

        now = time.monotonic()
        last_log = getattr(self, "_last_countdown_log", 0.0)
        if now - last_log >= 30.0:
            self._last_countdown_log = now
            remaining = max(0.0, self.bot._chair_rest_until - now)
            logger.info(
                f"[Chair Rest] Remaining: {remaining / 60.0:.1f} minute(s)."
            )
