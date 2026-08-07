import time
from src.states.base_state import State

class HuntingState(State):
    def on_enter(self):
        # Heartbeat diagnostic: tell the user we've entered hunting so they
        # can immediately see the FSM is alive even if on_frame raises
        # before logging.
        try:
            from src.utils.logger import logger
            logger.info("[FSM] HuntingState.on_enter")
        except Exception:
            pass

    def on_exit(self):
        pass

    def check_transitions(self):
        if self.bot.rune_solver.is_rune_enable(
            self.bot.img_frame_gray, self.bot.img_frame_debug) or \
            self.bot.rune_solver.is_rune_warning(
            self.bot.img_frame_gray, self.bot.img_frame_debug):
            # When "Rune enable" message appears on screen
            self.bot.screenshot_img_frame()

            return "finding_rune"

        else:
            return None

    def on_frame(self):
        # 1 Hz heartbeat so the log confirms "FSM is still running hunting"
        # even when all commands are (none,none,none) → keys=<idle>.  Users
        # were inferring "FSM dead" purely from the heartbeat "keys=<idle>"
        # emitted by KeyBoardController.
        try:
            from src.utils.logger import logger
            _now = time.time()
            _last = getattr(type(self), "_last_on_frame_log", -999.0)
            if _now - _last >= 1.0:
                type(self)._last_on_frame_log = _now
                logger.info(
                    "[FSM] HuntingState.on_frame fired. "
                    f"loc_player={getattr(self.bot,'loc_player',None)} "
                    f"cmd=({self.bot.cmd_move_x},{self.bot.cmd_move_y},"
                    f"{self.bot.cmd_action})"
                )
        except Exception:
            pass

        # Get commend from route map
        self.bot.update_cmd_by_route()

        # Check if reach goal on route map
        self.bot.check_reach_goal()

        # -----------------------------------------------------------------
        # Mob detection + attack synth: HOISTED to MapleStoryAutoBot.run_once
        # (see comment there).  We guard the state-local copy with a flag so
        # users who fork the state object or run outside of run_once don't
        # silently skip attack.  run_once sets bot._mob_detection_ran_this_frame
        # to True after calling update_cmd_by_mob_detection; here we call it
        # only if that flag is missing (e.g. legacy caller).
        # -----------------------------------------------------------------
        if not getattr(self.bot, "_mob_detection_ran_this_frame", False):
            # Get attack commend by detecting mobs near players
            self.bot.update_cmd_by_mob_detection()

        # If player stuck for too long, perform a random command
        if self.bot.is_player_stuck():
            self.bot.update_cmd_by_random()

        # send command to keyboard controller
        self.bot.kb.set_command(self.bot.cmd_move_x + ' ' + \
                                self.bot.cmd_move_y + ' ' + \
                                self.bot.cmd_action)
