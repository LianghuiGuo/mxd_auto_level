import time

# Local import
from src.states.base_state import State

class PatrolState(State):
    def __init__(self, name, bot):
        super().__init__(name, bot)
        self.bot = bot
        self.is_patrol_to_left = True # Patrol direction flag
        self.patrol_turn_point_cnt = 0 # Patrol tuning back counter

    def on_enter(self):
        pass

    def on_exit(self):
        pass

    def check_transitions(self):
        return None

    def on_frame(self):
        # 1 Hz heartbeat
        try:
            from src.utils.logger import logger
            _now = time.time()
            _last = getattr(type(self), "_last_on_frame_log", -999.0)
            if _now - _last >= 1.0:
                type(self)._last_on_frame_log = _now
                logger.info(
                    "[FSM] PatrolState.on_frame fired. "
                    f"loc_player={self.bot.loc_player} "
                    f"cmd=({self.bot.cmd_move_x},{self.bot.cmd_move_y},"
                    f"{self.bot.cmd_action})"
                )
        except Exception:
            pass

        # Clear any stale action from the previous frame FIRST (before mob
        # detection / periodic-attack decide this frame's action).  Otherwise
        # cmd_action latches to "attack" forever and the KeyBoardController
        # loop fires the attack key every frame — bypassing
        # patrol_attack_interval and attack-locking the character in place so
        # it never walks.  Mob detection or the interval check below will
        # re-raise "attack" this frame when appropriate.
        if self.bot.cmd_action == "attack":
            self.bot.cmd_action = "none"

        x, y = self.bot.loc_player
        h, w = self.bot.img_frame.shape[:2]
        loc_player_ratio = float(x)/float(w)
        left_ratio, right_ratio = self.bot.cfg["patrol"]["range"]

        # Check if we need to change patrol direction
        if self.is_patrol_to_left and loc_player_ratio < left_ratio:
            self.patrol_turn_point_cnt += 1
        elif (not self.is_patrol_to_left) and loc_player_ratio > right_ratio:
            self.patrol_turn_point_cnt += 1

        if self.patrol_turn_point_cnt > self.bot.cfg["patrol"]["turn_point_thres"]:
            self.is_patrol_to_left = not self.is_patrol_to_left
            self.patrol_turn_point_cnt = 0

        # Decide the patrol walk direction for THIS frame (used only when there
        # is no monster to attack).
        patrol_dir = "left" if self.is_patrol_to_left else "right"

        # Run monster detection FIRST (if it hasn't already been hoisted into
        # run_once this frame).  It sets cmd_action="attack" and points
        # cmd_move_x at the monster whenever one is inside the attack box.
        if not getattr(self.bot, "_mob_detection_ran_this_frame", False):
            self.bot.update_cmd_by_mob_detection()
            self.bot._mob_detection_ran_this_frame = True

        # Attacks are driven purely by monster detection.  The old
        # fixed-interval "blind attack" (patrol_attack_interval) has been
        # removed so the character only attacks when a monster is detected.
        #
        # CRITICAL: only overwrite cmd_move_x with the patrol walk direction
        # when there is NO attackable monster this frame.  Otherwise we would
        # clobber the attack-facing direction that update_cmd_by_mob_detection
        # just set (mob on the left, patrol walking right), which made the
        # character jitter its facing every frame and effectively never land an
        # attack.
        if not getattr(self.bot, "_has_attackable_target", False):
            self.bot.cmd_move_x = patrol_dir

        # If player stuck for too long, perform a random command — but never
        # override a real attack we just decided on from mob detection.
        if not getattr(self.bot, "_has_attackable_target", False) \
                and self.bot.is_player_stuck():
            self.bot.update_cmd_by_random()

        # send command to keyboard controller
        self.bot.kb.set_command(self.bot.cmd_move_x + ' ' + \
                                self.bot.cmd_move_y + ' ' + \
                                self.bot.cmd_action)
