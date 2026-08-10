import time

# Local import
from src.states.base_state import State

class PatrolState(State):
    def __init__(self, name, bot):
        super().__init__(name, bot)
        self.bot = bot
        self.is_patrol_to_left = True # Patrol direction flag
        self.patrol_turn_point_cnt = 0 # Patrol tuning back counter

        # --- auto-jump-when-stuck bookkeeping ---
        # We detect "pushing a walk direction but not actually advancing" by
        # sampling a reference position at the moment we START walking a given
        # direction and checking how far it has moved after jump_stuck_sec.
        self._jump_ref_pos = None       # reference position (minimap dot or screen x tuple)
        self._jump_ref_time = 0.0       # when the reference was taken
        self._jump_ref_dir = None       # which walk direction the reference belongs to
        self._jump_last_time = 0.0      # last time we issued an auto-jump (cooldown)

    def _maybe_auto_jump(self, walk_dir):
        '''
        Decide whether the character is blocked by terrain while patrol-walking
        ``walk_dir`` and, if so, set cmd_action="jump".  Returns True when a
        jump was issued this frame.

        Progress is measured purely on the on-screen player x (self.bot
        .loc_player).  The camera follows the character so a freely-walking
        character's x still wanders as it moves between screen thirds, whereas a
        character wedged in a wall/corner stays pinned to nearly the same x — so
        "x barely changed while we kept pushing the same direction for
        jump_stuck_sec" is a reliable blocked signal.

        NOTE: we deliberately do NOT use loc_player_minimap here — in YOLO
        patrol mode it is not refreshed and stays constant, which made moved==0
        every frame and spam-jumped forever (the bug this replaces).
        '''
        cfg = self.bot.cfg["patrol"]
        if not cfg.get("jump_enable", True):
            return False
        if walk_dir not in ("left", "right"):
            self._jump_ref_dir = None
            return False

        now = time.time()
        lp = getattr(self.bot, "loc_player", None) or (0, 0)
        x = float(lp[0])

        # (Re)baseline whenever the walk direction changed or we don't have a
        # reference yet — progress is only meaningful within one direction.
        if self._jump_ref_dir != walk_dir or self._jump_ref_pos is None:
            self._jump_ref_dir = walk_dir
            self._jump_ref_pos = x
            self._jump_ref_time = now
            return False

        moved = abs(x - self._jump_ref_pos)
        stuck_dist = float(cfg.get("jump_stuck_dist_screen", 25))

        # Real progress -> re-baseline and don't jump.
        if moved > stuck_dist:
            self._jump_ref_pos = x
            self._jump_ref_time = now
            return False

        # Not stuck long enough yet — keep waiting.
        if now - self._jump_ref_time < float(cfg.get("jump_stuck_sec", 1.2)):
            return False

        # Respect the jump cooldown so we don't spam ALT every frame.
        if now - self._jump_last_time < float(cfg.get("jump_cooldown", 1.0)):
            return False

        self.bot.cmd_action = "jump"
        self._jump_last_time = now
        # Re-baseline so we measure a fresh window after the hop.
        self._jump_ref_pos = x
        self._jump_ref_time = now
        try:
            from src.utils.logger import logger
            logger.info(
                f"[Patrol] auto-jump: blocked walking {walk_dir} "
                f"(x moved={moved:.1f} <= {stuck_dist} over "
                f"{cfg.get('jump_stuck_sec', 1.2)}s)."
            )
        except Exception:
            pass
        return True

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

        # Auto-jump over terrain: if we're walking (no monster to attack) but
        # the character isn't actually advancing, hop to get over the step /
        # ledge blocking the route.  Runs only while patrol-walking so it never
        # clobbers an attack.
        jumped = False
        if not getattr(self.bot, "_has_attackable_target", False):
            jumped = self._maybe_auto_jump(patrol_dir)
        else:
            # Attacking: the character SHOULD stand still to hit the mob, so its
            # x not moving is expected — reset the progress baseline so we don't
            # instantly auto-jump the moment the mob dies and we resume walking.
            self._jump_ref_dir = None

        # If player stuck for too long, perform a random command — but never
        # override a real attack we just decided on from mob detection, and not
        # in the same frame we already issued an auto-jump.
        if not jumped \
                and not getattr(self.bot, "_has_attackable_target", False) \
                and self.bot.is_player_stuck():
            self.bot.update_cmd_by_random()

        # send command to keyboard controller
        self.bot.kb.set_command(self.bot.cmd_move_x + ' ' + \
                                self.bot.cmd_move_y + ' ' + \
                                self.bot.cmd_action)
