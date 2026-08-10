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

    def _current_progress_pos(self):
        '''
        Return a scalar-ish reference for "how far along the route" the player
        is, preferring the minimap dot (absolute map coords, unaffected by the
        camera following the character) and falling back to the on-screen x.

        Returns a tuple (kind, value) where kind is "mm" or "screen" so we only
        compare like-with-like (their pixel scales differ a lot).
        '''
        mm = getattr(self.bot, "loc_player_minimap", None)
        if mm is not None and mm != (0, 0):
            return ("mm", (float(mm[0]), float(mm[1])))
        # fallback: on-screen x (camera-follow makes this less reliable but it's
        # better than nothing when the minimap dot isn't found).
        lp = getattr(self.bot, "loc_player", None) or (0, 0)
        return ("screen", (float(lp[0]), float(lp[1])))

    def _maybe_auto_jump(self, walk_dir):
        '''
        Decide whether the character is blocked by terrain while walking
        ``walk_dir`` and, if so, set cmd_action="jump".  Returns True when a
        jump was issued this frame.

        Trigger: the player keeps pushing the SAME direction but its real
        position (minimap dot when available, else screen x) moved less than
        cfg.patrol.jump_stuck_dist over cfg.patrol.jump_stuck_sec.
        '''
        cfg = self.bot.cfg["patrol"]
        if not cfg.get("jump_enable", True):
            return False
        if walk_dir not in ("left", "right"):
            return False

        now = time.time()
        kind, pos = self._current_progress_pos()

        # Reset the reference whenever the walk direction changed or we don't
        # have one yet — we only measure progress within a single direction.
        if self._jump_ref_dir != walk_dir or self._jump_ref_pos is None:
            self._jump_ref_dir = walk_dir
            self._jump_ref_pos = (kind, pos)
            self._jump_ref_time = now
            return False

        ref_kind, ref_pos = self._jump_ref_pos
        # If the position source changed (minimap dot appeared/disappeared),
        # re-baseline instead of comparing incomparable scales.
        if ref_kind != kind:
            self._jump_ref_pos = (kind, pos)
            self._jump_ref_time = now
            return False

        moved = abs(pos[0] - ref_pos[0]) + abs(pos[1] - ref_pos[1])
        # Minimap dots move only a few px per step, but the on-screen x can jump
        # tens of px as the character walks, so use a larger "not moving"
        # threshold for the screen fallback to avoid false auto-jumps while the
        # character is actually walking fine.
        stuck_dist = float(cfg.get("jump_stuck_dist", 3))
        if kind == "screen":
            stuck_dist = float(cfg.get("jump_stuck_dist_screen", 12))

        # Real progress -> re-baseline and don't jump.
        if moved > stuck_dist:
            self._jump_ref_pos = (kind, pos)
            self._jump_ref_time = now
            return False

        # Not moving yet, but hasn't been long enough — keep waiting.
        if now - self._jump_ref_time < float(cfg.get("jump_stuck_sec", 0.8)):
            return False

        # Blocked long enough: jump (respecting cooldown).
        if now - self._jump_last_time < float(cfg.get("jump_cooldown", 0.7)):
            return False

        self.bot.cmd_action = "jump"
        self._jump_last_time = now
        # Re-baseline so we measure a fresh window after the hop.
        self._jump_ref_pos = (kind, pos)
        self._jump_ref_time = now
        try:
            from src.utils.logger import logger
            logger.info(
                f"[Patrol] auto-jump: blocked walking {walk_dir} "
                f"(moved={moved:.1f} <= {stuck_dist} over "
                f"{cfg.get('jump_stuck_sec', 0.8)}s, src={kind})."
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
            jumped = self._maybe_auto_jump(self.bot.cmd_move_x)

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
