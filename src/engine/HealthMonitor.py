'''
Health Monitor take game windows as input to calculate player's HP/MP/EXP Bar percentage.
When player's HP/MP drop to specific threshold, it'd press key to drink potion 
'''

# Standard Import
import threading
import time
import cv2
import numpy as np

# Local Import
from src.utils.logger import logger
from src.utils.common import get_bar_percent
from src.input.KeyBoardController import press_key

class HealthMonitor:
    '''
    Independent health monitoring thread that can heal while other actions are running
    '''
    def __init__(self, cfg, kb_controller):
        self.cfg = cfg
        self.kb = kb_controller
        self.is_terminated = False
        self.enabled = True
        self.thread = None # health monitor thread

        # Health monitoring state
        self.hp_percent = 100
        self.mp_percent = 100
        self.exp_percent = 100

        # Timers
        self.t_last_heal = 0
        self.t_last_mp = 0
        self.t_last_hp_reduce = 0
        self.t_last_run = 0
        self.t_hp_watch_dog = time.time()

        # Frame data (will be updated by main thread)
        self.img_frame = None
        self.frame_lock = threading.Lock()

        # FPS settings
        self.fps_limit = self.cfg["health_monitor"]["fps_limit"]
        self.fps = 0

        # Debug information
        # hp/mp/exp bars loc and size, [(x,y,w,h), ...]
        self.loc_size_bars = [(0, 0, 0, 0),
                              (0, 0, 0, 0),
                              (0, 0, 0, 0)]

        # --- Locked bar ROIs -------------------------------------------------
        # The dynamic contour finder is fragile: transient white UI (damage
        # numbers, popups, buff bars) can sneak into the 3-bar set and get
        # mistaken for the HP bar, pinning HP at a constant value -> endless
        # healing on full HP.  To make this robust we LOCK the bar rectangles
        # once we get a trustworthy detection (exactly 3 long-thin bars AND the
        # left-most one actually contains red = a real HP bar).  After that we
        # ALWAYS read those fixed ROIs and never re-run the noisy contour
        # search, so no transient UI can hijack the HP reading.
        self._bars_locked = None          # list[(x,y,w,h)] once locked
        self._lock_fail_streak = 0        # consecutive locked-read failures

        # Manual bar ROIs (highest priority). If provided, we skip auto-detect
        # entirely and always read these fixed rectangles -> fully stable.
        self._bars_manual = False
        mb = self.cfg["health_monitor"].get("manual_bars")
        if mb and len(mb) == 3:
            self._bars_locked = [tuple(int(v) for v in r) for r in mb]
            self._bars_manual = True
            logger.info(
                f"[Health Monitor] Using manual bar ROIs: {self._bars_locked}")
        self._debug_dump = bool(self.cfg["health_monitor"].get("debug_dump", False))
        self._dbg_dump_done = False

        logger.info("[Health Monitor] Init done")

    def start(self):
        '''
        Start health monitoring thread
        '''
        if not self.is_terminated:
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            logger.info("[Health Monitor] Started")

    def stop(self):
        '''
        Stop health monitoring thread
        '''
        self.is_terminated = True
        if self.thread:
            self.thread.join()
            logger.info("[Health Monitor] Terminated")

    def enable(self):
        '''
        Enable health monitoring
        '''
        self.enabled = True

    def disable(self):
        '''
        Disable health monitoring
        '''
        self.enabled = False

    def update_frame(self, img_frame):
        '''
        Update frame data from main thread
        '''
        with self.frame_lock:
            self.img_frame = img_frame

    def get_hp_mp_exp_percent(self):
        '''
        Extracts the player's HP, MP, and EXP ratios from game frame.

        This function:
        - Crops the predefined HP, MP, and EXP bar regions from the game frame.
        - Identifies empty areas in each bar.
        - Computes the fill ratio for each bar as: 1 - (empty_pixels / total_pixels).

        Returns:
            tuple: (hp_percent, mp_percent, exp_percent), each a float between 0 and 1.
        '''
        if self.img_frame is None:
            return None, None, None

        with self.frame_lock:
            img_frame = self.img_frame.copy()

        # --- Fast path: bars already locked -> read fixed ROIs ---------------
        # Once locked we TRUST the position and only read the fill ratio.  We do
        # NOT re-verify "is this red" here on purpose: at very low HP the bar is
        # almost empty (little/no red), and requiring red would make us stop
        # reading exactly when healing matters most.  The lock itself was gated
        # on a real red HP bar (see below), and the bar never moves, so the
        # position stays valid for the whole session.
        if self._bars_locked is not None:
            Hf, Wf = img_frame.shape[:2]
            percents = []
            ok = True
            for (x, y, w, h) in self._bars_locked:
                if x < 0 or y < 0 or x + w > Wf or y + h > Hf or w <= 0 or h <= 0:
                    ok = False
                    break
                percents.append(get_bar_percent(img_frame[y:y+h, x:x+w]))
            if ok:
                return percents
            # Geometry no longer fits the frame.
            if self._bars_manual:
                logger.warning(
                    "[Health Monitor] manual_bars ROI is outside the frame "
                    f"{(Wf, Hf)}; fix health_monitor.manual_bars.")
                return (None, None, None)
            logger.warning(
                "[Health Monitor] Locked bar ROI no longer fits the frame "
                "(resolution changed?); re-detecting bars.")
            self._bars_locked = None
            return (None, None, None)

        # --- Debug: dump the bottom status-bar strip for manual measuring ----
        # NOTE: img_frame here is ALREADY the bottom UI strip (img_frame[ui_y_start:]),
        # so manual_bars coordinates are relative to THIS strip (x from 0, y from 0),
        # NOT the full client frame.
        if self._debug_dump and not self._dbg_dump_done:
            try:
                import os
                Hf, Wf = img_frame.shape[:2]
                os.makedirs("log", exist_ok=True)
                path = os.path.join("log", "health_bottom_bar.png")
                cv2.imwrite(path, img_frame)
                self._dbg_dump_done = True
                logger.info(
                    f"[Health Monitor] debug_dump: wrote {path} "
                    f"(this IS the bottom UI strip, size={(Wf, Hf)}). "
                    "Measure HP/MP/EXP bars in THIS image; coordinates are "
                    "already in the correct system. Put three [x,y,w,h] into "
                    "health_monitor.manual_bars (order HP, MP, EXP).")
            except Exception as e:
                logger.warning(f"[Health Monitor] debug_dump failed: {e}")

        # --- Colour-based bar localisation -----------------------------------
        # The old white-border contour finder was unreliable on this client:
        # the wide white chat box / status panel produced fat white blobs that
        # got mistaken for the HP bar (observed roi=(269,29,542,20), all-white
        # sample -> red_ratio 0.000 -> never heals).  MapleStory's three status
        # bars are RED (HP), BLUE (MP) and a YELLOW/ORANGE (EXP) strip.  We find
        # them directly by colour: this cannot be fooled by white UI.
        bars = self._detect_bars_by_colour(img_frame)
        if bars is None:
            cls = type(self)
            now = time.time()
            if now - getattr(cls, "_bar_dbg_t", 0.0) >= 3.0:
                cls._bar_dbg_t = now
                Hs, Ws = img_frame.shape[:2]
                logger.warning(
                    f"[Health Monitor] Could not locate HP/MP/EXP bars by "
                    f"colour in bottom strip (size={(Ws, Hs)}). "
                    "Enable health_monitor.debug_dump to measure manual_bars.")
            return (None, None, None)

        # Lock the HP/MP/EXP ROIs for the rest of the session.
        self._bars_locked = list(bars)
        self.loc_size_bars = list(bars)
        logger.info(f"[Health Monitor] Locked bar ROIs (HP/MP/EXP): {bars}")

        percent_bars = []
        for x, y, w, h in bars:
            percent_bars.append(get_bar_percent(img_frame[y:y+h, x:x+w]))
        return percent_bars

    @staticmethod
    def _detect_bars_by_colour(strip):
        '''
        Locate the HP (red), MP (blue) and EXP (yellow) bars inside the bottom
        UI strip purely by colour, then return their [x,y,w,h] ROIs ordered
        HP, MP, EXP.  Returns None if all three could not be found confidently.

        This is robust against the white chat box / panels that fooled the old
        white-contour finder.
        '''
        import numpy as _np
        b = strip[:, :, 0].astype(_np.int16)
        g = strip[:, :, 1].astype(_np.int16)
        r = strip[:, :, 2].astype(_np.int16)

        red_m  = ((r > 90) & (r - g > 40) & (r - b > 40)).astype(_np.uint8)
        blue_m = ((b > 90) & (b - g > 30) & (b - r > 40)).astype(_np.uint8)
        # EXP bar is a thin yellow/orange strip: high R & G, low B.
        yel_m  = ((r > 120) & (g > 100) & (b < 100) & (r - b > 40)).astype(_np.uint8)

        def _largest_bar(mask):
            # Close small gaps horizontally so a partially-filled bar stays one
            # blob, then take the widest long-thin component.
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
            m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            best = None
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if w < 8 or h < 1:
                    continue
                ar = w / float(h)
                if ar < 2.0:            # bars are wide & short
                    continue
                if best is None or w > best[2]:
                    best = (int(x), int(y), int(w), int(h))
            return best

        # A coloured blob marks only the FILLED part of a bar.  get_bar_percent
        # needs the WHOLE bar (filled + empty grey slot, bounded by the white
        # end-caps) to compute a fill ratio.  So from the filled blob we expand
        # left/right along the blob's centre row until we hit the white border,
        # yielding the full bar ROI.
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        Hs, Ws = gray.shape[:2]

        def _expand_to_white(blob):
            if blob is None:
                return None
            x, y, w, h = blob
            cy = min(max(y + h // 2, 0), Hs - 1)
            row = gray[cy]
            # walk left from blob start until white cap (or strip edge)
            xl = x
            while xl > 0 and row[xl - 1] < 240:
                xl -= 1
            xr = x + w - 1
            while xr < Ws - 1 and row[xr + 1] < 240:
                xr += 1
            nw = xr - xl + 1
            if nw < w:              # expansion failed, keep original
                xl, nw = x, w
            return (int(xl), int(y), int(nw), int(h))

        hp = _expand_to_white(_largest_bar(red_m))
        mp = _expand_to_white(_largest_bar(blue_m))
        ex = _expand_to_white(_largest_bar(yel_m))

        # HP and MP are the critical ones (drive healing). EXP is optional; if
        # missing, synthesise a placeholder so downstream indexing is safe.
        if hp is None or mp is None:
            return None
        if ex is None:
            ex = (mp[0], mp[1], mp[2], mp[3])
        return [hp, mp, ex]

    @staticmethod
    def _looks_like_hp_bar(bar_img, min_red_ratio=0.02):
        '''
        Heuristic: a real MapleStory HP bar has a RED filled section.  Return
        True if ANY non-trivial amount of the bar is red-dominant (R clearly
        greater than G and B).

        The goal is ONLY to reject the fully grey/white UI elements the contour
        finder occasionally mistakes for the HP bar (which pinned HP at a
        constant value and caused endless healing).  It must NOT reject a real
        but LOW HP bar: at e.g. 10/20 HP only ~half the (already short) bar is
        red, and after subtracting borders / empty section the red fraction can
        be small — so the threshold is deliberately tiny (2%).  A genuine fake
        (grey popup / damage number) has essentially 0% red and is still
        rejected.
        '''
        if bar_img is None or bar_img.size == 0:
            return False
        b = bar_img[:, :, 0].astype(np.int16)
        g = bar_img[:, :, 1].astype(np.int16)
        r = bar_img[:, :, 2].astype(np.int16)
        # Slightly looser red test than before (darker reds like (195,10,42)
        # BGR still qualify) so dim HP bars aren't missed.
        red = (r > 90) & (r - g > 30) & (r - b > 30)
        return float(red.mean()) >= min_red_ratio

    def _monitor_loop(self):
        '''
        Main monitoring loop running in separate thread
        '''
        while not self.is_terminated:
            try:
                if not self.enabled:
                    self.limit_fps()
                    continue

                # Get current time
                t_cur = time.time()

                # Get current HP/MP ratios
                hp_percent, mp_percent, exp_percent = self.get_hp_mp_exp_percent()
                if hp_percent is not None:
                    # Check if HP bar has reduced
                    if self.hp_percent > hp_percent:
                        self.t_last_hp_reduce = t_cur
                    self.hp_percent = hp_percent
                if mp_percent is not None:
                    self.mp_percent = mp_percent
                if exp_percent is not None:
                    self.exp_percent = exp_percent

                hp_thres = self.cfg["health_monitor"]["add_hp_percent"]
                mp_thres = self.cfg["health_monitor"]["add_mp_percent"]
                hp_cd    = self.cfg["health_monitor"]["add_hp_cooldown"]
                mp_cd    = self.cfg["health_monitor"]["add_mp_cooldown"]
                watchdog_timeout = self.cfg["health_monitor"]["return_home_watch_dog_timeout"]

                # Check if need to heal (with cooldown)
                if self.cfg["health_monitor"]["force_heal"]:
                    # Ignore cooldown and force keycontroller to heal first
                    if self.hp_percent < hp_thres:
                        if not self.kb.is_need_force_heal:
                            logger.info(f"[Health Monitor] Force heal triggered, "
                                        f"HP: {self.hp_percent:.1f}%")
                        self.kb.is_need_force_heal = True
                    else:
                        self.kb.is_need_force_heal = False
                else:
                    if (self.hp_percent <= hp_thres and
                        t_cur - self.t_last_heal > hp_cd):
                        self._heal()
                        logger.info(f"[Health Monitor] Auto heal triggered, HP: {self.hp_percent:.1f}%")
                        self.t_last_heal = t_cur

                # Check if no HP potion and need to return home
                if self.cfg["health_monitor"]["return_home_if_no_potion"]:
                    if self.hp_percent >= hp_thres:
                        self.t_hp_watch_dog = t_cur # reset watchdog
                    else:
                        # If watchdog timeout, use homing scroll to return home
                        if t_cur - self.t_hp_watch_dog > watchdog_timeout:
                            logger.warning(f"[Health Monitor] HP({self.hp_percent:.1f}%) < {hp_thres:.1f}% "
                                           f"for {round(t_cur - self.t_hp_watch_dog, 2)} seconds.")
                            logger.warning(f"[Health Monitor] Return home because potion is used up.")
                            press_key(self.cfg["key"]["return_home"]) # Return home
                            self.is_terminated = True # Terminate Health monitor
                            self.kb.is_terminated = True # Terminate AutoBot

                # Check if need MP (with cooldown)
                if (self.mp_percent <= mp_thres and t_cur - self.t_last_mp > mp_cd):
                    self._add_mp()
                    self.t_last_mp = t_cur
                    logger.info(f"[Health Monitor] Auto MP triggered, MP: {self.mp_percent:.1f}%")

                # Sleep to avoid excessive CPU usage
                self.limit_fps()

            except Exception as e:
                logger.error(f"[Health Monitor] {e}")
                self.limit_fps()

    def _heal(self):
        '''
        Execute heal action
        '''
        try:
            press_key(self.cfg["key"]["add_hp"], 0.05)
        except Exception as e:
            logger.error(f"[Health Monitor] Heal action failed: {e}")

    def _add_mp(self):
        '''
        Execute MP recovery action
        '''
        try:
            press_key(self.cfg["key"]["add_mp"], 0.05)
        except Exception as e:
            logger.error(f"[Health Monitor] MP action failed: {e}")

    def limit_fps(self):
        '''
        Limit FPS
        '''
        # If the loop finished early, sleep to maintain target FPS
        target_duration = 1.0 / self.fps_limit  # seconds per frame
        frame_duration = time.time() - self.t_last_run
        if frame_duration < target_duration:
            time.sleep(target_duration - frame_duration)

        # Update FPS
        self.fps = round(1.0 / (time.time() - self.t_last_run))
        self.t_last_run = time.time()
        # logger.info(f"FPS = {self.fps}")
