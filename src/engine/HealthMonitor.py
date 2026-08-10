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

        img_frame_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
        white_mask = cv2.inRange(img_frame_gray, 240, 255)
        # cv2.imshow("white_mask", white_mask)

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # NOTE: the caller (MapleStoryAutoLevelUp.run_once) already crops the
        # frame to the bottom UI band (img_frame[ui_y_start:, :]) before
        # calling update_frame(), so ``img_frame`` here is ALREADY just the
        # short HP/MP/EXP strip (e.g. 1296x90), NOT the full client.  Earlier
        # this filter assumed a full-height frame and demanded the bar sit in
        # the bottom 80% / be < 6% of height — both wrong for a 90px strip, so
        # nothing matched.  Here we only apply the SHAPE test (long & thin) and
        # a width test relative to the strip width; we do NOT constrain y or an
        # absolute height, because within the strip the bars can be anywhere
        # and are a large fraction of the strip's height.
        H_full, W_full = img_frame_gray.shape[:2]
        min_w = int(W_full * 0.03)           # each bar spans a few % of width
        max_w = int(W_full * 0.45)
        min_h = 2
        max_h = H_full                        # strip is short; don't cap height

        loc_size_bars = []
        cand_dbg = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h <= 0:
                continue
            ar = w / float(h)
            # long-and-thin AND width within the relative range
            if (3.0 < ar < 30.0 and
                    min_w <= w <= max_w and
                    min_h <= h <= max_h):
                loc_size_bars.append((x, y, w, h))
                cand_dbg.append((x, y, w, h, round(ar, 1)))

        # sort contours by x coordinate
        loc_size_bars = sorted(loc_size_bars, key=lambda bar: bar[0])

        # One-shot diagnostic so the user can see WHY bars were / weren't
        # found (prints the candidate rectangles the shape filter accepted).
        try:
            cls = type(self)
            if not getattr(cls, "_bar_dbg_done", False) or \
                    len(loc_size_bars) != 3:
                now = time.time()
                if now - getattr(cls, "_bar_dbg_t", 0.0) >= 3.0:
                    cls._bar_dbg_t = now
                    cls._bar_dbg_done = True
                    logger.info(
                        f"[Health Monitor] bar-detect: frame_wh="
                        f"{(W_full, H_full)} bottom_band_y>={y_bottom_band} "
                        f"w:[{min_w},{max_w}] h:[{min_h},{max_h}] "
                        f"found={len(loc_size_bars)} candidates={cand_dbg}"
                    )
        except Exception:
            pass

        if len(loc_size_bars) != 3:
            return (None, None, None)

        # Identity check on the HP bar (the left-most of the three).  The
        # contour finder can occasionally lock onto a NON-HP white-bordered UI
        # element (damage numbers, quest/notice popups, buff bars, …) that
        # sneaks into the 3-bar set.  Such a fake "HP bar" is mostly grey/white
        # inside, so get_bar_percent reports a constant low value (observed:
        # HP stuck at 15.9%), and the bot then drinks potions forever on full
        # HP.  A REAL HP bar's filled portion is red (R clearly > G and > B).
        # If the left-most bar has almost no red fill, treat this frame's read
        # as invalid (return None) so the monitor keeps its last good value and
        # does NOT heal.
        hp_x, hp_y, hp_w, hp_h = loc_size_bars[0]
        hp_img = img_frame[hp_y:hp_y + hp_h, hp_x:hp_x + hp_w]
        if not self._looks_like_hp_bar(hp_img):
            if not getattr(type(self), "_hp_fake_warned", False):
                type(self)._hp_fake_warned = True
                logger.warning(
                    "[Health Monitor] Left-most detected bar does not look "
                    "like an HP bar (little/no red fill).  Ignoring this HP "
                    "read to avoid drinking potions on full HP.  If this "
                    "repeats, the HP bar ROI / detection needs calibration."
                )
            return (None, None, None)

        # Update loc_size_bars
        self.loc_size_bars = loc_size_bars

        # Get bar filled ratio
        percent_bars = []
        for x, y, w, h in loc_size_bars:
            percent_bars.append(get_bar_percent(img_frame[y:y+h, x:x+w]))
        return percent_bars

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
