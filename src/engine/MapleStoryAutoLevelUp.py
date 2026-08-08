'''
Execute this script:
python mapleStoryAutoLevelUp.py --map cloud_balcony --monster brown_windup_bear,pink_windup_bear
'''
# Standard import
import time
import random
import argparse
import glob
import sys
import logging
import os
import datetime
import threading

# Library import
import numpy as np
import cv2
import yaml

# Local import
from src.utils.global_var import WINDOW_WORKING_SIZE
from src.utils.logger import logger
from src.utils.common import (find_pattern_sqdiff, draw_rectangle, screenshot, nms,
    load_image, get_mask, get_minimap_loc_size, get_player_location_on_minimap,
    is_mac, override_cfg, load_yaml, get_all_other_player_locations_on_minimap,
    click_in_game_window, mask_route_colors, to_opencv_hsv, debug_minimap_colors,
    activate_game_window, is_img_16_to_9, normalize_pixel_coordinate, resize_window
)
from src.input.KeyBoardController import KeyBoardController, press_key
from src.input.KeyBoardListener import KeyBoardListener
if is_mac():
    from src.input.GameWindowCapturorForMac import GameWindowCapturor
else:
    from src.input.GameWindowCapturor import GameWindowCapturor
from src.engine.HealthMonitor import HealthMonitor
from src.engine.Profiler import Profiler
from src.engine.RuneSolver import RuneSolver
from src.engine.FiniteStateMachine import FiniteStateMachine
from src.states.hunting import HuntingState
from src.states.finding_rune import FindingRuneState
from src.states.near_rune import NearRuneState
from src.states.solving_rune import SolvingRuneState
from src.states.auxiliary import AuxiliaryState
from src.states.patrol import PatrolState

class MapleStoryAutoBot:
    '''
    MapleStoryAutoBot
    '''
    def __init__(self, args):
        '''
        Init MapleStoryAutoBot
        '''
        self.args = args # User args
        self.cfg = None # Configuration
        self.idx_routes = 0 # Index of route map
        self.monsters_info = {} # monster information
        self.monsters = [] # monster detected in current frame
        self.fps = 0 # Frame per second
        self.red_dot_center_prev = None # previous other player location in minimap
        self.video_writer = None # For video recording feature
        self.color_code = {} # For color code instruction
        self.color_code_up_down = {} # Color code only contain 'up' and 'down'
        # Party-red-bar anti-false-positive (FP) guards -------------------------------------------------
        # Non-Artale / CN clients typically don't render the little above-character
        # "party bar" that Stage 1 relies on, so the relaxed HSV+contour filter
        # can lock on to STATIC UI red bars (top-right character panel HP bar,
        # bottom HP gauge, red decoration pixels, etc.).  A detector box that
        # never moves always fools the stuck watchdog into firing forever, so
        # we keep a short position track and disable Stage 1 once convicted.
        self._prb_track_frames   = []   # ring buffer of (prb_bar, prb_player) tuples
        self._prb_track_len      = 12   # ~0.5–1 s at typical loop rates
        self._prb_fp_suspected   = False
        self._prb_fp_disabled    = False  # once True, Stage 1 is skipped for rest of run
        self._prb_first_frame_t  = None   # time.time() of the very first Stage-1 "OK" frame
        self._prb_dbg_fp_saved   = False  # only save the convicted FP mask/crop once
        self.thread_auto_bot = None # thread for running autobot
        self.cmd_move_x = "none" # "left" "right"
        self.cmd_move_y = "none" # "up" "down"
        self.cmd_action = "none" # "jump" "attack" ....
        # Signals (for UI)
        self.image_debug_signal = None
        self.route_map_viz_signal = None
        # Flags
        self.is_first_frame = True # first frame flag
        self.is_terminated = False # Close all object and thread if True
        self.is_on_ladder = False # Character is on ladder or not
        self.is_show_debug_window = not args.disable_viz #
        self.is_need_show_debug_window = not args.disable_viz #
        self.is_disable_control = args.disable_control
        self.is_ui = args.is_ui # Whether is using UI framework to invoke engine
        self.is_frame_done = False #
        # Coordinate (top-left coordinate)
        self.loc_nametag = (0, 0) # nametag location on game screen
        self.loc_party_red_bar = (0, 0) # party red bar location on game screen
        self.loc_minimap = (0, 0) # minimap location on game screen
        self.loc_player = (0, 0) # player location on game screen
        self.loc_player_minimap = (0, 0) # player location on minimap
        self.loc_minimap_global = (0, 0) # minimap location on global map
        self.loc_player_global = (0, 0) # player location on global map
        self.loc_watch_dog = (0, 0) # watch dog location on global map
        # Images
        self.frame = None # raw image
        self.img_frame = None # game window frame
        self.img_frame_gray = None # game window frame graysale
        self.img_frame_debug = None # game window frame for visualization
        self.img_route = None # route map
        self.img_route_debug = None # route map for visualization
        self.img_minimap = np.zeros((10, 10, 3), dtype=np.uint8) # minimap on game screen
        # Timers
        self.t_last_frame = time.time() # Last frame timer, for fps calculation
        self.t_watch_dog = time.time() # Last movement timer
        self.t_last_teleport = time.time() # Last teleport timer
        self.t_last_attack = time.time() # Last attack timer for cooldown
        self.t_last_minimap_update = time.time()
        self.t_to_change_channel = time.time()
        # Images
        self.img_map = None
        self.img_routes = []
        self.img_nametag = None
        self.img_nametag_gray = None
        self.img_create_party_enable = None
        self.img_create_party_disable = None
        self.img_login_button = None

        # Database
        self.data = load_yaml("config/config_data.yaml")
        # Threads & Objects
        self.kb = None # Keyboard controller
        self.capture = None # Game window capturor
        self.health_monitor = None # Health monitor
        self.profiler = None # Profiler, for performance issue debugging
        self.rune_solver = None # Rune solver

        # Finite State Machine
        self.fsm = FiniteStateMachine()
        self.fsm.add_state(HuntingState    ("hunting"     , self))
        self.fsm.add_state(FindingRuneState("finding_rune", self))
        self.fsm.add_state(NearRuneState   ("near_rune"   , self))
        self.fsm.add_state(SolvingRuneState("solving_rune", self))
        self.fsm.add_state(AuxiliaryState  ("aux"         , self))
        self.fsm.add_state(PatrolState     ("patrol"      , self))
        self.fsm.add_transition("hunting", "finding_rune") # When saw a "Rune has created" messgae
        self.fsm.add_transition("finding_rune", "hunting") # After finding rune timeout
        self.fsm.add_transition("finding_rune", "near_rune") # When detect a nearby rune
        self.fsm.add_transition("finding_rune", "solving_rune") # When enter the arrow minimap
        self.fsm.add_transition("near_rune", "finding_rune") # After rune solving timeout
        self.fsm.add_transition("near_rune", "solving_rune") # When enter the arrow minimap
        self.fsm.add_transition("solving_rune", "hunting") # After rune solving
        self.fsm.set_init_state("hunting")

    def update_signals(self, image_debug_signal, route_map_viz_signal):
        '''
        Update signal from UI framework.
        For debug window viz
        '''
        self.image_debug_signal = image_debug_signal
        self.route_map_viz_signal = route_map_viz_signal

    def _ensure_nametag_template_loaded(self, log_level="cascade"):
        '''
        Lazy helper that guarantees ``self.img_nametag`` and
        ``self.img_nametag_gray`` are populated whenever the configured
        nametag (``cfg["nametag"]["name"]``) differs from the one currently
        loaded.

        * At ``load_config`` time we call it so the user sees a clear
          ``Loaded image: nametag/xxx.png`` message alongside the other
          resource loads.
        * At cascade-run time we call it so toggling ``enable`` or editing
          ``name`` via the **Advanced Settings** UI does **not** require a
          full restart — the next frame reloads the template.

        Returns True if a valid nametag template is available after this
        call, False otherwise.
        '''
        current_key = (
            bool(self.cfg["nametag"].get("enable", False)),
            str(self.cfg["nametag"].get("name", "") or ""),
        )
        enable_flag, name = current_key

        # Short-circuit if nothing changed and we already have a template.
        if getattr(self, "_last_nametag_cfg_key", None) == current_key and \
           self.img_nametag is not None and \
           self.img_nametag_gray is not None:
            return True

        # Always try to load if a real name is supplied, independent of the
        # enable flag.  The cascade (Stage 2) will decide whether to *use*
        # the result based on enable.  If name is "example" or empty we skip
        # (no one ships a real "example" player-nametag template).
        need_load = bool(name) and name != "example"
        if not need_load:
            self._last_nametag_cfg_key = current_key
            return False

        path_color = f"nametag/{name}.png"
        import os as _os
        file_exists = _os.path.isfile(path_color)
        if not file_exists:
            logger.error(
                f"[_ensure_nametag_template_loaded] Nametag template file "
                f"not found: {path_color!r}.  Expected a PNG at "
                f"``nametag/{name}.png``.  Stage 2 (nametag) of the player "
                "location cascade will be skipped until the file exists "
                "and nametag.name matches the filename (without .png)."
            )
            self.img_nametag = None
            self.img_nametag_gray = None
            self._last_nametag_cfg_key = current_key
            return False

        img = load_image(path_color)
        img_gray = load_image(path_color, cv2.IMREAD_GRAYSCALE)
        if img is None or img_gray is None:
            logger.error(
                f"[_ensure_nametag_template_loaded] load_image returned "
                f"None for {path_color!r} (corrupt / unsupported PNG?).  "
                "Stage 2 (nametag) skipped."
            )
            self.img_nametag = None
            self.img_nametag_gray = None
            self._last_nametag_cfg_key = current_key
            return False

        self.img_nametag = img
        self.img_nametag_gray = img_gray
        self._last_nametag_cfg_key = current_key
        if log_level == "init":
            logger.info(
                f"Loaded image: {path_color} (name={name!r}, "
                f"enable={enable_flag}, shape={self.img_nametag.shape})"
            )
        else:
            logger.info(
                f"[player_cascade] (Re)loaded nametag template "
                f"{path_color!r} (name={name!r}, enable={enable_flag})."
            )
        return True

    def load_config(self, cfg):
        '''
        load_config
        '''
        # Parse color code in config
        self.color_code = {
            tuple(map(int, k.split(','))): v
            for k, v in cfg["route"]["color_code"].items()
        }
        self.color_code_up_down = {
            tuple(map(int, k.split(','))): v
            for k, v in cfg["route"]["color_code_up_down"].items()
        }

        if cfg["bot"]["mode"] == "normal":
            map_name = cfg['bot']['map']
            # Check if the map is supported in config_data.yaml
            if map_name not in self.data["map_mobs_mapping"]:
                text = f"Invalid map name: {map_name}. "\
                        "Not supported in config/config_data.yaml."
                logger.error(text)
                return -1
                # raise RuntimeError(text)

            # Load map.png from minimaps/
            self.img_map = load_image(f"minimaps/{map_name}/map.png",
                                      cv2.IMREAD_COLOR)
            # Load route*.png from minimaps/
            route_files = sorted(glob.glob(f"minimaps/{map_name}/route*.png"))
            route_files = [p for p in route_files if not p.endswith("route_rest.png")]
            self.img_routes = []
            for route_file in route_files:
                img = cv2.cvtColor(load_image(route_file), cv2.COLOR_BGR2RGB)
                # Remove pixel in map that is color code
                img = mask_route_colors(self.img_map, img, cfg["route"]["color_code"])
                img = mask_route_colors(self.img_map, img, cfg["route"]["color_code_up_down"])
                self.img_routes.append(img)

            # Load monsters images from monster/<monster_name>
            for monster_name in self.data["map_mobs_mapping"][map_name]:
                imgs = []
                for file in glob.glob(f"monster/{monster_name}/{monster_name}*.png"):
                    # Add original image
                    img = load_image(file)
                    imgs.append((img, get_mask(img, (0, 255, 0))))
                    # Add flipped image
                    img_flip = cv2.flip(img, 1)
                    imgs.append((img_flip, get_mask(img_flip, (0, 255, 0))))
                if imgs:
                    self.monsters_info[monster_name] = imgs
                else:
                    logger.error(f"No images found in monster/{monster_name}/{monster_name}*")
                    return -1
                    # raise RuntimeError(f"No images found in monster/{monster_name}/{monster_name}*")
            logger.info(f"Loaded monsters: {list(self.monsters_info.keys())}")

        # Re-resolve nametag template at every load_config call (the UI
        # triggers this every time the user saves changes in the Advanced
        # Settings tab) so the user does NOT need to restart AutoBot after
        # editing nametag.name / nametag.enable.
        #
        # NOTE: This MUST happen after self.cfg is assigned (at the end of
        # load_config) if we want to read cfg["nametag"]["name"] from the
        # *new* config, not the previous one.  To keep the "Loaded image:"
        # message in the same relative order as other assets, we pass the
        # incoming ``cfg`` directly into the helper via a temporary swap.
        _orig_cfg = getattr(self, "cfg", None)
        try:
            self.cfg = cfg
            self._ensure_nametag_template_loaded(
                log_level="cascade" if _orig_cfg is not None else "init"
            )
        finally:
            # Restore (will be overwritten again below with the new cfg, so
            # this block is just safety for early-return paths above).
            if _orig_cfg is not None:
                self.cfg = _orig_cfg

        # Load misc image
        lang = cfg["system"]["language"]
        self.img_create_party_enable  = load_image(f"misc/party_button_create_enable_{lang}.png")
        self.img_create_party_disable = load_image(f"misc/party_button_create_disable_{lang}.png")
        self.img_login_button = load_image(f"misc/login_button_{lang}.png")

        # Normalized pixel coordinate configuration
        cfg['rune_warning_cn']['top_left'] = normalize_pixel_coordinate(
            cfg['rune_warning_cn']['top_left'], cfg['game_window']['size'])
        cfg['rune_warning_cn']['bottom_right'] = normalize_pixel_coordinate(
            cfg['rune_warning_cn']['bottom_right'], cfg['game_window']['size'])
        cfg['rune_warning_eng']['top_left'] = normalize_pixel_coordinate(
            cfg['rune_warning_eng']['top_left'], cfg['game_window']['size'])
        cfg['rune_warning_eng']['bottom_right'] = normalize_pixel_coordinate(
            cfg['rune_warning_eng']['bottom_right'], cfg['game_window']['size'])
        cfg['rune_enable_msg_cn']['top_left'] = normalize_pixel_coordinate(
            cfg['rune_enable_msg_cn']['top_left'], cfg['game_window']['size'])
        cfg['rune_enable_msg_cn']['bottom_right'] = normalize_pixel_coordinate(
            cfg['rune_enable_msg_cn']['bottom_right'], cfg['game_window']['size'])
        cfg['rune_enable_msg_eng']['top_left'] = normalize_pixel_coordinate(
            cfg['rune_enable_msg_eng']['top_left'], cfg['game_window']['size'])
        cfg['rune_enable_msg_eng']['bottom_right'] = normalize_pixel_coordinate(
            cfg['rune_enable_msg_eng']['bottom_right'], cfg['game_window']['size'])
        cfg['rune_solver']['arrow_box_coord'] = normalize_pixel_coordinate(
            cfg['rune_solver']['arrow_box_coord'], cfg['game_window']['size'])
        cfg['ui_coords']['login_button_top_left'] = normalize_pixel_coordinate(
            cfg['ui_coords']['login_button_top_left'], cfg['game_window']['size'])
        cfg['ui_coords']['login_button_bottom_right'] = normalize_pixel_coordinate(
            cfg['ui_coords']['login_button_bottom_right'], cfg['game_window']['size'])

        # Print mode on log
        logger.info(f"[load_config] Config AutoBot as {cfg['bot']['mode']} mode")

        # Update cfg
        self.cfg = cfg

        return 0 # load successfully

    def start(self):
        '''
        Start all threads
        '''
        # Start keyboard controller thread
        self.kb = KeyBoardController(self.cfg)
        if self.is_disable_control:
            self.kb.disable() # Disable keyboard controller for debugging

        # Start game window capturing thread
        if self.args.test_image == '':
            self.capture = GameWindowCapturor(self.cfg)
        else:
            self.capture = GameWindowCapturor(self.cfg, self.args.test_image)

        # Once GameWindowCapturor has resolved the *actual exact* window
        # title (e.g. '冒险岛怀旧服' instead of the YAML default substring
        # 'MapleStory Worlds'), propagate that exact title to the keyboard
        # controller so its foreground-window / activation logic uses the
        # correct, resolvable identifier.  Without this sync the controller
        # kept matching against the wrong token, which meant:
        #   (a) is_game_window_active returned False even when the user was
        #       looking at the correct game window (causing spurious HALTED
        #       warnings that blocked all key presses), and
        #   (b) ensure_game_window_active's FindWindow couldn't find the
        #       HWND because the YAML token was not an exact title match.
        if getattr(self.capture, "window_title", None):
            self.kb.set_window_title(self.capture.window_title)
        # Also propagate the resolved HWND (if any) from GameWindowCapturor
        # to KeyBoardController so the PostMessage keyboard backend can fire
        # WM_KEYDOWN/WM_KEYUP directly into the game's message queue,
        # bypassing the low-level keyboard hook that many Chinese
        # MapleStory anti-cheat shields attach to block SendInput.
        _cap_hwnd = getattr(self.capture, "window_hwnd", 0) or 0
        if _cap_hwnd != 0:
            try:
                self.kb.set_game_hwnd(_cap_hwnd)
            except Exception:
                pass

        # Start health monitoring thread
        self.health_monitor = HealthMonitor(self.cfg, self.kb)
        if self.cfg["health_monitor"]["enable"] and \
            not self.is_disable_control:
            self.health_monitor.start()

        # Init profiler
        self.profiler = Profiler(self.cfg)

        # Init rune solver
        self.rune_solver = RuneSolver(self.cfg)

        # Reset all timers
        self.t_last_frame = time.time()
        self.t_watch_dog = time.time()
        self.t_last_teleport = time.time()
        self.t_last_attack = time.time()
        self.t_last_minimap_update = time.time()
        self.t_to_change_channel = time.time()

        # Reset one-shot player-localisation flags so a Stop → Start cycle,
        # a map change (load_config re-run), or a route-index switch do not
        # keep stale diagnostics from a previous run.  If we don't clear
        # these the synth/projection paths never re-emit their helpful
        # hints and the "is this match good enough" gates stay poisoned.
        self._prb_track_frames      = []
        self._prb_fp_suspected      = False
        self._prb_fp_disabled       = False
        self._prb_first_frame_t     = None
        self._prb_dbg_fp_saved      = False
        self._prb_dbg_saved         = False
        self._minimap_global_synth  = False
        self._mm_global_synth_warned        = False
        self._loc_player_mm_proj_warned     = False
        self._cc_fallback_no_mm_warned      = False
        self._route_color_fallback_logged_at = -99999.0
        self._route_empty_patrol_logged_at   = -99999.0
        self._route_cmd_dbg_logged_at        = -99999.0
        self._last_loc_method       = None
        # Mob-detection + viz state.
        # _last_mob_full_counters keeps the most recent snapshot of BOX/FULL/GRAYSCALE
        # counts produced by update_cmd_by_mob_detection.  It is read by
        # update_info_on_img_frame_debug (to paint the DETECT[...] status line +
        # 1 Hz DETECTOR_VIZ log line) even on frames where update_cmd_by_mob_detection
        # is skipped (e.g. very old state or first frame).  Initialise here rather than
        # lazily so the viz layer never raises AttributeError/KeyError before
        # the detector has run once.
        self._last_mob_full_counters = {"box": -1,
                                       "full": -1,
                                       "grayscale": -1,
                                       "used_fb": False,
                                       "mode": "?",
                                       "box_wh": (None, None),
                                       "player": (0, 0),
                                       "nearest": None,
                                       "attack_dir": None,
                                       "cmd_action_now": None}
        # run_once sets this to True immediately after calling
        # update_cmd_by_mob_detection; the Hunting/Patrol/FindingRune state
        # handlers use it to avoid calling the detector a second time this frame.
        self._mob_detection_ran_this_frame = False
        # Nametag-anti-FP trackers.  CN clients often render the character's
        # own name TWICE: once above the character's head (the one we want)
        # and once in the top-right info panel in big bold red letters.
        # Because the text/font/colour is identical, template matching
        # usually picks the UI one (larger, sharper, fixed position) which
        # makes the tracker think the player never moves → stuck forever.
        self._nt_track_frames       = []
        self._nt_track_len          = 12
        self._nt_fp_disabled        = False
        self._nt_first_frame_t      = None
        self._nt_dbg_fp_saved       = False
        # Emergency motion fallback.  When both the route tracker AND the
        # random-stuck-rescue fail to produce a non-none move command for a
        # continuous window, we drop into a very simple left/right patrol
        # loop so at least the character walks around and attracts mobs.
        self._patrol_dir_toggled_at = time.time()
        self._patrol_current        = "right"  # left ↔ right alternator

        # Set init state
        if self.args.init_state != "":
            self.fsm.set_init_state(self.args.init_state) # For debugging
        elif self.cfg["bot"]["mode"] == "aux":
            self.fsm.set_init_state("aux")
        elif self.cfg["bot"]["mode"] == "patrol":
            self.fsm.set_init_state("patrol")
        else:
            self.fsm.set_init_state("hunting")

        # Start Auto Bot main thread
        self.thread_auto_bot = threading.Thread(target=self.loop)
        self.thread_auto_bot.start()
        self.is_first_frame = True

        logger.info("[MapleStoryAutoBot] Started")

    def pause(self):
        '''
        Terminate thread except main thread
        '''
        self.terminate_threads()

    def enable_viz(self):
        '''
        Enable AutoBot to generate debug image
        '''
        self.is_need_show_debug_window = True
        logger.debug("[enable_viz] is_show_debug_window = True")

    def disable_viz(self):
        '''
        Disable AutoBot to generate debug image
        '''
        self.is_need_show_debug_window = False
        logger.debug("[disable_viz] is_show_debug_window = False")

    def start_record(self):
        '''
        Start record
        '''
        # Prepare video writer if need to record
        if not self.is_show_debug_window:
            self.enable_viz()

        # Make sure video/ exist
        os.makedirs("video", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join("video", f"{timestamp}.mp4")

        # Get video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 codec
        self.video_writer = cv2.VideoWriter(path, fourcc, 10, WINDOW_WORKING_SIZE)

        logger.info(f"[start_record] Record video to {path}")

    def stop_record(self):
        '''
        Stop Record
        '''
        self.video_writer = None
        logger.info("[stop_record] Stop recording")

    def get_player_location_camera_center_fallback(self):
        '''
        Final fallback for player screen-location estimation when both
        ``party_red_bar`` (CN client renders it in the top-right UI panel,
        *not* above the character) and ``nametag`` (no custom template
        supplied yet) fail.

        Assumptions:
          * The camera is tightly centered on the character (standard
            MapleStory behaviour in non-town maps).
          * Therefore the character is approximately at
            ``(WINDOW_WORKING_SIZE.w//2, WINDOW_WORKING_SIZE.h//2 + vertical
            bias)`` — most 2D side-scrollers keep the character slightly
            *below* the true vertical center because the HUD occupies the
            bottom and ground features live in the lower half.

        Returns a ``(loc_player, method_tag)`` tuple where ``method_tag`` is
        used purely for diagnostic logging so the user can tell that the
        cascade fell back to this estimator.
        '''
        w_wrk, h_wrk = WINDOW_WORKING_SIZE
        # Slight vertical bias: aim ~58% of the way down the work window.
        # This is empirical but a lot safer than dead-center.
        cx = w_wrk // 2
        cy = int(h_wrk * 0.58)
        # Draw a distinguishable marker on the debug viz so the user can tell
        # we're on fallback and NOT actually seeing the character.
        cv2.putText(self.img_frame_debug, "CAMERA CENTER FALLBACK",
                    (max(0, cx - 120), max(20, cy - 60)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
        cv2.drawMarker(self.img_frame_debug, (cx, cy),
                       (0, 220, 255), cv2.MARKER_CROSS, 28, 2)
        return (cx, cy), "camera_center_fallback"

    def _detect_player_location_cascade(self):
        '''
        Three-stage player-location cascade so the bot never ends up with a
        completely stale ``loc_player`` that keeps the stuck watchdog firing
        forever on non-Artale clients.

        Stage 1 — party red bar (Artale default; best accuracy).
        Stage 2 — name tag: template matching against the character-name
                   image under ``nametag/<name>.png``.  Lazy-reloaded here
                   every frame so changes made in **Advanced Settings**
                   (``nametag.enable`` / ``nametag.name``) take effect
                   WITHOUT restarting the process.
        Stage 3 — camera-center approximation: cheap, coarse, keeps the
                   watchdog happy when 1 & 2 both fail.  Condition is
                   "``self.img_frame`` has valid content" — if the capture
                   pipeline is delivering frames we assume the camera is
                   still roughly centered on the character (standard
                   side-scroller behaviour).  Old code additionally gated on
                   minimap-dot visibility; the new tolerant
                   ``get_player_location_on_minimap`` makes that mostly
                   redundant, but we still prefer it if present.

        Returns ``(loc_player, loc_party_red_bar, method_used)``.
        ``method_used`` is a short tag for the debug viz/logs.
        '''
        method_used = "none"

        # --- Stage 1: party red bar ------------------------------------------
        # If the anti-FP tracker convicted Stage 1 of matching a static UI
        # element (top-right HP bar / red decoration) we skip it entirely for
        # the rest of this run — otherwise we end up stuck on a stale
        # ``loc_player`` that never moves, which keeps the watchdog firing
        # forever (the exact failure the user is seeing).
        prb_player, prb_bar = None, None
        if not self._prb_fp_disabled:
            prb_player, prb_bar = self.get_player_location_by_party_red_bar()
        if prb_player is not None and prb_bar is not None:
            return prb_player, prb_bar, "party_red_bar"

        # --- Stage 2: nametag ------------------------------------------------
        # Lazy-resolve: if the user changed nametag.enable or nametag.name
        # via the UI since the last frame, reload the template RIGHT NOW so
        # they don't need to click "restart bot" (or even close the UI).
        self._ensure_nametag_template_loaded(log_level="cascade")

        # Run even when ``nametag.enable`` is False — on CN/non-Artale
        # clients the red bar is top-right and Stage 1 always fails, so we
        # silently fall back to the nametag pipeline.  ``enable`` gate is
        # soft: if False we still run once (so the user can preview) but in
        # future the flag could toggle a strict enforcement.
        nt_player = None
        if self.img_nametag is not None and \
           self.img_nametag_gray is not None and \
           self.cfg["nametag"]["name"] != "example":
            try:
                nt_player = self.get_player_location_by_nametag()
            except Exception as e:
                logger.debug(f"[player_cascade] nametag raised: {e!r}")
                nt_player = None
        if nt_player is not None:
            method_used = "nametag" if self.cfg["nametag"]["enable"] else "nametag_auto_fallback"
            return nt_player, prb_bar, method_used

        # --- Stage 3: camera-center fallback --------------------------------
        # Primary gate: is the frame capture actually yielding real pixels?
        img_frame_ok = (getattr(self, "img_frame", None) is not None and
                        self.img_frame.size > 0)

        # Helper: minimap-dot visibility (soft gate; if not there the frame
        # gate is still authoritative for camera-center fallback).
        mm_dot_ok = (getattr(self, "loc_player_minimap", None) is not None and
                     self.loc_player_minimap != (0, 0))

        if img_frame_ok:
            if not mm_dot_ok and not getattr(self, "_cc_fallback_no_mm_warned", False):
                self._cc_fallback_no_mm_warned = True
                logger.debug(
                    "[player_cascade] Using camera-center fallback without "
                    "a visible minimap player dot.  Capture pipeline is OK "
                    "but minimap-dot colour may be calibrated to the wrong "
                    "BGR value for this client; inspect log/debug_minimap_"
                    "player_raw.png if navigation accuracy is poor."
                )
            cam_player, _tag = self.get_player_location_camera_center_fallback()
            return cam_player, prb_bar, "camera_center_fallback"

        # All stages failed (img_frame is None too = probably login screen,
        # lost HWND, or misconfigured window-capture parameters).
        return None, prb_bar, method_used or "all_stages_failed"

    def _is_nametag_false_positive(self, loc_nametag_abs, loc_player_abs):
        '''
        Decide whether a good-score nametag match is actually a STATIC UI
        panel element (CN clients render the character's own name in the
        top-right info panel using the exact same font/colour as the
        above-head nametag).

        Heuristics:
          (a) Geometry — reject if outside playfield (right 36% / bottom 20%).
          (b) Temporal — if position moved < 3 px across last 12 frames it
              is almost certainly a static UI element.

        Returns True when convicted as UI false-positive.
        '''
        if loc_nametag_abs is None or loc_player_abs is None:
            return True
        W, H = WINDOW_WORKING_SIZE
        nx, ny = loc_nametag_abs
        PLAY_Y_LO  = int(H * 0.06)
        PLAY_Y_HI  = int(H * 0.80)
        PLAY_X_LO  = int(W * 0.14)
        PLAY_X_HI  = int(W * 0.64)
        geom_violation = not (PLAY_X_LO <= nx <= PLAY_X_HI and
                              PLAY_Y_LO <= ny <= PLAY_Y_HI)

        self._nt_track_frames.append((loc_nametag_abs, loc_player_abs))
        if len(self._nt_track_frames) > self._nt_track_len:
            self._nt_track_frames.pop(0)
        if self._nt_first_frame_t is None:
            self._nt_first_frame_t = time.time()

        temp_violation = False
        if len(self._nt_track_frames) == self._nt_track_len:
            nt0_x, nt0_y = self._nt_track_frames[0][0]
            ntN_x, ntN_y = self._nt_track_frames[-1][0]
            pl0_x, pl0_y = self._nt_track_frames[0][1]
            plN_x, plN_y = self._nt_track_frames[-1][1]
            nt_disp  = abs(ntN_x - nt0_x) + abs(ntN_y - nt0_y)
            pl_disp  = abs(plN_x - pl0_x) + abs(plN_y - pl0_y)
            if nt_disp < 3 and pl_disp < 3:
                warmup_s = time.time() - self._nt_first_frame_t
                if warmup_s > 4.0:
                    temp_violation = True

        # NOTE on the temporal ("static") rule:
        #   The intent was to reject the character name rendered STATICALLY in
        #   the top-right info panel.  But a *real* above-head nametag is also
        #   static whenever the player simply stands still (just after F1, or
        #   while waiting).  The old code let a static match escalate all the
        #   way to a PERMANENT disable (_nt_fp_disabled), which produced the
        #   exact bug the user reported: "nametag works for ~3 s then never
        #   again".  Fix: a static match only causes a *soft, this-frame*
        #   fallback to camera-center; it must NEVER latch the permanent
        #   disable.  As soon as the player moves, the next frame's nametag is
        #   accepted again.  Only GEOMETRY violations (match landing in the
        #   right-info-panel / bottom UI band) are trustworthy enough to count
        #   toward the permanent disable.
        # Decision: only GEOMETRY convicts.  A static-but-in-playfield match is
        # accepted as the real (standing-still) player, so loc_player no longer
        # flickers between the true spot and the camera-center fallback while
        # the character waits.  temp_violation is retained purely for the
        # diagnostic log/crop below.
        is_fp = geom_violation
        if (geom_violation or temp_violation) and not self._nt_dbg_fp_saved \
                and len(self._nt_track_frames) >= 4:
            self._nt_dbg_fp_saved = True
            try:
                pad = 60
                cx0 = max(0, nx - pad)
                cy0 = max(0, ny - pad)
                cx1 = min(W, nx + self.img_nametag.shape[1] + pad)
                cy1 = min(H, ny + self.img_nametag.shape[0] + pad)
                if hasattr(self, "img_frame") and self.img_frame is not None:
                    crop = self.img_frame[cy0:cy1, cx0:cx1].copy()
                    rbx, rby = nx - cx0, ny - cy0
                    cv2.rectangle(crop, (rbx, rby),
                                  (rbx + self.img_nametag.shape[1],
                                   rby + self.img_nametag.shape[0]),
                                  (0, 0, 255), 2)
                    cv2.imwrite("log/debug_nametag_fp_crop.png", crop)
                    cv2.imwrite("log/debug_nametag_fp_full.png", self.img_frame)
            except Exception:
                pass
            why = []
            if geom_violation: why.append(f"geom(nametag@({nx},{ny}))")
            if temp_violation: why.append("static(12f<3px)")
            _why = " + ".join(why) if why else "unknown"
            logger.warning(
                f"[Stage2 nametag] CONVICTED false-positive ({_why}). "
                "Falling back to Stage 3 (camera center).  "
                "Diagnostic crop saved to log/debug_nametag_fp_*.png."
            )
        # Permanent disable is reserved for the reliable GEOMETRY signal only.
        # A purely temporal (static-position) match must not latch it, so a
        # player who stands still does not permanently lose nametag tracking.
        if geom_violation and len(self._nt_track_frames) >= self._nt_track_len * 2:
            self._nt_fp_disabled = True
        return is_fp

    def get_player_location_by_nametag(self):
        '''
        Detects the player's location based on the nametag position in the game window.
        ROOT-BUG FIXES (see task #1):
          (1) score >= diff_thres returns None instead of reusing stale loc_nametag.
          (2) cropped-ROI match result now correctly adds back CROP_DX/CROP_DY.
          (3) After good score, runs _is_nametag_false_positive temporal filter.

        This function works by:
        - Extracting a vertical region of interest (ROI) where the nametag is expected.
        - Padding the ROI to avoid template matching edge issues.
        - Using template matching to locate the nametag, split into left and right halves
          to improve robustness against partial occlusion.
        - Selecting the best match (left or right) based on score and cache status.
        - Computing the player's center position by applying a fixed offset to the nametag.

        Returns:
            loc_player (tuple): The (x, y) coordinates of the player's estimated location.
                Returns ``None`` if convicted as UI-panel false-positive.
        '''
        # CN 怀旧服: pre-crop to PLAYFIELD (exclude top-right status panel,
        # top 10% chrome, left 14% chat/minimap, bottom 20% HP/EXP bar area)
        # so the template-match cannot accidentally lock onto the static
        # character-info panel name.
        H_full, W_full = self.img_frame_gray.shape[:2]
        ui_y_start     = int(self.cfg["ui_coords"]["ui_y_start"])
        PFX_X_LO = int(W_full * 0.14)
        PFX_X_HI = int(W_full * 0.64)
        PFX_Y_LO = int(H_full * 0.10)
        PFX_Y_HI = max(PFX_Y_LO + 64, ui_y_start - int(H_full * 0.20))
        CROP_DX, CROP_DY = PFX_X_LO, PFX_Y_LO

        # Get camera region in the game window
        img_camera = self.img_frame_gray[PFX_Y_LO:PFX_Y_HI, PFX_X_LO:PFX_X_HI].copy()

        # Get nametag image and search image
        if self.cfg["nametag"]["mode"] == "white_mask":
            # Apply Gaussian blur for smoother white detection
            img_camera = cv2.GaussianBlur(img_camera, (3, 3), 0)
            img_nametag = cv2.GaussianBlur(self.img_nametag_gray, (3, 3), 0)
            lower_white, upper_white = (150, 255)
            img_roi = cv2.inRange(img_camera, lower_white, upper_white)
            img_nametag  = cv2.inRange(img_nametag, lower_white, upper_white)
        elif self.cfg["nametag"]["mode"] == "grayscale":
            img_roi = img_camera
            img_nametag = self.img_nametag_gray
        elif self.cfg["nametag"]["mode"] == "histogram_eq":
            # Apply histogram equalization
            img_nametag_eq = cv2.equalizeHist(self.img_nametag_gray)
            img_camera_eq = cv2.equalizeHist(img_camera)

            # Apply global (fixed) threshold
            _, img_nametag = cv2.threshold(img_nametag_eq, 150, 255, cv2.THRESH_BINARY)
            _, img_roi = cv2.threshold(img_camera_eq, 150, 255, cv2.THRESH_BINARY)
        else:
            logger.error(f"Unsupported nametag detection mode: {self.cfg['nametag']['mode']}")
            return None

        # If anti-FP tracker has already convicted Stage 2, skip all work and
        # let the cascade go straight to Stage 3 (camera center fallback).
        if getattr(self, "_nt_fp_disabled", False):
            return None

        # Guard: if nametag template is larger than the (already-cropped)
        # search region, matching is impossible.
        th, tw = img_nametag.shape[:2]
        ch, cw = img_roi.shape[:2]
        if th > ch or tw > cw:
            return None

        # Pad search region to deal with fail detection when player is at map edge
        (pad_y, pad_x) = self.img_nametag.shape[:2]
        img_roi = cv2.copyMakeBorder(
            img_roi,
            pad_y, pad_y, pad_x, pad_x,
            borderType=cv2.BORDER_REPLICATE  # replicate border for safe matching
        )

        # Get last frame name tag location.
        # BUG-FIX (task #1 item 2): previous code used self.loc_nametag
        # directly (absolute full-window coords) inside the cropped ROI,
        # which meant the "last result" hint pointed at completely the
        # wrong place when CROP_DX/CROP_DY were nonzero.  Now we subtract
        # the crop offsets before adding padding back.
        if self.is_first_frame or self.loc_nametag == (0, 0):
            last_result = None
        else:
            last_result = (
                (self.loc_nametag[0] - CROP_DX) + pad_x,
                (self.loc_nametag[1] - CROP_DY) + pad_y,
            )

        # Get number of splits
        h, w = img_nametag.shape
        num_splits = max(1, w // self.cfg["nametag"]["split_width"])
        w_split = w // num_splits

        # Get nametag's background mask
        mask = get_mask(self.img_nametag, (0, 255, 0))

        # Vertically split the nametag image
        nametag_splits = {}
        for i in range(num_splits):
            x_s = i * w_split
            x_e = (i + 1) * w_split if i < num_splits - 1 else w
            nametag_splits[f"{i+1}/{num_splits}"] = {
                "img": img_nametag[:, x_s:x_e],
                "mask": mask[:, x_s:x_e],
                "last_result": (
                    (last_result[0] + x_s, last_result[1]) if last_result else None
                ),
                "score_penalty": 0.0,
                "offset_x": x_s
            }

        # Match tempalte
        matches = []
        for tag_type, split in nametag_splits.items():
            loc, score, is_cached = find_pattern_sqdiff(
                img_roi,
                split["img"],
                last_result=split["last_result"],
                mask=split["mask"],
                global_threshold=self.cfg["nametag"]["global_diff_thres"]
            )
            w_match = split["img"].shape[1]
            h_match = split["img"].shape[0]
            score += split["score_penalty"]
            matches.append((tag_type, loc, score, w_match, h_match, is_cached, split["offset_x"]))

        # Select best match and fix offset:
        matches.sort(key=lambda x: (not x[5], x[2]))  # prefer cached, then low score
        tag_type, loc_in_padded, score, w_match, h_match, is_cached, offset_x = matches[0]

        # --- Unwind coordinate transforms (task #1 bug #2) ----------------
        # loc_in_padded lives in "cropped ROI + border-padding" space.
        # 1) subtract split-horizontal offset
        # 2) subtract border padding → back to cropped-playfield relative
        # 3) ADD BACK CROP_DX/CROP_DY → absolute full-window coords
        lx_rel_pad = loc_in_padded[0] - offset_x
        ly_rel_pad = loc_in_padded[1]
        lx_rel_crop = lx_rel_pad - pad_x
        ly_rel_crop = ly_rel_pad - pad_y
        loc_nametag_abs = (lx_rel_crop + CROP_DX, ly_rel_crop + CROP_DY)

        # --- Score gate (task #1 bug #1) ----------------------------------
        # ROOT FIX: when score >= diff_thres we MUST NOT fall back to the
        # stale self.loc_nametag (which on the very first run often stored
        # a non-cropped match against the top-right UI panel).  Instead we
        # return None so the cascade can try Stage 3 (camera center).
        DIFF_THRES = self.cfg["nametag"]["diff_thres"]
        if score >= DIFF_THRES:
            return None

        # Score is good → cache the absolute full-window nametag position.
        self.loc_nametag = loc_nametag_abs

        # Compute player center from nametag (still in full-window abs coords)
        loc_player_abs = (
            loc_nametag_abs[0] + w // 2,
            loc_nametag_abs[1] - self.cfg["nametag"]["offset"][1],
        )

        # --- Anti-FP temporal/geometry filter (task #1 bug #3) ------------
        if self._is_nametag_false_positive(loc_nametag_abs, loc_player_abs):
            return None

        # Draw name tag detection box for debugging
        draw_rectangle(self.img_frame_debug, loc_nametag_abs,
                       self.img_nametag.shape, (0, 255, 0), "")
        text = f"NameTag,{round(score, 2)}," + \
                f"{'cached' if is_cached else 'missed'}," + \
                f"{tag_type}"
        cv2.putText(self.img_frame_debug, text,
                    (loc_nametag_abs[0],
                     loc_nametag_abs[1] + self.img_nametag.shape[0] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return loc_player_abs

    def _is_party_red_bar_false_positive(self, prb_bar, prb_player, img_camera_h):
        '''
        Decide whether the biggest-contour red box that *looks* like a party
        red bar is actually a static UI element.

        Heuristics (CN/怀旧服 oriented — these are *conservative* rejections,
        i.e. we only kill candidates that are almost certainly UI):

          (a) **Screen-geometry sanity** — a real above-character bar should
              live inside the "playfield" region, NOT glued to the very top /
              very right edge of the frame.  Panels on the top-right are the
              canonical place clients put character info, HP bars and party
              member mini-bars.
          (b) **Temporal sanity** — character bars *jitter* (idle animation,
              knockback, walking, jumping, even just camera sub-pixel shifts).
              A box whose position (both bar top-left and derived player
              center) hasn't moved more than ~2 px across the last 12 frames
              is almost certainly a static UI panel decoration.
          (c) **Warm-up grace** — the bot starts with the character still, so
              rule (b) only kicks in after a short warm-up window (4 s).

        Returns ``True`` when we believe this is a UI false-positive and the
        caller should pretend Stage 1 returned ``(None, None)``.  Also writes
        a one-shot diagnostic crop to ``log/debug_party_red_fp_*.png`` the
        first time we convict, so the user can sanity-check.
        '''
        if prb_bar is None or prb_player is None:
            return True

        bx, by, bw, bh = prb_bar[0], prb_bar[1], prb_bar[2], prb_bar[3]
        W, H = WINDOW_WORKING_SIZE

        # --- (a) Geometry rejection (cheap, per-frame) --------------------
        #   Top 22 px   : real bars want clearance for nameplate + top border
        #   Right 36%  : top-right character panel area (CN client big HP)
        #   Bottom 24% : bottom HP/MP/EXP bars
        #   Left 14%   : reserved for mini-map + chat panel
        PLAY_Y_LO  = int(H * 0.06)
        PLAY_Y_HI  = int(H * 0.76)
        PLAY_X_LO  = int(W * 0.14)
        PLAY_X_HI  = int(W * 0.64)
        geom_violation = not (PLAY_X_LO <= bx <= PLAY_X_HI and
                              PLAY_Y_LO <= by <= PLAY_Y_HI)

        # Track temporal statistics (do this even on geom-fail so the buffer
        # stays warm if the detector is simply drifting between panels).
        self._prb_track_frames.append((prb_bar, prb_player))
        if len(self._prb_track_frames) > self._prb_track_len:
            self._prb_track_frames.pop(0)

        if self._prb_first_frame_t is None:
            self._prb_first_frame_t = time.time()

        # --- (b) Temporal rejection ---------------------------------------
        temp_violation = False
        if len(self._prb_track_frames) == self._prb_track_len:
            bx0, by0 = self._prb_track_frames[ 0][0][:2]
            bxN, byN = self._prb_track_frames[-1][0][:2]
            px0, py0 = self._prb_track_frames[ 0][1]
            pxN, pyN = self._prb_track_frames[-1][1]
            bar_dx   = abs(bxN - bx0)
            bar_dy   = abs(byN - by0)
            play_dx  = abs(pxN - px0)
            play_dy  = abs(pyN - py0)
            # Need AT LEAST 3 px of displacement in either (bar or player)
            # over the tracking window.  3 px is tiny (idle breathing causes
            # that much), so anything less is definitely static UI.
            if (bar_dx + bar_dy) < 3 and (play_dx + play_dy) < 3:
                warmup_s = time.time() - self._prb_first_frame_t
                if warmup_s > 4.0:
                    temp_violation = True

        is_fp = geom_violation or temp_violation

        # One-shot diagnostics — save a crop + a short log the FIRST time we
        # convict (not every frame).  Crop is taken from img_camera
        # (top-part) so the user can eyeball what box was picked.
        if is_fp and (geom_violation or temp_violation) and \
           not self._prb_dbg_fp_saved and len(self._prb_track_frames) >= 4:
            self._prb_dbg_fp_saved = True
            try:
                # Build a tiny diagnostic mosaic: a larger crop around the
                # convicted box, drawn with a red rectangle.
                pad_x, pad_y = 40, 28
                cx0 = max(0, bx - pad_x)
                cy0 = max(0, by - pad_y)
                cx1 = min(W, bx + bw + pad_x)
                cy1 = min(img_camera_h, by + bh + pad_y)
                if hasattr(self, "img_frame") and self.img_frame is not None:
                    crop = self.img_frame[cy0:cy1, cx0:cx1].copy()
                    # draw rect *within crop coords*
                    rbx, rby = bx - cx0, by - cy0
                    cv2.rectangle(crop, (rbx, rby), (rbx + bw, rby + bh),
                                  (0, 0, 255), 2)
                    cv2.imwrite("log/debug_party_red_fp_crop.png", crop)
                    cv2.imwrite("log/debug_party_red_fp_full.png",
                                self.img_frame)
            except Exception:  # noqa: BLE001 — debug write is best-effort
                pass
            why = []
            if geom_violation: why.append(f"geom(box@({bx},{by}) outside playfield)")
            if temp_violation: why.append("static(12 frames <3px displacement)")
            _why = " + ".join(why) if why else "unknown"
            logger.warning(
                f"[Stage1 party_red_bar] CONVICTED false-positive ({_why}). "
                "Stage 1 disabled for the rest of the run; falling back to "
                "Stage 2 (nametag) / Stage 3 (camera center).  Crop saved to "
                "log/debug_party_red_fp_*.png if you want to verify."
            )

        if is_fp:
            self._prb_fp_suspected = True
            # Stay "suspected" for a few frames; after 2 full tracking
            # windows of sustained FP, commit to disabling Stage 1.  This
            # dampens one-off flicker.
            if len(self._prb_track_frames) >= self._prb_track_len * 2:
                self._prb_fp_disabled = True
        return is_fp

    def get_player_location_by_party_red_bar(self):
        '''
        get_player_location_by_party_red_bar

        Detects the character's party health bar on-screen and estimates the
        character center position via a configured offset.

        Robustness improvements:
          * Red in HSV wraps around at H=0/180 in OpenCV, so we combine TWO
            inRange masks: H ∈ [0, 10] AND H ∈ [160, 180] (low-H bright red +
            high-H dark crimson).  Previously we only took the low-H slice and
            therefore missed ~50% of crimson party bars used in non-Artale
            clients (e.g. 冒险岛怀旧服 CN client).
          * Contour geometry filter is relaxed slightly (height 4–12 px, width
            up to 80 px, area >= 8, fill rate >= 0.55) so differently-styled
            bars across clients still pass.
          * On failure the function prints a one-shot diagnostic summary
            (mask hit-count, contour count, biggest contour's box geometry)
            so users can quickly tell whether the issue is "wrong HSV range"
            vs "no party was created at all".
        '''
        # Zero out minimap area in the img_frame
        img_frame = self.img_frame.copy()
        x, y = self.loc_minimap
        h, w = self.img_minimap.shape[:2]
        img_frame[y:y+h, x:x+w] = 0

        # Get camera area
        img_camera = img_frame[:self.cfg["ui_coords"]["ui_y_start"], :]

        # Convert to HSV
        img_hsv = cv2.cvtColor(img_camera, cv2.COLOR_BGR2HSV)

        # --- Build red mask with TWO Hue wraparound ranges -------------------
        # Standard-config lower_red/upper_red are the "0° side" of red.
        # We auto-generate the complementary "180° side" range by offsetting
        # Hue by +180° (keeping S/V identical) so crimson/dark-red bars are
        # detected regardless of which half of the hue circle the client
        # renders them in.
        lower_std = to_opencv_hsv(self.cfg["party_red_bar"]["lower_red"])
        upper_std = to_opencv_hsv(self.cfg["party_red_bar"]["upper_red"])
        # Build the complementary (high-H) range.  Hue 0° -> +180° -> 180°
        # in standard space = H=90 in OpenCV scale (90 * 2 on the 0–180 circle).
        h_lo_std, s_lo_std, v_lo_std = lower_std
        h_hi_std, s_hi_std, v_hi_std = upper_std
        # Complementary side: shift the entire Hue slice by 90 OpenCV ticks
        # (i.e. +180° in standard HSV space).
        lower_wrap = np.array([h_lo_std + 90, s_lo_std, v_lo_std], dtype=np.uint8)
        upper_wrap = np.array([min(179, h_hi_std + 90), s_hi_std, v_hi_std], dtype=np.uint8)

        mask_red_a = cv2.inRange(img_hsv, lower_std, upper_std)
        mask_red_b = cv2.inRange(img_hsv, lower_wrap, upper_wrap)
        mask_red = cv2.bitwise_or(mask_red_a, mask_red_b)

        # Optional: save mask for debugging on the very first frame or on
        # persistent failure.  Helps the user tune HSV range visually.
        if not hasattr(self, "_prb_dbg_saved"):
            try:
                cv2.imwrite("log/debug_party_red_mask_a.png", mask_red_a)
                cv2.imwrite("log/debug_party_red_mask_b.png", mask_red_b)
                cv2.imwrite("log/debug_party_red_mask.png",   mask_red)
                self._prb_dbg_saved = True
            except Exception:  # noqa: BLE001 — log dir may not exist, ignore
                pass

        # Find contours on mask_red
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Filter contour by the (relaxed) geometric traits of a red HP bar
        boxs = []
        all_boxes_stats = []
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            fill_rate = float(area) / max(1, bh * bw)
            all_boxes_stats.append((bx, by, bw, bh, round(area, 1), round(fill_rate, 2)))
            # Relaxed filter: height 4-12 px, width up to 80 px, fill_rate >= 0.55
            if (4 <= bh <= 12 and 1 <= bw <= 80 and area >= 8 and fill_rate >= 0.55):
                boxs.append((bx, by, bw, bh))

        if not boxs:
            # Diagnostics: show top-5 largest contours by bounding-box area
            all_boxes_stats.sort(key=lambda b: b[2] * b[3], reverse=True)
            top5 = all_boxes_stats[:5]
            non_zero = int(cv2.countNonZero(mask_red))
            logger.debug(
                f"[get_player_location_by_party_red_bar] Not found. "
                f"mask non-zero px={non_zero}; raw contours={len(contours)}; "
                f"top-5 boxes (x,y,w,h,area,fill): {top5}"
            )
            return None, None  # red bar not found

        # Sort box by area
        boxs.sort(key=lambda box: box[2] * box[3], reverse=True)

        # Consider the biggest area as party red bar
        bx, by, bw, bh = boxs[0]

        # Offset coordinate
        loc_party_red_bar = (bx, by)
        loc_player = (bx + self.cfg["party_red_bar"]["offset"][0],
                      by + self.cfg["party_red_bar"]["offset"][1])

        # Anti-FP check: if the best box sits in a top-right character panel
        # or has been static across ~0.5 s of frames, it's almost certainly
        # a UI blood bar / decoration — NOT the character's own party bar.
        # Treat Stage 1 as a miss so the cascade keeps going.
        full_bar_box = (bx, by, bw, bh)
        cam_h = img_camera.shape[0]
        if self._is_party_red_bar_false_positive(full_bar_box, loc_player, cam_h):
            return None, None

        # visualize for debug
        draw_rectangle(self.img_frame_debug, loc_party_red_bar,
                    (bh, bw), (0, 255, 0), "party red bar", thickness=1, text_height=0.4)

        return loc_player, loc_party_red_bar

    def get_player_location_on_global_map(self):
        '''
        get_player_location_on_global_map

        Resolves ``self.loc_player_global`` — the player's pixel position on
        ``self.img_route`` (the full-route overlay image).

        **Original upstream formula (Artale client only):**
            ``loc_player_global = loc_minimap_global + loc_player_minimap + offset``

        That relies on TWO independent CV matchers both succeeding every frame:
          * ``find_pattern_sqdiff(img_map, img_minimap)`` → where is the top-left
            white-boxed minimap glued on to the route map background
            (``loc_minimap_global``).
          * ``get_player_location_on_minimap()`` → yellow player dot *inside*
            the ``img_minimap`` crop (``loc_player_minimap``).

        **Why the CN / 怀旧服 client always ended up STUCK:**
          1. The upstream ``find_pattern_sqdiff`` **always returns a position**
             even when the match is garbage — its "fallback" is ``(0,0)``
             with ``score=1.0``.
          2. ``get_player_location_on_minimap`` on CN clients usually can't
             see the yellow dot (default BGR is tuned for Artale; capture
             pipelines also shift colours slightly) so it keeps returning
             ``None`` and leaves ``self.loc_player_minimap`` at the zero
             initialiser.
          3. Result: ``loc_player_global == (0+0+0, 0+0+0)`` or some other
             *frame-invariant constant*.  The stuck watchdog sees 0 movement
             for 10 s and keeps spamming random moves FOREVER — exactly what
             the user reported.

        **Refined behaviour (robust across clients):**
          (a) Trust ``loc_minimap_global`` only if the SQDIFF match score is
              **actually good** (``< 0.4``).  Otherwise synthesise a guess by
              *linearly mapping* the on-screen player position to the route
              map's dimensions — this is a 0-th order approximation but it
              changes every time the camera moves, which is enough to keep
              the stuck watchdog happy and the route-tracker following a
              roughly-correct colour code.
          (b) If ``loc_player_minimap`` is stale (still ``(0,0)`` because no
              yellow dot was ever observed) substitute a *screen-derived
              offset*: take the on-screen player position (from whatever
              cascade stage succeeded — nametag / camera centre / ...) and
              project it on to the "minimap inside route map" coordinate
              system using the ratio between the two frames.
        '''
        # ------------------------------------------------------------------
        # Step 1 — where on the ROUTE MAP does the minimap sit?
        # ------------------------------------------------------------------
        match_loc, score, _is_cached = find_pattern_sqdiff(
                                            self.img_map,
                                            self.img_minimap)
        if score < 0.4:
            # Strong template match: trust it.  This is the normal path on
            # clients where the minimap renders with sharp white borders and
            # the route-map was captured from the same build.
            self.loc_minimap_global = match_loc
        else:
            # Weak or garbage match.  Don't poison loc_player_global with a
            # random (0,0) anchor.  Synthesise a guess by linearly mapping
            # the on-screen player position to the CURRENT ROUTE MAP'S
            # extents — we use ``self.img_route`` (the active route_*.png
            # selected by ``idx_routes``) rather than ``self.img_map``
            # because:
            #   (a) mask_route_colors may resize img_map to match
            #       img_route, meaning their shapes CAN differ, and
            #   (b) get_nearest_color_code searches in img_route coords,
            #       so the projected point must live in that same space.
            # NOTE: WINDOW_WORKING_SIZE is (WIDTH, HEIGHT) as per global_var.py
            # and every other unpack site in this file (L528, L648, L919).
            # The code below was previously unpacked in the wrong order
            # (H, W) which projected an on-screen player sitting at the
            # middle of the frame onto the far-right of the route map,
            # making the nearest-color lookup return the wrong command.
            W_WIN_W, W_WIN_H = WINDOW_WORKING_SIZE   # (WIDTH, HEIGHT)
            # Use the active route image as the projection target, falling
            # back to img_map (e.g. during the first frame when img_route
            # hasn't been assigned yet).
            if getattr(self, "img_route", None) is not None:
                tgt_h, tgt_w = self.img_route.shape[:2]
            else:
                tgt_h, tgt_w = self.img_map.shape[:2]
            px, py = self.loc_player if self.loc_player is not None else \
                     (W_WIN_W // 2, int(W_WIN_H * 0.58))
            proj_gx = int(px / max(1, W_WIN_W) * tgt_w)
            proj_gy = int(py / max(1, W_WIN_H) * tgt_h)
            proj_gx = max(0, min(max(0, tgt_w - 1), proj_gx))
            proj_gy = max(0, min(max(0, tgt_h - 1), proj_gy))
            self.loc_minimap_global = (0, 0)
            self._minimap_global_synth = True
            if not getattr(self, "_mm_global_synth_warned", False):
                self._mm_global_synth_warned = True
                logger.warning(
                    "[get_player_location_on_global_map] minimap→route "
                    f"template match was poor (score={round(score, 3)} ≥ 0.4). "
                    "Falling back to a screen→route proportional projection "
                    f"(target=img_route {tgt_w}x{tgt_h}, "
                    f"projected_global=({proj_gx},{proj_gy})).  This keeps "
                    "the watchdog happy but route precision is reduced; for "
                    "best precision calibrate minimap.player_color and "
                    "re-capture minimaps/<map>/map.png from the current "
                    "client build."
                )

        # ------------------------------------------------------------------
        # Step 2 — where INSIDE the minimap is the player?
        # ------------------------------------------------------------------
        x_offset, y_offset = self.cfg["minimap"]["offset"]
        if getattr(self, "_minimap_global_synth", False) or \
           self.loc_player_minimap is None or \
           self.loc_player_minimap == (0, 0):
            # Project screen→route *again* — re-compute because the code
            # above might have chosen img_map as a fallback target on the
            # very first frame while the route image was not yet bound.
            # NOTE: WINDOW_WORKING_SIZE is (WIDTH, HEIGHT) as per global_var.py
            # and every other unpack site in this file (L528, L648, L919).
            # The code below was previously unpacked in the wrong order
            # (H, W) which projected an on-screen player sitting at the
            # middle of the frame onto the far-right of the route map,
            # making the nearest-color lookup return the wrong command.
            W_WIN_W, W_WIN_H = WINDOW_WORKING_SIZE   # (WIDTH, HEIGHT)
            if getattr(self, "img_route", None) is not None:
                tgt_h, tgt_w = self.img_route.shape[:2]
            else:
                tgt_h, tgt_w = self.img_map.shape[:2]
            px, py = self.loc_player if self.loc_player is not None else \
                     (W_WIN_W // 2, int(W_WIN_H * 0.58))
            proj_gx = int(px / max(1, W_WIN_W) * tgt_w)
            proj_gy = int(py / max(1, W_WIN_H) * tgt_h)
            proj_gx = max(0, min(max(0, tgt_w - 1), proj_gx))
            proj_gy = max(0, min(max(0, tgt_h - 1), proj_gy))
            # Goal: loc_minimap_global (0,0) + loc_player_minimap + offset
            # equals the projected pixel so that the downstream sum lands
            # exactly on (proj_gx, proj_gy).
            loc_player_minimap = (proj_gx - x_offset, proj_gy - y_offset)
            if not getattr(self, "_minimap_global_synth", False) and \
               not getattr(self, "_loc_player_mm_proj_warned", False):
                self._loc_player_mm_proj_warned = True
                logger.warning(
                    "[get_player_location_on_global_map] No visible "
                    "minimap player dot (loc_player_minimap is (0,0)). "
                    "Substituting an on-screen→minimap projection so the "
                    "stuck watchdog can still observe movement.  Calibrate "
                    "minimap.player_color (see debug_minimap_colors.py / "
                    "log/debug_minimap_player_raw.png) for best accuracy."
                )
        else:
            loc_player_minimap = self.loc_player_minimap

        loc_player_global = (
            self.loc_minimap_global[0] + loc_player_minimap[0] + x_offset,
            self.loc_minimap_global[1] + loc_player_minimap[1] + y_offset
        )

        # Draw local minimap rectangle
        camera_bottom_right = (
            self.loc_minimap_global[0] + self.img_minimap.shape[1],
            self.loc_minimap_global[1] + self.img_minimap.shape[0]
        )
        cv2.rectangle(self.img_route_debug, self.loc_minimap_global,
                      camera_bottom_right, (0, 255, 255), 1)
        cv2.putText(
            self.img_route_debug,
            f"Minimap,score({round(score, 2)})",
            (self.loc_minimap_global[0], self.loc_minimap_global[1]+15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (0, 255, 255), 1
        )

        # Draw player center
        cv2.circle(self.img_route_debug,
                   loc_player_global, radius=2,
                   color=(0, 255, 255), thickness=-1)

        return loc_player_global

    def get_nearest_color_code(self):
        '''
        Searches for the nearest color-coded action marker
        around the player on the route map.

        This function:
        - Scans each pixel in the search box to find nearest color code
        - Tracks the closest matching pixel using Manhattan distance (|dx| + |dy|).
        - Returns a dictionary containing the nearest matching
          pixel's position, color, action label, and distance.

        Returns:
            dict or None: Dictionary containing:
                - "pixel": (x, y) coordinate of the matched pixel
                - "color": matched RGB color tuple
                - "action": corresponding action string from config
                - "distance": Manhattan distance from player
            Returns None if no matching color is found within the region.

        **CN/怀旧服 compatibility note**
        ---------------------------------
        Upstream searched only a tiny ``search_range`` (default 10 px) box
        around the player.  That assumption breaks whenever
        ``get_player_location_on_global_map`` had to fall back to a
        *synthesised* coordinate (minimap→route template score ≥ 0.4, or
        missing yellow player dot).  In those situations the coordinate
        error is usually 20–300 px so a 10-px radius window always comes
        back empty → ``cmd_move_*`` stays at ``"none"`` → the character
        never moves, and because the synthesised coordinate jitters every
        frame the stuck-watchdog doesn't even fire a random rescue action.

        Fixed here by running **two passes**:
          Pass 1 — tight box of ``cfg.search_range`` (the upstream
                   behaviour; still preferred because it's O(range²) fast).
          Pass 2 — if Pass 1 yielded nothing, scan the **whole route
                   image** but still pick the *geometrically nearest*
                   colour-code pixel.  This is O(W·H) per frame but route
                   images are small (a few hundred px on each side), and on
                   CN clients where we can't trust the global-map position
                   it's the only way the bot actually walks along the
                   authored route.
        '''
        x0, y0 = self.loc_player_global
        h, w = self.img_route.shape[:2]
        base_range = int(self.cfg["route"]["search_range"])

        # ------------------------------------------------------------------
        # Helper: scan a single rectangular region and update the two
        # "nearest" accumulators passed in (mutable lists of length 1).
        # ------------------------------------------------------------------
        def _scan_region(x_min, y_min, x_max, y_max,
                         _nearest, _nearest_ud,
                         _min_dist, _min_dist_ud):
            for y in range(y_min, y_max):
                row = self.img_route[y]
                for x in range(x_min, x_max):
                    pixel = (int(row[x, 0]), int(row[x, 1]), int(row[x, 2]))
                    dist = abs(x - x0) + abs(y - y0)
                    if pixel in self.color_code and dist < _min_dist[0]:
                        _nearest[0] = {
                            "pixel": (x, y),
                            "color": pixel,
                            "command": self.color_code[pixel],
                            "distance": dist,
                        }
                        _min_dist[0] = dist
                    if pixel in self.color_code_up_down and dist < _min_dist_ud[0]:
                        _nearest_ud[0] = {
                            "pixel": (x, y),
                            "color": pixel,
                            "command": self.color_code_up_down[pixel],
                            "distance": dist,
                        }
                        _min_dist_ud[0] = dist

        nearest       = [None]
        nearest_ud    = [None]
        min_dist      = [float('inf')]
        min_dist_ud   = [float('inf')]

        # Pass 1 — tight window (fast, upstream behaviour)
        xr_lo = max(0, x0 - base_range)
        xr_hi = min(w, x0 + base_range)
        yr_lo = max(0, y0 - base_range)
        yr_hi = min(h, y0 + base_range)
        _scan_region(xr_lo, yr_lo, xr_hi, yr_hi,
                     nearest, nearest_ud, min_dist, min_dist_ud)

        # Draw the tight search box on the route debug overlay regardless
        # of success (the user wants to SEE where the tracker thinks they
        # are, even when it's wrong).
        draw_rectangle(
            self.img_route_debug,
            (xr_lo, yr_lo),
            ((xr_hi - xr_lo), (yr_hi - yr_lo)),
            (0, 0, 255), "", text_height=0.4, thickness=1,
        )

        # Pass 2 — if nothing matched, scan the whole route image.  Also
        # emit a one-shot WARNING (with periodic throttled replays) so the
        # user can tell we're doing a slow scan.  Throttle at ~once per 5 s
        # to avoid spamming.
        used_fallback = False
        if nearest[0] is None and nearest_ud[0] is None:
            used_fallback = True
            _scan_region(0, 0, w, h,
                         nearest, nearest_ud, min_dist, min_dist_ud)
            now = time.time()
            _last = getattr(self, "_route_color_fallback_logged_at", -99999)
            if now - _last > 5.0:
                self._route_color_fallback_logged_at = now
                d1 = None if nearest[0]    is None else nearest[0]   ["distance"]
                d2 = None if nearest_ud[0] is None else nearest_ud[0]["distance"]
                n_px   = None if nearest[0]    is None else nearest[0]   ["pixel"]
                n_cmd  = None if nearest[0]    is None else nearest[0]   ["command"]
                n_col  = None if nearest[0]    is None else nearest[0]   ["color"]
                ud_px  = None if nearest_ud[0] is None else nearest_ud[0]["pixel"]
                ud_cmd = None if nearest_ud[0] is None else nearest_ud[0]["command"]
                ud_col = None if nearest_ud[0] is None else nearest_ud[0]["color"]
                screen_loc = getattr(self, "loc_player", None)
                loc_method  = getattr(self, "_last_loc_method", "?")
                logger.warning(
                    "[get_nearest_color_code] Tight search window "
                    f"(range={base_range} px) around player_global=({x0},{y0}) "
                    "found no route colour-code pixels.  Fallback: scanned "
                    f"full route map ({w}×{h}).  "
                    f"main: dist={d1} pixel={n_px} color={n_col} cmd={n_cmd!r}; "
                    f"ud: dist={d2} pixel={ud_px} color={ud_col} cmd={ud_cmd!r}.  "
                    f"screen_loc_player={screen_loc} loc_method={loc_method}.  "
                    "This usually means minimap→route template matching "
                    "failed; re-capturing minimaps/<map>/map.png from the "
                    "current client build will dramatically improve route "
                    "precision and lower CPU cost."
                )

        # Draw a green line from the current player_global estimate to
        # whichever colour-code pixel finally "won" this frame.  If we
        # used the fallback (full-map) scan, draw the line in MAGENTA so
        # the user can tell at a glance that the tracker is no longer
        # trusting the projected coordinate.
        if nearest[0] is not None:
            line_color = (255, 0, 255) if used_fallback else (0, 255, 0)
            cv2.line(self.img_route_debug,
                     self.loc_player_global, nearest[0]["pixel"],
                     line_color, 1)
            cv2.putText(
                self.img_frame_debug, f"Route Action: {nearest[0]['command']}",
                (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA,
            )
            cv2.putText(
                self.img_frame_debug, f"Route Index: {self.idx_routes}",
                (650, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA,
            )
        if nearest_ud[0] is not None:
            line_color = (255, 128, 255) if used_fallback else (0, 0, 255)
            cv2.putText(
                self.img_frame_debug, f"Route Action: {nearest_ud[0]['command']}",
                (650, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA,
            )
            cv2.line(self.img_route_debug,
                     self.loc_player_global, nearest_ud[0]["pixel"],
                     line_color, 1)

        return nearest[0], nearest_ud[0]  # if not found return none

    def get_attack_range(self, is_left=True):
        '''
        get_attack_range
        '''
        if self.cfg["bot"]["attack"] == "aoe_skill":
            dx = self.cfg["aoe_skill"]["range_x"] // 2
            dy = self.cfg["aoe_skill"]["range_y"] // 2
            x0 = max(0, self.loc_player[0] - dx)
            x1 = min(self.img_frame.shape[1], self.loc_player[0] + dx)
            y0 = max(0, self.loc_player[1] - dy)
            y1 = min(self.img_frame.shape[0], self.loc_player[1] + dy)

        elif self.cfg["bot"]["attack"] == "directional":
            if is_left:
                x0 = self.loc_player[0] - self.cfg["directional_attack"]["range_x"]
                x1 = self.loc_player[0]
            else:
                x0 = self.loc_player[0]
                x1 = x0 + self.cfg["directional_attack"]["range_x"]
            y0 = self.loc_player[1] - self.cfg["directional_attack"]["range_y"] // 2
            y1 = y0 + self.cfg["directional_attack"]["range_y"]
        else:
            raise RuntimeError(f"Unsupported attack mode: {self.cfg['bot']['attack']}")

        return (x0, y0, x1, y1)

    def get_nearest_monster(self, is_left=True):
        '''
        Finds the nearest monster within the player's attack range.

        This function:
        - Defines an attack box relative to the player position,
            depending on the facing direction (`is_left`).
        - Iterates through all detected monsters and checks which ones overlap
          with the attack box.
        - Returns the closest valid monster that meets the overlap criteria.

        Args:
            is_left (bool): If True, assume the player is facing left;
                            adjusts attack box accordingly.
        Returns:
            dict or None: The nearest monster's info dict, or None if no valid match.
        '''

        x0, y0, x1, y1 = self.get_attack_range(is_left=is_left)

        nearest_monster = None
        min_distance = float('inf')
        for monster in self.monsters:
            mx1, my1 = monster["position"]
            mw, mh = monster["size"]
            mx2 = mx1 + mw
            my2 = my1 + mh

            # Calculate intersection
            ix1 = max(x0, mx1)
            iy1 = max(y0, my1)
            ix2 = min(x1, mx2)
            iy2 = min(y1, my2)

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter_area = iw * ih

            min_mob_area = min(img.shape[0]*img.shape[1] for _, imgs in self.monsters_info.items() for img, _ in imgs)
            inter_area_thres = min(min_mob_area, self.cfg['monster_detect']['max_mob_area_trigger'])
            if inter_area >= inter_area_thres:
                # Compute distance to player center
                monster_center = (mx1 + mw // 2, my1 + mh // 2)
                dx = monster_center[0] - self.loc_player[0]
                dy = monster_center[1] - self.loc_player[1]
                distance = abs(dx) + abs(dy)  # Manhattan distance

                if distance < min_distance:
                    min_distance = distance
                    nearest_monster = monster

        return nearest_monster

    def get_monsters_in_range(self, top_left, bottom_right):
        '''
        get_monsters_in_range
        '''
        x0, y0 = top_left
        x1, y1 = bottom_right

        img_roi = self.img_frame[y0:y1, x0:x1]

        # Shift player's location into ROI coordinate system
        px, py = self.loc_player
        px_in_roi = px - x0
        py_in_roi = py - y0

        # Define rectangle range around player (in ROI coordinate)
        char_x_min = max(0, px_in_roi - self.cfg["character"]["width"] // 2)
        char_x_max = min(img_roi.shape[1], px_in_roi + self.cfg["character"]["width"] // 2)
        char_y_min = max(0, py_in_roi - self.cfg["character"]["height"] // 2)
        char_y_max = min(img_roi.shape[0], py_in_roi + self.cfg["character"]["height"] // 2)

        monsters = []
        for monster_name, monster_imgs in self.monsters_info.items():
            for img_monster, mask_monster in monster_imgs:
                if self.cfg["bot"]["mode"] == "patrol":
                    pass # Don't detect monster using template in patrol mode
                elif self.cfg["monster_detect"]["mode"] == "template_free":
                    # Generate mask where pixel is exactly (0,0,0)
                    black_mask = np.all(img_roi == [0, 0, 0], axis=2).astype(np.uint8) * 255
                    # cv2.imshow("Black Pixel Mask", black_mask)

                    # Zero out mask inside this region (ignore player's own character)
                    black_mask[char_y_min:char_y_max, char_x_min:char_x_max] = 0

                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
                    closed_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
                    # cv2.imshow("Black Mask", closed_mask)

                    # draw player character bounding box
                    draw_rectangle(
                        self.img_frame_debug, (char_x_min+x0, char_y_min+y0),
                        (self.cfg["character"]["height"], self.cfg["character"]["width"]),
                        (255, 0, 0), "Character Box"
                    )

                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)

                    monsters = []
                    min_area = 1000
                    for i in range(1, num_labels):
                        x, y, w, h, area = stats[i]
                        if area > min_area:
                            monsters.append({
                                "name": "",
                                "position": (x0+x, y0+y),
                                "size": (h, w),
                                "score": 1.0,
                            })
                elif self.cfg["monster_detect"]["mode"] == "contour_only":
                    # Use only black lines contour to detect monsters
                    # Create masks (already grayscale)
                    mask_pattern = np.all(img_monster == [0, 0, 0], axis=2).astype(np.uint8) * 255
                    mask_roi = np.all(img_roi == [0, 0, 0], axis=2).astype(np.uint8) * 255

                    # Zero out mask inside this region (ignore player's own character)
                    mask_roi[char_y_min:char_y_max, char_x_min:char_x_max] = 0

                    # Apply Gaussian blur (soften the masks)
                    blur = self.cfg["monster_detect"]["contour_blur"]
                    img_monster_blur = cv2.GaussianBlur(mask_pattern, (blur, blur), 0)
                    img_roi_blur = cv2.GaussianBlur(mask_roi, (blur, blur), 0)

                    # Check template vs ROI size before matching
                    h_roi, w_roi = img_roi_blur.shape[:2]
                    h_temp, w_temp = img_monster_blur.shape[:2]

                    if h_temp > h_roi or w_temp > w_roi:
                        return []  # template bigger than roi, skip this matching

                    # Perform template matching
                    res = cv2.matchTemplate(img_roi_blur, img_monster_blur, cv2.TM_SQDIFF_NORMED)

                    # Apply soft threshold
                    match_locations = np.where(res <= self.cfg["monster_detect"]["diff_thres"])

                    h, w = img_monster.shape[:2]
                    for pt in zip(*match_locations[::-1]):
                        monsters.append({
                            "name": monster_name,
                            "position": (pt[0] + x0, pt[1] + y0),
                            "size": (h, w),
                            "score": res[pt[1], pt[0]],
                        })
                elif self.cfg["monster_detect"]["mode"] == "grayscale":
                    img_monster_gray = cv2.cvtColor(img_monster, cv2.COLOR_BGR2GRAY)
                    img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
                    res = cv2.matchTemplate(
                            img_roi_gray,
                            img_monster_gray,
                            cv2.TM_SQDIFF_NORMED,
                            mask=mask_monster)
                    match_locations = np.where(res <= self.cfg["monster_detect"]["diff_thres"])
                    h, w = img_monster.shape[:2]
                    for pt in zip(*match_locations[::-1]):
                        monsters.append({
                            "name": monster_name,
                            "position": (pt[0] + x0, pt[1] + y0),
                            "size": (h, w),
                            "score": res[pt[1], pt[0]],
                    })
                elif self.cfg["monster_detect"]["mode"] == "color":
                    res = cv2.matchTemplate(
                            img_roi,
                            img_monster,
                            cv2.TM_SQDIFF_NORMED,
                            mask=mask_monster)
                    match_locations = np.where(res <= self.cfg["monster_detect"]["diff_thres"])
                    h, w = img_monster.shape[:2]
                    for pt in zip(*match_locations[::-1]):
                        monsters.append({
                            "name": monster_name,
                            "position": (pt[0] + x0, pt[1] + y0),
                            "size": (h, w),
                            "score": res[pt[1], pt[0]],
                    })
                else:
                    logger.error(f"Unexpected camera localization mode: {self.cfg['monster_detect']['mode']}")
                    return []

        # Apply Non-Maximum Suppression to monster detection
        monsters = nms(monsters, iou_threshold=0.4)

        # Detect monster via health bar
        if self.cfg["monster_detect"]["with_enemy_hp_bar"]:
            # Create color mask for Monsters' HP bar
            mask = cv2.inRange(img_roi,
                               np.array(self.cfg["monster_detect"]["hp_bar_color"]),
                               np.array(self.cfg["monster_detect"]["hp_bar_color"]))

            # Find connected components (each cluster of green pixels)
            num_labels, labels, stats, centroids = \
                cv2.connectedComponentsWithStats(mask, connectivity=8)

            for i in range(1, num_labels):  # skip background (label 0)
                x, y, w, h, area = stats[i]
                if area < 3:  # small noise filter
                    continue

                # Guess a monster bounding box
                y += 10
                x = max(0, x)
                y = max(0, y)
                w = 70
                h = min(img.shape[0] for _, imgs in self.monsters_info.items() for img, _ in imgs)

                monsters.append({
                    "name": "Health Bar",
                    "position": (x0 + x, y0 + y),
                    "size": (h, w),
                    "score": 1.0,
                })

        # Debug
        # Draw attack detection range
        draw_rectangle(
            self.img_frame_debug, (x0, y0), (y1-y0, x1-x0),
            (255, 0, 0), "Mob Detection Box"
        )

        # Draw monsters bounding box
        for monster in monsters:
            if monster["name"] == "Health Bar":
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

            draw_rectangle(
                self.img_frame_debug, monster["position"], monster["size"],
                color, str(round(monster['score'], 2))
            )

        return monsters

    def get_img_frame(self):
        '''
        get_img_frame

        Tolerant preprocessing pipeline:
          1. Capture a raw frame from the window capturor.
          2. Strip the title bar using cfg[game_window][title_bar_height].
          3. Validate the aspect ratio against 16:9 (tolerant via config).
          4. Always resize the valid frame to WINDOW_WORKING_SIZE so the rest
             of the CV pipeline runs on a canonical resolution.

        Previously the strict pixel-perfect check on game_window.size caused
        false negatives when the user's client had a different 16:9
        resolution.  Now any 16:9-like client size works out of the box;
        non-16:9 sizes are allowed to proceed with a warning (so the user is
        informed but the bot is not bricked).
        '''
        # Get window game raw frame
        self.frame = self.capture.get_frame()
        if self.frame is None:
            logger.warning("Failed to capture game frame.")
            return None

        # Cut the title bar (the capture API usually returns the full window
        # including title bar; we drop `title_bar_height` rows from the top).
        frame_no_title = self.frame[self.cfg["game_window"]["title_bar_height"]:, :]

        # Test-image path: skip size validation entirely (frame may come from
        # screenshots taken at arbitrary resolutions).
        if self.args.test_image != "":
            return cv2.resize(frame_no_title, WINDOW_WORKING_SIZE,
                              interpolation=cv2.INTER_NEAREST)

        h, w = frame_no_title.shape[:2]
        ratio = w / h if h > 0 else 0.0
        ratio_tolerance = self.cfg["game_window"].get("ratio_tolerance", 0.08)
        expected_16_9 = 16.0 / 9.0
        expected_hw = tuple(self.cfg["game_window"]["size"])  # (h, w)
        is_16_9 = abs(ratio - expected_16_9) <= ratio_tolerance
        is_exact_size = (h, w) == expected_hw

        if is_exact_size:
            # Perfect match — fast path, no warnings.
            pass
        elif is_16_9:
            # Different 16:9 resolution (e.g. 1920x1080 client, 1280x720
            # client, etc.). Resize to working size is still safe because all
            # downstream CV ops are scale-invariant.
            logger.info(
                f"[get_img_frame] Client size {w}x{h} (16:9 ✓) differs from "
                f"canonical {expected_hw[1]}x{expected_hw[0]}; will be auto-"
                f"resized to the working resolution {WINDOW_WORKING_SIZE[0]}x"
                f"{WINDOW_WORKING_SIZE[1]}."
            )
        else:
            # Non-16:9 ratio.  The algorithm still runs after resizing, but
            # UI-coordinate constants (HP/MP bar locations, button templates,
            # minimap search regions) may be wrong — so we print a clear
            # ERROR-looking warning but still return a frame instead of
            # crashing the whole bot.
            logger.warning(
                f"[get_img_frame] Client size {w}x{h} (ratio {ratio:.3f}) is "
                f"NOT 16:9 (expected {expected_16_9:.3f} ± tolerance "
                f"{ratio_tolerance:.3f}). Downstream CV (party button, HP bar, "
                f"rune templates) may misbehave.\n"
                f"  -> Fix: In the game client, switch to **Windowed mode** "
                f"and pick the smallest available 16:9 resolution "
                f"(typically 1296x759, client area ~1282x693)."
            )

        # Always resize to the canonical working resolution so the rest of
        # the pipeline stays consistent.
        return cv2.resize(frame_no_title, WINDOW_WORKING_SIZE,
                   interpolation=cv2.INTER_NEAREST)

    def is_player_stuck(self):
        """
        Checks whether the player is stuck (not moving).

        Two mutually complementary detection modes are used depending on how
        the player was located this frame:

        **Mode A — precise global map position (default)**
          When the minimap → route template match actually succeeded (i.e.
          ``_minimap_global_synth`` is False and the nametag / party-red-bar
          stage produced a trusted location), compare the current global
          position against ``loc_watch_dog`` exactly like the original
          implementation did.  Movement exceeding ``watchdog.range`` resets
          the timer; idle longer than ``watchdog.timeout`` flags the player
          as stuck.

        **Mode B — synthesized / fallback position (CN client common case)**
          When we are *not* running in precise mode (camera_center_fallback,
          nametag false-positive triggered, minimap template match failed
          and we fell back to a screen→route proportional projection, or
          ``_minimap_global_synth`` is True) the player's *projected* global
          position is effectively anchored to the camera — which in
          MapleStory follows the character, so ``loc_player_global`` looks
          static even when the character is running perfectly well.  Using
          the original distance-based check in this scenario causes
          100%-repeatable false positives every ``watchdog.timeout``
          seconds.

          Instead, we treat the keyboard commands *being emitted* as the
          ground truth for motion:
            * If ``cmd_move_x`` or ``cmd_move_y`` is actively non-``none``
              (i.e. the bot is trying to walk in some direction) we consider
              the character "moving enough" and reset the watchdog timer.
              This lets a direction command keep the watchdog happy for as
              long as the character keeps pressing LEFT/RIGHT/UP/DOWN, which
              is exactly what we want in fallback mode.
            * If commands sit at ``none`` for longer than
              ``watchdog.timeout`` then the character *really is* idle and
              we return True (stuck) just like in precise mode.

        Returns:
            bool: True if the player is stuck, False otherwise.
        """
        current_time = time.time()

        # ------------------------------------------------------------------
        # Decide which detection mode we're in.  Any of these triggers
        # switches us to the command-based (Mode B) check.
        # ------------------------------------------------------------------
        loc_method = getattr(self, "_last_loc_method", None) or ""
        synth_mode = bool(getattr(self, "_minimap_global_synth", False)) or \
                     loc_method in (
                        "camera_center_fallback",
                        "nametag_false_positive_fallback",
                     )

        if not synth_mode:
            # Mode A — precise global map motion (original behaviour)
            dx = abs(self.loc_player_global[0] - self.loc_watch_dog[0])
            dy = abs(self.loc_player_global[1] - self.loc_watch_dog[1])
            if dx + dy > self.cfg["watchdog"]["range"]:
                self.loc_watch_dog = self.loc_player_global
                self.t_watch_dog = current_time
                return False
        else:
            # Mode B — use active move commands as motion proxy.
            #
            # NOTE: We intentionally inspect ``self.kb.cmd_left_right`` and
            # ``self.kb.cmd_up_down`` here rather than the "desired"
            # commands produced by update_cmd_by_route, because these are
            # the values *actually currently being dispatched* by the
            # keyboard-controller thread.  If the KB thread holds a LEFT
            # key pressed we want to give it credit.
            cmd_lr = getattr(getattr(self, "kb", None), "cmd_left_right", "none") or "none"
            cmd_ud = getattr(getattr(self, "kb", None), "cmd_up_down",   "none") or "none"
            if cmd_lr not in ("none", "stop") or cmd_ud not in ("none", "stop"):
                # Bot is actively sending a direction -> not stuck.
                self.loc_watch_dog = self.loc_player_global
                self.t_watch_dog = current_time
                return False

        dt = current_time - self.t_watch_dog
        if dt > self.cfg["watchdog"]["timeout"]:
            # watch dog idle for too long, player stuck
            self.loc_watch_dog = self.loc_player_global
            self.t_watch_dog = current_time
            logger.warning(
                f"[is_player_stuck] Player stuck for {round(dt, 2)} seconds "
                f"(mode={'cmd-based fallback' if synth_mode else 'global-map'} "
                f"loc_method={loc_method!r})."
            )
            return True
        return False

    def screenshot_img_frame(self):
        '''
        Save self.img_frame
        '''
        if self.img_frame is None:
            logger.error("[screenshot_img_frame] Failed, game window is not available")
        else:
            screenshot(self.img_frame, "img_frame")

        if self.img_frame_debug is None:
            pass
        else:
            screenshot(self.img_frame_debug, "img_frame_debug")

        if self.frame is None:
            pass
        else:
            screenshot(self.frame, "frame")

    def is_near_edge(self):
        '''
        Detects whether the player is near a teleport edge region

        This function:
        - Defines a rectangular search region around the player's current global location.
        - Scans for pixels matching a specific edge teleport color code within the region.
        - If matching pixels are found, it computes the average X position of those pixels.
        - Compares that average to the player's X position to determine whether the edge is on the left or right.

        Returns:
            str: One of:
                - "edge on left"
                - "edge on right"
                - "" (empty string if no edge is detected nearby)
        '''
        x0, y0 = self.loc_player_global
        h, w = self.img_route.shape[:2]
        h_trigger_box = self.cfg["edge_teleport"]["trigger_box_height"]
        w_trigger_box = self.cfg["edge_teleport"]["trigger_box_width"]
        x_min = max(0, x0 - w_trigger_box//2)
        x_max = min(w, x0 + w_trigger_box//2)
        y_min = max(0, y0 - h_trigger_box//2)
        y_max = min(h, y0 + h_trigger_box//2)

        # Debug: draw search box
        # draw_rectangle(
        #     self.img_route_debug,
        #     (x_min, y_min),
        #     (y_max - y_min, x_max - x_min),
        #     (0, 0, 255), "Edge Check", thickness=1, text_height=0.4
        # )

        # Find mask of matching pixels
        roi = self.img_route[y_min:y_max, x_min:x_max]
        mask = np.all(roi == self.cfg["edge_teleport"]["color_code"], axis=2)
        coords = np.column_stack(np.where(mask))

        # No edge pixel
        if coords.size == 0:
            return ""

        # Calculate mean position of matching pixels
        mean_x = np.mean(coords[:, 1])

        # Compare to roi center
        if mean_x < x0:
            return "edge on left"
        else:
            return "edge on right"

    def update_info_on_img_frame_debug(self):
        '''
        update_info_on_img_frame_debug
        '''
        # Print text at bottom left corner
        self.fps = round(1.0 / (time.time() - self.t_last_frame))
        text_y_interval = 23
        text_y_start = 460
        dt_screenshot = time.time() - self.kb.t_last_screenshot
        h, w = self.frame.shape[:2]
        text_list = [
            f"FPS: {self.fps}",
            f"State: {self.fsm.state.name}",
            f"Resolution: {h}x{w}, Ratio: {round(w/h, 2)}",
            f"Press 'F1' to {'pause' if self.kb.is_enable else 'start'} Bot",
            f"Press 'F2' to save screenshot{' : Saved' if dt_screenshot < 0.7 else ''}",
             "Press 'F12' to quit"]
        for idx, text in enumerate(text_list):
            cv2.putText(
                self.img_frame_debug, text,
                (10, text_y_start + text_y_interval*idx),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA
            )

        # Draw attack box on debug window
        if self.cfg["bot"]["attack"] == "aoe_skill":
            x0, y0, x1, y1 = self.get_attack_range()
            draw_rectangle(
                self.img_frame_debug, (x0, y0),
                (y1-y0, x1-x0),
                (0, 0, 255), "Attack Range"
            )
        elif self.cfg["bot"]["attack"] == "directional":
            x0, y0, x1, y1 = self.get_attack_range(is_left=True)
            draw_rectangle(
                self.img_frame_debug, (x0, y0),
                (y1-y0, x1-x0),
                (0, 0, 255), "Attack Range(Left)"
            )
            x0, y0, x1, y1 = self.get_attack_range(is_left=False)
            draw_rectangle(
                self.img_frame_debug, (x0, y0),
                (y1-y0, x1-x0),
                (0, 0, 255), "Attack Range(Right)"
            )

        # Draw minimap rectangle on img debug
        draw_rectangle(
            self.img_frame_debug,
            self.loc_minimap,
            self.img_minimap.shape[:2],
            (0, 0, 255), "minimap",thickness=2
        )

        # Don't draw minimap in patrol mode
        if self.cfg["bot"]["mode"] in ["patrol", "aux"]:
            return

        # Compute crop region with boundary check
        crop_w, crop_h = 80, 80
        x0 = max(0, self.loc_player_global[0] - crop_w // 2)
        y0 = max(0, self.loc_player_global[1] - crop_h // 2)
        x1 = min(self.img_route_debug.shape[1], x0 + crop_w)
        y1 = min(self.img_route_debug.shape[0], y0 + crop_h)

        # Check if valid crop region
        if x1 <= x0 or y1 <= y0:
            return

        # Crop region
        mini_map_crop = self.img_route_debug[y0:y1, x0:x1]
        mini_map_crop = cv2.resize(mini_map_crop,
                                (int(mini_map_crop.shape[1] * 3),
                                 int(mini_map_crop.shape[0] * 3)),
                                interpolation=cv2.INTER_NEAREST)
        # Paste into top-right corner of self.img_frame_debug
        h_crop, w_crop = mini_map_crop.shape[:2]
        h_frame, w_frame = self.img_frame_debug.shape[:2]
        x_paste = w_frame - w_crop - 10  # 10px margin from right
        y_paste = 10
        self.img_frame_debug[y_paste:y_paste + h_crop, x_paste:x_paste + w_crop] = mini_map_crop

        # Draw border around minimap
        cv2.rectangle(
            self.img_frame_debug,
            (x_paste, y_paste),
            (x_paste + w_crop, y_paste + h_crop),
            color=(255, 255, 255),   # White border
            thickness=2
        )

        # Draw HP/MP/EXP bar on debug window
        percent_bars = [self.health_monitor.hp_percent,
                      self.health_monitor.mp_percent,
                      self.health_monitor.exp_percent]
        for i, bar_name in enumerate(["HP", "MP", "EXP"]):
            x_s, y_s = (250, 30)
            # Print bar ratio on debug window
            cv2.putText(self.img_frame_debug,
                        f"{bar_name}: {percent_bars[i]:.1f}%",
                        (x_s, y_s + 30*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            # Draw bar on debug window
            x_s, y_s = (410, 13)
            x, y, w, h = self.health_monitor.loc_size_bars[i]
            self.img_frame_debug[y_s+30*i:y_s+h+30*i, x_s:x_s+w] = \
                self.img_frame[self.cfg["ui_coords"]["ui_y_start"]:, :][y:y+h, x:x+w]

        # ================================================================
        # Debug viz overlay — MOB DETECTION SUMMARY (always drawn).
        # Users often report "I don't see any detection boxes so the
        # detector mustn't be running", but the detector was running and
        # just returned N=0 inside a too-small search box.  We fix that
        # perception gap by always printing a single coloured line on
        # the debug canvas that summarises the current detect state.
        #   - GREEN  : BOX_count>0 → detector working, mobs found.
        #   - YELLOW : BOX_count=0 but FULL_count>0 → "mobs on screen
        #              but outside the narrow attack box".
        #   - PURPLE : BOX_count=0, FULL_count=0, grayscale_count>0 →
        #              default detect mode missed them; grayscale caught
        #              them.
        #   - RED    : everything 0 → detector really is blind to this
        #              client's sprites.  Switch to test mode.
        # ================================================================
        def _draw_mob_status_line(text, color_bgr):
            try:
                if self.img_frame_debug is None: return
                overlay = self.img_frame_debug
                # Draw the line at y≈20 (below route text which is at ~40)
                cv2.putText(overlay, text, (8, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2)
            except Exception:
                pass
        try:
            _status_box = len(self.monsters) if hasattr(self, 'monsters') else -1
            _status_counters = getattr(self, "_last_mob_full_counters", None) or {}
            full_count = _status_counters.get("full", -1)
            grayscale_count = _status_counters.get("grayscale", -1)
            if _status_box > 0:
                color_box = (0, 220, 0)   # green
                tag = f"BOX_mobs={_status_box} attack-ready"
            elif full_count > 0:
                color_box = (0, 220, 220)  # yellow
                tag = (f"BOX=0 FULL={full_count} "
                       f"({'used_fallback_box' if _status_counters.get('used_fb') else 'pending_move'})")
            elif grayscale_count > 0:
                color_box = (220, 0, 220)  # purple
                tag = f"BOX=0 FULL=0 GRAYSCALE={grayscale_count} (mode mismatch)"
            else:
                color_box = (0, 0, 220)   # red
                tag = "BOX=0 FULL=0 → detector blind; press F2 for debug crop"
            detect_mode = "?"
            try:
                detect_mode = str(self.cfg["monster_detect"].get("mode", "?"))
            except Exception:
                pass
            _draw_mob_status_line(
                f"DETECT[{detect_mode}]: {tag}",
                color_box,
            )
            # Also print to log at ~1 Hz so users who have cv2 windows
            # minimised still see the status.
            cls = type(self)
            if not hasattr(cls, "_viz_mob_status_t"):
                cls._viz_mob_status_t = 0.0
            _now = time.time()
            if _now - cls._viz_mob_status_t >= 1.0:
                cls._viz_mob_status_t = _now
                logger.info(
                    "[DETECTOR_VIZ] "
                    f"mode={detect_mode!r} BOX={_status_box} FULL={full_count} "
                    f"GRAYSCALE={grayscale_count} debug_canvas="
                    f"{'ready' if self.img_frame_debug is not None else 'NONE'} "
                    f"emit_to_cv2={getattr(self,'_should_emit_debug_to_cv2', None)}"
                )
        except Exception:
            pass

    def update_img_frame_debug(self):
        '''
        update_img_frame_debug

        Draws the Game Window Debug CV window.  VIZ FIX: honours the new
        `_should_emit_debug_to_cv2` flag (which respects user explicit
        disable_viz) while still letting F2 screenshots succeed.
        '''
        # Always print command on screen first (even if we're not emitting
        # to cv2 windows, the F2 screenshot needs this info visible).
        try:
            if self.img_frame_debug is not None:
                cv2.putText(self.img_frame_debug,
                            f"Cmd: {self.cmd_move_x} {self.cmd_move_y} {self.cmd_action}",
                            (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        except Exception:
            pass
        # Only actually create/show cv2 windows if the user left viz ON.
        if getattr(self, '_should_emit_debug_to_cv2',
                   bool(self.is_show_debug_window)):
            try:
                cv2.imshow("Game Window Debug",
                    self.img_frame_debug[:self.cfg["ui_coords"]["ui_y_start"], :])
            except Exception:
                pass
            # Update FPS timer
            self.t_last_frame = time.time()

    def ensure_is_in_party(self):
        '''
        ensure_is_in_party

        Press the ``party`` hotkey to open the party window, then try to click
        the "Create Party" button if it is visible.

        Robustness notes:
          * If the "create" button is not found, instead of blindly assuming
            "we're already in party" (the old behaviour that silently masked
            missing-party-red-bar bugs), we additionally try to look for the
            **disabled** state of the same button — if neither is found we
            print a diagnostic warning so users can either create a party
            manually, adjust the create_party_button_thres, or provide a
            region-correct button template under ``misc/``.
        '''
        # open party window
        press_key(self.cfg["key"]["party"])

        # Wait party window to show up
        time.sleep(0.5)

        # Update image frame
        self.img_frame = self.get_img_frame()

        if self.img_frame is None:
            logger.warning(
                "[ensure_is_in_party] Skipped: game frame is unavailable "
                "(wrong window size / capture not ready). Party status is "
                "unchanged; the bot will retry on next frame if needed."
            )
            # Close the party window anyway so the UI doesn't block gameplay.
            press_key(self.cfg["key"]["party"])
            return

        lang = self.cfg["system"]["language"]
        thres = self.cfg['party_red_bar'][f'create_party_button_{lang}_thres']

        # Find the 'create party' button (both states: enabled + disabled)
        loc_enable, score_enable, _ = find_pattern_sqdiff(
                        self.img_frame, self.img_create_party_enable)
        loc_disable, score_disable, _ = find_pattern_sqdiff(
                        self.img_frame, self.img_create_party_disable)
        found_enable  = score_enable  < thres
        found_disable = score_disable < thres

        if found_enable and score_enable <= score_disable:
            logger.info(f"[ensure_is_in_party] Find party enable button({round(score_enable, 2)}), creating party...")
            h, w = self.img_create_party_enable.shape[:2]
            click_in_game_window(self.capture.window_title,
                (loc_enable[0] + w // 2,
                 loc_enable[1] + h // 2 + self.cfg['game_window']['title_bar_height'])
            )
            # Give the server ~1.2 s to respond and create the party red bar.
            # It's OK if we slightly delay startup; without this sleep the
            # very next frame often runs red-bar detection before the bar
            # was actually rendered on the client.
            time.sleep(1.2)
        elif found_disable:
            logger.info(
                f"[ensure_is_in_party] 'Create party' button looks disabled "
                f"(score {round(score_disable, 2)}). Assuming the character "
                "is already in a party; continuing."
            )
        else:
            # Neither match → cannot definitively tell if we're in a party.
            # Best-effort: try clicking at the enable location anyway (as a
            # soft fallback) and emit a clear diagnostic pointing the user to
            # manual party creation if the red bar still doesn't show up.
            logger.warning(
                "[ensure_is_in_party] Neither 'Create Party' button template "
                f"matched.  enabled_template_score={round(score_enable, 3)} "
                f"(threshold <{thres}), disabled_template_score="
                f"{round(score_disable, 3)}.\n"
                "  -> Likely cause: button template under misc/ is for a "
                "different server / resolution, or language setting "
                f"(system.language={lang!r}) does not match the client.\n"
                "  -> Quick fix: PRESS THE PARTY HOTKEY MANUALY IN-GAME "
                "('P' by default) and click 'Create Party' by hand.  Once "
                "the party red HP bar appears above your character, the "
                "bot will be able to locate you and start hunting on the "
                "next restart."
            )
            # Fallback click-at-enable-location (harmless if there's nothing)
            if score_enable < 0.25:
                h, w = self.img_create_party_enable.shape[:2]
                logger.info(
                    "[ensure_is_in_party] Fallback: clicking near best "
                    f"'create party' candidate (score {round(score_enable, 2)})."
                )
                click_in_game_window(self.capture.window_title,
                    (loc_enable[0] + w // 2,
                     loc_enable[1] + h // 2 + self.cfg['game_window']['title_bar_height'])
                )
                time.sleep(1.2)

        # close party window
        press_key(self.cfg["key"]["party"])
        time.sleep(0.2)

    def channel_change(self):
        '''
        channel_change
        '''
        logger.info("[channel_change] Start")

        window_title = self.capture.window_title
        ui_coords = self.cfg["ui_coords"]
        click_in_game_window(window_title, ui_coords["menu"])
        time.sleep(1)
        click_in_game_window(window_title, ui_coords["channel"])
        time.sleep(1)
        click_in_game_window(window_title, ui_coords["random_channel"])
        time.sleep(1)
        click_in_game_window(window_title, ui_coords["random_channel_confirm"])
        time.sleep(1)

        loc_login_button = None
        while loc_login_button is None and not self.is_terminated:
            try:
                self.img_frame = self.get_img_frame()
                loc_login_button = self.get_login_button_location()
                if loc_login_button is None:
                    logger.info("Waiting for login button to show up...")
            except Exception as e:
                logger.warning(f"Exception occurred while waiting for login button: {e}")
                if not is_mac():
                    resize_window(window_title, width=1296, height=759)
                logger.info("Retrying login button detection...")

            time.sleep(3)
        logger.info(f"login_button button found: {loc_login_button}")

        time.sleep(3)  # wait the screen to be brighter

        # Click login button
        click_in_game_window(window_title, loc_login_button)
        time.sleep(2)

        # Click "Select Character"
        click_in_game_window(window_title, ui_coords["select_character"])
        time.sleep(5)

        self.kb.enable()
        self.kb.set_command("none none none")
        self.kb.release_all_key()

        self.ensure_is_in_party() # Make sure player is in party

        self.fsm.set_init_state("hunting")
        self.t_last_attack = time.time() # Update timer

    def terminate_threads(self):
        '''
        terminate all threads
        '''
        # Terminate keyboard controller
        if self.kb is not None:
            self.kb.is_terminated = True
        # Terminate game window capturor
        if self.capture is not None:
            self.capture.stop()
        # Terminate health monitor
        if self.health_monitor is not None:
            self.health_monitor.stop()
        self.is_terminated = True
        logger.info(f"[terminate_threads] Terminated all threads")

    def get_attack_direction(self, monster_left, monster_right):
        '''
        get_attack_direction
        '''
        # Compute distance for left
        distance_left = float('inf')
        if monster_left is not None:
            mx, my = monster_left["position"]
            mw, mh = monster_left["size"]
            center_left = (mx + mw // 2, my + mh // 2)
            distance_left = abs(center_left[0] - self.loc_player[0]) + \
                            abs(center_left[1] - self.loc_player[1])
        # Compute distance for right
        distance_right = float('inf')
        if monster_right is not None:
            mx, my = monster_right["position"]
            mw, mh = monster_right["size"]
            center_right = (mx + mw // 2, my + mh // 2)
            distance_right = abs(center_right[0] - self.loc_player[0]) + \
                            abs(center_right[1] - self.loc_player[1])
        # Choose attack direction and nearest monster
        attack_direction = None
        # nearest_monster = None

        # Additional validation: check if monster is actually on the correct side
        def is_monster_on_correct_side(monster, direction):
            if monster is None:
                return False
            mx, my = monster["position"]
            mw, mh = monster["size"]
            monster_center_x = mx + mw // 2
            player_x = self.loc_player[0]

            if direction == "left":
                return monster_center_x < player_x  # Monster should be left of player
            else:  # direction == "right"
                return monster_center_x > player_x  # Monster should be right of player

        # Only choose direction if there's a clear winner and monster is on correct side
        if monster_left is not None and monster_right is None and \
            is_monster_on_correct_side(monster_left, "left"):
            attack_direction = "left"
            # nearest_monster = monster_left
        elif monster_right is not None and monster_left is None and \
            is_monster_on_correct_side(monster_right, "right"):
            attack_direction = "right"
            # nearest_monster = monster_right
        elif monster_left is not None and monster_right is not None:
            # Both sides have monsters, check distance and side validation
            left_valid = is_monster_on_correct_side(monster_left, "left")
            right_valid = is_monster_on_correct_side(monster_right, "right")

            if left_valid and not right_valid:
                attack_direction = "left"
                # nearest_monster = monster_left
            elif right_valid and not left_valid:
                attack_direction = "right"
                # nearest_monster = monster_right
            elif left_valid and right_valid and distance_left < distance_right - 50:
                attack_direction = "left"
                # nearest_monster = monster_left
            elif left_valid and right_valid and distance_right < distance_left - 50:
                attack_direction = "right"
                # nearest_monster = monster_right
            # If both valid but distances too close, don't attack to avoid confusion

        # Debug attack direction selection
        if monster_left is not None or monster_right is not None:
            left_side_ok = is_monster_on_correct_side(monster_left, "left") if monster_left else False
            right_side_ok = is_monster_on_correct_side(monster_right, "right") if monster_right else False
            debug_text = f"L:{distance_left:.0f}({left_side_ok}) R:{distance_right:.0f}({right_side_ok}) Dir:{attack_direction}"
            cv2.putText(self.img_frame_debug, debug_text,
                        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        return attack_direction

    def is_need_change_channel(self, loc_other_players):
        '''
        is_need_change_channel
        '''
        # Calculate center value
        xs = [x for (x, _) in loc_other_players]
        ys = [y for (_, y) in loc_other_players]
        if len(xs) == 0 or len(ys) == 0:
            return False
        center_x, center_y = (np.mean(xs), np.mean(ys))
        if np.isnan(center_x) or np.isnan(center_y):
            return False
        center = (int(np.mean(xs)), int(np.mean(ys)))
        #logger.info(f"[is_need_change_channel] Center of mass = {center}")

        # Change channel
        mode = self.cfg["channel_change"]["mode"]
        if mode == "true":
            logger.warning("[is_need_change_channel] Player detected, immediately change channel.")
            return True
        elif mode == "pixel":
            if self.red_dot_center_prev is None:
                self.red_dot_center_prev = center
            else:
                dx = abs(center[0] - self.red_dot_center_prev[0])
                dy = abs(center[1] - self.red_dot_center_prev[1])
                total = dx + dy
                logger.debug(f"[is_need_change_channel] Movement dx={dx}, dy={dy}, total={total}")
                thres = self.cfg["channel_change"]["other_player_move_thres"]
                if total > thres:
                    logger.warning(f"Other player movement > {thres} pixel detected. "
                                "Trigger channel change.")
                    return True
        else:
            logger.error(f"[is_need_change_channel] Unsupported mode: {mode}")

        return False

    def is_time_to_change_channel(self):
        '''
        is_time_to_change_channel
        '''
        if not self.cfg["scheduled_channel_switching"]["enable"]:
            return False
        dt = time.time() - self.t_to_change_channel
        if dt > self.cfg["scheduled_channel_switching"]["interval_seconds"]:
            self.t_to_change_channel = time.time()
            return True
        return False

    def get_login_button_location(self):
        '''
        get_login_button_location
        '''
        # Extract the region where the login button should appear
        x0, y0 = self.cfg["ui_coords"]["login_button_top_left"]
        x1, y1 = self.cfg["ui_coords"]["login_button_bottom_right"]
        img_roi = self.img_frame[y0:y1, x0:x1]

        # Draw rectange on debug image
        draw_rectangle(self.img_frame_debug, (x0, y0),
                       (y1-y0, x1-x0), (0, 255, 0), "login_button box")

        # Find the 'login' button
        loc, score, _ = find_pattern_sqdiff(
                        img_roi, self.img_login_button)
        if score < self.cfg["ui_coords"]["login_button_thres"]:
            h, w = self.img_login_button.shape[:2]
            logger.info(f"[get_login_button_location] Found login button with score({score})")
            return (x0 + loc[0] + w // 2,
                    y0 + loc[1] + h // 2 + self.cfg['game_window']['title_bar_height'])
        else:
            return None

    def update_cmd_by_route(self):
        # get color code from img_route
        color_code, color_code_up_down = self.get_nearest_color_code()
        # Use color_code and color_code_up_down to complement each other
        # To prevent character stuck at the end of ladder, we use two color color pixels
        # and let them complement with each other, to ensure smoothy ladder climbing
        if color_code and color_code_up_down:
            if color_code["distance"] < color_code_up_down["distance"]:
                self.cmd_move_x, self.cmd_move_y, self.cmd_action = color_code["command"].split()
                _, cmd, _ = color_code_up_down["command"].split()
                if self.cmd_move_y == "none" and self.is_on_ladder:
                    self.cmd_move_y = cmd # only complement cmd_move_y when player is on ladder
            else:
                self.cmd_move_x, self.cmd_move_y, self.cmd_action = color_code_up_down["command"].split()
                cmd, _, _ = color_code["command"].split()
                if self.cmd_move_x == "none" and self.is_on_ladder:
                    self.cmd_move_x = cmd # only complement cmd_move_x when player is on ladder
        elif color_code:
            self.cmd_move_x, self.cmd_move_y, self.cmd_action = color_code["command"].split()
        elif color_code_up_down:
            self.cmd_move_x, self.cmd_move_y, self.cmd_action = color_code_up_down["command"].split()
        else:
            # ------------------------------------------------------------------
            # No route colour-code was found.  This is extremely rare now that
            # get_nearest_color_code falls back to a full-route scan, but if
            # it DOES happen (empty route image, wrong map selected, some
            # other bug) we must not leave cmd_move_* at their previous
            # values — otherwise the bot walks in one direction forever or
            # sits still forever.
            #
            # Emergency left/right patrol: toggle direction every 3 s so we
            # at least produce motion and give mob-detection + random-rescue
            # a chance to kick in.
            now = time.time()
            if now - self._patrol_dir_toggled_at > 3.0:
                self._patrol_dir_toggled_at = now
                self._patrol_current = "left" if self._patrol_current == "right" \
                                                  else "right"
            self.cmd_move_x = self._patrol_current
            self.cmd_move_y = "none"
            self.cmd_action = "none"
            _last = getattr(self, "_route_empty_patrol_logged_at", -99999)
            if now - _last > 5.0:
                self._route_empty_patrol_logged_at = now
                logger.warning(
                    "[update_cmd_by_route] get_nearest_color_code returned "
                    "no match after both tight-window and full-route scans. "
                    f"Falling back to emergency {self._patrol_current} patrol "
                    "so the character keeps moving.  This usually indicates "
                    "minimaps/<map>/route_*.png is empty or contains no "
                    "colour-code pixels for the configured color_code map."
                )

        # teleport away from edge to avoid falling off cliff
        if self.is_near_edge() and \
            time.time() - self.t_last_teleport > self.cfg["teleport"]["cooldown"]:
            self.cmd_action = "teleport"
            self.t_last_teleport = time.time() # update timer

        # Use teleport while walking
        if self.cfg['teleport']['is_use_teleport_to_walk'] and \
            time.time() - self.t_last_teleport > self.cfg['teleport']['cooldown']:
            self.cmd_action = "teleport"
            self.t_last_teleport = time.time() # update timer

        # replace teleport to jump if user doesn't set teleport key
        if self.cfg["key"]["teleport"] == "" and self.cmd_action == "teleport":
            self.cmd_action = "jump"

        # ------------------------------------------------------------------
        # Rate-limited (every 3 s) diagnostic dump so the user can confirm
        # that the route-tracker is actually producing *something* even when
        # the synthesis fallback is on.  Logs: *committed* move commands
        # (after all the if/elif/else branches above have run) + which
        # colour-code pixel produced them (if any) + current loc_method.
        # NOTE: this is intentionally placed at the END of the function so
        # the printed cmd_move_x/y/action values reflect the commands we
        # are about to send to KeyBoardController this frame, not the stale
        # values from the previous frame.
        now = time.time()
        _dbg_last = getattr(self, "_route_cmd_dbg_logged_at", -99999)
        if now - _dbg_last > 3.0:
            self._route_cmd_dbg_logged_at = now
            def _summ(c):
                if c is None: return "None"
                return (f"d={c.get('distance')} px={c.get('pixel')} "
                        f"col={c.get('color')} cmd={c.get('command')!r}")
            _loc_m = getattr(self, "_last_loc_method", "?")
            _lpg = getattr(self, "loc_player_global", None)
            _lp  = getattr(self, "loc_player", None)
            logger.info(
                "[update_cmd_by_route] DIAG: "
                f"cmd_move_x={self.cmd_move_x!r} cmd_move_y={self.cmd_move_y!r} "
                f"cmd_action={self.cmd_action!r} | "
                f"color={_summ(color_code)} | color_ud={_summ(color_code_up_down)} | "
                f"loc_method={_loc_m} loc_player_global={_lpg} screen_loc_player={_lp}"
            )

    def update_cmd_by_mob_detection(self):
        # Get monster search box
        margin = self.cfg["monster_detect"]["search_box_margin"]
        if self.cfg["bot"]["attack"] == "aoe_skill":
            dx = self.cfg["aoe_skill"]["range_x"] // 2 + margin
            dy = self.cfg["aoe_skill"]["range_y"] // 2 + margin
            cooldown = self.cfg["aoe_skill"]["cooldown"]
        elif self.cfg["bot"]["attack"] == "directional":
            dx = self.cfg["directional_attack"]["range_x"] + margin
            dy = self.cfg["directional_attack"]["range_y"] + margin
            cooldown = self.cfg["directional_attack"]["cooldown"]
        else:
            raise RuntimeError(f"Unsupported attack mode: {self.cfg['bot']['attack']}")
        x0 = max(0                      , self.loc_player[0] - dx)
        x1 = min(self.img_frame.shape[1], self.loc_player[0] + dx)
        y0 = max(0                      , self.loc_player[1] - dy)
        y1 = min(self.img_frame.shape[0], self.loc_player[1] + dy)

        # Get monsters in the search box
        self.monsters = self.get_monsters_in_range((x0, y0), (x1, y1))

        # Check if no mob to attack
        if len(self.monsters) == 0:
            # ------------------------------------------------------------
            # ATK MISS#1 early-return diagnostic (0.5 Hz, LRU-cached last
            # reason).  Prints the exact parameters that caused the empty
            # result so the user can distinguish:
            #   - search box too small (wh < 300x200 and dx/dy < 250)
            #   - player location was off (camera_center_fallback projecting
            #     player outside playfield)
            #   - full-screen scan ALSO returned 0 → contour/grayscale/
            #     color detect mode template/threshold completely wrong for
            #     the current client sprite.
            # ------------------------------------------------------------
            try:
                cls = type(self)
                last_dbg_empty = getattr(cls, "_at_mob_dbg_empty_t", 0.0)
                last_full_scan = getattr(cls, "_at_mob_dbg_full_t", 0.0)
                now = time.time()
                box_str = (f"[{x0:.0f},{y0:.0f}→{x1:.0f},{y1:.0f} "
                           f"wh={x1-x0:.0f}x{y1-y0:.0f}]"
                           if isinstance(x0,(int,float)) else "?")
                # Run a one-time full-frame scan every ~4 seconds to
                # decouple "search box too small" from "detect template
                # totally mismatches".
                full_wh = None
                full_count = -1
                full_samples = None
                all_mobs = None
                grayscale_count = -1
                grayscale_samples = None
                if now - last_full_scan >= 4.0:
                    cls._at_mob_dbg_full_t = now
                    try:
                        H, W = self.img_frame.shape[:2]
                        full_wh = (W, H)
                        all_mobs = self.get_monsters_in_range((0, 0), (W, H))
                        full_count = len(all_mobs)
                        if full_count:
                            full_samples = [(m.get("name","?"),
                                             m.get("position",(0,0)),
                                             m.get("size",(0,0)))
                                            for m in all_mobs[:3]]
                        # --------------------------------------------------
                        # ATK MISS#3: when full-count is still 0 (contour_only
                        # or default mode totally mismatches the Chinese client's
                        # mushroom sprites), temporarily fall back once to
                        # grayscale detect mode (usually a bit more tolerant)
                        # and compare counts (record so diagnostics tell us if we
                        # should permanently swap modes entirely.
                        # --------------------------------------------------
                        if full_count == 0:
                            try:
                                try:
                                    orig_mode = self.cfg["monster_detect"]["mode"]
                                    _saved_th = self.cfg["monster_detect"].get("diff_thres", 0.8)
                                    self.cfg["monster_detect"]["mode"] = "grayscale"
                                    self.cfg["monster_detect"]["diff_thres"] = max(0.85, _saved_th)
                                    try:
                                        mobs_gray = self.get_monsters_in_range((0, 0), (W, H))
                                        grayscale_count = len(mobs_gray)
                                        if grayscale_count > 0:
                                            grayscale_samples = [(m.get("name","?"),
                                                                  m.get("position",(0,0)),
                                                                  m.get("size",(0,0)))
                                                                 for m in mobs_gray[:3]]
                                    finally:
                                        self.cfg["monster_detect"]["mode"] = orig_mode
                                        self.cfg["monster_detect"]["diff_thres"] = _saved_th
                                except Exception:
                                    grayscale_count = -2
                            except Exception:
                                pass
                    except Exception:
                        pass
                # ------------------------------------------------------------
                # ATK MISS#2 auto fallback: whenever BOX_count == 0 but a recent
                # FULL_count > 0, treat the full-frame list as the current
                # monsters list and skip the early return.  This lets the bot
                # attack even if the monster is far outside the default
                # directional_attack range (e.g. a freshly spawned at the
                # other end of a long platform) — get_nearest_monster
                # below will correctly pick the closest one and issue a move
                # command *towards* the mob first, then attack when in range.
                # ------------------------------------------------------------
                _used_fallback_box = False
                if full_count > 0 and all_mobs is not None:
                    try:
                        self.monsters = all_mobs
                        _used_fallback_box = True
                    except Exception:
                        pass
                elif grayscale_count > 0:
                    # grayscale fallback found mobs when contour/default didn't
                    try:
                        # Re-run once with grayscale mode *this frame* so
                        # self.monsters is populated.  We re-run because
                        # the cached `mobs_gray` list may be stale (we
                        # sampled full-scan only every 4s; the actual list is for
                        # the current frame here).
                        H, W = self.img_frame.shape[:2]
                        orig_mode = self.cfg["monster_detect"]["mode"]
                        orig_th   = self.cfg["monster_detect"].get("diff_thres", 0.8)
                        try:
                            self.cfg["monster_detect"]["mode"] = "grayscale"
                            self.cfg["monster_detect"]["diff_thres"] = max(0.85, orig_th)
                            _m = self.get_monsters_in_range((0, 0), (W, H))
                            self.monsters = _m
                            _used_fallback_box = True
                        finally:
                            self.cfg["monster_detect"]["mode"] = orig_mode
                            self.cfg["monster_detect"]["diff_thres"] = orig_th
                    except Exception:
                        pass

                        # ------------------------------------------------------------
                if now - last_dbg_empty >= 2.0:
                    cls._at_mob_dbg_empty_t = now
                    try:
                        md_mode = self.cfg["monster_detect"]["mode"]
                    except Exception:
                        md_mode = "?"
                    try:
                        a_rng = (self.cfg["directional_attack"].get("range_x","?"),
                                 self.cfg["directional_attack"].get("range_y","?"))
                    except Exception:
                        a_rng = ("?","?")
                    try:
                        img_shape = tuple(getattr(self.img_frame, "shape", (0,0,0))[:2])
                    except Exception:
                        img_shape = (0,0)
                    extra = []
                    if _used_fallback_box:
                        extra.append("used_fallback_box=yes")
                    # VIZ: cache FULL/GRAYSCALE counters on the class so
                    # the on-screen status line (drawn by update_info_on_img
                    # can display them for users who only look at the cv2
                    # window / F2 screenshots.
                    try:
                        try:
                            _bw = int(x1-x0) if isinstance(x0,(int,float)) else None
                            _bh = int(y1-y0) if isinstance(y0,(int,float)) else None
                        except Exception:
                            _bw, _bh = None, None
                        cached = {"full": full_count,
                                  "grayscale": grayscale_count,
                                  "used_fb": bool(_used_fallback_box),
                                  "box": 0,
                                  "mode": md_mode,
                                  "box_wh": (_bw, _bh),
                                  "player": tuple(int(x) for x in getattr(self,'loc_player',(0,0))),
                        }
                        self._last_mob_full_counters = cached
                    except Exception:
                        self._last_mob_full_counters = {"full": full_count,
                                                         "grayscale": grayscale_count,
                                                         "used_fb": False,
                                                         "box": 0}
                    logger.info(
                        "[update_cmd_by_mob_detection] DIAG_EMPTY: "
                        f"mode={self.cfg['bot']['attack']!r} "
                        f"detect_mode={md_mode!r} "
                        f"margin={margin!r} "
                        f"directional_attack.range_xy={a_rng!r} "
                        f"search_box={box_str} "
                        f"img_frame_wh={(img_shape[1],img_shape[0])} "
                        f"player_loc={tuple(int(x) for x in getattr(self,'loc_player',(0,0)))} "
                        f"BOX_count=0 "
                        + (f"FULL_count={full_count} FULL_wh={full_wh} "
                           f"FULL_samples={full_samples!r} "
                           if full_count >= 0 else "")
                        + (f"grayscale_count={grayscale_count} "
                           f"grayscale_samples={grayscale_samples!r} "
                           if grayscale_count >= 0 else "")
                        + (" ".join(extra) if extra else "")
                    )
            except Exception:
                pass
            # If neither fallback fired, we still have len(self.monsters)==0
            # → keep the original behaviour (skip attack this frame).
            if len(self.monsters) == 0:
                return

        # Update attack command
        attack_direction_str = None
        nearest_info = None
        if self.cfg["bot"]["attack"] == "aoe_skill":
            # Summarise nearest monster (only useful for DIAG, not actual
            # attack command since AoE ignores direction)
            if self.monsters:
                nearest_info = (self.monsters[0].get("name","?"),
                                self.monsters[0].get("position", (0,0)))
            if time.time() - self.t_last_attack > cooldown:
                self.cmd_action = "attack"
                self.t_last_attack = time.time()

        elif self.cfg["bot"]["attack"] == "directional":
            # Get nearest monster to player
            monster_left  = self.get_nearest_monster(is_left = True)
            monster_right = self.get_nearest_monster(is_left = False)
            # Determine attack direction
            attack_direction = self.get_attack_direction(monster_left, monster_right)
            attack_direction_str = attack_direction
            if monster_left is not None:
                nearest_info = (monster_left.get("name","?"),
                                monster_left.get("position",(0,0)), "L")
            if monster_right is not None:
                # Pick the closer of the two for diagnostics.
                nr = (monster_right.get("name","?"),
                       monster_right.get("position",(0,0)), "R")
                if nearest_info is None:
                    nearest_info = nr
            # Attack Command
            if time.time() - self.t_last_attack > cooldown and attack_direction is not None:
                self.cmd_action = "attack"
                self.t_last_attack = time.time()
                # Set up attack direction
                self.cmd_move_x = attack_direction

        # VIZ: cache the non-empty-box counters on the instance so the
        # on-screen status line can show BOX=N FULL=… when detection
        # actually found something (vs DIAG_EMPTY path which only updates
        # the cache when BOX=0).
        try:
            md_mode = str(self.cfg["monster_detect"].get("mode", "?"))
        except Exception:
            md_mode = "?"
        try:
            _cached = getattr(self, "_last_mob_full_counters", {}) or {}
            _cached["box"] = len(self.monsters)
            _cached["mode"] = md_mode
            if nearest_info is not None:
                _cached["nearest"] = nearest_info
            _cached["attack_dir"] = attack_direction_str
            _cached["cmd_action_now"] = self.cmd_action
            self._last_mob_full_counters = _cached
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Rate-limited DIAG log: lets the user instantly tell whether:
        #   (A) monster detection found anything in the attack box, or
        #   (B) everything was found but cooldown blocked the cmd_action, or
        #   (C) we found nothing at all (the search box was empty)
        # The log runs at ~0.5 Hz so the log file stays readable.
        try:
            cls = type(self)
            last_dbg = getattr(cls, "_at_mob_dbg_t", 0.0)
            now = time.time()
            if now - last_dbg >= 2.0:
                try:
                    cls._at_mob_dbg_t = now
                    if isinstance(x0,(int,float)):
                        box = (f"[{x0:.0f},{y0:.0f}→{x1:.0f},{y1:.0f} "
                               f"wh={x1-x0:.0f}x{y1-y0:.0f}]")
                    else:
                        box = "?"
                    dt_cd = time.time() - self.t_last_attack
                    logger.info(
                        "[update_cmd_by_mob_detection] DIAG: "
                        f"mode={self.cfg['bot']['attack']!r} "
                        f"search_box={box} "
                        f"monsters_in_range={len(self.monsters)} "
                        f"nearest={nearest_info!r} "
                        f"attack_dir={attack_direction_str!r} "
                        f"cmd_action_now={self.cmd_action!r} "
                        f"cd_left={max(0, cooldown - dt_cd):.1f}s "
                        f"t_last_attack={dt_cd:.1f}s_ago "
                        f"player_loc={tuple(int(x) for x in getattr(self,'loc_player',(0,0)))}"
                    )
                except Exception:
                    pass
        except Exception:
            pass

    def update_cmd_by_random(self):
        '''
        update_cmd_by_random - pick a random action except 'up' and teleport command
        '''
        self.cmd_move_x = random.choice(["left", "right", "none"])
        self.cmd_move_y = random.choice(["down", "none"])
        self.cmd_action = random.choice(["jump", "none"])
        logger.warning("[update_cmd_by_random]"\
                    f"{self.cmd_move_x} {self.cmd_move_y} {self.cmd_action}")

    def check_reach_goal(self):
        if self.cmd_action == "goal":
            # Switch to next route map
            self.idx_routes = (self.idx_routes+1)%len(self.img_routes)
            logger.debug(f"Change to new route:{self.idx_routes}")

    def run_once(self):
        '''
        Process one game window frame
        '''
        # ------------------------------------------------------------------
        # TOP-LEVEL GUARD: catch *every* unhandled exception raised during
        # this frame, log a full traceback + 1Hz summary, and return -1 so
        # the outer loop() knows to re-initialize anything it needs to.
        #
        # Why we need this:
        #   Previously a single raised exception anywhere in run_once (e.g.
        #   _last_mob_full_counters being unset, img_route_debug being None
        #   during the viz layer, a NoneType.shape access from a stale
        #   capture) would propagate up to the Qt signal handler / main
        #   while-loop and SILENTLY KILL subsequent frames:
        #     - HuntingState.on_frame might still fire its 1Hz log ONCE
        #       from inside fsm.do_state_stuff(), then never again.
        #     - KeyBoardController's own 3s HEARTBEAT keeps firing so the
        #       user sees "bot not frozen" but sees cmd=(none,none,none)
        #       forever and infers "bot not moving not attacking".
        #
        # This one try/except makes the first failure visible immediately
        # in the log (full traceback) and keeps the bot looping so it
        # can self-heal next frame (loc_player refreshes, capture renews
        # buffers, etc.).
        # ------------------------------------------------------------------
        try:
            return self._run_once_impl()
        except Exception as _e_run_once:
            import traceback
            _tb = traceback.format_exc()
            _cls = type(self)
            _now = time.time()
            _last_tb = getattr(_cls, "_run_once_last_tb_logged_at", -999.0)
            _last_summ = getattr(_cls, "_run_once_last_summ_logged_at", -999.0)
            # Full traceback: at most once every 8 s to avoid spamming.
            if _now - _last_tb >= 8.0:
                _cls._run_once_last_tb_logged_at = _now
                logger.error(
                    "[run_once] Unhandled exception during frame:\n"
                    + _tb
                )
            # Short summary line: 1 Hz so user can see "still failing".
            if _now - _last_summ >= 1.0:
                _cls._run_once_last_summ_logged_at = _now
                logger.warning(
                    f"[run_once] frame aborted: {type(_e_run_once).__name__}: "
                    f"{_e_run_once} (see full traceback above if recent)"
                )
            # Reset counters so caller doesn't treat us as healthy this frame.
            self.is_frame_done = False
            self._mob_detection_ran_this_frame = False
            return -1

    def _run_once_impl(self):
        '''
        Real implementation of run_once — kept separate so a single
        try/except in run_once() can guard the entire frame pipeline
        without masking individual per-stage try/excepts inside.
        '''
        # Start profiler for performance debugging

        self.profiler.start()

        # Check if need viz window
        self.is_show_debug_window = self.is_need_show_debug_window

        # Grayscale game window (always - these are the real inputs for all
        # the detectors, regardless of whether viz is ON.)
        # Get game window frame
        img_frame = self.get_img_frame()
        if img_frame is None:
            # ------------------------------------------------------------------
            # CAPTURE-FAIL WATCHDOG (1 Hz log + 2 consecutive threshold).
            # Why this is needed:
            #   Before this guard, `get_img_frame() is None` would silently
            #   return -1 every frame, so:
            #     - FSM's HuntingState.on_frame fired exactly once (when the
            #       capture was still alive during init).
            #     - run_once's outer try/except never fires (early-return
            #       isn't an exception).
            #     - KeyBoardController's own 3 s heartbeat keeps spamming
            #       "keys=<idle> backend=keybd_sc_letter" forever, so the
            #       user sees "bot not frozen but moving=0 attacking=0" and
            #       has no idea the capture died.
            #   Now we print a 1 Hz line with capture internals + activate
            #   the game window (the common reason for "frame = None" is the
            #   user alt+tabbed away or the window was minimised) until the
            #   capture resumes.
            # ------------------------------------------------------------------
            _now = time.time()
            _cls = type(self)
            if not hasattr(_cls, "_capture_fail_cnt"):
                _cls._capture_fail_cnt = 0
                _cls._capture_fail_last_log = -999.0
            _cls._capture_fail_cnt += 1
            if _now - _cls._capture_fail_last_log >= 1.0:
                _cls._capture_fail_last_log = _now
                try:
                    _cap_thread_alive = (self.capture is not None
                                         and getattr(self.capture, "is_running", True)
                                         if self.capture is not None else False)
                    _hwnd = getattr(self.capture, "hwnd", None)
                    _wt = getattr(self.capture, "window_title", None)
                except Exception:
                    _cap_thread_alive = "?"
                    _hwnd = None
                    _wt = None
                logger.warning(
                    "[run_once] get_img_frame returned None — "
                    f"consecutive_fail={_cls._capture_fail_cnt} "
                    f"capture_thread_alive={_cap_thread_alive} "
                    f"hwnd={_hwnd} window_title={_wt!r}. "
                    "Activating game window and retrying next frame. "
                    "HINT: if this repeats, the game window was minimised / "
                    "switched to another desktop / the capture thread died — "
                    "restore the game window to its normal 16:9 size."
                )
            if not is_mac():
                activate_game_window(self.capture.window_title)
            return -1 # Wait for game window to be ready
        else:
            # capture healthy: reset the fail counter.
            try:
                type(self)._capture_fail_cnt = 0
            except Exception:
                pass
            self.img_frame = img_frame

        # Grayscale game window
        self.img_frame_gray = cv2.cvtColor(self.img_frame, cv2.COLOR_BGR2GRAY)

        # ================================================================
        # VIZ FIX: ALWAYS build the debug viz canvas, even when
        # is_show_debug_window is False.  This fixes 3 user-visible bugs:
        #   (1) User presses F2 → screenshot_img_frame() was saving a
        #       None / empty debug canvas, even though monster detection
        #       was actually running.
        #   (2) User toggles viz OFF → ON mid-run → the newly-opened cv2
        #       window was empty for one full frame.
        #   (3) User complains "no detection boxes are shown" because they
        #       never noticed the tiny "Enable Viz" Qt checkbox, ran with
        #       defaults, and assumed detection wasn't happening at all.
        #
        # Extra RAM cost per frame is exactly one BGR copy (e.g. 1296x759
        # × 3 bytes ≈ 2.8 MB), which is negligible compared to the rest
        # of the OpenCV allocations made every frame.
        # ================================================================
        self.img_frame_debug = self.img_frame.copy()
        if not hasattr(self, "img_route_debug") or self.img_route_debug is None:
            # img_route_debug is normally lazily created inside
            # get_player_location_on_global_map() when viz is ON; create
            # it here unconditionally so route-global overlaps still
            # render even when the user starts with viz OFF.
            try:
                h, w = self.img_routes[0].shape[:2]
                self.img_route_debug = self.img_routes[0].copy()
            except Exception:
                self.img_route_debug = None
        # For display layer: still HONOUR the user's explicit viz toggle.
        # If they turned it OFF via --disable-viz / disable_viz() call,
        # we stop emitting to cv2 windows at the very end, but we keep
        # drawing to the buffers so F2 screenshots + on-demand toggles
        # show the full picture.
        # In UI (PySide6) mode the debug frame is delivered to the embedded
        # "Game Window Viz" QLabel via Qt signals (see loop()).  We must NOT
        # also call cv2.imshow() there: the Qt thread has no cv2.waitKey()
        # message pump, so the extra "Game Window Debug" OpenCV window hangs
        # and Windows marks it "未响应" (not responding).  So cv2 emission is
        # only enabled for the standalone/legacy CLI path (not is_ui).
        self._should_emit_debug_to_cv2 = bool(self.is_show_debug_window) and not self.is_ui

        # Get current route image
        if self.cfg["bot"]["mode"] == "normal":
            self.img_route = self.img_routes[self.idx_routes]
            if self.is_show_debug_window:
                self.img_route_debug = cv2.cvtColor(self.img_route, cv2.COLOR_RGB2BGR)

        self.profiler.mark("Image Preprocessing")

        ###################
        ### Get Minimap ###
        ###################
        # Get minimap coordinate and size on game window
        minimap_result = get_minimap_loc_size(self.img_frame)
        if minimap_result is None:
            if time.time() - self.t_last_minimap_update > 30:
                # Unable to get minimap for 30 seconds -> assume it's login screen
                loc_login_button = self.get_login_button_location()
                if loc_login_button:
                    logger.info("Found login button on screen. Proceed to login.")
                    click_in_game_window(self.capture.window_title,
                                         loc_login_button)
                    time.sleep(3)
                    click_in_game_window(self.capture.window_title,
                                         self.cfg["ui_coords"]["select_character"])
                    time.sleep(2)
        else:
            x, y, w, h = minimap_result
            # Shrink minimap boardary by one pixel to avoid pixel leaking to minimap
            x += 1
            y += 1
            w -= 2
            h -= 2
            # update minimap image
            self.loc_minimap = (x, y)
            self.img_minimap = self.img_frame[y:y+h, x:x+w]
            self.t_last_minimap_update = time.time()

        self.profiler.mark("Get Minimap Location and Size")

        # Update health monitor with current frame
        self.health_monitor.update_frame(self.img_frame[self.cfg["ui_coords"]["ui_y_start"]:, :])

        #################################
        ### Player Location Detection ###
        #################################
        #
        # NOTE: the two sub-steps below are ordered carefully.
        # Step A computes ``loc_player_minimap`` (yellow player dot on the
        # top-left minimap).  Step B then runs a THREE-STAGE cascade to find
        # the player on-screen.  Stage 3 of the cascade (camera-center
        # fallback) is ONLY permitted if we actually SEE a valid minimap
        # player-dot in Step A — that way we never fabricate a player screen
        # location when both the visual HW capture *and* the minimap are
        # returning garbage (e.g. login screen, lost HWND, ...).

        # ---- (A) Minimap player dot (run FIRST so cascade Stage 3 can use it)
        loc_player_minimap = get_player_location_on_minimap(
                                self.img_minimap,
                                minimap_player_color=self.cfg["minimap"]["player_color"])
        if loc_player_minimap:
            self.loc_player_minimap = loc_player_minimap

        # Also refresh "other players on minimap" dot list (used later by
        # channel-change and PvP logic).
        loc_other_players = get_all_other_player_locations_on_minimap(
                                self.img_minimap,
                                self.cfg['minimap']['other_player_color'])

        # ---- (B) On-screen player location (3-stage cascade)
        # New: always run the cascade; Stage 2 (nametag) auto-activates when
        # Stage 1 (party red bar) fails, and Stage 3 (camera-center) acts as
        # the final guardrail.  This removes the brittle if/else that forced
        # CN/怀旧服 users to "opt in" before the bot could locate them.
        loc_player, loc_party_red_bar, method_used = \
            self._detect_player_location_cascade()
        if loc_party_red_bar is not None:
            self.loc_party_red_bar = loc_party_red_bar

        # Print the method used once per startup (and whenever it changes) so
        # the user can tell at a glance why the bot is / isn't stuck.
        last_method = getattr(self, "_last_loc_method", None)
        if last_method != method_used:
            self._last_loc_method = method_used
            if method_used == "camera_center_fallback":
                logger.warning(
                    "[Player Location] Falling back to CAMERA CENTER (no "
                    "red-bar or nametag template matched your client).  The "
                    "bot should now move but hunting accuracy will be "
                    "reduced.  Fix: make a nametag template for your "
                    "character's name (see usage notes)."
                )
            elif method_used in ("nametag", "nametag_auto_fallback"):
                logger.info(
                    f"[Player Location] Using nametag detection "
                    f"({method_used})."
                )
            elif method_used == "party_red_bar":
                logger.info("[Player Location] Using party red-bar detection.")
            else:
                logger.warning(
                    f"[Player Location] All detection stages failed "
                    f"(method={method_used}); keeping last known position.  "
                    "Stuck watchdog will keep firing until at least one "
                    "stage succeeds."
                )

        # Update player location
        if loc_player is not None:
            # Check if character is on ladder
            dx = abs(loc_player[0] - self.loc_player[0])
            dy = abs(loc_player[1] - self.loc_player[1])
            if self.is_on_ladder:
                if dx > 3: # Leave ladder if there is horizontal move
                    self.is_on_ladder = False
            else:
                if dx < 3 and dy != 0:
                    self.is_on_ladder = True
            # logger.info((self.is_on_ladder, dx, dy))
            # Update player location
            self.loc_player = loc_player

        # Draw player center for debugging, plus the detection method used
        cv2.circle(self.img_frame_debug,
                self.loc_player, radius=3,
                color=(0, 0, 255), thickness=-1)
        # Label method used (top-left corner, under minimap / HP bar text)
        method_color = {
            "party_red_bar":         (0,   255, 0),
            "nametag":               (0,   255, 255),
            "nametag_auto_fallback": (0,   200, 255),
            "camera_center_fallback": (0,  220, 255),
            "all_stages_failed":     (0,   0,   255),
            "none":                  (128, 128, 128),
        }.get(method_used, (200, 200, 200))
        cv2.putText(self.img_frame_debug,
                    f"LOC: {method_used}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, method_color, 2)

        # Debug (minimap color analysis — kept from upstream)
        # if self.is_first_frame:
        #     logger.info("Running minimap color analysis...")
        #     debug_minimap_colors(self.img_minimap, other_player_color)

        # Get player location on global map
        if self.cfg["bot"]["mode"] in ["patrol", "aux"]:
            self.loc_player_global = self.loc_player_minimap
        else:
            self.loc_player_global = self.get_player_location_on_global_map()

        self.profiler.mark("Player Location Detection")

        ######################
        ### Change Channel ###
        ######################
        if self.cfg['channel_change']['enable'] and \
            self.is_need_change_channel(loc_other_players):
            self.kb.set_command("none none none")
            self.kb.release_all_key()
            self.kb.disable()
            time.sleep(1)
            self.channel_change()
            self.red_dot_center_prev = None
            return 0

        if self.is_time_to_change_channel():
            self.kb.set_command("none none none")
            self.kb.release_all_key()
            self.kb.disable()
            time.sleep(1)
            self.channel_change()
            return 0

        self.profiler.mark("Change Channel")

        #######################
        ### Attack WatchDog ###
        ####################### Check if last attack is timeout
        dt = time.time() - self.t_last_attack
        if self.cfg['bot']['mode'] == 'normal' and \
            dt > self.cfg["watchdog"]["last_attack_timeout"]:
            logger.info(f"[Attack Timeout] Last attack timeout for {round(dt, 2)} seconds")
            cfg_action = self.cfg["watchdog"]["last_attack_timeout_action"]
            if cfg_action == "change_channel":
                logger.info("[Attack Timeout] Change channel!")
                self.channel_change()
            elif cfg_action == "go_home":
                logger.info("[Attack Timeout] Return home!")
                press_key(self.cfg["key"]["return_home"])
                # Terminate Autobot
                self.is_terminated = True
                self.kb.is_terminated = True
            else:
                logger.info(f"Unsupported timeout mode: {cfg_action}")

        self.profiler.mark("Attack WatchDog")

        ######################
        ### State Behavior ###
        ######################
        self.fsm.do_state_stuff()

        self.is_first_frame = False

        self.profiler.mark("State per-frame behavior")

        # ================================================================
        # Mob detection + attack command synthesis (runs EVERY frame,
        # regardless of FSM state and regardless of whether
        # is_show_debug_window is True/False).
        # ================================================================
        try:
            self.update_cmd_by_mob_detection()
            self._mob_detection_ran_this_frame = True
        except Exception as _e_mob:
            logger.warning(f"[run_once] update_cmd_by_mob_detection raised: {_e_mob}")
            self._mob_detection_ran_this_frame = False

        # Also paint the debug viz layer (DETECT[...] line, monster boxes,
        # command overlay) every frame — F2 screenshots / Qt signal below
        # need these pixels rendered even if we don't call cv2.imshow().
        try:
            self.update_info_on_img_frame_debug()
        except Exception as _e_dbg:
            logger.warning(f"[run_once] update_info_on_img_frame_debug raised: {_e_dbg}")

        #####################
        ### Debug Windows ###
        #####################
        # Only the *cv2 window emission* is gated by the viz flag.
        if getattr(self, '_should_emit_debug_to_cv2',
                   bool(self.is_show_debug_window)):
            try:
                self.update_img_frame_debug()  # calls cv2.imshow internally
            except Exception:
                pass

            # Save debug window to video
            if self.video_writer:
                try:
                    self.video_writer.write(self.img_frame_debug)
                except Exception:
                    pass

            # Resize img_route_debug for better visualization
            if self.cfg["bot"]["mode"] == "normal":
                try:
                    self.img_route_debug = cv2.resize(
                                self.img_route_debug, (0, 0),
                                fx=self.cfg["minimap"]["debug_window_upscale"],
                                fy=self.cfg["minimap"]["debug_window_upscale"],
                                interpolation=cv2.INTER_NEAREST)
                except Exception:
                    pass

            self.profiler.mark("Debug Window Show")

            # Update FPS timer
            self.t_last_frame = time.time()

            # Print profiler result
            if self.cfg["profiler"]["enable"] and \
                self.profiler.total_frames % self.cfg["profiler"]["print_frequency"] == 0:
                logger.info('\n' + self.profiler.report())

        return 0 # frame done

    def loop(self):
        '''
        Auto Bot main loop
        Only run when call autobot from UI framework and AutoBotController
        '''
        # Make sure player is in party
        if not is_mac():
            activate_game_window(self.capture.window_title)
            time.sleep(0.3)
            self.ensure_is_in_party()

        _loop_cls = type(self)
        _loop_last_summ_t = -99999.0

        while not self.kb.is_terminated:

            t_start = time.time()

            # Process one game window frame
            self.is_frame_done = False
            ret = self.run_once()

            # Only proceed if the frame is valid
            if ret == 0:
                # Draw image on debug window
                if self.is_show_debug_window and self.is_ui:
                    try:
                        if self.img_frame_debug is not None:
                            img_frame_debug_emit = self.img_frame_debug[:
                                self.cfg["ui_coords"]["ui_y_start"], :].copy()
                            self.image_debug_signal.emit(img_frame_debug_emit)
                        if self.img_route_debug is not None:
                            img_route_debug_emit = self.img_route_debug.copy()
                            self.route_map_viz_signal.emit(img_route_debug_emit)
                    except Exception as _e_emit:
                        logger.warning(f"[loop] Qt debug viz emit failed: {_e_emit}")
                try:
                    _loop_cls._loop_last_ret_was_skip = False
                except Exception:
                    pass
            else:
                # ------------------------------------------------------------------
                # LOOP-LEVEL DIAGNOSTIC: if run_once keeps returning ret != 0
                # (either early -1 from a dead capture, or -1 from an unhandled
                # exception that the run_once() guard caught + summarised), we
                # emit a 1 Hz line from the loop so users don't stare at "KB
                # heartbeat = alive + keys=<idle>" for hours without knowing
                # why the bot isn't advancing.
                #   ret == -1 + capture_fail_cnt >= 1  → capture died.
                #   ret == -1 + no capture warning      → exception inside
                #       run_once_impl (already printed by run_once guard).
                # ------------------------------------------------------------------
                try:
                    _cap_fail_cnt = getattr(_loop_cls, "_capture_fail_cnt", 0)
                except Exception:
                    _cap_fail_cnt = "?"
                _now = time.time()
                if _now - _loop_last_summ_t >= 1.0:
                    _loop_last_summ_t = _now
                    logger.warning(
                        f"[loop] run_once returned {ret!r} — frame skipped. "
                        f"capture_consecutive_fail={_cap_fail_cnt}. "
                        "HINT: check the immediately preceding log for a "
                        "'[run_once] get_img_frame returned None ...' line "
                        "(= game window minimised / capture died) or a "
                        "'[run_once] Unhandled exception' traceback "
                        "(= a code error inside the frame handler)."
                    )
                try:
                    _loop_cls._loop_last_ret_was_skip = True
                except Exception:
                    pass

            self.is_frame_done = True

            # Cap FPS to save system resource
            frame_duration = time.time() - t_start
            target_duration = 1.0 / self.cfg["system"]["fps_limit_main"]
            if frame_duration < target_duration:
                time.sleep(target_duration - frame_duration)

def main(args):
    '''
    This main function works as a fake autoBotController
    This function will only be called when the using terminal to
    run this script
    '''
    #####################
    ### Init Auto Bot ###
    #####################
    try:
        mapleStoryAutoBot = MapleStoryAutoBot(args)
    except Exception as e:
        logger.error(f"MapleStoryAutoBot Init failed: {e}")
        sys.exit(1)
    else:
        logger.info("MapleStoryAutoBot Init Successfully")

    ####################
    ### Apply Config ###
    ####################
    # Load defautl yaml config
    cfg = load_yaml("config/config_default.yaml")
    # Override with platform config
    if is_mac():
        cfg = override_cfg(cfg, load_yaml("config/config_macOS.yaml"))
    # Override with user customized config
    cfg = override_cfg(cfg, load_yaml(f"config/config_{args.cfg}.yaml"))
    # Dump config to log for debugging
    logger.debug(yaml.dump(cfg, sort_keys=False,
                 indent=2, default_flow_style=False))
    # autoBot load config
    mapleStoryAutoBot.load_config(cfg)

    #####################
    ### Start AutoBot ###
    #####################
    try:
        mapleStoryAutoBot.start() # Start all threads in autoBot
    except Exception as e:
        logger.error(f"MapleStoryAutoBot start failed: {e}")
        mapleStoryAutoBot.terminate_threads() # Terminate all threads
        sys.exit(1)
    else:
        logger.info("MapleStoryAutoBot Start Successfully")

    # Start record game window for debugging
    if args.record:
        mapleStoryAutoBot.start_record()

    kb_listener = KeyBoardListener(is_autobot=True)
    kb_listener.register_func_key_handler('f1', mapleStoryAutoBot.kb.toggle_enable)
    kb_listener.register_func_key_handler('f2', mapleStoryAutoBot.screenshot_img_frame)
    kb_listener.register_func_key_handler('f12', mapleStoryAutoBot.terminate_threads)

    # While loop
    while not mapleStoryAutoBot.is_terminated:
        # Show debug image on window
        if mapleStoryAutoBot.is_frame_done:
            if mapleStoryAutoBot.img_frame_debug is not None:
                cv2.imshow("Game Window Debug",
                    mapleStoryAutoBot.img_frame_debug[:
                        mapleStoryAutoBot.cfg["ui_coords"]["ui_y_start"], :])

            if mapleStoryAutoBot.img_route_debug is not None:
                cv2.imshow("Route Map Debug", mapleStoryAutoBot.img_route_debug)

        cv2.waitKey(1)

        time.sleep(0.01)

    #########################
    ### Terminate AutoBot ###
    #########################
    mapleStoryAutoBot.terminate_threads() # Terminate all threads

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--disable_control',
        action='store_true',
        help='Disable simulated keyboard input'
    )

    parser.add_argument(
        '--cfg',
        type=str,
        default='custom',
        help='Choose customized config yaml file in config/'
    )

    parser.add_argument(
        '--debug',
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        '--record',
        action="store_true",
        help="Record debug window"
    )

    parser.add_argument(
        '--disable_viz',
        action="store_true",
        help="Disable viz debug window"
    )

    parser.add_argument(
        '--test_image',
        default="",
        help="Pass in image in test/XXX.png"
    )

    parser.add_argument(
        '--init_state',
        default="",
        help="choose the init_state"
    )

    args = parser.parse_args()
    args.is_ui = False # Always set False for command line

    # Set logger level
    if args.debug:
        logger.set_level(logging.DEBUG)

    main(args)
