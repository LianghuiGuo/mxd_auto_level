'''
KeyBoardController
Simulate user keyboard input to control character in the game 
'''
# Standard Import
import threading
import time

# Library import
import pyautogui
from pynput import keyboard

# Local import
from src.utils.logger import logger
from src.utils.common import is_mac

if is_mac():
    import Quartz
else:
    import pygetwindow as gw

pyautogui.PAUSE = 0  # remove delay

def key_down(key):
    '''
    Press key down
    '''
    try:
        pyautogui.keyDown(key)
    except pyautogui.FailSafeException:
        logger.warning("[key_down] pyautogui failsafe triggered during key_down.")
        recover_mouse()

def key_up(key):
    '''
    Release key
    '''
    try:
        pyautogui.keyUp(key)
    except pyautogui.FailSafeException:
        logger.warning("[key_up] pyautogui failsafe triggered during key_up.")
        recover_mouse()

def recover_mouse():
    '''
    Move mouse back to center to avoid pyautogui failsafe
    '''
    pyautogui.FAILSAFE = False # Temp disasble failsafe to avoid nested exception

    screen_w, screen_h = pyautogui.size()
    pyautogui.moveTo(screen_w // 2, screen_h // 2)
    time.sleep(0.2) # Give it a moment to "cool down"

    pyautogui.FAILSAFE = True # Recover failsafe

def press_key(key, duration=0.05):
    '''
    Simulates a key press for a specified duration
    '''
    if key:
        key_down(key)
        time.sleep(duration)
        key_up(key)

class KeyBoardController():
    '''
    KeyBoardController

    The controller owns two window-title identifiers:

    ``cfg_token``
      - The token loaded from the YAML config (``cfg["game_window"]["title"]``).
        May be a substring like ``"MapleStory Worlds"`` or a list of tokens.

    ``window_title`` (the legacy attribute, now holds the **resolved exact
    window title**)
      - Populated later by ``set_window_title(...)`` once
        ``GameWindowCapturor`` has successfully enumerated the exact title
        (e.g. ``"冒险岛怀旧服"``, a value that is NOT predictable from the
        YAML token when the user has a CN/TW/KR client).

    Both identifiers are tried during ``is_game_window_active`` /
    ``ensure_game_window_active`` so either exact-match or substring-match
    will correctly identify the game client.
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        self.cmd_action = "none"
        self.cmd_up_down = "none"
        self.cmd_left_right = "none"
        self.cmd_up_down_last = ""
        self.cmd_left_right_last = ""

        # Keep the original YAML token(s) for substring matching later.
        # It's important that this preserves the *original* list/tuple/str
        # shape because the rest of the code expects string containment.
        _tok = cfg["game_window"]["title"]
        self.cfg_tokens = list(_tok) if isinstance(_tok, (list, tuple)) else [_tok]
        # window_title starts as the first token; it is *overwritten* with
        # the exact window-title string once GameWindowCapturor finds it.
        self.window_title = self.cfg_tokens[0] if self.cfg_tokens else ""

        self.fps = 0 # Frame per seconds
        # Timer
        self.t_last_up = 0.0
        self.t_last_down = 0.0
        self.t_last_toggle = 0.0
        self.t_last_screenshot = 0.0
        self.t_last_jump_down = 0.0
        self.t_last_run = time.time()
        self.t_last_skill = 0.0 # Last time character perform action(attack, cast spell, ...)
        self.t_last_buff_cast = [0] * len(self.cfg["buff_skill"]["keys"]) # Last time cast buff skill
        # Flags
        self.is_enable = True
        self.is_need_force_heal = False
        self.is_terminated = False
        # Parameters
        self.debounce_interval = self.cfg["system"]["key_debounce_interval"]
        self.fps_limit = self.cfg["system"]["fps_limit_keyboard_controller"]

        # use 'ctrl', 'alt' for mac, because it's hard to get around
        # macOS's security settings
        if is_mac():
            self.toggle_key = keyboard.Key.ctrl
            self.screenshot_key = keyboard.Key.alt
            self.terminate_key = keyboard.Key.esc
        else:
            self.toggle_key = keyboard.Key.f1
            self.screenshot_key = keyboard.Key.f2
            self.terminate_key = keyboard.Key.f12

        # set up attack key
        self.attack_key = ""
        if cfg["bot"]["attack"] == "aoe_skill":
            self.attack_key = cfg["key"]["aoe_skill"]
        elif cfg["bot"]["attack"] == "directional":
            self.attack_key = cfg["key"]["directional_attack"]
        else:
            raise ValueError(f"Unexpected attack type: {cfg['bot']['attack']}")

        # Start keyboard control thread
        threading.Thread(target=self.run, daemon=True).start()

        logger.info("[KeyBoardController] Init done")

    def set_window_title(self, exact_title: str):
        '''
        Called after ``GameWindowCapturor`` successfully resolves the exact
        foreground-window title (e.g. ``"冒险岛怀旧服"``) so the keyboard
        controller can match against it exactly (and fall back to substring
        tokens as well).

        Logs the update on first change so users can verify the link between
        the capture thread and this controller.
        '''
        if not exact_title or exact_title == self.window_title:
            return
        old_title = self.window_title
        self.window_title = exact_title
        # Prepend exact title to cfg_tokens so exact-match wins first (and
        # duplicates are removed downstream).
        new_tokens = [exact_title] + [t for t in self.cfg_tokens if t != exact_title]
        self.cfg_tokens = new_tokens
        logger.info(
            f"[KeyBoardController] window title updated: {old_title!r} -> "
            f"{exact_title!r}; search tokens = {self.cfg_tokens!r}"
        )

    def _title_matches(self, candidate: str) -> bool:
        '''
        Return True if ``candidate`` matches one of the configured tokens
        (substring case-insensitive) or equals the exact resolved title.
        '''
        if not candidate:
            return False
        low = candidate.lower()
        for token in self.cfg_tokens:
            if not token:
                continue
            if token == candidate:            # exact match
                return True
            if token.lower() in low:          # substring match (case-insensitive)
                return True
        return False

    def toggle_enable(self):
        '''
        toggle_enable
        '''
        self.is_enable = not self.is_enable
        logger.info(f"Player pressed F1, is_enable:{self.is_enable}")

        # Make sure all key are released
        self.release_all_key()

    def disable(self):
        '''
        disable keyboard controlller
        '''
        self.is_enable = False

    def enable(self):
        '''
        enable keyboard controlller
        '''
        self.is_enable = True

    def set_command(self, new_command):
        '''
        Set keyboard command
        '''
        self.cmd_left_right, self.cmd_up_down, self.cmd_action = new_command.split()

    def is_game_window_active(self):
        '''
        Check if the game window is currently the active (foreground) window.

        Matching is performed against every token in ``self.cfg_tokens`` using
        **exact equality or case-insensitive substring match** via
        ``_title_matches``.  This avoids false negatives when the resolved
        exact title (e.g. ``"冒险岛怀旧服"``) is different from the YAML
        substring token (e.g. ``"MapleStory Worlds"``).

        Returns a tuple ``(is_active, active_window_title_or_None)`` so the
        caller can log what was actually in front when this check failed.
        '''
        if is_mac():
            active_window = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID
            )
            for window in active_window:
                window_name = window.get(Quartz.kCGWindowName, '')
                if window_name and self._title_matches(window_name):
                    return True, window_name
            return False, None
        else:
            try:
                active_window = gw.getActiveWindow()
                if not active_window:
                    return False, "<none>"
                title = getattr(active_window, "title", "") or ""
                if title and self._title_matches(title):
                    return True, title
                return False, title
            except Exception:
                return False, "<exception>"

    def ensure_game_window_active(self):
        '''
        Best-effort foreground activation of the game window.

        Tries, in order:
          1. If already active, return True immediately.
          2. pygetwindow.activate() on every window whose title matches a
             known token (exact match or substring via ``_title_matches``,
             not just the single exact ``window_title`` attribute).
          3. win32gui.FindWindow with the exact resolved title; if that
             returns 0, fall back to an EnumWindows scan using the same
             multi-token matcher to locate the HWND even when the exact
             window-title string differs from the configured token.
          4. For any HWND found: if iconic, SW_RESTORE; then
             SetForegroundWindow.

        Returns True if the game became active, False otherwise.
        '''
        is_active, _ = self.is_game_window_active()
        if is_active:
            return True

        # --- Stage 1: pygetwindow, matched via all tokens -------------------
        try:
            all_wins = gw.getAllWindows() or []
            for w in all_wins:
                t = getattr(w, "title", "") or ""
                if self._title_matches(t):
                    try:
                        w.activate()
                        time.sleep(0.05)
                        if self.is_game_window_active()[0]:
                            return True
                    except Exception:
                        # pygetwindow raises for weird HWNDs; keep going.
                        pass
        except Exception:
            pass

        # --- Stage 2: win32gui EnumWindows fallback ------------------------
        try:
            import win32gui  # already a project dependency, imported lazily
            import win32con
        except Exception:
            return False

        def _find_hwnd_via_tokens():
            found_hwnd = [0]
            def cb(hwnd, _):
                try:
                    if not win32gui.IsWindowVisible(hwnd):
                        return
                    t = win32gui.GetWindowText(hwnd)
                    if self._title_matches(t):
                        found_hwnd[0] = hwnd
                except Exception:
                    pass
            try:
                win32gui.EnumWindows(cb, None)
            except Exception:
                pass
            return found_hwnd[0]

        hwnd = 0
        try:
            if self.window_title:
                hwnd = win32gui.FindWindow(None, self.window_title)
            if hwnd == 0:
                # Exact title FindWindow failed (usual case when the exact
                # title hasn't been propagated yet); scan all windows.
                hwnd = _find_hwnd_via_tokens()
            if hwnd != 0:
                if win32gui.IsIconic(hwnd):
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                time.sleep(0.05)
                return self.is_game_window_active()[0]
        except Exception:
            pass
        return False

    def release_all_key(self):
        '''
        Release all key
        '''
        key_up("left")
        key_up("right")
        key_up("up")
        key_up("down")
        # Also release attack keys to stop any ongoing attacks
        key_up(self.attack_key)

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

    def run(self):
        '''
        run
        '''
        # Rate-limited logging helpers so the log window doesn't flood.
        t_next_focus_warn = 0.0
        t_first_action_banner = True

        def do_action_key(kind, key):
            nonlocal t_first_action_banner
            if t_first_action_banner:
                # Print a one-time banner so users can tell that the
                # controller is actually dispatching keys (not just looping).
                is_active, foreground = self.is_game_window_active()
                logger.info(
                    "[KeyBoardController] Sending first action to game. "
                    f"active={is_active}, foreground_window={foreground!r}, "
                    f"window_title_token={self.window_title!r}"
                )
                t_first_action_banner = False
            press_key(key)

        while not self.is_terminated:
            # --- Preconditions -------------------------------------------------
            if not self.is_enable:
                self.limit_fps()
                continue

            # Ensure game window stays in the foreground (PyAutoGUI sends to
            # the foreground window globally; if the user clicks the Qt UI
            # all subsequent presses are lost).
            is_active, active_title = self.is_game_window_active()
            if not is_active:
                activated = self.ensure_game_window_active()
                if not activated:
                    # Only complain at ~0.5 Hz so the log stays readable.
                    now = time.time()
                    if now >= t_next_focus_warn:
                        t_next_focus_warn = now + 2.0
                        logger.warning(
                            "[KeyBoardController] Game window is not in the "
                            f"foreground and couldn't be activated.  Expected "
                            f"title containing {self.window_title!r}; current "
                            f"front window is {active_title!r}.  Keys are "
                            "HALTED until the game window regains focus."
                        )
                    self.limit_fps()
                    continue

            # Buff skill
            for i, buff_skill_key in enumerate(self.cfg["buff_skill"]["keys"]):
                cooldown = self.cfg["buff_skill"]["cooldown"][i]
                if time.time() - self.t_last_buff_cast[i] >= cooldown and \
                    time.time() - self.t_last_skill > self.cfg["buff_skill"]["action_cooldown"]:
                    do_action_key("buff", buff_skill_key)
                    logger.info(f"[Buff] Press buff skill key: '{buff_skill_key}' (cooldown: {cooldown}s)")
                    # Reset timers
                    self.t_last_buff_cast[i] = time.time()
                    self.t_last_skill = time.time()
                    break

            # Force Heal
            if self.is_need_force_heal:
                self.cmd_action = "add_hp"

            ##########################
            ### Left-Right Command ###
            ##########################
            if self.cmd_left_right == "left":
                key_up("right")
                key_down("left")
            elif self.cmd_left_right == "right":
                key_up("left")
                key_down("right")
            elif self.cmd_left_right == "stop":
                key_up("left")
                key_up("right")
            elif self.cmd_left_right == "none":
                if self.cmd_left_right_last != "none":
                    key_up("left")
                    key_up("right")
            else:
                logger.error("[KeyBoardController] Unsupported left-right command: "
                             f"{self.cmd_left_right}")
            self.cmd_left_right_last = self.cmd_left_right

            #######################
            ### Up-Down Command ###
            #######################
            if self.cmd_up_down == "up":
                key_up("down")
                key_down("up")
            elif self.cmd_up_down == "down":
                key_up("up")
                key_down("down")
            elif self.cmd_up_down == "stop":
                key_up("up")
                key_up("down")
            elif self.cmd_up_down == "none":
                if self.cmd_up_down_last != "none":
                    key_up("up")
                    key_up("down")
            else:
                logger.error("[KeyBoardController] Unsupported up-down command: "
                             f"{self.cmd_up_down}")
            self.cmd_up_down_last = self.cmd_up_down

            ######################
            ### Action Command ###
            ######################
            if self.cmd_action == "jump":
                do_action_key("jump", self.cfg["key"]["jump"])
            elif self.cmd_action == "teleport":
                do_action_key("teleport", self.cfg["key"]["teleport"])
            elif self.cmd_action == "attack":
                do_action_key("attack", self.attack_key)
                self.t_last_skill = time.time()
            elif self.cmd_action == "add_hp":
                do_action_key("add_hp", self.cfg["key"]["add_hp"])
                self.cmd_action = "none"  # Reset command
            elif self.cmd_action == "add_mp":
                do_action_key("add_mp", self.cfg["key"]["add_mp"])
                self.cmd_action = "none"  # Reset command
            elif self.cmd_action == "goal":
                pass
            elif self.cmd_action == "none":
                pass
            else:
                logger.error("[KeyBoardController] Unsupported action command: "
                             f"{self.cmd_action}")

            self.limit_fps()

        self.release_all_key() # Prevent key keep press down after termination

        logger.info("[KeyBoardController] terminated")
