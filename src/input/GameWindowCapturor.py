'''
Execute this script:
python mapleStoryAutoLevelUp.py --map cloud_balcony --monster brown_windup_bear,pink_windup_bear
'''
# Standard import
import time
import threading

# Libarary Import
from windows_capture import WindowsCapture, Frame, InternalCaptureControl
import cv2

# local import
from src.utils.logger import logger
from src.utils.common import get_game_window_title_by_token, load_image, resize_window

class GameWindowCapturor:
    '''
    GameWindowCapturor
    '''
    def __init__(self, cfg, test_image_name = None):
        self.cfg = cfg
        self.frame = None
        self.lock = threading.Lock()
        self.is_terminated = False
        self.fps = 0
        self.fps_limit = cfg["system"]["fps_limit_window_capturor"]
        self.t_last_run = 0.0
        self.capture_control = None
        self.window_title = ""

        # If use test image as input, disable the whole capture thread
        if test_image_name is not None:
            self.frame = load_image(f"test/{test_image_name}.png")
            return

        # Build a priority-ordered list of window-title search keywords.
        # 1. User's configured token(s) from YAML (supports both str and list)
        # 2. Targeted fallbacks covering common MapleStory clients.
        #    NOTE: Do NOT use the bare token "MapleStory" — it is too broad and
        #    matches our own Qt window titled "MapleStory AutoLevelUp".
        user_token = cfg["game_window"]["title"]
        search_tokens = list(user_token) if isinstance(user_token, (list, tuple)) else [user_token]
        GENERIC_FALLBACK_TOKENS = [
            "MapleStory Worlds",   # Artale (official Artale global client)
            "冒险岛怀旧服",          # 中文怀旧服 (user's current client)
            "冒险岛怀旧服v",         # 中文怀旧服版本号变体
            "冒险岛",               # 新枫之谷/台服/国服中文服通用
            "新楓之谷",             # 台服繁体
            "Artale",              # Artale client (another common title)
        ]
        # Deduplicate while preserving priority order (user config first, then fallback)
        seen = set()
        final_tokens = []
        for t in search_tokens + GENERIC_FALLBACK_TOKENS:
            if t not in seen:
                seen.add(t)
                final_tokens.append(t)

        # Get game window title (try each keyword, first match wins)
        self.window_title = get_game_window_title_by_token(final_tokens)

        if self.window_title is None:
            raise RuntimeError(
                "[GameWindowCapturor] Unable to find game window.\n"
                f"    - User config token : {user_token!r}\n"
                f"    - All tried keywords: {final_tokens}\n"
                "    -> Hint: in the UI, open [Advanced Settings] -> game_window\n"
                "              and set `title` to a unique substring of your game window.\n"
                "              Or use: python -c \"import win32gui; titles=[]; win32gui.EnumWindows("
                "lambda h,_: titles.append(win32gui.GetWindowText(h)), None); "
                "print([t for t in titles if t])\"  to list all visible window titles."
            )
        else:
            logger.info(f"[GameWindowCapturor] Found game window title: {self.window_title!r}")

        # Also resolve the HWND now so downstream code (KeyBoardController's
        # PostMessage backend, GameWindowCapturor's own future helpers) can
        # target the window directly without re-enumerating every time.
        self.window_hwnd = 0
        try:
            import win32gui  # already a project dependency (see common.py)
            self.window_hwnd = int(win32gui.FindWindow(None, self.window_title))
        except Exception:
            self.window_hwnd = 0
        if self.window_hwnd != 0:
            logger.info(
                f"[GameWindowCapturor] Resolved HWND {hex(self.window_hwnd)} "
                f"for {self.window_title!r}"
            )

        resize_window(self.window_title, width=1296, height=759, hwnd=self.window_hwnd or None)

        # Create capture handler
        self.capture = WindowsCapture(window_name=self.window_title)
        self.capture.event(self.on_frame_arrived)
        self.capture.event(self.on_closed)

        # Start capturing thread
        self.capture_control = self.capture.start_free_threaded()

        logger.info("[GameWindowCapturor] Init done")

    def on_frame_arrived(self, frame: Frame,
                         capture_control: InternalCaptureControl):
        '''
        Frame arrived callback: store frame into buffer with lock.
        '''
        with self.lock:
            self.frame = frame.frame_buffer
        self.limit_fps()

    def on_closed(self):
        '''
        Capture closed callback.
        '''
        logger.warning("[GameWindowCapturor] closed.")
        cv2.destroyAllWindows()

    def get_frame(self):
        '''
        Safely get latest game window frame.
        '''
        with self.lock:
            if self.frame is None:
                return None
            return cv2.cvtColor(self.frame, cv2.COLOR_BGRA2BGR)

    def stop(self):
        '''
        Stop capturing thread
        '''
        if self.capture_control is not None:
            self.capture_control.stop()
        logger.info("[GameWindowCapturor] Terminated")

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
