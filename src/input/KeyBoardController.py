'''
KeyBoardController
Simulate user keyboard input to control character in the game 
'''
# Standard Import
import ctypes
import random as _py_random
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

# ---------------------------------------------------------------------------
# Optional driver-level virtual Xbox gamepad backend (vgamepad + ViGEmBus).
#
# Why: private-server anti-cheat modules often block user-mode keyboard
# synthesis (keybd_event / SendInput / WH_KEYBOARD_LL / PostMessage) but
# almost none of them filter a *real* HID XInput gamepad that the ViGEmBus
# kernel driver exposes.  The left thumbstick X/Y axis sends movement, and
# A/B/X/Y/LB/RB/Start buttons emulate jump/attack/potion/teleport/buff keys
# — the only action the user has to take (one time) is:
#
#   1. Install ViGEmBus driver from https://vigem.org/   (reboot-free with
#      latest signed installer; needed once per machine).
#   2. ``pip install vgamepad``
#   3. Either:
#        a. set env var ``MAPLEBOT_USE_VGAMEPAD=1``  (simplest; no YAML edit)
#        b. or add ``key:  use_vgamepad: true``  to your customized YAML
#   4. Inside MapleStory → [Game Options] → [Gamepad Settings]: select the
#      newly-appeared "Controller (Xbox 360 For Windows)" and map:
#        * Move left/right/up/down  →  Left Stick (X/Y axis)
#        * Jump                     →  A
#        * Attack / basic attack    →  X
#        * Use HP potion            →  Y
#        * Use MP potion            →  B
#        * Teleport                 →  LB or RB (whichever fits; also
#                                      bound in YAML if you need a different
#                                      button)
#      (or alternatively just turn on "Gamepad Input" inside MapleStory
#      when it asks — the default mapping is usually enough).
#
# If vgamepad is not installed, or ViGEmBus driver is not present, the
# import / create() calls fail silently and the backend is never advertised
# or used — no behaviour change for vanilla installations.
# ---------------------------------------------------------------------------
_VGP = None       # None until enable_vgamepad() succeeds
_VGP_ENABLED = False
_VGP_ERR = None   # first init error string for HEARTBEAT reporting

def enable_vgamepad():
    """Try to initialise the vgamepad backend.

    Returns ``True`` when the pad is available and button/axis calls below
    are safe to make.  Subsequent calls after the first success are
    idempotent."""
    global _VGP, _VGP_ENABLED, _VGP_ERR
    if _VGP_ENABLED:
        return True
    if is_mac():
        _VGP_ERR = "vgamepad unavailable on macOS"
        return False
    try:
        import vgamepad  # type: ignore  # optional dep
    except Exception as e:
        _VGP_ERR = f"vgamepad import failed: {e}"
        return False
    try:
        pad = vgamepad.VX360Gamepad()
    except Exception as e:
        _VGP_ERR = f"VX360Gamepad() failed (ViGEmBus driver probably not installed): {e}"
        return False
    _VGP = pad
    _VGP_ENABLED = True
    _VGP_ERR = None
    # Initialise pad state: no buttons pressed, stick centered, triggers 0.
    try:
        pad.update()
    except Exception:
        pass
    logger.info(
        "[KeyBoardController] vgamepad backend enabled (ViGEm virtual "
        "Xbox 360 pad).  Movement now goes through the left stick; button "
        "calls A/B/X/Y/LB/RB/Start are routed to gamepad buttons.  If the "
        "character still doesn't move: open MapleStory game options and "
        "turn on Gamepad input / map the left-stick X/Y axis to "
        "left/right/up/down."
    )
    return True

def vgamepad_available() -> bool:
    return bool(_VGP_ENABLED and _VGP is not None)

def _vgp_thumb_x(val: float):
    """Set left-stick X axis.  ``val`` ∈ [-1.0 ... +1.0], negative = left."""
    if not vgamepad_available(): return
    try:
        import vgamepad as vg  # type: ignore
        v = max(-1.0, min(1.0, float(val)))
        _VGP.left_joystick_float(x_value_float=v, y_value_float=None) \
            if False else None   # placeholder; we call explicit int setter below
        # The int setter is available on all vgamepad versions; range ±32767
        iv = int(round(v * 32767.0))
        _VGP.left_joystick(x_value=iv, y_value=None) if False else None
        # Use a direct union call to avoid the optional y= API mismatch:
        #   left_joystick_float(x_value_float=..., y_value_float=currentY)
        # We track Y in a module-level box so X changes don't zero Y out.
        cur_y = _VGP_Y[0]
        _VGP.left_joystick_float(x_value_float=v, y_value_float=cur_y)
        _VGP.update()
    except Exception:
        pass

def _vgp_thumb_y(val: float):
    """Set left-stick Y axis.  ``val`` ∈ [-1.0 ... +1.0], positive = UP.

    Note: on many vgamepad builds the Y axis sign is inverted (negative
    inputs tilt the stick upward).  We correct that internally so callers
    can keep the intuitive ``+1 = up`` semantics."""
    if not vgamepad_available(): return
    try:
        v = max(-1.0, min(1.0, float(val)))
        # Invert sign for the underlying driver because XInput legacy.
        hw = - v
        cur_x = _VGP_X[0]
        _VGP.left_joystick_float(x_value_float=cur_x, y_value_float=hw)
        _VGP.update()
    except Exception:
        pass

# Module-level boxes holding the last-written stick components so updating
# one axis does not wipe out the other (vgamepad's setters overwrite both
# axes in the same report — we have to reconstruct the full report each
# call using our cached values).
_VGP_X = [0.0]
_VGP_Y = [0.0]

# Button map: YAML key-name → vgamepad.XUSB_BUTTON.*  This gets populated
# at KeyBoardController __init__ time from cfg["key"]["vgamepad_buttons"],
# and is used by press_key / key_down / key_up / release_all_key to also
# fire the mapped gamepad button alongside the keyboard press.
_VGP_BUTTON_MAP: dict = {}   # {lowercase_keystr: XUSB_GAMEPAD_* constant}
_VGP_BUTTONS_DOWN: set = set()  # currently-held gamepad buttons

def _build_vgp_constants_map():
    """Lazily import vgamepad and return {const_name: value}.

    Returns ``None`` if vgamepad is not importable or not enabled yet.
    Caller MUST wrap in try/except."""
    if not vgamepad_available():
        return None
    try:
        import vgamepad as vg  # type: ignore
        return {
            "A":     vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
            "B":     vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
            "X":     vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
            "Y":     vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
            "LB":    vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
            "RB":    vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
            "LS":    vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB,
            "RS":    vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB,
            "START": vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
            "BACK":  vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK,
            "GUIDE": vg.XUSB_BUTTON.XUSB_GAMEPAD_GUIDE,
            "DPAD_UP":    vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
            "DPAD_DOWN":  vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
            "DPAD_LEFT":  vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
            "DPAD_RIGHT": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
        }
    except Exception:
        return None

def _try_vgp_button(key, press: bool):
    """Helper used by key_down / key_up / press_key to also press/release
    the gamepad button that the YAML mapped to ``key`` (if any).

    Returns True when a button was actually pressed/released, False
    otherwise (silent no-op for unmapped keys / disabled backend)."""
    if not key or not vgamepad_available() or not _VGP_BUTTON_MAP:
        return False
    try:
        ks = str(key).strip().lower()
        if not ks: return False
        code = _VGP_BUTTON_MAP.get(ks)
        if code is None: return False
        if press:
            _vgp_press_button(code)
            _VGP_BUTTONS_DOWN.add(code)
        else:
            _vgp_release_button(code)
            _VGP_BUTTONS_DOWN.discard(code)
        return True
    except Exception:
        return False

def _vgp_set_axis(x, y):
    """Atomic update of both thumb axes (avoids one axis zeroing the other)."""
    if not vgamepad_available(): return
    xv = max(-1.0, min(1.0, float(x) if x is not None else _VGP_X[0]))
    yv = max(-1.0, min(1.0, float(y) if y is not None else _VGP_Y[0]))
    if x is not None: _VGP_X[0] = xv
    if y is not None: _VGP_Y[0] = yv
    try:
        _VGP.left_joystick_float(x_value_float=_VGP_X[0], y_value_float=-_VGP_Y[0])
        _VGP.update()
    except Exception:
        pass

# Gamepad-button aliases (matches a real Xbox 360 pad layout).  Each
# helper wraps press/release/update into a single best-effort call that
# never raises.
def _vgp_press_button(btn_code):
    if not vgamepad_available(): return False
    try:
        import vgamepad as vg  # type: ignore
        _VGP.press_button(button=btn_code)
        _VGP.update()
        return True
    except Exception: return False

def _vgp_release_button(btn_code):
    if not vgamepad_available(): return False
    try:
        import vgamepad as vg  # type: ignore
        _VGP.release_button(button=btn_code)
        _VGP.update()
        return True
    except Exception: return False


def press_key(key):
    '''
    Compat-only alias for ``press_key(key, 0.05)`` with single-arg callers.
    The real variadic implementation lives later in the file under the same
    name (Python allows late re-binding of module-level function
    references by overwriting ``globals()``), and after the module finishes
    loading the variadic ``press_key`` below takes over for all callers.
    '''
    _press_key_real(key, 0.05)

# ---------------------------------------------------------------------------
# Windows-only: alternative keyboard backends (PostMessage → scan-code
# keybd_event → native SendInput(scan) → PyAutoGUI)
# ---------------------------------------------------------------------------
# PyAutoGUI uses ``SendInput`` on Windows, which most private MapleStory
# clients block via a low-level keyboard hook (nProtect / custom anti-cheat
# shield).  To keep the bot functional even when SendInput is blacklisted we
# provide three additional backends ordered by "bypass rate" (highest first):
#
# 1. ``PostMessage`` to the game HWND with ``WM_KEYDOWN/WM_KEYUP`` — posts
#    window messages directly into the target's message queue, bypassing the
#    global input hook chain.  Requires a known HWND and the window must
#    NOT be minimised.  Fails when the game uses DirectInput / raw-input
#    (reads the kernel HID device state instead of the MSG loop).
# 2. ``keybd_event`` with ``KEYEVENTF_SCANCODE`` — fires at the *hardware
#    scan-code* layer rather than the virtual-key layer.  Anti-cheats that
#    only inspect the virtual-key table miss these events entirely.
#    ``keybd_event`` is deprecated but still present in user32.dll and
#    internally remaps to SendInput, so some very aggressive shields still
#    see it.
# 3. Native ``SendInput`` called *directly* via ctypes with
#    KEYEVENTF_SCANCODE.  Same underlying API as PyAutoGUI, but we set the
#    INPUT structure ourselves so that it contains a scan-code rather than
#    relying on the VK→scan remap that PyAutoGUI uses.  Anti-cheats that
#    blacklist PyAutoGUI's specific call-pattern often allow this call.
# 4. PyAutoGUI (SendInput via library wrapper) — highest compatibility but
#    lowest bypass rate.
#
# All backends degrade gracefully when unavailable, so the module still
# imports cleanly on macOS, Wine, or Windows SKUs that lack some of the
# user32 entry points.  ``_last_backend_used`` records which backend
# successfully fired last, so the KeyBoardController heartbeat can surface
# it in logs (handy for confirming *which* backend a shield blocked).
# ---------------------------------------------------------------------------

_WIN_OK = False
_USER32 = None
try:
    if not is_mac():
        # ``windll.user32`` binds symbols lazily at *call* time rather than
        # at *import* time, so missing entry points (Windows 7 / Wine)
        # raise when invoked rather than crashing the import.  Exactly what
        # we want here.
        _USER32 = ctypes.windll.user32
        try:
            _WIN32U = ctypes.windll.win32u  # Windows 10+ exports NtUser* here
        except Exception:
            _WIN32U = None
        try:
            _KERNEL32 = ctypes.windll.kernel32
        except Exception:
            _KERNEL32 = None
        _WIN_OK = True
except Exception:  # pragma: no cover — defensive guard
    _WIN_OK = False
    _USER32 = None
    _WIN32U = None
    _KERNEL32 = None

# keybd_event + SendInput flags
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_SCANCODE = 0x0008
_KEYEVENTF_UNICODE  = 0x0004   # 发送 wScan = 字符(UCS-2 LE)的"输入法合成路径"——一些私服钩子不碰这个，因为是 IMM/CTFMON 在用
_MAPVK_VK_TO_VSC = 0   # uCode=virtual-key, return value=scan-code

# Native SendInput INPUT struct layout (KEYBDINPUT variant).
# MSDN: INPUT.type = INPUT_KEYBOARD (1)  →  union field ki is active.
_INPUT_KEYBOARD = 1
class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk",       ctypes.c_ushort),
        ("wScan",     ctypes.c_ushort),
        ("dwFlags",   ctypes.c_ulong),
        ("time",      ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]
class _INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("ki", _KEYBDINPUT)]
    _anonymous_ = ("_u",)
    _fields_ = [("type", ctypes.c_ulong), ("_u", _U)]

# PostMessage constants (used by WM-msg backend)
_WM_KEYDOWN = 0x0100
_WM_KEYUP   = 0x0101
def _MAKELPARAM(rep: int, scan: int, ext: int, ctx: int, prev: int, trans: int) -> int:
    """Build the LPARAM for WM_KEYDOWN / WM_KEYUP according to Microsoft docs."""
    # Bits:  0-15 repeat count (1 for down, 1 for up)
    #       16-23 scan code
    #       24    extended-key flag
    #       25-28 reserved (0)
    #       29    context code
    #       30    previous key state
    #       31    transition state (0=down, 1=up)
    l = int(rep & 0xFFFF)
    l |= (int(scan & 0xFF) << 16)
    l |= (int(ext & 0x1) << 24)
    l |= (int(ctx & 0x1) << 29)
    l |= (int(prev & 0x1) << 30)
    l |= (int(trans & 0x1) << 31)
    if l >= 0x80000000:
        l -= 0x100000000  # signed 32-bit
    return l

# ---------------------------------------------------------------------------
# PyAutoGUI virtual-key name -> Windows VK code table.  We only enumerate the
# keys the bot actually uses; unknown names fall back to ord(...) which works
# for plain ASCII characters (a-z, 0-9, punctuation).
# ---------------------------------------------------------------------------
_PYAUTOGUI_NAME_TO_VK = {
    "escape":   0x1B,  "esc":     0x1B,
    "enter":    0x0D,  "return":  0x0D,  "\n":      0x0D,
    "space":    0x20,
    "tab":      0x09,
    "left":     0x25,  "up":      0x26,  "right":   0x27,  "down":    0x28,
    "insert":   0x2D,  "delete":  0x2E,  "home":    0x24,  "end":     0x23,
    "pageup":   0x21,  "pagedown":0x22,
    "f1":       0x70,  "f2":      0x71,  "f3":      0x72,  "f4":      0x73,
    "f5":       0x74,  "f6":      0x75,  "f7":      0x76,  "f8":      0x77,
    "f9":       0x78,  "f10":     0x79,  "f11":     0x7A,  "f12":     0x7B,
    "ctrl":     0x11,  "lctrl":   0xA2,  "rctrl":   0xA3,
    "control":  0x11,  "lcontrol":0xA2,  "rcontrol":0xA3,
    "alt":      0x12,  "lalt":    0xA4,  "ralt":    0xA5,
    "shift":    0x10,  "lshift":  0xA0,  "rshift":  0xA1,
    "win":      0x5B,  "lwin":    0x5B,  "rwin":    0x5C,
    "capslock": 0x14,  "numlock": 0x90,  "scrolllock": 0x91,
    "printscreen": 0x2C, "pause": 0x13,
}

# Extended-key flag: arrow keys, Home/End, PgUp/PgDn, Insert/Delete,
# RCtrl, RAlt are "extended" on IBM AT keyboards and the scan-code path
# needs KEYEVENTF_EXTENDEDKEY (0x0001) set in dwFlags otherwise some
# games treat them as numpad keys instead of dedicated cursor keys.
_EXTENDED_VK_SET = frozenset([
    0x25, 0x26, 0x27, 0x28,            # arrows
    0x2D, 0x2E, 0x24, 0x23, 0x21, 0x22,# ins/del/home/end/pgup/pgdn
    0xA2, 0xA5,                         # RCtrl, RAlt
])
_KEYEVENTF_EXTENDEDKEY = 0x0001

def _key_name_to_vk(key):
    """Convert a PyAutoGUI key name ("left", "a", "f1" ...) to a Windows
    virtual-key code.  Returns ``None`` when the mapping can't be resolved."""
    if key is None:
        return None
    if isinstance(key, int):
        return key
    k = str(key).lower()
    if k in _PYAUTOGUI_NAME_TO_VK:
        return _PYAUTOGUI_NAME_TO_VK[k]
    if len(k) == 1:
        v = ord(k.upper())
        if 32 <= v < 127:
            return v
    return None

def _vk_to_sc(vk) -> int:
    """Virtual-key → hardware scan-code via ``MapVirtualKeyW``.  Returns 0 on
    failure (caller should skip the scan-code backend in that case)."""
    if not _WIN_OK or _USER32 is None or vk is None:
        return 0
    try:
        return int(_USER32.MapVirtualKeyW(int(vk) & 0xFFFFFFFF, _MAPVK_VK_TO_VSC))
    except Exception:
        return 0

def _is_extended_key(vk) -> bool:
    """True for VK codes that need KEYEVENTF_EXTENDEDKEY on the scan path."""
    try:
        return int(vk) in _EXTENDED_VK_SET
    except Exception:
        return False

# The resolved game HWND — shared between key_down / key_up (PostMessage
# needs it).  KeyBoardController propagates the HWND here once
# GameWindowCapturor has found a valid window.
_RESOLVED_GAME_HWND = [0]  # boxed so module-level helpers can mutate it
_RESOLVED_TITLE_TOKENS: list = []

# Diagnostic telemetry exposed to KeyBoardController.run() heartbeat:
#   _last_backend_used – one of "PostMessage"/"keybd_scan"/"native_SendInput"/
#                         "PyAutoGUI"/"macOS"/"failed"
#   _last_backend_ts   – unix time the flag was last updated
#   _consecutive_fails – counter used by KeyBoardController's once-per-startup
#                         "your anti-cheat blocks *everything*" warning.
_last_backend_used = ["unknown"]
_last_backend_ts   = [0.0]
_consecutive_fails = [0]

def _set_backend(name: str, success: bool):
    """Atomically record which backend most recently fired."""
    _last_backend_used[0] = name if success else "failed"
    _last_backend_ts[0]   = time.time()
    if success:
        _consecutive_fails[0] = 0
    else:
        _consecutive_fails[0] += 1

def get_last_backend_info():
    """Return (backend_name, last_update_unix_time, consecutive_failures) to
    the caller (used by the keyboard-controller heartbeat).  All values are
    safe to read on any thread."""
    return (str(_last_backend_used[0]),
            float(_last_backend_ts[0]),
            int(_consecutive_fails[0]))

def register_game_hwnd(hwnd: int):
    """Called from KeyBoardController / GameWindowCapturor after the game
    HWND has been resolved.  Allows PostMessage backend to target the
    correct window.  Passing 0 clears the cached value."""
    _RESOLVED_GAME_HWND[0] = int(hwnd or 0)

def register_game_title_tokens(tokens):
    """Called from KeyBoardController so the PostMessage backend can
    re-discover the HWND on-the-fly if ``register_game_hwnd`` hasn't fired
    yet or the window was recycled."""
    _RESOLVED_TITLE_TOKENS.clear()
    for t in tokens or []:
        if t:
            _RESOLVED_TITLE_TOKENS.append(str(t))

def _ensure_game_hwnd() -> int:
    """Return the game HWND if known; otherwise try a quick EnumWindows
    sweep using the registered title tokens.  Returns 0 if nothing was
    found (PostMessage backend is skipped in that case)."""
    if _RESOLVED_GAME_HWND[0] != 0:
        try:
            if _WIN_OK and _USER32.IsWindow(_RESOLVED_GAME_HWND[0]):
                return _RESOLVED_GAME_HWND[0]
        except Exception:
            pass
        _RESOLVED_GAME_HWND[0] = 0
    if not _WIN_OK or not _RESOLVED_TITLE_TOKENS:
        return 0
    try:
        found = [0]
        EnumWindows = _USER32.EnumWindows
        EnumWindowsProc = ctypes.WINFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        GetWindowTextW = _USER32.GetWindowTextW
        GetWindowTextLengthW = _USER32.GetWindowTextLengthW
        IsWindowVisible = _USER32.IsWindowVisible
        def cb(hwnd, _lparam):
            if not IsWindowVisible(hwnd):
                return True
            length = GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value
            low = title.lower()
            for tok in _RESOLVED_TITLE_TOKENS:
                if not tok:
                    continue
                if title == tok or tok.lower() in low:
                    found[0] = hwnd
                    return False
            return True
        EnumWindows(EnumWindowsProc(cb), None)
        if found[0] != 0:
            _RESOLVED_GAME_HWND[0] = found[0]
            return found[0]
    except Exception:
        pass
    return 0

# ---------------------------------------------------------------------------
# Attached-thread input attachment (AttachThreadInput + SendMessage
# synchronous WM_KEYDOWN/WM_KEYUP).
#
# Why this works when PostMessage(keybd_event/SendInput) doesn't:
#   * Some private-server anti-cheat modules hook *global* keybd_event /
#     SendInput / low-level keyboard proc (WH_KEYBOARD_LL) but they do
#     NOT sit inside each process' own GetMessage loop — so if we first
#     AttachThreadInput() our thread to the game-window's foreground
#     thread, then call SendMessage() *synchronously* (not PostMessage),
#     the delivered WM_KEYDOWN/WM_KEYUP messages run INSIDE the game's
#     message pump with a properly-attached input queue — most
#     WH_KEYBOARD (per-process, non-LL) hooks still fire, but a lot of
#     private servers never install a per-process WH_KEYBOARD, they only
#     hook WH_KEYBOARD_LL, which SendMessage bypasses entirely.
#
# This is a "hard" attachment: we AttachThreadInput, send down+up, then
# detach, because permanently attaching input queues between processes
# causes deadlocks if one of them blocks.
# ---------------------------------------------------------------------------
_WM_KEYFIRST = 0x0100
WM_KEYDOWN = 0x0100
WM_KEYUP   = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP   = 0x0105
WM_CHAR       = 0x0102   # TranslateMessage() 会把 WM_KEYDOWN 翻译成 WM_CHAR（按键对应的字符）
WM_DEADCHAR   = 0x0103
WM_SYSCHAR    = 0x0106
WM_SYSDEADCHAR = 0x0107

def _lparam_from_vk_sc(vk: int, sc: int, is_down: bool, repeat: int = 1) -> int:
    """Build a 32-bit LPARAM for WM_KEYDOWN/WM_KEYUP that matches what a
    real keyboard driver generates (scan code in bits 16-23, extended-flag
    bit 24, transition-state bit 31 for key-up)."""
    try:
        vki = int(vk) & 0xFFFFFFFF
        sci = int(sc) & 0xFF
        ext = 0x01000000 if _is_extended_key(vki) else 0x00000000
        rep = int(repeat) & 0xFFFF
        lparam = rep | (sci << 16) | ext
        if not is_down:
            # KEYUP: previous-key-state bit 30 = 1, transition-state bit 31 = 1
            lparam |= 0xC0000000
            lparam &= 0xFFFFFFFF
        return lparam
    except Exception:
        return 0

def _send_key_attached(vk: int, is_down: bool, hwnd: int) -> bool:
    """Attach to the game window's foreground-thread input queue, then
    SendMessage(WM_KEYDOWN or WM_KEYUP) synchronously.  Returns True on
    apparent success, False otherwise.  Always detaches before returning
    (even on error) to avoid cross-process queue deadlocks.

    Additionally, on KEYDOWN we also fire:
      * WM_SYSKEYDOWN (for Alt-combined code paths — some private-server
        WndProcs dispatch movement from SYS variants because legacy
        MapleStory used the Alt key for "sit" and treated direction
        keys as menu-navigation code paths), and
      * WM_CHAR with the ToUnicode()-translated character (this is what
        TranslateMessage() normally does inside a message pump; a few
        private-server clients read movement from the WM_CHAR path
        instead of WM_KEYDOWN to hook input more tightly).
    """
    if not _WIN_OK or _USER32 is None or _KERNEL32 is None or vk is None or hwnd is None or hwnd == 0:
        return False
    try:
        vki = int(vk) & 0xFFFFFFFF
        sc  = _vk_to_sc(vki)
        if sc == 0:
            # Fall back to zero: Windows will still dispatch the message
            # for many VK codes (letters, numbers) even with sc=0.
            sc = 0
        GetWindowThreadProcessId = _USER32.GetWindowThreadProcessId
        GetCurrentThreadId       = _KERNEL32.GetCurrentThreadId
        AttachThreadInput        = _USER32.AttachThreadInput
        SendMessageW             = _USER32.SendMessageW
        ToUnicodeEx              = _USER32.ToUnicodeEx
        AttachThreadInput.restype = ctypes.c_bool
        AttachThreadInput.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_bool]
        GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        GetWindowThreadProcessId.restype  = ctypes.c_uint
        SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_int64]
        SendMessageW.restype  = ctypes.c_int64

        game_tid = GetWindowThreadProcessId(ctypes.c_void_p(hwnd), None)
        my_tid   = GetCurrentThreadId()
        attached = False
        if game_tid != 0 and game_tid != my_tid:
            # AttachThreadInput fails when called with the same TID (harmless),
            # and fails silently if the target thread isn't in a
            # get-message/dispatch-message loop — caller will see False and
            # fall through to scan-code / native-SendInput / PyAutoGUI.
            ok = bool(AttachThreadInput(ctypes.c_uint(game_tid),
                                        ctypes.c_uint(my_tid),
                                        ctypes.c_bool(True)))
            if not ok:
                return False
            attached = True
        try:
            sent_any = False
            # (A) Core: WM_KEYDOWN / WM_KEYUP with a realistic LPARAM.
            msg = WM_KEYDOWN if is_down else WM_KEYUP
            lparam = _lparam_from_vk_sc(vki, sc, is_down)
            SendMessageW(ctypes.c_void_p(hwnd), ctypes.c_uint(msg),
                         ctypes.c_size_t(vki), ctypes.c_int64(lparam))
            sent_any = True

            # (B) Bonus: WM_SYSKEYDOWN/SYSKEYUP — legacy MapleStory clients
            # sometimes handle movement from the SYS-key path because the
            # Alt (menu) modifier used to live on the same dispatch table
            # as jump + direction.  Don't fire SYS-UP separately without
            # a matching DOWN because some WndProcs interpret orphan UP
            # as a "menu was cancelled" internal side-effect.
            try:
                sys_msg = WM_SYSKEYDOWN if is_down else WM_SYSKEYUP
                SendMessageW(ctypes.c_void_p(hwnd), ctypes.c_uint(sys_msg),
                             ctypes.c_size_t(vki), ctypes.c_int64(lparam))
                sent_any = True
            except Exception:
                pass

            # (C) On DOWN only: WM_CHAR with ToUnicode translated UCS-2 LE
            # code point (what TranslateMessage would normally produce
            # inside the game's DispatchMessage).  Even direction keys
            # have a WM_CHAR translation (U+0000 or the VK itself); some
            # clients read movement from the WM_CHAR branch to avoid
            # per-process WH_KEYBOARD hooks swallowing WM_KEYDOWN.
            if is_down:
                try:
                    # Translate VK+sc → UCS-2 LE.  Don't worry if it's 0;
                    # Windows still dispatches WM_CHAR with wParam=0 and
                    # many legacy clients treat this as "a key happened".
                    wsz = ctypes.create_unicode_buffer(8)
                    kbd_state = (ctypes.c_ubyte * 256)()
                    n_chars = 0
                    try:
                        n_chars = int(ToUnicodeEx(ctypes.c_uint(vki),
                                                  ctypes.c_uint(sc),
                                                  ctypes.cast(kbd_state, ctypes.POINTER(ctypes.c_ubyte)),
                                                  wsz, ctypes.c_int(8),
                                                  ctypes.c_uint(0),
                                                  ctypes.c_void_p(0)))
                    except Exception:
                        n_chars = 0
                    wparam_char = 0
                    if n_chars > 0 and wsz.value:
                        # Send WM_CHAR for *each* translated codepoint (usually 1,
                        # sometimes 2 for surrogate pairs — direction keys
                        # never produce them, but we handle it anyway).
                        for ch in wsz.value[:max(n_chars, 1)]:
                            wparam_char = ord(ch) & 0xFFFFFFFF
                            lparam_char = _lparam_from_vk_sc(vki, sc, True)  # same bits as the DOWN that produced it
                            SendMessageW(ctypes.c_void_p(hwnd), ctypes.c_uint(WM_CHAR),
                                         ctypes.c_size_t(wparam_char),
                                         ctypes.c_int64(lparam_char))
                    else:
                        # No translation — fire WM_CHAR with wParam = VK
                        # code itself (legacy clients key off this value
                        # to detect "a keyboard event happened" even when
                        # ToUnicode produced nothing printable).
                        lparam_char = _lparam_from_vk_sc(vki, sc, True)
                        SendMessageW(ctypes.c_void_p(hwnd), ctypes.c_uint(WM_CHAR),
                                     ctypes.c_size_t(vki),
                                     ctypes.c_int64(lparam_char))
                    sent_any = True
                except Exception:
                    pass

            return sent_any
        finally:
            if attached and game_tid != 0 and game_tid != my_tid:
                try:
                    AttachThreadInput(ctypes.c_uint(game_tid),
                                      ctypes.c_uint(my_tid),
                                      ctypes.c_bool(False))
                except Exception:
                    pass
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Backend 6 — per-thread keyboard-state manipulation (SetKeyboardState).
#
# ⚠️  This is the most aggressive path we have for private-server clients
# that:
#   * disable all gamepad/DirectInput (so vgamepad does nothing), AND
#   * lock keyboard bindings (so letter-keys can't substitute arrows), AND
#   * hook user32.dll keybd_event / SendInput / WH_KEYBOARD /
#     WH_KEYBOARD_LL aggressively enough that Backends 1–5 are all dropped.
#
# Why it works:
#   Windows keeps a per-thread 256-byte "keyboard state table" (stored in
#   the thread's TIB → user32!gptinfo → akeys[256]).  Both GetKeyState and
#   GetKeyboardState read *only* from this table — they never look at the
#   actual HID/PS/2 hardware buffer.  When two threads are attached via
#   AttachThreadInput, they *SHARE* this table — meaning: if we
#   SetKeyboardState on our bot thread while attached to the game-window
#   thread, the game's next call to GetKeyState(VK_LEFT) immediately sees
#   0x80 (pressed) / 0x00 (not pressed) without any synthetic input ever
#   being generated by keybd_event / SendInput.  No user-mode hook in the
#   game process can detect this; there was no message, no callback, no
#   INPUT struct.
# ---------------------------------------------------------------------------
def _set_keyboard_state_attached(vk: int, is_down: bool, hwnd: int) -> bool:
    """Attach to the game window thread, SetKeyboardState the key state of
    ``vk`` directly in the shared per-thread key-state table, then detach.

    VK indexing into the 256-byte array is exactly ``vk & 0xFF`` per MSDN
    (GetKeyboardState returns a 256-byte array indexed by virtual-key
    code 0..255).  Bit 0x80 = pressed; bit 0x01 = toggled (we leave
    toggle alone — only write the high bit for down/up)."""
    if not _WIN_OK or _USER32 is None or _KERNEL32 is None or vk is None or hwnd is None or hwnd == 0:
        return False
    try:
        vki = int(vk) & 0xFF
        GetWindowThreadProcessId = _USER32.GetWindowThreadProcessId
        GetCurrentThreadId       = _KERNEL32.GetCurrentThreadId
        AttachThreadInput        = _USER32.AttachThreadInput
        GetKeyboardState         = _USER32.GetKeyboardState
        SetKeyboardState         = _USER32.SetKeyboardState
        AttachThreadInput.restype = ctypes.c_bool
        AttachThreadInput.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_bool]
        GetKeyboardState.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]
        GetKeyboardState.restype  = ctypes.c_bool
        SetKeyboardState.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]
        SetKeyboardState.restype  = ctypes.c_bool

        game_tid = GetWindowThreadProcessId(ctypes.c_void_p(hwnd), None)
        my_tid   = GetCurrentThreadId()
        # Attach first — attachment causes the keyboard-state table merge.
        attached = False
        if game_tid != 0 and game_tid != my_tid:
            ok = bool(AttachThreadInput(ctypes.c_uint(game_tid),
                                        ctypes.c_uint(my_tid),
                                        ctypes.c_bool(True)))
            if not ok:
                return False
            attached = True
        try:
            # (1) Read the current 256-byte table so we only mutate the
            #     single VK slot (leaving toggle state / modifier keys
            #     untouched — otherwise we'd mess up Shift/Ctrl/NumLock).
            ks_arr = (ctypes.c_ubyte * 256)()
            ok = bool(GetKeyboardState(ctypes.cast(ks_arr, ctypes.POINTER(ctypes.c_ubyte))))
            if not ok:
                # Can't read current state — initialise to all-zeros;
                # writing 0x80 will still make GetKeyState return pressed
                # because the *high* bit is the pressed-flag regardless.
                ks_arr = (ctypes.c_ubyte * 256)()
            # Toggle bit 0x01 is "CapsLock/NumLock toggled ON".  We never
            # change it — only the high nibble (pressed state).
            if is_down:
                ks_arr[vki] = ks_arr[vki] | 0x80
            else:
                ks_arr[vki] = ks_arr[vki] & 0x7F
            return bool(SetKeyboardState(ctypes.cast(ks_arr, ctypes.POINTER(ctypes.c_ubyte))))
        finally:
            if attached and game_tid != 0 and game_tid != my_tid:
                try:
                    AttachThreadInput(ctypes.c_uint(game_tid),
                                      ctypes.c_uint(my_tid),
                                      ctypes.c_bool(False))
                except Exception:
                    pass
    except Exception:
        return False


def _native_sendinput_sc(vk: int, is_down: bool) -> bool:
    """Low-level helper: fire a native ``SendInput`` with scan-code flags.

    Builds the INPUT struct ourselves (instead of delegating to pyautogui)
    so we can mark it as extended-key when needed and be 100% sure we're
    using the scan-code path.  Returns True on success (SendInput returned
    the number of inputs we asked it to submit, i.e. 1), False otherwise.
    """
    if not _WIN_OK or _USER32 is None or vk is None:
        return False
    sc = _vk_to_sc(vk)
    if sc == 0:
        return False
    dw_flags = _KEYEVENTF_SCANCODE
    if _is_extended_key(vk):
        dw_flags |= _KEYEVENTF_EXTENDEDKEY
    if not is_down:
        dw_flags |= _KEYEVENTF_KEYUP
    ki = _KEYBDINPUT(wVk=0,         # scan-code path: wVk MUST be 0
                     wScan=sc & 0xFFFF,
                     dwFlags=dw_flags,
                     time=0,
                     dwExtraInfo=None)
    inp = _INPUT()
    inp.type = _INPUT_KEYBOARD
    inp.ki   = ki
    cb_size = ctypes.sizeof(_INPUT)
    try:
        n = _USER32.SendInput(
            ctypes.c_uint(1),
            ctypes.byref(inp),
            ctypes.c_int(cb_size))
        return int(n) == 1
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Noise-padded SendInput (Backend 4a): _native_sendinput_ex
#
# Rationale: some private-server anti-cheats detect *bot* SendInput() calls
# by looking for suspicious call signatures:
#   * nInputs == 1                  (real USB keyboards usually deliver
#                                    INPUT arrays of 2~4 because the HID
#                                    report also carries modifier state)
#   * all cInputs use the exact same dwFlags (no UNICODE padding between
#     them / no zero-struct padding).
#   * deterministic timing between calls (no 1..4 ms USB interrupt jitter).
#
# This backend generates a 3-INPUT array with:
#   [0] zero-initialized keyboard struct (HID driver padding)
#   [1] the *real* scancode down/up event
#   [2] KEYEVENTF_UNICODE with wScan=0 (输入法合成器"空字符"——很多
#       外挂钩子直接跳过 UNICODE 结构，因为它们假定只在 IMM/CTFMON
#       进程里才会出现，但 Windows 内核允许任何进程发 UNICODE)
# 1..5 ms jitter BEFORE each call (once per bot key_down/key_up
# frame), so the timing distribution looks indistinguishable from a
# real USB 125 Hz / 1000 Hz keyboard.
# ---------------------------------------------------------------------------

def _native_sendinput_ex(vk: int, is_down: bool) -> bool:
    """Noise-padded variant of _native_sendinput_sc that mimics real USB
    HID timing + INPUT-batch signatures.  Intended to defeat cheap
    'SendInput nInputs=1' signature detectors on private MapleStory
    servers."""
    if not _WIN_OK or _USER32 is None or vk is None:
        return False
    sc = _vk_to_sc(vk)
    if sc == 0:
        return False
    dw_flags = _KEYEVENTF_SCANCODE
    if _is_extended_key(vk):
        dw_flags |= _KEYEVENTF_EXTENDEDKEY
    if not is_down:
        dw_flags |= _KEYEVENTF_KEYUP

    # Build a 3-INPUT array at *consecutive* memory addresses — the
    # Windows kernel reads them sequentially, so a plain Python list of
    # _INPUT() objects is fine as long as we pass the first byref +
    # correct sizeof.  We use (INPUT * 3) array for guaranteed contiguity.
    Arr3 = (_INPUT * 3)
    arr = Arr3()
    # slot 0: zeroed padding (real HID batch headers often carry a
    #         previously-reported modifier-packet that didn't change so
    #         the struct is 0 — kernel accepts this harmlessly).
    arr[0].type     = _INPUT_KEYBOARD
    arr[0].ki.wVk   = 0
    arr[0].ki.wScan = 0
    arr[0].ki.dwFlags = 0
    arr[0].ki.time = 0
    # slot 1: the actual down/up scan-code event
    arr[1].type         = _INPUT_KEYBOARD
    arr[1].ki.wVk       = 0
    arr[1].ki.wScan     = sc & 0xFFFF
    arr[1].ki.dwFlags   = dw_flags
    arr[1].ki.time      = 0
    # slot 2: KEYEVENTF_UNICODE with wScan=0 (NUL char).  This structure
    # is fully valid and processed by the IMM text composer (which
    # discards NUL) — hooks that only look at KEYEVENTF_SCANCODE/VK
    # never see it, but Windows still sees nInputs=3.
    arr[2].type       = _INPUT_KEYBOARD
    arr[2].ki.wVk     = 0
    arr[2].ki.wScan   = 0  # NUL character
    arr[2].ki.dwFlags = _KEYEVENTF_UNICODE
    arr[2].ki.time    = 0

    # 1..5 ms jitter BEFORE calling SendInput.  This randomises the
    # inter-call timing distribution enough to defeat simple "delta-t <
    # 0.5 ms = bot" fingerprint detectors on private servers.
    try:
        time.sleep(_py_random.uniform(0.001, 0.005))
    except Exception:
        pass

    cb = ctypes.sizeof(_INPUT)
    try:
        n = _USER32.SendInput(ctypes.c_uint(3),
                              ctypes.cast(arr, ctypes.POINTER(_INPUT)),
                              ctypes.c_int(cb))
        # Accept if >= 1 of the 3 inputs was inserted (the scan-code
        # one — UNICODE/NUL might be dropped by IMM, but that's fine).
        return int(n) >= 1
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Direct-syscall SendInput (Backend 4b): _ntusersendinput_direct
#
# Rationale: the *vast majority* of private-server shields only hook the
# user32.dll export "SendInput" (i.e. a 5-byte trampoline / inline IAT
# patch).  They almost *never* also hook the actual Windows kernel syscall
# entry point (win32u!NtUserSendInput / ntdll!NtUserSendInput) because
# that would require a kernel driver, which a private-server distributed
# .exe installer cannot ship.
#
# This backend:
#   * loads win32u.dll (Windows 10+; falls back to user32 on older builds)
#   * calls GetProcAddress("NtUserSendInput") directly
#   * calls it with the same UINT/cInputs/PINPUT/cbSize arguments.
#
# Because we resolve and call the *syscall stub* — bypassing the user32
# SendInput wrapper where the shield hook lives — the injected key event
# reaches the Windows kernel with no in-process interception at all.
# ---------------------------------------------------------------------------
_NTUSER_PROC = [None]  # cached result: ctypes function pointer or None
_NTUSER_ERR  = [None]  # cached error string so HEARTBEAT can show why off

def _resolve_ntusersendinput():
    """One-time resolver for NtUserSendInput.  Cached in module-level box."""
    if _NTUSER_PROC[0] is not None:
        return _NTUSER_PROC[0]
    if not _WIN_OK:
        return None
    candidate_libs = []
    if _WIN32U is not None: candidate_libs.append(_WIN32U)
    if _USER32  is not None: candidate_libs.append(_USER32)
    # Try loading win32u explicitly (it's not always in ctypes.windll
    # on older Python/Windows builds).
    try:
        lib = ctypes.WinDLL("win32u.dll", use_last_error=True)
        if lib not in candidate_libs:
            candidate_libs.insert(0, lib)
    except Exception:
        pass
    try:
        lib = ctypes.WinDLL("user32.dll", use_last_error=True)
        if lib not in candidate_libs:
            candidate_libs.append(lib)
    except Exception:
        pass
    for lib in candidate_libs:
        try:
            raw = getattr(lib, "NtUserSendInput", None)
            if raw is None:
                try:
                    GetProcAddress = ctypes.windll.kernel32.GetProcAddress
                    GetModuleHandleW = ctypes.windll.kernel32.GetModuleHandleW
                    GetModuleHandleW.restype = ctypes.c_void_p
                    GetProcAddress.restype = ctypes.c_void_p
                    GetProcAddress.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
                    # Find which DLL we were looking at via handle lookup
                    names = []
                    if lib is _WIN32U: names.append(b"win32u.dll\0")
                    if lib is _USER32:  names.append(b"user32.dll\0")
                    for nm in names:
                        h = GetModuleHandleW(nm)
                        if not h: continue
                        addr = GetProcAddress(h, b"NtUserSendInput\0")
                        if addr:
                            # wrap as function pointer
                            NT = ctypes.WINFUNCTYPE(ctypes.c_uint,     # UINT return = cInputs inserted
                                                   ctypes.c_uint,     # cInputs
                                                   ctypes.c_void_p,   # pInputs (INPUT array)
                                                   ctypes.c_int)      # cbSize
                            proc = NT(addr)
                            _NTUSER_PROC[0] = proc
                            return proc
                except Exception:
                    continue
            if raw is not None:
                raw.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_int]
                raw.restype  = ctypes.c_uint
                _NTUSER_PROC[0] = raw
                return raw
        except Exception:
            continue
    _NTUSER_ERR[0] = "NtUserSendInput not exported from win32u.dll / user32.dll on this Windows build"
    return None


def _ntusersendinput_direct(vk: int, is_down: bool) -> bool:
    """Call NtUserSendInput directly (bypasses user32!SendInput wrapper
    where most private-server hooks live)."""
    proc = _resolve_ntusersendinput()
    if proc is None or not _WIN_OK or vk is None:
        return False
    sc = _vk_to_sc(vk)
    if sc == 0:
        return False
    dw_flags = _KEYEVENTF_SCANCODE
    if _is_extended_key(vk):
        dw_flags |= _KEYEVENTF_EXTENDEDKEY
    if not is_down:
        dw_flags |= _KEYEVENTF_KEYUP
    inp = _INPUT()
    inp.type     = _INPUT_KEYBOARD
    inp.ki.wVk   = 0
    inp.ki.wScan = sc & 0xFFFF
    inp.ki.dwFlags = dw_flags
    inp.ki.time = 0
    cb = ctypes.sizeof(_INPUT)
    try:
        # Same jitter as backend 4a: random 1-5 ms before the syscall.
        time.sleep(_py_random.uniform(0.001, 0.005))
        n = proc(ctypes.c_uint(1),
                 ctypes.cast(ctypes.byref(inp), ctypes.c_void_p),
                 ctypes.c_int(cb))
        return int(n) == 1
    except Exception:
        return False


def key_down(key):
    '''
    Press key down.

    Windows backend order (first-success wins):
      1. PostMessage WM_KEYDOWN (target HWND message queue)
      2. AttachThreadInput + SendMessage(WM_KEYDOWN) synchronous
      3. keybd_event with KEYEVENTF_SCANCODE
      4. native ctypes SendInput with KEYEVENTF_SCANCODE
      5a.✨ _native_sendinput_ex noise-padded nInputs=3 (USB HID 仿真)
      5b.✨ _ntusersendinput_direct 直接 win32u!NtUserSendInput syscall
      6. PyAutoGUI.keyDown (SendInput via pyautogui wrapper)

    On macOS the code-path is unchanged.
    '''
    if is_mac():
        try:
            pyautogui.keyDown(key)
            _set_backend("macOS", True)
        except pyautogui.FailSafeException:
            logger.warning("[key_down] pyautogui failsafe triggered.")
            recover_mouse()
            _set_backend("macOS", False)
        return

    vk = _key_name_to_vk(key)

    # ------------------------------------------------------------------
    # False-positive self-check helper (gas-verify).
    # PostMessage / SendMessage / SetKeyboardState can all "succeed"
    # from an API-return-value perspective while never actually reaching
    # the game's GetAsyncKeyState / GetKeyState reader.  After each of
    # those backends we call this helper; if it returns False we treat
    # the backend as a fail and keep falling through.
    # ------------------------------------------------------------------
    def _gas_verify_down(vki: int) -> bool:
        try:
            if _USER32 is None: return False
            # Sleep 2-4 ms so that a recently-queued PostMessage/WM_KEYDOWN
            # has time to reach the kernel and flip the async-key state.
            time.sleep(_py_random.uniform(0.002, 0.004))
            s = _USER32.GetAsyncKeyState(ctypes.c_int(vki & 0xFF))
            # bit 0x8000 = currently pressed; bit 0x0001 = pressed at least
            # once since the last GetAsyncKeyState call.  Either counts as
            # the OS-level synthetic event having actually landed.
            return bool(s & 0x8000) or bool(s & 0x0001)
        except Exception:
            return False
    def _gas_verify_up(vki: int) -> bool:
        try:
            if _USER32 is None: return False
            time.sleep(_py_random.uniform(0.002, 0.004))
            s = _USER32.GetAsyncKeyState(ctypes.c_int(vki & 0xFF))
            # UP: the "currently pressed" high bit MUST be 0.  We accept
            # either (a) high bit == 0 or (b) low bit == 1 and high bit
            # == 0 (transition happened recently), as long as the bit
            # that the game actually uses for "is key held right now" is
            # off.
            return not bool(s & 0x8000)
        except Exception:
            return False

    # Backend #0 (NEW — promoted because gas_map confirms it's the ONLY
    # one that consistently flips GetAsyncKeyState on this client).
    #   keybd_event with KEYEVENTF_SCANCODE.
    #
    # ATK#3 dual-path: for non-direction keys (letter/space/ctrl used for
    # jump + attack), private-server shields often white-list ONLY the 4
    # arrow scan-codes and drop everything else.  So for any VK that is
    # NOT in {VK_LEFT,VK_UP,VK_RIGHT,VK_DOWN} we fire *both* the scan-code
    # path AND a direct SetKeyboardState-attached fallback in the same
    # backend call, returning success if *either* passes the GAS check.
    # This keeps attack/jump alive even when their specific scan codes
    # are blacklisted while arrows work.
    if _WIN_OK and vk is not None and _USER32 is not None:
        sc = _vk_to_sc(vk)
        # arrow direction VKs — known to be accepted by the shield in
        # the user's current build (gas_map bit 0 == P for them), so use
        # the plain scan-code path they trust.
        _ARROW_VKS = (0x25, 0x26, 0x27, 0x28)  # LEFT,UP,RIGHT,DOWN
        try:
            vki = int(vk) & 0xFF
        except Exception:
            vki = 0
        is_arrow = bool(vki in _ARROW_VKS)
        if sc != 0 or not is_arrow:
            try:
                # Path 0A — scan-code via keybd_event (works for arrows
                # and for letters when the shield is lenient).
                if sc != 0:
                    fl = _KEYEVENTF_SCANCODE
                    if _is_extended_key(vk):
                        fl |= _KEYEVENTF_EXTENDEDKEY
                    if not is_down:
                        fl |= _KEYEVENTF_KEYUP
                    _USER32.keybd_event(ctypes.c_byte(0),
                                        ctypes.c_byte(sc & 0xFF),
                                        ctypes.c_uint(fl),
                                        ctypes.c_void_p(0))
                passed_0a = _gas_verify_down(vki) if is_down else \
                            _gas_verify_up   (vki)

                if passed_0a and is_arrow:
                    _set_backend("keybd_sc", True)
                    return
                if passed_0a and not is_arrow:
                    _set_backend("keybd_sc_letter", True)
                    return

                # Path 0B — for non-arrow keys that failed scan-code
                # injection (shield drops the scan code before it reaches
                # the OS async table), try SetKeyboardState_attached
                # immediately in the same backend.  This works for exactly
                # the same reason it worked as Backend 6: the game thread
                # internal GetKeyState reads the shared per-thread state
                # table, not the global async table.
                if not is_arrow:
                    hwnd = _ensure_game_hwnd()
                    if hwnd != 0:
                        try:
                            if _set_keyboard_state_attached(vki, is_down=is_down, hwnd=int(hwnd)):
                                # No GAS verify here — see Backend 6 note
                                # about per-thread vs global table.
                                _set_backend("keybd_sc_fallback_stkbd", True)
                                return
                        except Exception:
                            pass
                    # Still in backend 0 for non-arrow, but 0A + 0B
                    # both failed? fall through; the PostMessage/
                    # SendMessage/native/etc backends below will try.
            except Exception:
                pass

    # Backend #1: PostMessage WM_KEYDOWN (may generate false positives —
    # verify with GetAsyncKeyState before accepting as success).
    if _WIN_OK and vk is not None:
        hwnd = _ensure_game_hwnd()
        if hwnd != 0:
            try:
                sc = _vk_to_sc(vk)
                lparam_down = _MAKELPARAM(rep=1, scan=sc, ext=0, ctx=0, prev=0, trans=0)
                res = _USER32.PostMessageW(ctypes.c_void_p(hwnd), _WM_KEYDOWN,
                                           ctypes.c_size_t(int(vk) & 0xFFFFFFFF),
                                           ctypes.c_ssize_t(lparam_down))
                if res != 0 and _gas_verify_down(int(vk)):
                    _set_backend("PostMessage", True)
                    return
            except Exception:
                pass

    # Backend #2: AttachThreadInput + SendMessage(synchronous) WM_KEYDOWN
    # (bypasses WH_KEYBOARD_LL global hook in many private servers)
    if _WIN_OK and vk is not None:
        hwnd = _ensure_game_hwnd()
        if hwnd != 0:
            try:
                if _send_key_attached(int(vk), is_down=True, hwnd=hwnd):
                    if _gas_verify_down(int(vk)):
                        _set_backend("Attached_SendMessage", True)
                        return
            except Exception:
                pass

    # Backend #3 (was #4): native SendInput (scan-code path)
    if _WIN_OK and vk is not None:
        if _native_sendinput_sc(int(vk), is_down=True) and _gas_verify_down(int(vk)):
            _set_backend("native_SendInput", True)
            return

    # Backend #4a (was #4a): ✨ noise-padded native SendInput (nInputs=3
    # batch + 1..5ms random jitter + KEYEVENTF_UNICODE NUL padding)
    if _WIN_OK and vk is not None:
        if _native_sendinput_ex(int(vk), is_down=True) and _gas_verify_down(int(vk)):
            _set_backend("native_SendInput_ex", True)
            return

    # Backend #4b (was #4b): ✨ direct-syscall NtUserSendInput
    if _WIN_OK and vk is not None:
        if _ntusersendinput_direct(int(vk), is_down=True) and _gas_verify_down(int(vk)):
            _set_backend("NtUserSendInput_direct", True)
            return

    # Backend #6 (was #6): ✨✨ per-thread SetKeyboardState attached to
    # the game window's thread — note: SetKeyboardState only writes the
    # *per-thread* state table, not the global async table, so the GAS
    # check (global) is expected to miss it.  We therefore accept the
    # SetKeyboardState return value directly (True means the API accepted
    # the write; the game's GetKeyState inside the attached thread will
    # see it).
    if _WIN_OK and vk is not None:
        hwnd = _ensure_game_hwnd()
        if hwnd != 0:
            try:
                if _set_keyboard_state_attached(int(vk), is_down=True, hwnd=int(hwnd)):
                    _set_backend("SetKeyboardState_attached", True)
                    return
            except Exception:
                pass

    # Backend #5 (last-resort): PyAutoGUI (SendInput via library)
    try:
        pyautogui.keyDown(key)
        if _gas_verify_down(int(vk) if vk is not None else 0) or vk is None:
            _set_backend("PyAutoGUI", True)
        else:
            # PyAutoGUI call didn't actually flip the global async-key
            # state either — still mark it as used so logs are honest,
            # but count as a failure for the consec-fail counter.
            _set_backend("PyAutoGUI", False)
    except pyautogui.FailSafeException:
        logger.warning("[key_down] pyautogui failsafe triggered.")
        recover_mouse()
        _set_backend("PyAutoGUI", False)
    except Exception:
        _set_backend("PyAutoGUI", False)


def key_up(key):
    '''
    Release key (mirrors key_down's backend order).
    '''
    if is_mac():
        try:
            pyautogui.keyUp(key)
            _set_backend("macOS", True)
        except pyautogui.FailSafeException:
            logger.warning("[key_up] pyautogui failsafe triggered.")
            recover_mouse()
            _set_backend("macOS", False)
        return

    vk = _key_name_to_vk(key)

    # --- reuse gas-verify helpers (mirror of key_down) -----------------
    def _gas_verify_down_up(vki: int, expect_down: bool) -> bool:
        try:
            if _USER32 is None: return False
            time.sleep(_py_random.uniform(0.002, 0.004))
            s = _USER32.GetAsyncKeyState(ctypes.c_int(vki & 0xFF))
            high_now = bool(s & 0x8000)
            if expect_down:
                return high_now or bool(s & 0x0001)
            # UP: expect "currently pressed" bit to be OFF
            return (not high_now)
        except Exception:
            return False

    # Backend #0 (promoted): keybd_event SCANCODE | KEYUP  (mirrors the
    # key_down Backend #0 — dual-path 0A scan-code + 0B stkbd-attached
    # fallback for non-arrow keys whose scan codes the shield drops).
    if _WIN_OK and vk is not None and _USER32 is not None:
        sc = _vk_to_sc(vk)
        _ARROW_VKS = (0x25, 0x26, 0x27, 0x28)  # LEFT,UP,RIGHT,DOWN
        try:
            vki = int(vk) & 0xFF
        except Exception:
            vki = 0
        is_arrow = bool(vki in _ARROW_VKS)
        if sc != 0 or not is_arrow:
            try:
                # Path 0A — scan-code via keybd_event
                if sc != 0:
                    fl = _KEYEVENTF_SCANCODE | _KEYEVENTF_KEYUP
                    if _is_extended_key(vk):
                        fl |= _KEYEVENTF_EXTENDEDKEY
                    _USER32.keybd_event(ctypes.c_byte(0),
                                        ctypes.c_byte(sc & 0xFF),
                                        ctypes.c_uint(fl),
                                        ctypes.c_void_p(0))
                passed_0a = _gas_verify_down_up(vki, expect_down=False)
                if passed_0a and is_arrow:
                    _set_backend("keybd_sc", True)
                    return
                if passed_0a and not is_arrow:
                    _set_backend("keybd_sc_letter", True)
                    return
                # Path 0B — for non-arrow keys whose scan-code release
                # event was dropped, fall back to shared-state-keyboard-
                # table manipulation (SetKeyboardState_attached).
                if not is_arrow:
                    hwnd = _ensure_game_hwnd()
                    if hwnd != 0:
                        try:
                            if _set_keyboard_state_attached(vki, is_down=False, hwnd=int(hwnd)):
                                _set_backend("keybd_sc_fallback_stkbd", True)
                                return
                        except Exception:
                            pass
            except Exception:
                pass

    # Backend #1: PostMessage WM_KEYUP (with GAS self-check to avoid
    # false positives that would mask the keybd_event backend).
    if _WIN_OK and vk is not None:
        hwnd = _ensure_game_hwnd()
        if hwnd != 0:
            try:
                sc = _vk_to_sc(vk)
                lparam_up = _MAKELPARAM(rep=1, scan=sc, ext=0, ctx=0, prev=1, trans=1)
                res = _USER32.PostMessageW(ctypes.c_void_p(hwnd), _WM_KEYUP,
                                           ctypes.c_size_t(int(vk) & 0xFFFFFFFF),
                                           ctypes.c_ssize_t(lparam_up))
                if res != 0 and _gas_verify_down_up(int(vk), expect_down=False):
                    _set_backend("PostMessage", True)
                    return
            except Exception:
                pass

    # Backend #2: AttachThreadInput + SendMessage WM_KEYUP
    if _WIN_OK and vk is not None:
        hwnd = _ensure_game_hwnd()
        if hwnd != 0:
            try:
                if _send_key_attached(int(vk), is_down=False, hwnd=hwnd):
                    if _gas_verify_down_up(int(vk), expect_down=False):
                        _set_backend("Attached_SendMessage", True)
                        return
            except Exception:
                pass

    # Backend #3: native SendInput (scan-code path)
    if _WIN_OK and vk is not None:
        if _native_sendinput_sc(int(vk), is_down=False) and \
           _gas_verify_down_up(int(vk), expect_down=False):
            _set_backend("native_SendInput", True)
            return

    # Backend #4a: ✨ noise-padded native SendInput
    if _WIN_OK and vk is not None:
        if _native_sendinput_ex(int(vk), is_down=False) and \
           _gas_verify_down_up(int(vk), expect_down=False):
            _set_backend("native_SendInput_ex", True)
            return

    # Backend #4b: ✨ direct-syscall NtUserSendInput
    if _WIN_OK and vk is not None:
        if _ntusersendinput_direct(int(vk), is_down=False) and \
           _gas_verify_down_up(int(vk), expect_down=False):
            _set_backend("NtUserSendInput_direct", True)
            return

    # Backend #6: ✨✨ SetKeyboardState attached — mirror of key_down
    # (API-return-only trusted, GAS check is skipped because the per-thread
    # state table isn't the global async-key state table).
    if _WIN_OK and vk is not None:
        hwnd = _ensure_game_hwnd()
        if hwnd != 0:
            try:
                if _set_keyboard_state_attached(int(vk), is_down=False, hwnd=int(hwnd)):
                    _set_backend("SetKeyboardState_attached", True)
                    return
            except Exception:
                pass

    # Backend #5 (last-resort): PyAutoGUI keyUp
    try:
        pyautogui.keyUp(key)
        ok = True
        if vk is not None:
            ok = _gas_verify_down_up(int(vk), expect_down=False) or vk is None
        if ok:
            _set_backend("PyAutoGUI", True)
        else:
            _set_backend("PyAutoGUI", False)
    except pyautogui.FailSafeException:
        logger.warning("[key_up] pyautogui failsafe triggered.")
        recover_mouse()
        _set_backend("PyAutoGUI", False)
    except Exception:
        _set_backend("PyAutoGUI", False)

def recover_mouse():
    '''
    Move mouse back to center to avoid pyautogui failsafe
    '''
    pyautogui.FAILSAFE = False # Temp disasble failsafe to avoid nested exception

    screen_w, screen_h = pyautogui.size()
    pyautogui.moveTo(screen_w // 2, screen_h // 2)
    time.sleep(0.2) # Give it a moment to "cool down"

    pyautogui.FAILSAFE = True # Recover failsafe

def _press_key_real(key, duration=0.05):
    '''
    Simulates a key press for a specified duration.

    When the vgamepad backend is available, also fires the gamepad button
    mapped to ``key`` (if any) *during* the same window, holding both
    inputs until ``duration`` expires.  This "belt & suspenders" approach
    lets MapleStory receive the command on both HID pipelines so even if
    the anti-cheat shield kills the keyboard path the gamepad axis/button
    event still lands.
    '''
    if key:
        _try_vgp_button(key, press=True)
        key_down(key)
        time.sleep(duration)
        key_up(key)
        _try_vgp_button(key, press=False)

# Expose duration-capable press_key as the public symbol (existing callers
# ``press_key(k, 0.1)`` / ``press_key(k)`` all resolve to this one).
#
# NOTE: Python module loading guarantees that the ``press_key(key)``
# placeholder defined at the top of this file exists during ``key_down`` /
# ``key_up`` function definitions (they don't call press_key so no circular
# reference problem here).  Once the parser reaches this statement we
# overwrite the symbol globally so later callers see the variadic version,
# and we also replace ``_press_key_real`` references in the earlier
# placeholder to avoid any stale closures.
def press_key(*args, **kwargs):
    """Unified public ``press_key`` supporting ``(key)`` or ``(key, duration)``."""
    _press_key_real(*args, **kwargs)
# Replace any internal references to the earlier compat shim by overwriting
# the module's own globals table.  This avoids "call-early" issues in code
# that captured the old function reference during import-time side effects.
try:
    import sys as _sys
    _m = _sys.modules[__name__]
    if getattr(_m, "press_key", None) is not press_key:
        setattr(_m, "press_key", press_key)
    del _m, _sys
except Exception:
    pass


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
        # Once-per-startup "anti-cheat blocks everything" warning.  The KB
        # backend telemetry increments a consecutive-failure counter; once
        # it crosses a threshold we emit a single actionable warning so
        # users don't stare at "keys=LEFT but character stands still" logs
        # for 10 minutes without guidance.
        self._kb_block_warned = False

        # Optional ViGEm virtual-gamepad backend:
        #   Enabled when EITHER
        #     (a) env var MAPLEBOT_USE_VGAMEPAD is truthy ("1", "true", "yes", ...)
        #     (b) cfg["key"]["use_vgamepad"] is truthy
        #   Disabled (default) when neither is set — and even if one *is*
        #   set, enable_vgamepad() may still fail gracefully (missing
        #   vgamepad package / ViGEmBus driver not installed) and the bot
        #   continues on the keyboard backends.
        self.vgamepad_enabled = False
        try:
            import os as _os
            env_on = str(_os.environ.get("MAPLEBOT_USE_VGAMEPAD", "")).strip().lower()
            env_on = env_on in ("1", "true", "yes", "on", "y")
            cfg_on = False
            try:
                _kc = self.cfg.get("key") or {}
                cfg_on = bool(_kc.get("use_vgamepad", False))
            except Exception:
                pass
            if env_on or cfg_on:
                self.vgamepad_enabled = enable_vgamepad()
        except Exception:
            self.vgamepad_enabled = False
        # Build default button map: YAML-key → gamepad-button.  Users can
        # override in cfg["key"]["vgamepad_buttons"] = {"jump":"A", ...}
        self._vgp_default_map = {
            # logical YAML key name  →  Xbox 360 pad button
            "jump":               "A",
            "directional_attack": "X",
            "aoe_skill":          "X",          # both attack modes → X
            "add_hp":             "Y",
            "add_mp":             "B",
            "teleport":           "LB",         # bumper = convenient skill
            "return_home":        "RB",         # opposite bumper
            "party":              "START",
        }
        # Build the lowercase-keystring → XUSB_BUTTON.* map and push it
        # into the module-level _VGP_BUTTON_MAP so the free functions
        # key_down/key_up/press_key can look it up.
        try:
            constants = _build_vgp_constants_map() if vgamepad_available() else None
            if constants:
                overrides = {}
                try:
                    _kc = self.cfg.get("key") or {}
                    overrides = dict(_kc.get("vgamepad_buttons") or {})
                except Exception:
                    overrides = {}
                mapping = {}
                for yaml_name, btn_name in {**self._vgp_default_map, **overrides}.items():
                    try:
                        keystr = str(self.cfg["key"].get(yaml_name, "")).strip().lower()
                    except Exception:
                        keystr = ""
                    if not keystr: continue
                    bcn = str(btn_name).strip().upper()
                    code = constants.get(bcn)
                    if code is None: continue
                    mapping[keystr] = int(code)
                # Also allow the user to explicitly map raw key strings
                # directly:  {"f":"X", "space":"A", ...}
                try:
                    direct = {}
                    try:
                        _kc = self.cfg.get("key") or {}
                        direct = dict(_kc.get("vgamepad_raw_keys") or {})
                    except Exception:
                        direct = {}
                    for raw_key, btn_name in direct.items():
                        ks = str(raw_key).strip().lower()
                        bcn = str(btn_name).strip().upper()
                        code = constants.get(bcn)
                        if code is None: continue
                        mapping[ks] = int(code)
                except Exception:
                    pass
                # Export to module level so free functions can see it.
                global _VGP_BUTTON_MAP
                _VGP_BUTTON_MAP.clear()
                _VGP_BUTTON_MAP.update(mapping)
                if mapping:
                    logger.info(
                        "[KeyBoardController] vgamepad button map built: "
                        f"{len(mapping)} keyboard-keys → pad buttons.  Keys in "
                        f"map = {sorted(list(mapping.keys()))!r}"
                    )
        except Exception:
            pass

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

        # Propagate title tokens *and* resolved HWND (if available) to the
        # module-level PostMessage / scan-code backends so they know which
        # window to target before any explicit ``set_window_title`` call.
        if not is_mac():
            try:
                register_game_title_tokens(self.cfg_tokens)
            except Exception:
                pass

        # Start keyboard control thread
        threading.Thread(target=self.run, daemon=True).start()

        logger.info("[KeyBoardController] Init done")

    def set_game_hwnd(self, hwnd):
        '''
        Register the resolved game HWND with the PostMessage backend.

        This is best-effort: passing ``None`` / ``0`` / an invalid HWND is
        silently ignored (the PostMessage backend falls back to an
        EnumWindows sweep on each key_down/key_up call instead).  Invalid
        HWNDs (e.g. the game client was closed and reopened) are caught by
        ``_ensure_game_hwnd`` which validates them with ``IsWindow``.
        '''
        try:
            h = int(hwnd or 0)
            register_game_hwnd(h)
            if h != 0:
                logger.info(
                    "[KeyBoardController] registered game HWND "
                    f"{hex(h)} with PostMessage keyboard backend"
                )
        except Exception:
            pass

    def set_window_title(self, exact_title: str):
        '''
        Called after ``GameWindowCapturor`` successfully resolves the exact
        foreground-window title (e.g. ``"冒险岛怀旧服"``) so the keyboard
        controller can match against it exactly (and fall back to substring
        tokens as well).

        In addition, we re-push the updated token list at the PostMessage
        backend and try a FindWindow lookup so the *HWND* is also already
        primed by the time any key press needs to be synthesised.

        Logs the update on first change so users can verify the link between
        the capture thread and this controller.
        '''
        if not exact_title or exact_title == self.window_title:
            # Still refresh tokens on the backend side even if the title
            # already matches (the caller may be re-confirming the window
            # after a capture-restart, and the module-level EnumWindows
            # sweep benefits from fresh tokens).
            if not is_mac():
                try:
                    register_game_title_tokens(self.cfg_tokens)
                except Exception:
                    pass
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

        if not is_mac():
            # Keep module-level PostMessage/scan-code backends in sync.
            try:
                register_game_title_tokens(self.cfg_tokens)
            except Exception:
                pass
            # Try a cheap FindWindow (exact title) — this gives the
            # PostMessage backend a warm HWND without relying on the
            # caller to call set_game_hwnd explicitly.
            try:
                import win32gui  # already a project dep; imported lazily
                hwnd = win32gui.FindWindow(None, self.window_title)
                if hwnd != 0:
                    self.set_game_hwnd(hwnd)
            except Exception:
                pass

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
            # IMPORTANT: "foreground" is NOT the same as "owns the keyboard
            # focus queue".  Many private-server / anti-cheat clients (e.g.
            # 冒险岛怀旧服) only accept synthetic keystrokes after we have
            # explicitly attached to their input thread and called SetFocus
            # + a TOPMOST bounce.  When is_active is already True we used to
            # early-return and SKIP that whole sequence — the visible symptom
            # is exactly what the user reported: pressing F1 no longer aligns
            # / raises the game window, and the character stops responding to
            # synthetic keys even though the window looks focused.  So we now
            # run the force-focus routine here too — but ONLY ONCE per
            # continuous "active" streak.  Running it every second was a
            # mistake: the AttachThreadInput + TOPMOST bounce briefly
            # re-asserts the window/input state, which INTERRUPTS a key that
            # the bot is trying to *hold down* for continuous walking.  The
            # visible symptom is exactly what the user reported: on start the
            # character takes one small step left and then freezes (each 1 Hz
            # focus bounce cancels the held LEFT key).  So we latch a flag and
            # only force-focus the first time we observe the window active;
            # it is reset whenever the window loses focus (see below), so a
            # genuine focus loss still re-triggers the full sequence.
            if not getattr(self, "_force_focus_done_while_active", False):
                self._force_focus_done_while_active = True
                try:
                    self._force_focus_game_window()
                except Exception:
                    pass
            return True

        # Window is NOT active right now → clear the latch so that once we
        # regain focus we force-focus exactly once again.
        self._force_focus_done_while_active = False

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
            import win32process
            import win32api
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
                # ============================================================
                # Robust foreground activation sequence (Experience 676134).
                #
                # SetForegroundWindow is *not* sufficient for many private
                # servers because Windows enforces "foreground permission"
                # rules: only the thread that currently owns the foreground
                # window may transfer it — otherwise SetForegroundWindow
                # silently "flashes the taskbar button" and the *real*
                # foreground stays on the Qt UI / Python terminal.  The
                # symptom the user sees: keys are sent but Python's Qt
                # window blinks instead of the game consuming them.
                #
                # Steps we run (ordered by likely success):
                #   1. ASFW_ANY: grant the game process foreground rights
                #   2. If iconic: SW_RESTORE
                #   3. AttachThreadInput: our thread <-> game UI thread so
                #      we can steal focus from inside its input queue
                #   4. BringWindowToTop + SetForegroundWindow + SetFocus
                #   5. (Briefly) HWND_TOPMOST then HWND_NOTOPMOST so the
                #      OS compositor keeps the window above everything even
                #      if the game's internal renderer doesn't call
                #      SetForegroundWindow itself
                #   6. Detach thread input (CRITICAL — if we leave the two
                #      threads attached one hang will freeze both processes)
                # ============================================================
                try:
                    # 1. Grant foreground permission to ALL processes so
                    #    the game's window thread doesn't get denied when
                    #    we hand it focus.
                    ASFW_ANY = -1
                    try:
                        _USER32.AllowSetForegroundWindow(ASFW_ANY)
                    except Exception:
                        # AllowSetForegroundWindow may be absent on very
                        # old Windows builds; treat as non-fatal.
                        pass

                    if win32gui.IsIconic(hwnd):
                        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

                    # 3. AttachThreadInput: our thread <-> game UI thread
                    game_tid, _ = win32process.GetWindowThreadProcessId(hwnd)
                    my_tid   = win32api.GetCurrentThreadId()
                    attached = False
                    if game_tid and game_tid != my_tid:
                        try:
                            win32process.AttachThreadInput(my_tid, game_tid, True)
                            attached = True
                        except Exception:
                            attached = False

                    try:
                        # 4. Bring + set-foreground + set-focus
                        try:
                            win32gui.BringWindowToTop(hwnd)
                        except Exception:
                            pass
                        try:
                            win32gui.SetForegroundWindow(hwnd)
                        except Exception:
                            pass
                        try:
                            # SetFocus accepts child HWND; if the game is
                            # composed of a parent + rendering child both
                            # calls may fail but that's OK (we still
                            # transferred input-queue ownership above).
                            win32gui.SetFocus(hwnd)
                        except Exception:
                            pass

                        # 5. TOPMOST bounce: 20 ms topmost, then un-topmost
                        try:
                            win32gui.SetWindowPos(hwnd,
                                                  win32con.HWND_TOPMOST,
                                                  0, 0, 0, 0,
                                                  win32con.SWP_NOMOVE |
                                                  win32con.SWP_NOSIZE)
                            time.sleep(0.02)
                            win32gui.SetWindowPos(hwnd,
                                                  win32con.HWND_NOTOPMOST,
                                                  0, 0, 0, 0,
                                                  win32con.SWP_NOMOVE |
                                                  win32con.SWP_NOSIZE |
                                                  win32con.SWP_NOACTIVATE)
                        except Exception:
                            pass
                    finally:
                        # 6. CRITICAL: always detach.  If the two thread
                        #    input queues stay attached forever and one
                        #    thread blocks on a kernel sync object, the
                        #    other thread will freeze with it.
                        if attached:
                            try:
                                win32process.AttachThreadInput(my_tid,
                                                               game_tid,
                                                               False)
                            except Exception:
                                pass

                    time.sleep(0.08)
                    return self.is_game_window_active()[0]
                except Exception:
                    # Fallback to a simple SetForegroundWindow if the
                    # robust sequence raised anywhere unexpected.
                    try:
                        if win32gui.IsIconic(hwnd):
                            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                        win32gui.SetForegroundWindow(hwnd)
                        time.sleep(0.05)
                    except Exception:
                        pass
                    return self.is_game_window_active()[0]
        except Exception:
            pass
        return False

    def _force_focus_game_window(self):
        '''
        Unconditionally (re)assert real keyboard focus on the game window.

        This is the same robust foreground+focus sequence used inside
        ensure_game_window_active's fallback path, but factored out so it
        can also run when the window merely *looks* active (is_active True).

        Steps: locate HWND via tokens -> AllowSetForegroundWindow(ASFW_ANY)
        -> restore if iconic -> AttachThreadInput(our<->game) ->
        BringWindowToTop + SetForegroundWindow + SetFocus -> brief
        HWND_TOPMOST bounce -> detach (always).

        Returns True on best-effort success, False otherwise.  Safe to call
        every ~1 s; all failures are swallowed.
        '''
        try:
            import win32gui
            import win32con
            import win32process
            import win32api
        except Exception:
            return False

        hwnd = 0
        try:
            if self.window_title:
                hwnd = win32gui.FindWindow(None, self.window_title)
            if hwnd == 0:
                found = [0]
                def _cb(h, _):
                    try:
                        if not win32gui.IsWindowVisible(h):
                            return
                        if self._title_matches(win32gui.GetWindowText(h)):
                            found[0] = h
                    except Exception:
                        pass
                try:
                    win32gui.EnumWindows(_cb, None)
                except Exception:
                    pass
                hwnd = found[0]
        except Exception:
            return False

        if hwnd == 0:
            return False

        try:
            try:
                _USER32.AllowSetForegroundWindow(-1)  # ASFW_ANY
            except Exception:
                pass
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

            game_tid, _ = win32process.GetWindowThreadProcessId(hwnd)
            my_tid = win32api.GetCurrentThreadId()
            attached = False
            if game_tid and game_tid != my_tid:
                try:
                    win32process.AttachThreadInput(my_tid, game_tid, True)
                    attached = True
                except Exception:
                    attached = False
            try:
                for _fn in (lambda: win32gui.BringWindowToTop(hwnd),
                            lambda: win32gui.SetForegroundWindow(hwnd),
                            lambda: win32gui.SetFocus(hwnd)):
                    try:
                        _fn()
                    except Exception:
                        pass
                try:
                    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                          win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                    time.sleep(0.02)
                    win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                                          win32con.SWP_NOMOVE | win32con.SWP_NOSIZE |
                                          win32con.SWP_NOACTIVATE)
                except Exception:
                    pass
            finally:
                if attached:
                    try:
                        win32process.AttachThreadInput(my_tid, game_tid, False)
                    except Exception:
                        pass
            return True
        except Exception:
            return False

    def release_all_key(self):
        '''
        Release all key (keyboard + any held gamepad buttons + stick zero).
        '''
        key_up("left")
        key_up("right")
        key_up("up")
        key_up("down")
        # Also release attack keys to stop any ongoing attacks
        key_up(self.attack_key)
        # Release any gamepad buttons we think are still held (belt &
        # suspenders: even on keyboard-only runs, a stray held button
        # would keep the character walking infinitely without this).
        try:
            if vgamepad_available():
                # Center left stick
                _vgp_set_axis(0.0, 0.0)
                # Release every button we ever tracked
                global _VGP_BUTTONS_DOWN
                for code in list(_VGP_BUTTONS_DOWN):
                    try:
                        _vgp_release_button(code)
                    except Exception:
                        pass
                _VGP_BUTTONS_DOWN.clear()
        except Exception:
            pass

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
        t_next_kb_diag     = 0.0   # next time we emit the 3-s "key status" line

        def do_action_key(kind, key):
            nonlocal t_first_action_banner
            if t_first_action_banner:
                # Print a one-time banner so users can tell that the
                # controller is actually dispatching keys (not just looping).
                is_active, foreground = self.is_game_window_active()
                try:
                    _kb = _key_name_to_vk(key) if key else None
                except Exception:
                    _kb = None
                logger.info(
                    "[KeyBoardController] Sending first action to game. "
                    f"kind={kind!r} key={key!r} VK={_kb!r} "
                    f"active={is_active}, foreground_window={foreground!r}, "
                    f"window_title_token={self.window_title!r}"
                )
                t_first_action_banner = False
            # Rate-limited per-kind debug log: emits at ~2 Hz even if the
            # bot is firing attack every frame so we can tell *which* key
            # name is being used for attack (user couldn't rebind it earlier).
            try:
                last_key = getattr(KeyBoardController, f"_dbg_last_{kind}", (-1.0, None))
                now = time.time()
                if now - last_key[0] >= 0.5 or last_key[1] != key:
                    try:
                        _vk = _key_name_to_vk(key) if key else None
                    except Exception:
                        _vk = None
                    logger.info(
                        f"[KeyBoardController] do_action_key: kind={kind!r} "
                        f"key={key!r} VK={_vk!r}"
                    )
                    setattr(KeyBoardController, f"_dbg_last_{kind}", (now, key))
            except Exception:
                pass
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
            #
            # Decide the target left-stick X component (-1.0 / 0.0 / +1.0) as
            # we walk through the branches so we can push it to the driver-
            # level virtual pad *once* at the end (avoids redundant USB
            # reports every frame).
            target_x = None
            if self.cmd_left_right == "left":
                key_up("right")
                key_down("left")
                target_x = -1.0
            elif self.cmd_left_right == "right":
                key_up("left")
                key_down("right")
                target_x = +1.0
            elif self.cmd_left_right == "stop":
                key_up("left")
                key_up("right")
                target_x = 0.0
            elif self.cmd_left_right == "none":
                if self.cmd_left_right_last != "none":
                    key_up("left")
                    key_up("right")
                    target_x = 0.0
            else:
                logger.error("[KeyBoardController] Unsupported left-right command: "
                             f"{self.cmd_left_right}")
            self.cmd_left_right_last = self.cmd_left_right
            # Push X axis to vgamepad if enabled
            if target_x is not None:
                _vgp_set_axis(target_x, None)

            #######################
            ### Up-Down Command ###
            #######################
            target_y = None
            if self.cmd_up_down == "up":
                key_up("down")
                key_down("up")
                target_y = +1.0
            elif self.cmd_up_down == "down":
                key_up("up")
                key_down("down")
                target_y = -1.0
            elif self.cmd_up_down == "stop":
                key_up("up")
                key_up("down")
                target_y = 0.0
            elif self.cmd_up_down == "none":
                if self.cmd_up_down_last != "none":
                    key_up("up")
                    key_up("down")
                    target_y = 0.0
            else:
                logger.error("[KeyBoardController] Unsupported up-down command: "
                             f"{self.cmd_up_down}")
            self.cmd_up_down_last = self.cmd_up_down
            # Push Y axis to vgamepad if enabled
            if target_y is not None:
                _vgp_set_axis(None, target_y)

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

            # --- 3-s periodic keyboard diagnostic heartbeat ---------------
            # The user reported "bot says it sent keys but the character
            # never moved".  This low-frequency (0.33 Hz) heartbeat log
            # tells them *exactly* which keys the controller is pressing
            # right now, whether the game window is still in the
            # foreground, and — crucially — which of the four Windows
            # backends successfully fired last (PostMessage / keybd_scan /
            # native_SendInput / PyAutoGUI / failed).  If
            # ``backend=failed`` the OS / anti-cheat rejected *every*
            # synthetic input path.
            now2 = time.time()
            if now2 >= t_next_kb_diag:
                t_next_kb_diag = now2 + 3.0
                is_act, fg_title = self.is_game_window_active()
                held = []
                if self.cmd_left_right == "left":  held.append("LEFT")
                if self.cmd_left_right == "right": held.append("RIGHT")
                if self.cmd_up_down   == "up":     held.append("UP")
                if self.cmd_up_down   == "down":   held.append("DOWN")
                if self.cmd_action not in ("none", "goal"):
                    held.append(f"ACT:{self.cmd_action}")
                held_str = "+".join(held) if held else "<idle>"
                try:
                    backend, _bts, consec = get_last_backend_info()
                except Exception:
                    backend, consec = "unknown", 0
                # Report vgamepad state alongside keyboard backend:
                #   vgp=(off)          → not configured / not installed
                #   vgp=enabled+Ax,Y   → ViGEm driver + vgamepad both OK
                #   vgp=err:"msg"      → enable_vgamepad() ran but failed
                #                       (user forgot to install vgamepad or the
                #                        ViGEmBus kernel driver)
                try:
                    if not self.vgamepad_enabled and not _VGP_ENABLED and not _VGP_ERR:
                        vgp_str = "vgp=(off)"
                    elif self.vgamepad_enabled:
                        xv = round(_VGP_X[0], 2) if vgamepad_available() else 0.0
                        yv = round(_VGP_Y[0], 2) if vgamepad_available() else 0.0
                        vgp_str = f"vgp=ON LX={xv:+1.2f} LY={yv:+1.2f} btns={len(_VGP_BUTTONS_DOWN)}"
                    else:
                        vgp_str = f"vgp=err:{_VGP_ERR!r}"
                except Exception:
                    vgp_str = "vgp=(?)"

                # Report NtUserSendInput backend availability (syscall-path
                # bypass — if this is "ok" and backend still shows fail, the
                # problem is in the game window (focus / per-process hook),
                # not in user32.  If this is "err:…" it's a Windows build
                # without the export; harmless.
                try:
                    _ = _resolve_ntusersendinput()
                    if _NTUSER_PROC[0] is not None:
                        nt_str = "nt=ok"
                    else:
                        nt_str = "nt=NA" if _NTUSER_ERR[0] is None else f"nt=err:{_NTUSER_ERR[0]!r}"
                except Exception:
                    nt_str = "nt=(?)"

                # GetAsyncKeyState self-test (only when we are IDLE and not
                # sending any movement/action — this lets us distinguish:
                #   (A) "the synthetic keystroke was successfully injected
                #        into the Windows input queue" (GetAsyncKeyState
                #        reports a &1 short-live bit)
                #                             — vs —
                #   (B) "the Windows queue has it, but the game's own
                #        GetKeyState / WindowProc isn't reading it (per-
                #        process hook / focus / DPI issue)"
                #
                # We test VK_LEFT (0x25, arrow left — not a letter, not a
                # modifier) because it's one of the keys the bot actually
                # sends for movement, so the scan-code path we exercise is
                # the real one.
                #
                # DIR3 upgrade: instead of testing a single backend, we
                # exercise **5 representative backends** in order and
                # report a 5-char bitmap like "PxFxF":
                #   0: keybd_event sc (Backend 3)
                #   1: native_SendInput sc (Backend 4)
                #   2: native_SendInput_ex noise batch (Backend 4a)
                #   3: NtUserSendInput direct syscall (Backend 4b)
                #   4: SetKeyboardState attached (Backend 6)
                # We read the result via GetAsyncKeyState(VK_LEFT) bit 0x8000
                # (currently pressed) OR bit 0x0001 (was pressed since last
                # call) — either counts as a successful "OS accepted the
                # synthetic event".  A per-test clear ensures stale bits
                # from previous heartbeats don't bleed over.
                gas_str = "gas=skip"
                gas_map  = "gas_map=-----"
                if not held and self.cmd_action in ("none", "goal") and is_act and _WIN_OK:
                    try:
                        VK_LEFT = 0x25
                        if _USER32 is None:
                            gas_str = "gas=(?)"
                        else:
                            up_before = _USER32.GetAsyncKeyState(VK_LEFT) & 0x8000
                            if up_before != 0:
                                gas_str = "gas=userHeld"
                                gas_map = "gas_map=HHHHH"
                            else:
                                # Clear any residual "was pressed" bit by
                                # reading once — that's the documented
                                # behaviour of GetAsyncKeyState's low bit.
                                _ = _USER32.GetAsyncKeyState(VK_LEFT)
                                # --- per-backend test harness ---
                                # Each test returns True if the OS-level
                                # async-key state reflects the injection.
                                hwnd_test = _ensure_game_hwnd()
                                results = []
                                def _gas_inject(down_fn, up_fn):
                                    try:
                                        # Clear residual bit between tests.
                                        _ = _USER32.GetAsyncKeyState(VK_LEFT)
                                        ok_down = down_fn()
                                        time.sleep(0.008)
                                        st1 = _USER32.GetAsyncKeyState(VK_LEFT)
                                        ok_up = up_fn()
                                        time.sleep(0.004)
                                        st2 = _USER32.GetAsyncKeyState(VK_LEFT)
                                        pressed = bool(st1 & 0x8000) or bool(st2 & 0x0001)
                                        return bool(ok_down and ok_up and pressed)
                                    except Exception:
                                        return False
                                # (0) Backend 3 — keybd_event SCANCODE
                                def b3_down():
                                    try:
                                        sc = _vk_to_sc(VK_LEFT); fl = _KEYEVENTF_SCANCODE
                                        if _is_extended_key(VK_LEFT): fl |= _KEYEVENTF_EXTENDEDKEY
                                        _USER32.keybd_event(ctypes.c_byte(0), ctypes.c_byte(sc & 0xFF),
                                                            ctypes.c_uint(fl), ctypes.c_void_p(0))
                                        return True
                                    except Exception: return False
                                def b3_up():
                                    try:
                                        sc = _vk_to_sc(VK_LEFT); fl = _KEYEVENTF_SCANCODE | _KEYEVENTF_KEYUP
                                        if _is_extended_key(VK_LEFT): fl |= _KEYEVENTF_EXTENDEDKEY
                                        _USER32.keybd_event(ctypes.c_byte(0), ctypes.c_byte(sc & 0xFF),
                                                            ctypes.c_uint(fl), ctypes.c_void_p(0))
                                        return True
                                    except Exception: return False
                                results.append(_gas_inject(b3_down, b3_up) or _gas_inject(b3_down, b3_up))
                                # (1) Backend 4 — native SendInput sc
                                results.append(_gas_inject(lambda: _native_sendinput_sc(VK_LEFT, True),
                                                           lambda: _native_sendinput_sc(VK_LEFT, False)))
                                # (2) Backend 4a — noise-padded SendInput
                                results.append(_gas_inject(lambda: _native_sendinput_ex(VK_LEFT, True),
                                                           lambda: _native_sendinput_ex(VK_LEFT, False)))
                                # (3) Backend 4b — NtUserSendInput direct syscall
                                results.append(_gas_inject(lambda: _ntusersendinput_direct(VK_LEFT, True),
                                                           lambda: _ntusersendinput_direct(VK_LEFT, False)))
                                # (4) Backend 6 — SetKeyboardState attached.
                                # ⚠️  Critical: SetKeyboardState writes
                                # the *per-thread* 256-byte keyboard-state
                                # table (the one GetKeyState() reads), NOT
                                # the global async-key state table that
                                # GetAsyncKeyState() reads.  So we verify
                                # this backend by:
                                #   1. AttachThreadInput to the game
                                #      window's thread (shares state tables)
                                #   2. SetKeyboardState to set/clear VK_LEFT
                                #   3. Call GetKeyState(VK_LEFT) — which now
                                #      reads the shared table thanks to the
                                #      attachment — and accept it as success
                                #      if high bit is 0x80 for down / 0x00
                                #      for up.
                                #
                                # Before this fix the diagnostic used
                                # GetAsyncKeyState (global) which always
                                # returned 0 for this backend — a false
                                # negative that made SetKeyboardState look
                                # useless when it actually worked perfectly
                                # for games that read via GetKeyState.
                                def _b6_verify(vki, expect_down):
                                    if _USER32 is None or hwnd_test == 0:
                                        return False
                                    try:
                                        GetWindowThreadProcessId = _USER32.GetWindowThreadProcessId
                                        GetCurrentThreadId = _KERNEL32.GetCurrentThreadId if _KERNEL32 is not None else None
                                        AttachThreadInput = _USER32.AttachThreadInput
                                        GetKeyState = _USER32.GetKeyState
                                        g_tid = GetWindowThreadProcessId(ctypes.c_void_p(hwnd_test), None)
                                        m_tid = GetCurrentThreadId() if GetCurrentThreadId is not None else 0
                                        attached = False
                                        if g_tid != 0 and m_tid != 0 and g_tid != m_tid:
                                            ok = bool(AttachThreadInput(ctypes.c_uint(g_tid),
                                                                        ctypes.c_uint(m_tid),
                                                                        ctypes.c_bool(True)))
                                            attached = bool(ok)
                                        try:
                                            # Sleep a tiny bit so Windows
                                            # actually commits the state-table
                                            # write (state tables can be
                                            # lazily flushed between attach
                                            # boundaries).
                                            time.sleep(0.003)
                                            s = GetKeyState(ctypes.c_int(vki & 0xFF))
                                            # GetKeyState return: short int.
                                            # High bit set => key is pressed.
                                            pressed = bool(s & 0x8000) or bool(s < 0)
                                            return bool(pressed == bool(expect_down))
                                        finally:
                                            if attached and g_tid != 0 and m_tid != 0 and g_tid != m_tid:
                                                try:
                                                    AttachThreadInput(ctypes.c_uint(g_tid),
                                                                      ctypes.c_uint(m_tid),
                                                                      ctypes.c_bool(False))
                                                except Exception:
                                                    pass
                                    except Exception:
                                        return False
                                def b6_down():
                                    if hwnd_test == 0: return False
                                    if not _set_keyboard_state_attached(VK_LEFT, True, hwnd=int(hwnd_test)):
                                        return False
                                    return _b6_verify(VK_LEFT, True)
                                def b6_up():
                                    if hwnd_test == 0: return False
                                    if not _set_keyboard_state_attached(VK_LEFT, False, hwnd=int(hwnd_test)):
                                        return False
                                    return _b6_verify(VK_LEFT, False)
                                results.append(b6_down() and b6_up())
                                # Render bitmap: P/F/S where S="skipped/fn returned False"
                                rendered = []
                                pass_cnt = 0
                                for ok in results:
                                    if ok is True:
                                        rendered.append("P"); pass_cnt += 1
                                    elif ok is False:
                                        rendered.append("F")
                                    else:
                                        rendered.append("-")
                                gas_map = ("gas_map=" + "".join(rendered) +
                                           f" keys=keybd/native/ex/ntuser/stkbd passed={pass_cnt}/5")
                                if pass_cnt >= 3:
                                    gas_str = "gas=ok"
                                elif pass_cnt >= 1:
                                    gas_str = "gas=partial"
                                else:
                                    gas_str = "gas=FAIL"
                    except Exception:
                        gas_str = "gas=(?)"
                        gas_map = "gas_map=?????"

                logger.info(
                    "[KeyBoardController] HEARTBEAT: "
                    f"keys={held_str} "
                    f"cmd=({self.cmd_left_right},{self.cmd_up_down},{self.cmd_action}) "
                    f"focus={is_act} foreground={fg_title!r} "
                    f"backend={backend} conseq_fail={consec} "
                    f"{nt_str} {gas_str} {gas_map} "
                    f"{vgp_str}"
                )
                # ----------------------------------------------------------
                # Once-per-startup "every backend is failing" warning.
                # Trigger threshold:
                #   * consec >= 30 failures AND
                #   * there are keys to send (i.e. we are not idle) AND
                #   * focus is True (keys are being sent to a plausible
                #     target — avoids false warning when user is in UI)
                # We also gate it to 30s after startup (ignore the first
                # 30s of "window not ready yet" churn).
                # ----------------------------------------------------------
                if not self._kb_block_warned and \
                   now2 - self.t_last_run > 30.0 and \
                   consec >= 30 and \
                   (held or self.cmd_action not in ("none", "goal")) and \
                   is_act:
                    self._kb_block_warned = True
                    logger.warning(
                        "[KeyBoardController] ALL Windows keyboard-input "
                        "backends have failed for 30+ consecutive key "
                        f"injections (last backend={backend!r}).  This "
                        "almost always means the MapleStory client's "
                        "anti-cheat shield (nProtect / XignCode / a custom "
                        "low-level keyboard hook) is blocking all synthetic "
                        "input paths.  Suggested workarounds in order of "
                        "effort:\n"
                        "  1. REBIND DIRECTION KEYS INSIDE THE GAME: open "
                        "MapleStory → [Game Options] → [Keyboard Settings] "
                        "and change [Move Left/Right/Up/Down] from the "
                        "arrow keys to letter keys like A / D / W / S.  "
                        "Many shields only blacklist the standard arrow-key "
                        "scan-codes; letter keys go through on the exact "
                        "same backend call.  Then update config "
                        "config_default.yaml key.left/right/up/down to "
                        "match.\n"
                        "  2. Try running Python / the terminal *without* "
                        "elevated admin rights (some shields treat admin-"
                        "level synthetic input as suspicious; non-admin "
                        "sometimes works *better* on private servers).\n"
                        "  3. Install a driver-level virtual HID keyboard: "
                        "vgamepad (pip install vgamepad, exposes a real "
                        "XInput controller that DirectInput-style shields "
                        "cannot tell apart from a physical gamepad) or "
                        "Interception (requires a signed driver but bypasses "
                        "user-mode hooks entirely).  Ask the bot author to "
                        "enable this backend if you go this route.\n"
                        "You can press F1 at any time to pause key "
                        "injection while you try these fixes."
                    )

            self.limit_fps()

        self.release_all_key() # Prevent key keep press down after termination

        logger.info("[KeyBoardController] terminated")
