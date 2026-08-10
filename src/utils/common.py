'''
Utility functions
'''
# Standard Import
import cv2
import datetime
import os
import platform
import smtplib
from email.message import EmailMessage
import imaplib
import mimetypes
import email
from collections import defaultdict
import time

# Libarary Import
import numpy as np
import yaml
import pyautogui
import pygetwindow as gw
from ruamel.yaml import YAML

# macOS Import
if platform.system() == 'Darwin':
    import Quartz
else:
    import win32gui
    import win32con

# Local import
from src.utils.logger import logger
from src.utils.global_var import WINDOW_WORKING_SIZE

OS_NAME = platform.system()

def is_mac():
    return OS_NAME == 'Darwin'

def is_windows():
    return OS_NAME == 'Windows'

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        logger.info(f"Load yaml: {path}")
        data = yaml.safe_load(f) or {}
        return convert_lists_to_tuples(data)

def load_yaml_with_comments(path):
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.load(f)

    field_comments = defaultdict(dict)
    section_comments = {}

    for title, sub in data.items():
        # Extract section comment (before key)
        if sub.ca.comment and sub.ca.comment[1]:
            section_comment_lines = [line.value.strip('#').strip() for line in sub.ca.comment[1]]
            section_comments[title] = "\n".join(section_comment_lines)

        # Extract field-level comments
        if hasattr(sub, 'ca'):
            for key in sub:
                comment = sub.ca.items.get(key)
                if comment and comment[2]:
                    field_comments[title][key] = comment[2].value.strip('#').strip()

    return data, dict(field_comments), section_comments

def save_yaml(data, path):
    data = convert_tuples_to_lists(data)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False)
    logger.info(f"Save yaml: {path}")

def get_cfg_diff(base, current):
    """
    Recursively compute the diff between base and current configs.
    Return only the values from current that are different.
    """
    diff = {}
    for key in current:
        if key not in base:
            diff[key] = current[key]
        elif isinstance(current[key], dict) and isinstance(base.get(key), dict):
            sub_diff = get_cfg_diff(base[key], current[key])  # recursive call
            if sub_diff:
                diff[key] = sub_diff
        else:
            norm_current = normalize(current[key])
            norm_base = normalize(base.get(key))
            if norm_current != norm_base:
                diff[key] = current[key]
    return diff

def normalize(value):
    """
    Normalize value for comparison:
    - Convert tuples to lists
    - Recursively normalize lists and dicts
    """
    if isinstance(value, tuple):
        return [normalize(v) for v in value]
    elif isinstance(value, list):
        return [normalize(v) for v in value]
    elif isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    else:
        return value

def convert_tuples_to_lists(obj):
    if isinstance(obj, dict):
        return {k: convert_tuples_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, list):
        return [convert_tuples_to_lists(i) for i in obj]
    else:
        return obj

def override_cfg(base, override):
    '''
    override_cfg (in-place)
    Modifies `base` directly by overriding keys from `override`.
    '''
    for k, v in override.items():
        if (
            k in base and isinstance(base[k], dict)
            and isinstance(v, dict)
        ):
            override_cfg(base[k], v)  # recursive override
        else:
            base[k] = v  # direct override or new key
    return base

def convert_lists_to_tuples(obj):
    if isinstance(obj, list):
        return tuple(convert_lists_to_tuples(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: convert_lists_to_tuples(v) for k, v in obj.items()}
    else:
        return obj

def load_image(path, mode=cv2.IMREAD_COLOR):
    '''
    Load image from disk and verify existence.
    '''
    if not os.path.exists(path):
        logger.error(f"Image not found: {path}")
        raise FileNotFoundError(f"Image not found: {path}")

    # Load image
    img = cv2.imread(path, mode)
    if img is None:
        logger.error(f"Failed to load image file: {path}")
        raise ValueError(f"Failed to load image: {path}")

    logger.info(f"Loaded image: {path}")

    return img

def nms(monsters, iou_threshold=0.3):
    '''
    Apply Non-Maximum Suppression (NMS) to remove overlapping detections.

    Parameters:
    - monsters: List of dictionaries, each representing a detected monster with:
        - "position": (x, y) top-left corner
        - "size": (width, height)
        - "score": similarity/confidence score from template matching
    - iou_threshold: Float, intersection-over-union threshold to suppress overlapping boxes

    Returns:
    - List of filtered monster dictionaries after applying NMS
    '''
    boxes = []
    for m in monsters:
        x, y = m["position"]
        # NOTE: "size" is stored as (h, w) throughout the detector, so unpack
        # in that order (previously this was (w, h), which swapped the axes
        # and made IoU meaningless for non-square templates).
        h, w = m["size"]
        # [x1, y1, x2, y2, score, original_data]
        boxes.append([x, y, x + w, y + h, m["score"], m])

    # Sort so the *best* match is processed first.  Template matching here
    # uses TM_SQDIFF_NORMED where a LOWER score is a better match, so we must
    # sort ascending.  Sorting descending (the old behaviour) kept the worst
    # overlapping detections and let dense false-positive clusters survive.
    boxes.sort(key=lambda x: x[4])

    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best[5])  # original monster_info

        boxes = [b for b in boxes if get_iou(best, b) < iou_threshold]

    return keep

def get_iou(box1, box2):
    '''
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Each box is expected to be a tuple or list with at least 4 values:
    (x1, y1, x2, y2), where:
        - (x1, y1) is the top-left corner
        - (x2, y2) is the bottom-right corner

    Returns:
        A float representing the IoU value (0.0 ~ 1.0).
        If there is no overlap, returns 0.0.
    '''
    x1, y1, x2, y2 = box1[:4]
    x1_p, y1_p, x2_p, y2_p = box2[:4]

    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union = area1 + area2 - inter_area

    return inter_area / union

def screenshot(img, suffix="screenshot"):
    '''
    Save the given image as a screenshot file.

    Parameters:
    - img: numpy array (image to save).

    Behavior:
    - Saves the image to the "screenshot/" directory with the current timestamp as filename.
    '''

    if img is None:
        return

    # ensure directory exists
    os.makedirs("screenshot", exist_ok=True)

    # Generate timestamp string
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"screenshot/{timestamp}_{suffix}.png"
    cv2.imwrite(filename, img)
    logger.info(f"[screenshot] save to {filename}")

def draw_rectangle(img, top_left, size, color, text,
                   thickness=2, text_height=0.7):
    '''
    Draws a rectangle with an text label.

    Parameters:
    - img: The image on which to draw (numpy array).
    - top_left: Tuple (x, y), the top-left corner of the rectangle.
    - size: Tuple (height, width) of the rectangle.
    - color: Tuple (B, G, R), color of the rectangle and text.
    - text: String to display above the rectangle.
    '''
    bottom_right = (top_left[0] + size[1],
                    top_left[1] + size[0])
    cv2.rectangle(img, top_left, bottom_right, color, thickness)
    cv2.putText(img, text, (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, text_height, color, thickness)

def pad_to_size(img, size, pad_value=0):
    '''
    pad_to_size

    Returns the image padded to (height, width) = ``size`` with ``pad_value``.
    If ``img`` is ``None`` or not a valid numpy image, returns ``None`` so the
    caller can short-circuit instead of crashing downstream on ``.shape``.
    '''
    if img is None or not hasattr(img, "shape"):
        return None
    h_img, w_img = img.shape[:2]
    h_target, w_target = size

    pad_h = max(0, h_target - h_img)
    pad_w = max(0, w_target - w_img)

    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(
            img,
            top   = pad_h // 2,
            bottom= pad_h - pad_h // 2,
            left  = pad_w // 2,
            right = pad_w - pad_w // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_value
        )

    return img


def find_pattern_sqdiff(
        img, img_pattern,
        last_result=None,
        mask=None,
        local_search_radius=50,
        global_threshold=0.4
    ):
    '''
    Perform masked template matching using SQDIFF_NORMED method.

    Gracefully handles ``None`` / invalid inputs: returns a safe fallback of
    ``((0, 0), 1.0, False)`` instead of raising, so the caller (usually
    UI/auto-detection code) can continue running and surface the issue via
    logs instead of crashing the entire bot thread.
    '''
    if img is None or img_pattern is None or \
       not hasattr(img, "shape") or not hasattr(img_pattern, "shape"):
        # Callers typically check `min_val < threshold`; 1.0 is the worst
        # possible SQDIFF_NORMED score so nothing will ever match.
        return (0, 0), 1.0, False

    # Padding if img is smaller than pattern
    padded = pad_to_size(img, img_pattern.shape[:2])
    if padded is None:
        return (0, 0), 1.0, False
    img = padded

    # search last result location first to speedup
    h, w = img_pattern.shape[:2]
    if last_result is not None and global_threshold > 0.0:
        lx, ly = last_result
        x0 = max(0, lx - local_search_radius)
        y0 = max(0, ly - local_search_radius)
        x1 = min(img.shape[1], lx + local_search_radius + w)
        y1 = min(img.shape[0], ly + local_search_radius + h)

        img_roi = img[y0:y1, x0:x1]
        if img_roi.shape[0] >= h and img_roi.shape[1] >= w:
            res = cv2.matchTemplate(
                    img_roi,
                    img_pattern,
                    cv2.TM_SQDIFF_NORMED,
                    mask=mask
            )
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if min_val < global_threshold:
                return (x0 + min_loc[0], y0 + min_loc[1]), min_val, True

    # Global fallback
    res = cv2.matchTemplate(
            img,
            img_pattern,
            cv2.TM_SQDIFF_NORMED,
            mask=mask
    )

    # Replace -inf/+inf/nan to 1.0 to avoid numerical error
    res = np.nan_to_num(res, nan=1.0, posinf=1.0, neginf=1.0)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    return min_loc, min_val, False

def get_mask(img, ignore_pixel_color):
    '''
    get_mask
    '''
    mask = np.all(img == ignore_pixel_color, axis=2).astype(np.uint8) * 255
    mask = cv2.bitwise_not(mask)
    return mask

def to_opencv_hsv(color_hsv):
    """
    Convert HSV from standard scale:
    - Hue: 0–360
    - Saturation: 0–100
    - Value: 0–100
    to OpenCV HSV format:
    - Hue: 0–179
    - Saturation/Value: 0–255

    Args:
        color_hsv (tuple/list/np.ndarray): HSV in standard scale (H, S, V)

    Returns:
        np.ndarray: HSV in OpenCV scale
    """
    h, s, v = color_hsv
    h_opencv = round(h / 360 * 179)
    s_opencv = round(s / 100 * 255)
    v_opencv = round(v / 100 * 255)
    return np.array([h_opencv, s_opencv, v_opencv], dtype=np.uint8)

def to_standard_hsv(color_hsv):
    """
    Convert HSV from OpenCV scale to standard HSV scale.
    """
    h, s, v = color_hsv
    h_std = h / 179 * 360
    s_std = s / 255 * 100
    v_std = v / 255 * 100
    return (h_std, s_std, v_std)

def get_minimap_loc_size(img_frame, manual_roi=None,
                         max_w_ratio=0.18, max_h_ratio=0.18):
    '''
    Detects the location and size of the minimap within the game frame.

    manual_roi : [x, y, w, h] | None
        If provided (e.g. from config ``minimap.manual_roi``), skip ALL
        automatic border detection and return exactly this rectangle.  The
        minimap sits at a fixed spot in this client, so hand-measuring it once
        is the most reliable option — no black frame, no extra strip below.
        The rectangle is clamped to the frame bounds.

    max_w_ratio / max_h_ratio : float
        Upper bound on the detected minimap size as a fraction of the FRAME
        width/height.  The minimap always lives in the top-left corner and is
        small (~12% of the frame width, ~12% of its height on this client).
        Maps whose surrounding scenery is the same brown/beige as the minimap
        border (e.g. 明珠港郊外: beaches + dirt) make the connected-component
        border detector bleed OUTWARD into the game art, returning a box whose
        bottom-right corner spills far past the real minimap.  Rejecting /
        clamping candidates larger than these ratios keeps the ROI glued to the
        actual minimap instead of the surrounding terrain.

    The function works by:
    - Thresholding the image get pure white(255,255,255) pixels.
    - Using connected components to find white-bordered regions.
    - Filtering candidates based on expected minimap size and margin rules:
        - Top, bottom, left, right margins must be 1px white lines.

    Returns:
        (x, y, w, h): Top-left coordinate and width/height of the minimap.
                    Returns None if not found.
    '''
    # --- Manual override: fixed hand-measured rectangle --------------------
    if manual_roi is not None:
        try:
            H_frame, W_frame = img_frame.shape[:2]
            mx, my, mw, mh = (int(v) for v in manual_roi)
            mx = max(0, min(mx, W_frame - 1))
            my = max(0, min(my, H_frame - 1))
            mw = max(1, min(mw, W_frame - mx))
            mh = max(1, min(mh, H_frame - my))
            return mx, my, mw, mh
        except Exception:
            pass  # bad config -> fall through to auto-detection

    # Try strict pure-white first, then fall back to a "near-white" border
    # threshold.  Some clients (e.g. 冒险岛怀旧服) frame the minimap with a
    # multi-layer beige/near-white border whose outermost line is NOT exactly
    # (255,255,255); the strict test then finds nothing and both the route
    # recorder and normal-mode minimap→map matching silently fail.  The
    # near-white pass (every channel >= 250) catches that inner near-white
    # rim while still being specific enough not to match arbitrary scenery.
    for white_min in (255, 250):
        white = np.array([255, 255, 255])
        lo = np.array([white_min, white_min, white_min])
        mask_white = cv2.inRange(img_frame, lo, white)

        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(mask_white, connectivity=8)

        def _is_white_line(pixels):
            # A border line counts as white if (near-)white on every pixel.
            return bool(np.all(np.all(pixels >= white_min, axis=-1)))

        for i in range(1, num_labels):
            x0, y0, rw, rh, area = stats[i]

            # Filter out small blobs
            if rw < 100 or rh < 100:
                continue

            x1 = x0 + rw - 1
            y1 = y0 + rh - 1

            # Check 1px (near-)white top and bottom margins
            if not (_is_white_line(img_frame[y0, x0:x0+rw]) and
                    _is_white_line(img_frame[y1, x0:x0+rw])):
                continue

            # Check 1px (near-)white left and right margins
            if not (_is_white_line(img_frame[y0:y0+rh, x0]) and
                    _is_white_line(img_frame[y0:y0+rh, x1])):
                continue

            # Create a mask of non-white pixels (content inside the border)
            mask_minimap = np.any(img_frame[y0:y0+rh, x0:x0+rw] < white_min,
                                  axis=2).astype(np.uint8)

            coords = cv2.findNonZero(mask_minimap)
            if coords is None:
                continue  # skip empty block
            x_minimap, y_minimap, w_minimap, h_minimap = cv2.boundingRect(coords)

            # Offset by original x0, y0 to get coords in original image
            x_minimap += x0
            y_minimap += y0

            # Same sane-size clamp as the brown-frame path.
            w_minimap = min(w_minimap, int(img_frame.shape[1] * max_w_ratio))
            h_minimap = min(h_minimap, int(img_frame.shape[0] * max_h_ratio))
            return x_minimap, y_minimap, w_minimap, h_minimap

    # --- Fallback: dark-brown hollow frame (冒险岛怀旧服 minimap) -----------
    # This client draws the minimap panel inside a thin dark-brown border
    # (~(102,120,137) BGR) with a beige title strip on top and the actual
    # map area below.  Neither pure-white nor near-white borders exist, so
    # the two passes above find nothing.  The dark-brown border, however,
    # forms a clean CLOSED rectangle (a hollow box: high w/h, very low fill
    # ratio).  Detect that box, then return the darker map-content area
    # inside it (skipping the lighter title strip).
    brown_lo = np.array([90, 80, 60])
    brown_hi = np.array([165, 150, 130])
    mask_brown = cv2.inRange(img_frame, brown_lo, brown_hi)
    # The border is a thin, anti-aliased line so its pixels are broken up into
    # many tiny components; dilate to bridge the gaps into one closed frame.
    mask_brown = cv2.dilate(mask_brown, np.ones((3, 3), np.uint8), iterations=2)
    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(mask_brown, connectivity=8)
    Hf, Wf = img_frame.shape[:2]
    max_w = int(Wf * max_w_ratio)
    max_h = int(Hf * max_h_ratio)
    best = None
    for i in range(1, num_labels):
        x0, y0, rw, rh, area = stats[i]
        if rw < 100 or rh < 80:
            continue
        # Reject components that are clearly too big to be the minimap: those
        # are the border colour bleeding into the surrounding game art.  The
        # real minimap frame stays comfortably under max_w/max_h.
        if rw > max_w or rh > max_h:
            continue
        # Minimap sits in the top-left corner of the frame.
        if x0 > int(img_frame.shape[1] * 0.35) or \
           y0 > int(img_frame.shape[0] * 0.35):
            continue
        fill = area / float(rw * rh)
        # A hollow border frame is mostly empty inside -> low fill ratio.
        if fill > 0.30:
            continue
        if best is None or (rw * rh) > (best[2] * best[3]):
            best = (x0, y0, rw, rh)
    if best is not None:
        x0, y0, rw, rh = best
        inner = img_frame[y0:y0+rh, x0:x0+rw]
        # Row brightness: the beige title strip on top AND the light bottom
        # border are both bright (~150-250); the actual map content in between
        # is dark (~55-100).  Take the CONTIGUOUS dark band as the map area so
        # we crop off BOTH the title strip above and the extra light strip/
        # border below (the "下方多出来一块" the user saw).
        row_mean = inner.reshape(rh, -1, 3).mean(axis=(1, 2))
        # IMPORTANT: use the LONGEST CONTIGUOUS dark run, not first/last dark
        # row.  The beige title strip contains isolated dark rows (the "小地图"
        # map-name text) and the light bottom border also has stray dark rows;
        # taking dark_rows[0]/[-1] then swallowed the title strip above and the
        # light strip below (the exact "黑框 + 下方多出来一块" the user saw).
        # The real map content is one solid dark band, so pick that run.
        def _longest_dark_run(vals, thres=120):
            best_s = best_e = -1
            best_len = 0
            s = None
            for i, v in enumerate(vals):
                if v < thres:
                    if s is None:
                        s = i
                else:
                    if s is not None and (i - s) > best_len:
                        best_len, best_s, best_e = i - s, s, i - 1
                    s = None
            if s is not None and (len(vals) - s) > best_len:
                best_s, best_e = s, len(vals) - 1
            return best_s, best_e

        # Use the longest dark run ONLY to locate where the beige title strip
        # ends (its start = top of the real map content).  Do NOT use its END
        # as the map bottom: the player's YELLOW dot and bright platforms are
        # *bright*, so they break the dark run — using its end cropped the map
        # so tightly that most of the yellow dot was cut off (only ~24 of 171
        # px survived), and on slopes / edges the remaining pixels dropped
        # below the 4-px detection floor, freezing route recording (the
        # "走斜坡就不录制" / "最开始能录斜坡" regression).  Instead, keep the
        # map extending down to the frame's bottom border minus a thin margin,
        # so the full dot and platforms stay inside the crop.
        run_s, run_e = _longest_dark_run(row_mean)
        if run_s < 0:
            map_y0 = int(rh * 0.42)  # fallback: skip ~top 42% (title strip)
        else:
            map_y0 = run_s
        BORDER = 2  # dark-brown frame is ~1-2 px thick
        map_y1 = rh - 1 - BORDER

        # Left/right: just trim the thin border, keep full width (the dot can
        # sit anywhere horizontally, including near the edges on slopes).
        map_x0 = BORDER
        map_x1 = rw - 1 - BORDER

        x_minimap = x0 + map_x0
        y_minimap = y0 + map_y0
        w_minimap = map_x1 - map_x0 + 1
        h_minimap = map_y1 - map_y0 + 1
        # Final clamp: never let the ROI exceed the sane size cap even if the
        # component squeaked through (e.g. the dark run ran a little long).
        w_minimap = min(w_minimap, max_w)
        h_minimap = min(h_minimap, max_h)
        if w_minimap > 40 and h_minimap > 20:
            return x_minimap, y_minimap, w_minimap, h_minimap

    # logger.warning("Minimap not found in the game frame.")
    return None  # minimap not found

def get_player_location_on_minimap(img_minimap, minimap_player_color=(136, 255, 255),
                                    debug_label="player"):
    """
    Detects the player's position on the minimap.

    The function works by:
      * Iterating over a sequence of **per-channel tolerances** (10 → 20 →
        40 → 60) so the player dot is found even if windows-capture
        introduces subtle BGR channel shifts (very common when the client
        runs in 16:9 vs 4:3, or when the capture pipeline converts from
        BGRA to BGR on the fly).
      * For each tolerance, building an inRange mask and requiring at least
        4 pixels to match (avoid single-pixel noise).
      * Taking the pixel-wise centroid of all matched pixels as the final
        minimap coordinate.

    Parameters
    ----------
    img_minimap : ndarray
        Cropped minimap image (BGR).
    minimap_player_color : tuple[int, int, int]
        Reference BGR colour for the player dot (default bright yellow
        ``(136, 255, 255)``).
    debug_label : str
        Short label used only for the one-shot diagnostic log/dump when all
        tolerances fail (helps disambiguate "player dot" vs "other-player
        dot" in user reports).

    Returns
    -------
    (x, y) | None
        Player location in **minimap** coordinates (not game-screen coords)
        or None when fewer than 4 pixels match all tried tolerances.
    """
    tolerances = [10, 20, 40, 60]
    ref_bgr = tuple(map(int, minimap_player_color))
    last_mask = None
    last_n = 0
    for tol in tolerances:
        lower_bgr = tuple(max(0, c - tol) for c in ref_bgr)
        upper_bgr = tuple(min(255, c + tol) for c in ref_bgr)
        mask = cv2.inRange(img_minimap, lower_bgr, upper_bgr)
        n = int(cv2.countNonZero(mask))
        last_mask = mask
        last_n = n
        if n >= 4:
            # Compute the centroid directly from the mask.  Using
            # cv2.findNonZero(...).mean(axis=0)[0] was fragile: depending on
            # the OpenCV build / mask shape the reduction could collapse to a
            # scalar, and the subsequent avg[0] then raised
            # "IndexError: invalid index to scalar variable".  np.where is
            # unambiguous and returns row (y) / col (x) index arrays.
            ys, xs = np.where(mask > 0)
            return (int(round(float(xs.mean()))), int(round(float(ys.mean()))))

    # --- Robust "yellowness" fallback --------------------------------------
    # The reference-colour + tolerance approach fails when the client's player
    # dot differs a lot from the configured BGR (e.g. this user's dot is
    # (50,255,238) but default is (136,255,255) — the Blue channel alone is
    # 86 apart, outside even the widest ±60 window, so on darker frames like a
    # slope the dot vanished and route recording stalled).  A yellow dot is
    # defined structurally: GREEN and RED both high, BLUE clearly lower.  Match
    # that directly so ANY yellow player dot is found regardless of exact hue.
    b = img_minimap[:, :, 0].astype(np.int16)
    g = img_minimap[:, :, 1].astype(np.int16)
    r = img_minimap[:, :, 2].astype(np.int16)
    yellow = ((g >= 170) & (r >= 170) &
              (g.astype(np.int32) + r - 2 * b >= 150)).astype(np.uint8)
    n_y = int(yellow.sum())
    if n_y >= 4:
        ys, xs = np.where(yellow > 0)
        return (int(round(float(xs.mean()))), int(round(float(ys.mean()))))

    # --- Diagnostic path: all tolerances failed. ---------------------------
    # Dump the last (widest-tolerance) mask so the user can visualise what
    # the detector sees and then correct ``minimap.player_color`` in YAML.
    # We do this exactly once per process to avoid spam.
    if not hasattr(get_player_location_on_minimap, "_dbg_dumped"):
        get_player_location_on_minimap._dbg_dumped = True  # type: ignore[attr-defined]
        try:
            import os as _os
            _os.makedirs("log", exist_ok=True)
            fname = f"log/debug_minimap_{debug_label}_ref{ref_bgr[0]}_{ref_bgr[1]}_{ref_bgr[2]}.png"
            cv2.imwrite(fname, last_mask)
            # Also dump the raw minimap so colour can be eyeballed.
            fname_raw = f"log/debug_minimap_{debug_label}_raw.png"
            cv2.imwrite(fname_raw, img_minimap)
            # Try to give a hint: if the mask captured anything at all (<4
            # pixels) suggest the tolerance is too low; otherwise suggest
            # that minimap.player_color BGR value is wrong.
            if last_n > 0:
                hint = (
                    f"Found only {last_n} pixels (need ≥4) with the widest "
                    f"tolerance ±{tolerances[-1]}.  Try increasing "
                    "``minimap.player_color_tolerance`` in config or tuning "
                    f"the BGR reference (current={ref_bgr})."
                )
            else:
                hint = (
                    "Zero pixels matched any tolerance — "
                    f"``minimap.player_color`` (BGR {ref_bgr}) is almost "
                    "certainly wrong for this client.  Use the debugging "
                    "command to sample the *actual* player dot BGR from "
                    "``log/debug_minimap_player_raw.png``."
                )
            logger.debug(
                f"[get_player_location_on_minimap] Failed to locate "
                f"{debug_label!r} dot.  {hint}  Mask & raw minimap saved to "
                f"log/debug_minimap_{debug_label}_*.png."
            )
        except Exception:  # noqa: BLE001 — diagnostics must never crash caller
            pass

    return None

def get_all_other_player_locations_on_minimap(img_minimap, red_bgr=(0, 0, 255)):
    '''
    Detect red dot (0,0,255) and calculate the center to define as other player position.
    '''
    red_bgr = tuple(map(int, red_bgr))
    # 智能選擇容錯範圍：從較小開始，如果檢測不到就增加
    tolerances = [10, 20, 30, 40]  # 嘗試不同的容錯範圍
    
    for tolerance in tolerances:
        lower_bgr = tuple(max(0, c - tolerance) for c in red_bgr)
        upper_bgr = tuple(min(255, c + tolerance) for c in red_bgr)

        # 使用範圍檢測
        mask = cv2.inRange(img_minimap, lower_bgr, upper_bgr)
        coords = cv2.findNonZero(mask)

        if coords is not None and len(coords) >= 3:
            logger.debug(f"Found {len(coords)} red pixels with tolerance {tolerance}")
            logger.debug(f"Color range: {lower_bgr} to {upper_bgr}")
            # cv2.findNonZero normally returns shape (N, 1, 2), so pt[0] is
            # [x, y].  On some degenerate frames it can come back as (N, 2)
            # (pt[0] is then a scalar np.int32), which made `tuple(pt[0])`
            # raise "'numpy.int32' object is not iterable" and crash the whole
            # frame loop every tick (bot appeared frozen).  Flatten to (N, 2)
            # so unpacking is always correct regardless of the returned shape.
            pts = np.asarray(coords).reshape(-1, 2)
            return [(int(x), int(y)) for x, y in pts]  # List of (x, y)

    # 如果所有容錯範圍都檢測不到，記錄調試信息
    logger.debug(f"Red dot detection failed with all tolerances: {tolerances}")
    return []

def debug_minimap_colors(img_minimap, target_color=(0, 0, 255)):
    """
    調試函數：分析小地圖中的顏色分布，幫助找到正確的紅色點顏色值
    """
    # 保存原始小地圖
    cv2.imwrite("debug_minimap_original.png", img_minimap)
    
    # 分析顏色分布
    h, w = img_minimap.shape[:2]
    colors_found = {}
    
    # 掃描整個小地圖，統計顏色
    for y in range(0, h, 2):  # 每2個像素取一個樣本以提高效率
        for x in range(0, w, 2):
            color = tuple(img_minimap[y, x])
            if color not in colors_found:
                colors_found[color] = 0
            colors_found[color] += 1
    
    # 找出最常見的顏色（排除黑色和白色）
    sorted_colors = sorted(colors_found.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("=== Minimap Color Analysis ===")
    logger.info(f"Target color (BGR): {target_color}")
    logger.info("Top 10 most common colors:")
    
    for i, (color, count) in enumerate(sorted_colors[:10]):
        if color != (0, 0, 0) and color != (255, 255, 255):  # 排除純黑和純白
            logger.info(f"  {i+1}. BGR{color}: {count} pixels")
            
            # 檢查是否接近目標顏色
            diff = sum(abs(c1 - c2) for c1, c2 in zip(color, target_color))
            if diff < 50:  # 如果顏色差異小於50
                logger.info(f"    *** Close to target color! Difference: {diff} ***")
    
    # 創建不同容錯範圍的檢測結果
    for tolerance in [10, 20, 30, 40, 50]:
        lower_bgr = tuple(max(0, c - tolerance) for c in target_color)
        upper_bgr = tuple(min(255, c + tolerance) for c in target_color)
        mask = cv2.inRange(img_minimap, lower_bgr, upper_bgr)
        coords = cv2.findNonZero(mask)
        count = len(coords) if coords is not None else 0
        logger.info(f"Tolerance {tolerance}: Found {count} pixels")
        cv2.imwrite(f"debug_red_detection_tolerance_{tolerance}.png", mask)
    
    return sorted_colors

def get_bar_percent(img):
    '''
    Get HP/MP/EXP bar ratio with given bar image

    Return: float [0.0 - 1.0]
    '''
    # Sample a horizontal line at the vertical center of the bar
    h, w = img.shape[:2]
    line_pixels = img[h // 2, :]

    # Get left white boundary of bar
    lb = 0
    while lb < w and np.all(line_pixels[lb] >= 255):
        lb += 1

    # Get right white boundary of bar
    rb = w - 1
    while rb > lb and np.all(line_pixels[rb] >= 255):
        rb -= 1

    # Sanity check
    if rb <= lb:
        return 0.0

    # Get unfill pixel count in bar
    unfill_pixel_cnt = 0
    tolerance = 10
    for i in range(lb, rb + 1):
        r, g, b = line_pixels[i]
        if  abs(int(r) - int(g)) <= tolerance and \
            abs(int(r) - int(b)) <= tolerance and \
            int(r) > 0:
            unfill_pixel_cnt += 1

    # Compute fill ratio
    total_width = rb - lb + 1
    fill_width = total_width - unfill_pixel_cnt
    fill_ratio = fill_width / total_width if total_width > 0 else 0.0
    return fill_ratio*100

def nms_matches(matches, iou_thresh=0.0):
    '''
    Apply non-maximum suppression to remove overlapping matches.

    Args:
        matches: List of tuples (idx, loc, score, shape)
        iou_thresh: IoU threshold to trigger suppression (default 0.0 = any overlap)

    Returns:
        List of filtered matches (same format as input)
    '''
    filtered = matches.copy()
    i = 0
    while i < len(filtered):
        j = i + 1
        while j < len(filtered):
            _, loc_i, score_i, shape_i = filtered[i]
            _, loc_j, score_j, shape_j = filtered[j]

            box_i = (loc_i[0], loc_i[1],
                     loc_i[0] + shape_i[1], loc_i[1] + shape_i[0])
            box_j = (loc_j[0], loc_j[1],
                     loc_j[0] + shape_j[1], loc_j[1] + shape_j[0])

            if get_iou(box_i, box_j) > iou_thresh:
                if score_i > score_j:
                    filtered.pop(i)
                    i -= 1
                    break
                else:
                    filtered.pop(j)
                    j -= 1
            j += 1
        i += 1

    return filtered

def get_window_region_mac(window_title):
    '''
    Get window region on macOS using Quartz
    '''
    window_list = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID
    )
    # Get all exist windows
    all_titles = []
    for window in window_list:
        title = window.get(Quartz.kCGWindowName, '')
        owner = window.get(Quartz.kCGWindowOwnerName, '')
        if title:
            all_titles.append(f"{title} (Owner: {owner})")
    logger.debug(f"all_titles: {all_titles}")
    for window in window_list:
        if window.get(Quartz.kCGWindowName, '') == window_title:
            bounds = window.get(Quartz.kCGWindowBounds, {})
            return {
                "left": int(bounds.get('X', 0)),
                "top": int(bounds.get('Y', 0)),
                "width": int(bounds.get('Width', 0)),
                "height": int(bounds.get('Height', 0))
            }
    return None


def click_in_game_window(window_title, coord):
    '''
    Mouse click on a game window coordinate
    '''
    # game_window = gw.getWindowsWithTitle(window_title)[0]
    # win_left, win_top = game_window.left, game_window.top

    # If mac then coord / 2 and y position + 3
    if is_mac():
        coord = (coord[0] // 2, coord[1] // 2 + 10)

    if is_mac():
        # macOS implementation using Quartz
        region = get_window_region_mac(window_title)
        if region is None:
            text = f"Cannot find window: {window_title}"
            logger.error(text)
            raise RuntimeError(text)
        win_left, win_top = region["left"], region["top"]
    else:
        # Windows implementation using pygetwindow
        game_window = gw.getWindowsWithTitle(window_title)[0]
        win_left, win_top = game_window.left, game_window.top

    loc_click = (win_left + coord[0], win_top + coord[1])
    pyautogui.click(loc_click)
    logger.info(f"[click_in_game_window] click at {loc_click}")

def send_email(email_addr, password,
               to, subject, body, attachment_path):
    '''
    send_email
    '''
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = email_addr
    msg['To'] = to

    # Attach PNG image
    with open(attachment_path, 'rb') as f:
        file_data = f.read()
        maintype, subtype = mimetypes.guess_type(attachment_path)[0].split('/')
        filename = f.name.split("/")[-1]
        msg.add_attachment(file_data, maintype=maintype, subtype=subtype, filename=filename)

    # Send Email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(email_addr, password)
        smtp.send_message(msg)
        logger.info(f"[send_email] {subject} to {to}")

def check_inbox(email_addr, password, token):
    '''
    Check inbox for replies containing the expected token in the subject
    '''
    imap = imaplib.IMAP4_SSL("imap.gmail.com")
    imap.login(email_addr, password)
    imap.select("inbox")

    # IMAP search: only look for subjects that contain token
    status, messages = imap.search(None, f'(SUBJECT "{token}")')
    if status != "OK":
        logger.error("Search failed")
        imap.logout()
        return None

    for num in messages[0].split():
        status, data = imap.fetch(num, '(RFC822)')
        msg = email.message_from_bytes(data[0][1])
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_payload(decode=True).decode()
                imap.logout()
                return body.strip()

    imap.logout()
    return None

def mask_route_colors(img_map, img_route, color_code):
    """
    Masks all pixels in img_route where img_map contains any route color.
    Pixels at those positions in img_route are set to black (0,0,0).
    """
    # Parse color_code keys to list of RGB tuples
    target_colors = [tuple(map(int, color_str.split(','))) for color_str in color_code.keys()]

    # Ensure dimensions match
    if img_map.shape[:2] != img_route.shape[:2]:
        logger.warning("[mask_route_colors] Resizing img_map from "
                       f"{img_map.shape} to {img_route.shape}")
        img_map = cv2.resize(img_map, (img_route.shape[1], img_route.shape[0]))

    # Build mask for each color
    mask = np.zeros(img_map.shape[:2], dtype=bool)
    for color in target_colors:
        matches = np.all(img_map == color, axis=-1)
        mask |= matches

    # Apply mask to img_route (set those pixels to black)
    img_route[mask] = (0, 0, 0)

    return img_route

def activate_game_window(window_title):
    '''
    activate_game_window
    This function only support Windows OS
    '''
    hwnd = win32gui.FindWindow(None, window_title)
    if hwnd == 0:
        raise Exception(f"Cannot find window with title: {window_title}")

    try:
        # Try to restore the window first
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

        # Try to set foreground
        win32gui.SetForegroundWindow(hwnd)

        logger.info(f"[activate_game_window] Set game window to foreground")
    except:
        # If SetForegroundWindow fails, try alternative methods
        win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        win32gui.BringWindowToTop(hwnd)
        win32gui.SetActiveWindow(hwnd)

# Titles containing these substrings are almost certainly NOT the MapleStory
# game client. They include our own Qt control UI, PyInstaller windows,
# config editors, etc. Matching is case-insensitive substring on the full
# window title.
_WINDOW_TITLE_BLACKLIST_SUBSTRINGS = (
    "maplestory autolevelup",   # This program's own Qt UI window
    "autolevelup",              # Any auto-leveling UI window
    ".yaml",                    # YAML editor
    ".yml",
    "notepad",
    "visual studio",
    "pycharm",
    "command prompt",
    "powershell",
    "windows powershell",
    "file explorer",
    "settings",
    "task manager",
    "config",
)


def _window_title_is_blacklisted(title: str) -> bool:
    """Return True if a window title clearly cannot be the game client."""
    if not title:
        return True  # empty title (system dialogs / hidden windows)
    t_lower = title.lower()
    return any(blk in t_lower for blk in _WINDOW_TITLE_BLACKLIST_SUBSTRINGS)


def get_game_window_title_by_token(token):
    '''
    Find the first top-level, **visible**, game-client-looking window whose
    title contains ``token`` (case-insensitive substring match).

    Supports a **single keyword string** or a **list of fallback keyword
    strings**.  When a list is given, keywords are tried in order and the
    first non-empty match wins.  This makes it trivial to support
    multi-region / multi-language client titles such as
    ``["MapleStory Worlds", "冒险岛怀旧服", ...]``.

    Additional filters applied regardless of token match:
      * Window must be visible (``IsWindowVisible``).
      * Window must not be blacklisted (``_WINDOW_TITLE_BLACKLIST_SUBSTRINGS``)
        so we never accidentally attach to the AutoBot's own Qt UI window,
        a text editor, PowerShell, etc.

    Only works in Windows OS.
    '''
    tokens = token if isinstance(token, (list, tuple)) else [token]

    def _find_one(kw):
        matches = []
        def callback(hwnd, _matches):
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd)
            if _window_title_is_blacklisted(title):
                return
            if kw.lower() in title.lower():
                _matches.append(title)
        win32gui.EnumWindows(callback, matches)
        return matches[0] if matches else None

    for kw in tokens:
        found = _find_one(kw)
        if found:
            return found
    return None

def is_img_16_to_9(img, cfg):
    """
    Check if image aspect ratio is approximately 16:9.
    """
    tolerance = cfg["game_window"]["ratio_tolerance"]
    h, w = img.shape[:2]
    return abs(w/h - 16/9) <= tolerance

def normalize_pixel_coordinate(coord, window_size):
    '''
    Normalize pixel coordinate from current window size to standard (693x1282).
    '''
    h_win, w_win = window_size
    h_std, w_std = (693, 1282)

    # Standard size, no need to normalize
    if h_win == h_std and w_win == w_std:
        return coord

    scale_y = h_std / h_win
    scale_x = w_std / w_win

    x, y = coord
    norm_y = round(y * scale_y)
    norm_x = round(x * scale_x)

    logger.info("[normalize_pixel_coordinate] "\
                f"Normalized coord{coord} to coord{(norm_x, norm_y)}")

    return (norm_x, norm_y)

def resize_window(window_title, width=1296, height=759, hwnd=None):
    '''
    Resize a top-level window identified by ``window_title`` (exact match) or
    by a caller-provided ``hwnd`` (Win32 HWND, takes precedence over
    ``window_title`` when non-zero) to ``width`` x ``height``.

    Resilient behaviour:
      * If the window cannot be found, log a warning and return.
      * If the window already has the target client-area size (within a small
        tolerance), skip MoveWindow entirely — this avoids spurious
        ``ERROR_ACCESS_DENIED`` on game clients that block external sizing.
      * If the OS rejects MoveWindow (ERROR_ACCESS_DENIED / any
        ``pywin32`` error) with error code 5 (access denied) or other codes,
        log a **diagnostic warning** but **do not raise**.  Upstream callers
        can proceed because ``get_img_frame`` tolerates non-canonical client
        sizes (it auto-resizes to ``WINDOW_WORKING_SIZE`` on the CV side).

    Returns True if the window ended up at the requested size (either because
    it already was, or because MoveWindow succeeded); False otherwise.
    '''
    if hwnd is None or int(hwnd or 0) == 0:
        hwnd = win32gui.FindWindow(None, window_title)
    else:
        hwnd = int(hwnd)
        # Caller-provided HWND should still be a live window before we try to
        # touch it; otherwise fall back to FindWindow by title for safety.
        try:
            if not win32gui.IsWindow(hwnd):
                hwnd = win32gui.FindWindow(None, window_title)
        except Exception:
            hwnd = win32gui.FindWindow(None, window_title)
    if hwnd == 0:
        logger.warning(f"[resize_window] Cannot find exact-title window: {window_title!r}")
        return False

    # Read current outer-window rectangle (left, top, right, bottom) in screen
    # coordinates. ``GetClientRect`` would give client-only size but without
    # the border so it's harder to reconstruct the full outer size MoveWindow
    # expects. We stick with GetWindowRect + a size tolerance for the outer
    # frame (typically ~38 px for title + border on Windows 11).
    rect = win32gui.GetWindowRect(hwnd)
    x0, y0, x1, y1 = rect
    cur_w_outer = x1 - x0
    cur_h_outer = y1 - y0

    # Fast path: already the requested size (±2 px tolerance for rounding).
    if abs(cur_w_outer - width) <= 2 and abs(cur_h_outer - height) <= 2:
        logger.info(
            f"[resize_window] Window {window_title!r} is already {width}x{height}; "
            "skipping MoveWindow call."
        )
        return True

    # Actually move/resize.  Wrap in try/except because many game clients
    # (especially protected / anti-cheat ones) reject cross-process
    # SetWindowPos / MoveWindow with ERROR_ACCESS_DENIED (winerror 5).
    BORDER_APPROX = 38  # Windows 11 typical title + border height (rough)
    try:
        win32gui.MoveWindow(hwnd, x0, y0, width, height, True)
    except Exception as exc:  # noqa: BLE001 — pywin32 raises generic pywintypes.error
        # pywintypes.error unpacks as (winerror, funcname, msg)
        winerr = getattr(exc, "winerror", None)
        exc_args = getattr(exc, "args", None)
        if isinstance(exc_args, tuple) and len(exc_args) >= 1 and winerr is None:
            winerr = exc_args[0]
        if winerr == 5:
            client_area_h = max(1, cur_h_outer - BORDER_APPROX)
            ratio = cur_w_outer / client_area_h
            ratio_ok = abs(ratio - (16/9)) <= 0.08
            fix_hint = (
                " -> Proceeding anyway: CV pipeline auto-resizes, but detection "
                "is most reliable if you first set the game to 16:9 windowed "
                "mode (1280x720 / 1296x759) in the game settings menu."
                if ratio_ok else
                " -> Current aspect ratio does NOT look 16:9.  PLEASE set the "
                "game client to 16:9 windowed mode smallest resolution first "
                "(skip AutoBot resize by clicking 'Start' afterwards)."
            )
            logger.warning(
                f"[resize_window] Windows rejected MoveWindow on {window_title!r} "
                f"with ERROR_ACCESS_DENIED (winerror 5). Game likely ran with "
                f"higher integrity level (Admin / anti-cheat protection) than "
                f"the AutoBot python process.  Current outer: "
                f"{cur_w_outer}x{cur_h_outer} (target {width}x{height})."
                + fix_hint
            )
        else:
            logger.warning(
                f"[resize_window] Unexpected error while resizing "
                f"{window_title!r}: {exc!r}.  Proceeding anyway."
            )
        return False

    logger.info(f"已將「{window_title}」調整為 {width}x{height}")
    return True
