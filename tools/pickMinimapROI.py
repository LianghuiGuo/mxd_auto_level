'''
Interactively pick the minimap ROI (manual_roi) for config.

Why: on some clients (e.g. 冒险岛怀旧服) the minimap border is not pure white
and the auto-detector leaves a black frame / an extra strip below.  Instead of
fighting the auto-detector, hand-measure the minimap ONCE with the mouse and
paste the printed rectangle into config as ``minimap.manual_roi``.

Usage:
    python -m tools.pickMinimapROI --cfg custom

Then:
    1. A window shows the current game frame.
    2. Drag a box around the minimap's MAP CONTENT area only
       (exclude the "小地图" title strip on top and the outer border).
    3. Press ENTER / SPACE to confirm (or 'c' to cancel).
    4. Copy the printed line into config/config_custom.yaml under `minimap:`.
'''
import time
import argparse

import cv2

from src.utils.global_var import WINDOW_WORKING_SIZE
from src.utils.logger import logger
from src.utils.common import load_yaml, override_cfg
from src.input.GameWindowCapturor import GameWindowCapturor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None,
                        help="custom config name under config/ (e.g. custom)")
    args = parser.parse_args()

    cfg = load_yaml("config/config_default.yaml")
    if args.cfg:
        cfg_custom = load_yaml(f"config/config_{args.cfg}.yaml")
        cfg = override_cfg(cfg, cfg_custom)

    cap = GameWindowCapturor(cfg)

    logger.info("Waiting for a valid game frame...")
    frame = None
    t0 = time.time()
    while time.time() - t0 < 10:
        frame = cap.get_frame()
        if frame is not None:
            break
        time.sleep(0.1)

    if frame is None:
        logger.error("Could not capture a game frame. Is the game window open "
                     "and visible?")
        return

    logger.info(f"Captured frame shape = {frame.shape[1]}x{frame.shape[0]}")
    logger.info("Drag a box around the minimap MAP CONTENT (no title, no "
                "border).  ENTER/SPACE to confirm, 'c' to cancel.")

    roi = cv2.selectROI("Pick minimap ROI (ENTER=ok, c=cancel)", frame,
                        showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = (int(v) for v in roi)
    if w <= 0 or h <= 0:
        logger.error("No region selected (or cancelled).")
        return

    # Show the exact crop so the user can confirm it's clean.
    crop = frame[y:y+h, x:x+w]
    cv2.imshow("Selected minimap (any key to close)", crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n================ COPY INTO config/config_custom.yaml ================")
    print("minimap:")
    print(f"  manual_roi: [{x}, {y}, {w}, {h}]")
    print("====================================================================\n")
    logger.info(f"manual_roi = [{x}, {y}, {w}, {h}]")


if __name__ == "__main__":
    main()
