"""
Semi-automatic pre-labeller for the YOLO monster dataset.

WHY THIS EXISTS
---------------
Hand-drawing every bounding box for hundreds of screenshots is the most
expensive part of building a YOLO dataset.  This script bootstraps that work:
it runs the *existing* colour template matching (the same cv2.matchTemplate /
TM_SQDIFF_NORMED approach the live bot uses) over each screenshot and writes an
initial YOLO-format label file per image.  You then open the images in a
labelling tool (e.g. labelImg) and only need to FIX the boxes — delete false
positives, add missed monsters, correct classes — instead of drawing from
scratch.

It is deliberately self-contained (no import of src/, which pulls in
pyautogui/pywin32) so it runs anywhere.

USAGE
-----
1. Drop game screenshots into  ml/dataset/images/train/  (and val/).
2. Run:   python3 ml/prelabel.py
3. It writes  ml/dataset/labels/train/<same_name>.txt  for each image.
4. Open in labelImg (format: YOLO, classes from ml/data.yaml) and fix.

YOLO label format (one line per box):
    <class_index> <cx> <cy> <w> <h>
all normalised to [0,1] relative to image width/height.
"""

import glob
import os

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# Matching threshold for TM_SQDIFF_NORMED (lower = stricter).  We keep this
# fairly permissive on purpose: this is a PRE-label, and it is much faster for a
# human to delete an extra box than to draw a missing one.  Tune if you get too
# much noise.
DIFF_THRES = 0.30
IOU_THRES = 0.4


def load_classes():
    with open(os.path.join(HERE, "data.yaml"), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    # names may be {0: name} dict or a list
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def get_mask(img, color=(0, 255, 0)):
    """Mask OUT the green background so it does not contribute to matching."""
    exact = cv2.inRange(img, np.array(color), np.array(color))
    return cv2.bitwise_not(exact)


def load_templates(classes):
    """Return {class_index: [(template_img, mask), ...]} from monster/<name>/."""
    templates = {}
    for idx, name in enumerate(classes):
        imgs = []
        pattern = os.path.join(REPO, "monster", name, f"{name}*.png")
        for f in sorted(glob.glob(pattern)):
            t = cv2.imread(f)
            if t is None:
                continue
            m = get_mask(t)
            imgs.append((t, m))
            # also add horizontally-flipped variant (monsters face both ways)
            tf = cv2.flip(t, 1)
            imgs.append((tf, get_mask(tf)))
        if imgs:
            templates[idx] = imgs
        else:
            print(f"[warn] no templates found for class '{name}' at {pattern}")
    return templates


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def nms(dets):
    """dets: list of (score, x1, y1, x2, y2, cls). Lower score = better."""
    dets = sorted(dets, key=lambda d: d[0])
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if iou(best[1:5], d[1:5]) < IOU_THRES]
    return keep


def detect(frame, templates):
    dets = []
    for cls, imgs in templates.items():
        for t, m in imgs:
            h, w = t.shape[:2]
            if frame.shape[0] < h or frame.shape[1] < w:
                continue
            res = cv2.matchTemplate(frame, t, cv2.TM_SQDIFF_NORMED, mask=m)
            res = np.nan_to_num(res, nan=1.0, posinf=1.0)
            ys, xs = np.where(res <= DIFF_THRES)
            # keep only the strongest few points per template to bound work
            pts = sorted(zip(ys, xs), key=lambda p: res[p[0], p[1]])[:40]
            for y, x in pts:
                dets.append((float(res[y, x]), x, y, x + w, y + h, cls))
    return nms(dets)


def main():
    classes = load_classes()
    templates = load_templates(classes)
    if not templates:
        print("No templates loaded — nothing to do.")
        return

    total_imgs = 0
    total_boxes = 0
    for split in ("train", "val"):
        img_dir = os.path.join(HERE, "dataset", "images", split)
        lbl_dir = os.path.join(HERE, "dataset", "labels", split)
        os.makedirs(lbl_dir, exist_ok=True)
        for img_path in sorted(glob.glob(os.path.join(img_dir, "*.*"))):
            if img_path.endswith(".gitkeep"):
                continue
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            H, W = frame.shape[:2]
            dets = detect(frame, templates)

            stem = os.path.splitext(os.path.basename(img_path))[0]
            out = os.path.join(lbl_dir, stem + ".txt")
            with open(out, "w", encoding="utf-8") as f:
                for score, x1, y1, x2, y2, cls in dets:
                    cx = (x1 + x2) / 2.0 / W
                    cy = (y1 + y2) / 2.0 / H
                    bw = (x2 - x1) / float(W)
                    bh = (y2 - y1) / float(H)
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            total_imgs += 1
            total_boxes += len(dets)
            print(f"{img_path}: {len(dets)} boxes -> {out}")

    print(f"\nDone. Pre-labelled {total_imgs} images, {total_boxes} boxes total.")
    print("Next: open the images in labelImg (YOLO format) and FIX the boxes.")


if __name__ == "__main__":
    main()
