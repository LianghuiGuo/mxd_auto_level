"""
Synthetic dataset generator for YOLO monster detection.

Instead of hand-labelling screenshots, this pastes the green-screen monster
sprites from monster/<class>/ onto REAL game-background screenshots at random
positions / counts / flips / slight scales, and writes the YOLO label file for
free (we know exactly where we pasted each sprite).

WHY PASTE ONTO REAL BACKGROUNDS
-------------------------------
A detector trained on sprites over blank/random backgrounds tends to key on the
sprite-vs-background edge and generalises poorly.  Pasting onto actual game
backgrounds (ml/backgrounds/*.png) teaches it the real clutter it must ignore
(fences, posters, grass, HP bars, etc.), which is exactly where template
matching produced false positives.

USAGE
-----
1. Put a few clean game screenshots (few/no monsters) into ml/backgrounds/.
2. Ensure monster/<class>/*.png exist for every class in ml/data.yaml.
3. Run:  python3 ml/synth.py --n 400
   -> writes images+labels into ml/dataset/{images,labels}/{train,val}.

NOTE: sprites are cut using the exact green key color (0,255,0) that mob_maker
uses as background (their alpha is fully opaque, so we can't rely on alpha).
"""

import argparse
import glob
import os
import random

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GREEN = (0, 255, 0)


def scan_monster_classes():
    """Every monster/<name>/ folder that actually contains <name>*.png sprites,
    sorted alphabetically for a stable, reproducible class index."""
    out = []
    for d in sorted(glob.glob(os.path.join(REPO, "monster", "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        if glob.glob(os.path.join(d, f"{name}*.png")):
            out.append(name)
    return out


def load_classes():
    """Return the class-name list (index = position).

    If data.yaml has `auto_scan: true`, the class list is (re)built by scanning
    monster/ so adding a new monster folder needs zero manual edits.  The freshly
    scanned list is written back into data.yaml `names:` so training AND runtime
    inference share one stable index mapping.
    """
    path = os.path.join(HERE, "data.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if cfg.get("auto_scan"):
        classes = scan_monster_classes()
        if not classes:
            raise SystemExit("auto_scan is on but no monster/<name>/<name>*.png "
                             "sprites were found under monster/.")
        cfg["names"] = {i: n for i, n in enumerate(classes)}
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        print(f"[auto_scan] {len(classes)} classes written to data.yaml: "
              f"{classes}")
        return classes

    names = cfg["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def load_sprites(classes):
    """Return {class_idx: [(bgr, alpha_mask_uint8), ...]}."""
    sprites = {}
    for idx, name in enumerate(classes):
        lst = []
        for f in sorted(glob.glob(os.path.join(REPO, "monster", name,
                                               f"{name}*.png"))):
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            bgr = img[:, :, :3]
            # mask = pixels that are NOT the green key
            keep = (~np.all(bgr == GREEN, axis=2)).astype(np.uint8) * 255
            # tighten: erode 1px to kill green fringe
            keep = cv2.erode(keep, np.ones((2, 2), np.uint8))
            lst.append((bgr, keep))
        if lst:
            sprites[idx] = lst
        else:
            print(f"[warn] no sprites for class '{name}'")
    return sprites


def load_backgrounds():
    bgs = []
    for f in sorted(glob.glob(os.path.join(HERE, "backgrounds", "*.*"))):
        if f.endswith(".gitkeep"):
            continue
        im = cv2.imread(f)
        if im is not None:
            bgs.append(im)
    return bgs


def paste(bg, sprite_bgr, mask, x, y):
    h, w = mask.shape
    roi = bg[y:y + h, x:x + w]
    if roi.shape[:2] != (h, w):
        return
    m3 = (mask > 0)[:, :, None]
    roi[:] = np.where(m3, sprite_bgr, roi)


def gen_one(bg, sprites, classes, out_img, out_lbl, max_mobs=6):
    canvas = bg.copy()
    H, W = canvas.shape[:2]
    lines = []
    n = random.randint(1, max_mobs)
    for _ in range(n):
        cls = random.choice(list(sprites.keys()))
        sbgr, smask = random.choice(sprites[cls])

        # random horizontal flip
        if random.random() < 0.5:
            sbgr = cv2.flip(sbgr, 1)
            smask = cv2.flip(smask, 1)

        # slight random scale (0.85 - 1.2) to add size variance
        scale = random.uniform(0.85, 1.2)
        sh, sw = smask.shape
        nw, nh = max(6, int(sw * scale)), max(6, int(sh * scale))
        sbgr = cv2.resize(sbgr, (nw, nh), interpolation=cv2.INTER_NEAREST)
        smask = cv2.resize(smask, (nw, nh), interpolation=cv2.INTER_NEAREST)

        if W - nw <= 1 or H - nh <= 1:
            continue
        # bias vertical placement to the ground band (lower ~60% of screen)
        x = random.randint(0, W - nw - 1)
        y = random.randint(int(H * 0.35), H - nh - 1)

        paste(canvas, sbgr, smask, x, y)

        cx = (x + nw / 2.0) / W
        cy = (y + nh / 2.0) / H
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw / W:.6f} {nh / H:.6f}")

    cv2.imwrite(out_img, canvas)
    with open(out_lbl, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="total images to make")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--max_mobs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    classes = load_classes()
    sprites = load_sprites(classes)
    bgs = load_backgrounds()
    if not sprites:
        raise SystemExit("No sprites found under monster/.")
    if not bgs:
        raise SystemExit("No backgrounds found in ml/backgrounds/. Add some "
                         "clean game screenshots there first.")

    n_val = int(args.n * args.val_frac)
    for i in range(args.n):
        split = "val" if i < n_val else "train"
        img_dir = os.path.join(HERE, "dataset", "images", split)
        lbl_dir = os.path.join(HERE, "dataset", "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        bg = random.choice(bgs)
        stem = f"synth_{i:05d}"
        gen_one(bg, sprites, classes,
                os.path.join(img_dir, stem + ".png"),
                os.path.join(lbl_dir, stem + ".txt"),
                max_mobs=args.max_mobs)

    print(f"Generated {args.n} synthetic images "
          f"({n_val} val / {args.n - n_val} train) from "
          f"{len(bgs)} backgrounds and {len(classes)} classes.")
    print("Next: python3 ml/train.py")


if __name__ == "__main__":
    main()
