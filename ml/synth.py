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


# The player is a first-class detection target too (YOLO replaces the fragile
# nametag / party-red template matching for player localisation).  We always
# put it LAST so appending/removing monsters never shifts its class index.
PLAYER_CLASS = "player"
PLAYER_DIR = os.path.join(REPO, "ml", "player")


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


def _greenish_mask(bgr):
    """Boolean mask of green-screen background pixels.

    We can't rely on an EXACT (0,255,0) match: mob_maker sprites are
    anti-aliased, so the ring of pixels between the sprite and the pure-green
    background are partially-green (e.g. (0,240,20) in BGR).  Those transition
    pixels are only ~40% "pure green" on round sprites like slime, so exact
    matching left a thick green halo/box around the mob in the synthetic image
    (which taught the detector that "slime == green box" and wrecked real-game
    recall).  Treat any pixel with a dominant green channel and low red/blue as
    background so the whole key — including the anti-aliased fringe — is removed.
    """
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    # ONLY the pure green-screen key, not the mob's own (yellowish) green body.
    # Screen key is ~(0,255,0): both R and B are near-zero.  Slime's body green
    # is ~(9,158,107) — its high red (~107) keeps it OUT of this mask.  Keep the
    # red/blue ceilings tight so anti-aliased fringe is removed but body colors
    # survive.
    return (g > 120) & (r < 70) & (b < 70) & (g - r > 90) & (g - b > 90)


def sprite_mask(img):
    """Return (bgr, keep_mask_uint8) from a loaded sprite image.

    The kept region is the sprite itself; the (green-screen) background is
    removed.  Robust to both cut-out styles:
      * transparent PNG  -> start from the alpha channel (hand-cut player).
      * green-screen PNG  -> start from "everything".
    In BOTH cases we ALSO subtract green-screen pixels, because mob_maker mob
    PNGs ship WITH a fully-opaque alpha channel over a green background — so the
    alpha alone keeps the green box.  Removing greenish pixels here is what
    actually cuts the mob out.
    """
    if img is None:
        return None
    bgr = img[:, :, :3]
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        keep = (alpha > 10)
    else:
        keep = np.ones(bgr.shape[:2], dtype=bool)
    # Always drop green-screen background (incl. anti-aliased fringe).
    keep = keep & (~_greenish_mask(bgr))
    keep = (keep.astype(np.uint8)) * 255
    # tighten: erode 1px to kill any 1px green fringe still clinging to edges
    keep = cv2.erode(keep, np.ones((2, 2), np.uint8))
    return bgr, keep


def load_player_sprites():
    """Load hand-cut player sprites from ml/player/*.png (alpha or green key).
    Returns a list of (bgr, mask) or [] if none exist."""
    lst = []
    for f in sorted(glob.glob(os.path.join(PLAYER_DIR, "*.png"))):
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        sm = sprite_mask(img)
        if sm is not None:
            lst.append(sm)
    return lst


def load_classes(data_path=None, explicit_classes=None, dataset_dir="dataset",
                 add_player="auto"):
    """Return the class-name list (index = position) and (re)write it into the
    dataset's data.yaml.

    Parameters
    ----------
    data_path : path to the data.yaml to read/update (default ml/data.yaml).
    explicit_classes : if given, use EXACTLY these monster class names (in this
        order) instead of scanning monster/ — this is how you build a
        fixed-species model (e.g. blue_snail/red_snail/slime).
    dataset_dir : the `path:` value written into data.yaml (relative to ml/), so
        a specialised model can use its own images/labels folder.
    add_player : "auto" appends the player class iff ml/player/ has sprites;
        True forces it; False never adds it.
    """
    if data_path is None:
        data_path = os.path.join(HERE, "data.yaml")
    cfg = {}
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    if explicit_classes is not None:
        classes = list(explicit_classes)
        # validate sprites exist so we fail early with a clear message
        for name in classes:
            if not glob.glob(os.path.join(REPO, "monster", name, f"{name}*.png")):
                raise SystemExit(
                    f"No sprites found for requested class '{name}' at "
                    f"monster/{name}/{name}*.png")
        want_player = (add_player is True) or (
            add_player == "auto" and bool(glob.glob(
                os.path.join(PLAYER_DIR, "*.png"))))
        if want_player and PLAYER_CLASS not in classes:
            classes.append(PLAYER_CLASS)
        cfg["path"] = dataset_dir
        cfg.setdefault("train", "images/train")
        cfg.setdefault("val", "images/val")
        cfg["auto_scan"] = False
        cfg["names"] = {i: n for i, n in enumerate(classes)}
        with open(data_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        print(f"[classes] {len(classes)} fixed classes written to "
              f"{data_path}: {classes}")
        return classes

    if cfg.get("auto_scan"):
        classes = scan_monster_classes()
        if not classes:
            raise SystemExit("auto_scan is on but no monster/<name>/<name>*.png "
                             "sprites were found under monster/.")
        # Append the player class LAST (stable index) whenever the user has put
        # at least one cut-out sprite into ml/player/.
        if (add_player is True) or (add_player == "auto" and glob.glob(
                os.path.join(PLAYER_DIR, "*.png"))):
            classes.append(PLAYER_CLASS)
        cfg["names"] = {i: n for i, n in enumerate(classes)}
        with open(data_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        print(f"[auto_scan] {len(classes)} classes written to {data_path}: "
              f"{classes}")
        return classes

    names = cfg["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def load_sprites(classes):
    """Return {class_idx: [(bgr, mask_uint8), ...]}.

    The `player` class (if present) is loaded from ml/player/*.png; every other
    class is loaded from monster/<name>/<name>*.png.
    """
    sprites = {}
    for idx, name in enumerate(classes):
        if name == PLAYER_CLASS:
            lst = load_player_sprites()
            if not lst:
                print(f"[warn] no sprites in {PLAYER_DIR} for class 'player'")
            else:
                sprites[idx] = lst
            continue
        lst = []
        for f in sorted(glob.glob(os.path.join(REPO, "monster", name,
                                               f"{name}*.png"))):
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            sm = sprite_mask(img)
            if sm is not None:
                lst.append(sm)
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


def _paste_one(canvas, cls, sbgr, smask, W, H):
    """Flip/scale/place a single sprite and return its YOLO label line (or None
    if it couldn't fit)."""
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
        return None
    # bias vertical placement to the ground band (lower ~60% of screen)
    x = random.randint(0, W - nw - 1)
    y = random.randint(int(H * 0.35), H - nh - 1)

    paste(canvas, sbgr, smask, x, y)

    cx = (x + nw / 2.0) / W
    cy = (y + nh / 2.0) / H
    return f"{cls} {cx:.6f} {cy:.6f} {nw / W:.6f} {nh / H:.6f}"


def gen_one(bg, sprites, classes, out_img, out_lbl, max_mobs=6):
    canvas = bg.copy()
    H, W = canvas.shape[:2]
    lines = []

    # The player class index (if it exists) — in a real game screen the player
    # is always present exactly once, so paste it on EVERY image.
    player_idx = classes.index(PLAYER_CLASS) if PLAYER_CLASS in classes else None
    mob_indices = [i for i in sprites if i != player_idx]

    # 1) always place exactly one player (when we have player sprites)
    if player_idx is not None and player_idx in sprites:
        sbgr, smask = random.choice(sprites[player_idx])
        line = _paste_one(canvas, player_idx, sbgr, smask, W, H)
        if line:
            lines.append(line)

    # 2) place a random number of monsters
    if mob_indices:
        for _ in range(random.randint(1, max_mobs)):
            cls = random.choice(mob_indices)
            sbgr, smask = random.choice(sprites[cls])
            line = _paste_one(canvas, cls, sbgr, smask, W, H)
            if line:
                lines.append(line)

    cv2.imwrite(out_img, canvas)
    with open(out_lbl, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="total images to make")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--max_mobs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classes", default=None,
                    help="comma-separated monster class names to build a "
                         "FIXED-species model, e.g. "
                         "--classes blue_snail,red_snail,slime . "
                         "Omit to auto-scan all monster/ folders.")
    ap.add_argument("--data", default="data.yaml",
                    help="data.yaml filename under ml/ to read/write "
                         "(use a separate one per specialised model).")
    ap.add_argument("--dataset", default="dataset",
                    help="dataset folder name under ml/ for the generated "
                         "images/labels (use a separate one per model).")
    ap.add_argument("--no-player", action="store_true",
                    help="never add the 'player' class even if ml/player/ has "
                         "sprites.")
    args = ap.parse_args()
    random.seed(args.seed)

    explicit = None
    if args.classes:
        explicit = [c.strip() for c in args.classes.split(",") if c.strip()]

    data_path = os.path.join(HERE, args.data)
    add_player = False if args.no_player else "auto"
    classes = load_classes(data_path=data_path, explicit_classes=explicit,
                           dataset_dir=args.dataset, add_player=add_player)
    sprites = load_sprites(classes)
    bgs = load_backgrounds()
    if not sprites:
        raise SystemExit("No sprites found under monster/.")
    if not bgs:
        raise SystemExit("No backgrounds found in ml/backgrounds/. Add some "
                         "clean game screenshots there first.")

    ds_root = os.path.join(HERE, args.dataset)
    n_val = int(args.n * args.val_frac)
    for i in range(args.n):
        split = "val" if i < n_val else "train"
        img_dir = os.path.join(ds_root, "images", split)
        lbl_dir = os.path.join(ds_root, "labels", split)
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
          f"{len(bgs)} backgrounds and {len(classes)} classes "
          f"into ml/{args.dataset}/.")
    print(f"Next: python3 ml/train.py --data {args.data} "
          f"--name {os.path.splitext(args.data)[0]}")


if __name__ == "__main__":
    main()
