"""
Train a YOLO monster detector for the MapleStory bot.

Prerequisites:
  pip install ultralytics
  Labelled dataset under ml/dataset/{images,labels}/{train,val}
  Class definition in ml/data.yaml

Usage:
  python3 ml/train.py                 # default: yolo11n, 100 epochs, 640 imgsz
  python3 ml/train.py --epochs 200 --model yolov8n.pt --imgsz 640

On a machine WITH an NVIDIA GPU this uses CUDA automatically.  On CPU it still
works with the nano model, just slower.

After training, the best weights are copied to  models/mob_yolo.pt , which is
the path the engine loads by default (config: monster_detect.yolo_model_path).
"""

import argparse
import os
import shutil

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def make_abs_data_yaml():
    """Ultralytics resolves a RELATIVE `path:` against its own global datasets
    dir (e.g. D:\\...\\datasets), not against data.yaml's location, which makes
    training fail with 'images not found, missing path ...\\datasets\\dataset\\
    images\\val'.  Rewrite `path:` to the ABSOLUTE ml/dataset dir into a temp
    yaml so training works the same on every machine/OS without hand-editing.
    Also validates that train/val actually contain images."""
    src = os.path.join(HERE, "data.yaml")
    with open(src, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["path"] = os.path.join(HERE, "dataset")

    # sanity-check that images exist so we fail with a clear message
    for split in ("train", "val"):
        d = os.path.join(cfg["path"], cfg.get(split, f"images/{split}"))
        n = len([x for x in os.listdir(d)]) if os.path.isdir(d) else 0
        if n == 0:
            raise SystemExit(
                f"No images in {d}.\n"
                f"Run the synthetic-data generator first:\n"
                f"    python ml/synth.py --n 800")

    out = os.path.join(HERE, "data.abs.yaml")
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    # yolo11n.pt is the current-gen nano model; yolov8n.pt also fine.
    ap.add_argument("--model", default="yolo11n.pt",
                    help="base weights to fine-tune from")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None,
                    help="e.g. '0' for first GPU, 'cpu' to force CPU. "
                         "Default: auto-detect.")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics is not installed. Run: pip install ultralytics")

    data_yaml = make_abs_data_yaml()
    model = YOLO(args.model)

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=os.path.join(HERE, "runs"),
        name="mob_detector",
        exist_ok=True,
    )

    # Locate best.pt and copy to models/mob_yolo.pt for the engine to load.
    best = os.path.join(HERE, "runs", "mob_detector", "weights", "best.pt")
    if os.path.exists(best):
        os.makedirs(os.path.join(REPO, "models"), exist_ok=True)
        dest = os.path.join(REPO, "models", "mob_yolo.pt")
        shutil.copy(best, dest)
        print(f"\nCopied best weights -> {dest}")
        print("Set config monster_detect.mode: 'yolo' to use it.")
    else:
        print(f"[warn] could not find {best}; check training output.")

    return results


if __name__ == "__main__":
    main()
