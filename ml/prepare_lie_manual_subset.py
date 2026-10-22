#!/usr/bin/env python3
"""Sample a small bootstrap subset for manual lie-panel annotation.

Copies a fixed number of already-cropped panel frames from
``ml/lie_real_dataset`` into ``ml/lie_manual_dataset``, preserving the
whole-video train/val split from ``ml/lie_videos_config.json``.

Existing machine pre-labels are NOT copied — the local annotator starts blank
on purpose.

    python3 ml/prepare_lie_manual_subset.py --per-video 10
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _video_stem(filename: str) -> str:
    return Path(filename).stem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=HERE / "lie_real_dataset",
        help="source dataset with images/{train,val}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "lie_manual_dataset",
        help="manual annotation workspace",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "lie_videos_config.json",  # same file as detect_lie_panels output
    )
    parser.add_argument("--per-video", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete existing output images/labels before copying",
    )
    args = parser.parse_args()

    if args.per_video < 1:
        parser.error("--per-video must be >= 1")
    if not args.config.is_file():
        parser.error(f"config not found: {args.config}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    val_names = set(config.get("val", []))
    recordings = [entry["filename"] for entry in config.get("recordings", [])]
    if not recordings:
        parser.error("no recordings in config")

    if args.reset and args.output.exists():
        shutil.rmtree(args.output)

    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    stems = sorted((_video_stem(name) for name in recordings), key=len, reverse=True)
    by_video: dict[str, list[Path]] = {stem: [] for stem in stems}
    for split in ("train", "val"):
        image_dir = args.source / "images" / split
        if not image_dir.is_dir():
            continue
        for path in sorted(image_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            for stem in stems:
                if path.name.startswith(stem + "_") or path.stem == stem:
                    by_video[stem].append(path)
                    break

    summary = {"per_video": args.per_video, "seed": args.seed, "videos": []}
    total = 0
    for filename in recordings:
        stem = _video_stem(filename)
        pool = by_video.get(stem, [])
        if not pool:
            print(f"WARN: no source frames for {filename}")
            continue
        pick = pool if len(pool) <= args.per_video else rng.sample(pool, args.per_video)
        pick = sorted(pick)
        split = "val" if filename in val_names else "train"
        for src in pick:
            dst = args.output / "images" / split / src.name
            shutil.copy2(src, dst)
            # Ensure no stale label from a previous run for this filename.
            for suffix in (".txt", ".meta.json"):
                stale = args.output / "labels" / split / f"{src.stem}{suffix}"
                if stale.exists():
                    stale.unlink()
            total += 1
        summary["videos"].append(
            {
                "filename": filename,
                "split": split,
                "available": len(pool),
                "sampled": len(pick),
            }
        )
        print(f"{filename}: {len(pick)}/{len(pool)} -> {split}")

    (args.output / "classes.txt").write_text("lie_shape\n", encoding="utf-8")
    (args.output / "subset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Total sampled frames: {total}")
    print(f"Output: {args.output}")
    print("Next: python3 ml/lie_annotator/server.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
