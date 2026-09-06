#!/usr/bin/env python3
"""Local browser UI for manually labelling lie-detector panel crops.

Why this exists
---------------
Background-diff pre-labels on the rocky lie panel are often unusable (near
full-image boxes).  For a small bootstrap set it is faster to draw boxes by
hand than to fight the pre-labeller or stand up CVAT.

This tool is intentionally small: one class (``lie_shape``), YOLO txt on disk,
keyboard-driven.  It is for ~50–150 carefully labelled frames, not a full
annotation platform.

    # 1) sample a bootstrap subset (optional but recommended)
    python3 ml/prepare_lie_manual_subset.py --per-video 10

    # 2) open the local annotator
    python3 ml/lie_annotator/server.py
    # browser -> http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _yolo_path(labels_root: Path, split: str, stem: str) -> Path:
    return labels_root / split / f"{stem}.txt"


def _meta_path(labels_root: Path, split: str, stem: str) -> Path:
    return labels_root / split / f"{stem}.meta.json"


def list_items(images_root: Path, labels_root: Path) -> list[dict]:
    items: list[dict] = []
    for split in ("train", "val"):
        image_dir = images_root / split
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            stem = image_path.stem
            label_path = _yolo_path(labels_root, split, stem)
            meta_path = _meta_path(labels_root, split, stem)
            boxes: list[list[float]] = []
            status = "unset"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                status = str(meta.get("status", "unset"))
            if label_path.is_file():
                boxes = _read_yolo(label_path)
                if status == "unset":
                    status = "empty" if not boxes else "done"
            items.append(
                {
                    "id": f"{split}/{image_path.name}",
                    "split": split,
                    "name": image_path.name,
                    "stem": stem,
                    "status": status,
                    "box_count": len(boxes),
                }
            )
    return items


def _read_yolo(path: Path) -> list[list[float]]:
    boxes: list[list[float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields")
        class_id = int(parts[0])
        values = [float(v) for v in parts[1:]]
        if class_id != 0 or any(not 0.0 <= v <= 1.0 for v in values):
            raise ValueError(f"{path}:{line_number}: invalid YOLO row")
        boxes.append([0.0, *values])
    return boxes


def _write_yolo(path: Path, boxes: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for box in boxes:
        if len(box) != 5:
            raise ValueError(f"box must be [cls,cx,cy,w,h], got {box}")
        class_id = int(box[0])
        cx, cy, bw, bh = (float(v) for v in box[1:])
        if class_id != 0 or any(not 0.0 <= v <= 1.0 for v in (cx, cy, bw, bh)):
            raise ValueError(f"out-of-range YOLO box: {box}")
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")


def _write_meta(path: Path, status: str, note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"status": status, "note": note}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def make_handler(images_root: Path, labels_root: Path, load_existing: bool):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # quieter console
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: object) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(code, raw, "application/json; charset=utf-8")

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)

            if path in ("/", "/index.html"):
                html = (STATIC / "index.html").read_bytes()
                self._send(200, html, "text/html; charset=utf-8")
                return

            if path == "/api/list":
                items = list_items(images_root, labels_root)
                if not load_existing:
                    # Treat machine pre-labels as absent unless a meta status exists.
                    for item in items:
                        meta = _meta_path(labels_root, item["split"], item["stem"])
                        if not meta.is_file():
                            item["status"] = "unset"
                            item["box_count"] = 0
                stats = {
                    "total": len(items),
                    "done": sum(1 for i in items if i["status"] == "done"),
                    "empty": sum(1 for i in items if i["status"] == "empty"),
                    "unset": sum(1 for i in items if i["status"] == "unset"),
                    "boxes": sum(i["box_count"] for i in items if i["status"] == "done"),
                }
                self._json(200, {"items": items, "stats": stats, "class_name": "lie_shape"})
                return

            match = re.fullmatch(r"/api/image/(train|val)/([^/]+)", path)
            if match:
                split, name = match.group(1), match.group(2)
                image_path = images_root / split / name
                if not image_path.is_file():
                    self._json(404, {"error": "image not found"})
                    return
                data = image_path.read_bytes()
                suffix = image_path.suffix.lower()
                ctype = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".bmp": "image/bmp",
                    ".webp": "image/webp",
                }.get(suffix, "application/octet-stream")
                self._send(200, data, ctype)
                return

            match = re.fullmatch(r"/api/label/(train|val)/([^/]+)", path)
            if match:
                split, name = match.group(1), match.group(2)
                stem = Path(name).stem
                label_path = _yolo_path(labels_root, split, stem)
                meta_path = _meta_path(labels_root, split, stem)
                status = "unset"
                boxes: list[list[float]] = []
                note = ""
                if meta_path.is_file():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    status = str(meta.get("status", "unset"))
                    note = str(meta.get("note", ""))
                if label_path.is_file() and (load_existing or meta_path.is_file()):
                    boxes = _read_yolo(label_path)
                    if status == "unset":
                        status = "empty" if not boxes else "done"
                self._json(
                    200,
                    {
                        "split": split,
                        "name": name,
                        "status": status,
                        "boxes": boxes,
                        "note": note,
                    },
                )
                return

            if path.startswith("/static/"):
                rel = path[len("/static/") :]
                file_path = (STATIC / rel).resolve()
                if not str(file_path).startswith(str(STATIC.resolve())) or not file_path.is_file():
                    self._json(404, {"error": "not found"})
                    return
                data = file_path.read_bytes()
                ctype = "text/css" if file_path.suffix == ".css" else "application/javascript"
                self._send(200, data, ctype)
                return

            self._json(404, {"error": f"unknown path {path}"})

        def do_PUT(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            match = re.fullmatch(r"/api/label/(train|val)/([^/]+)", path)
            if not match:
                self._json(404, {"error": "unknown path"})
                return
            split, name = match.group(1), match.group(2)
            stem = Path(name).stem
            payload = self._read_json()
            status = str(payload.get("status", "done"))
            if status not in {"done", "empty", "unset"}:
                self._json(400, {"error": "status must be done|empty|unset"})
                return
            boxes = payload.get("boxes", [])
            if not isinstance(boxes, list):
                self._json(400, {"error": "boxes must be a list"})
                return
            try:
                if status == "unset":
                    for path_to_delete in (
                        _yolo_path(labels_root, split, stem),
                        _meta_path(labels_root, split, stem),
                    ):
                        if path_to_delete.is_file():
                            path_to_delete.unlink()
                elif status == "empty":
                    _write_yolo(_yolo_path(labels_root, split, stem), [])
                    _write_meta(_meta_path(labels_root, split, stem), "empty")
                else:
                    _write_yolo(_yolo_path(labels_root, split, stem), boxes)
                    _write_meta(_meta_path(labels_root, split, stem), "done")
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(
                200,
                {
                    "ok": True,
                    "split": split,
                    "name": name,
                    "status": status,
                    "box_count": 0 if status != "done" else len(boxes),
                },
            )

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        default=HERE.parent / "lie_manual_dataset" / "images",
        help="images/{train,val} root",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=HERE.parent / "lie_manual_dataset" / "labels",
        help="labels/{train,val} root (YOLO txt + optional .meta.json)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--load-existing",
        action="store_true",
        help="load existing YOLO txt even without a manual .meta.json "
        "(off by default because machine pre-labels are usually bad)",
    )
    args = parser.parse_args()

    if not (STATIC / "index.html").is_file():
        parser.error(f"missing UI file: {STATIC / 'index.html'}")
    if not args.images.is_dir():
        parser.error(
            f"images root not found: {args.images}\n"
            "Run: python3 ml/prepare_lie_manual_subset.py --per-video 10"
        )

    args.labels.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (args.labels / split).mkdir(parents=True, exist_ok=True)

    handler = make_handler(args.images.resolve(), args.labels.resolve(), args.load_existing)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    items = list_items(args.images, args.labels)
    print(f"Lie manual annotator")
    print(f"  images : {args.images}")
    print(f"  labels : {args.labels}")
    print(f"  frames : {len(items)}")
    print(f"  open   : http://{args.host}:{args.port}")
    print("  keys   : drag=draw box | click=select | Del=delete box | S=save")
    print("           A/←=prev | D/→=next | E=mark empty(neg) | U=unset")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
