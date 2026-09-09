"""Compare lie-detector tracker iterations from replay summaries and CSV runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Optional


def metrics_from_csv(path: Path) -> Optional[dict]:
    errors: list[float] = []
    switches = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("error_px", "")
            if raw:
                errors.append(float(raw))
            if str(row.get("identity_switched", "")).lower() in {"1", "true"}:
                switches += 1
    if not errors:
        return None
    return {
        "within_40px_ratio": sum(1 for e in errors if e <= 40.0) / len(errors),
        "median_error_px": statistics.median(errors),
        "identity_switches": switches,
        "evaluated_error_frames": len(errors),
    }


def load_summary(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["label"]: row for row in payload["results"]}


def cell(row: Optional[dict]) -> str:
    if not row or row.get("within_40px_ratio") is None:
        return "n/a"
    return (
        f"{row['within_40px_ratio']:.3f}/"
        f"{row['median_error_px']:.1f}/"
        f"{row['identity_switches']}"
    )


def frame_weighted(rows: list[Optional[dict]]) -> Optional[float]:
    frames = hits = 0.0
    for row in rows:
        if not row or row.get("within_40px_ratio") is None:
            continue
        count = row["evaluated_error_frames"]
        frames += count
        hits += row["within_40px_ratio"] * count
    return None if not frames else hits / frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="named replay summary JSON",
    )
    parser.add_argument(
        "--csv-run",
        action="append",
        default=[],
        metavar="NAME=GLOB_TEMPLATE",
        help="named CSV run, template with {n} for the clip number",
    )
    parser.add_argument("--clips", default="1,2,3,4,5,6,7,8,9,10,11")
    args = parser.parse_args()

    clips = [int(part) for part in args.clips.split(",")]
    columns: list[tuple[str, dict[str, dict]]] = []
    for spec in args.summary:
        name, _, path = spec.partition("=")
        columns.append((name, load_summary(Path(path))))
    for spec in args.csv_run:
        name, _, template = spec.partition("=")
        table: dict[str, dict] = {}
        for clip in clips:
            path = Path(template.format(n=clip))
            if path.exists():
                found = metrics_from_csv(path)
                if found:
                    table[f"录屏{clip}"] = found
        columns.append((name, table))

    header = f"{'video':10}" + "".join(f"{name:>24}" for name, _ in columns)
    print(header)
    print("-" * len(header))
    for clip in clips:
        label = f"录屏{clip}"
        line = f"{label:10}"
        for _, table in columns:
            line += f"{cell(table.get(label)):>24}"
        print(line)
    print("-" * len(header))
    line = f"{'weighted':10}"
    for _, table in columns:
        value = frame_weighted([table.get(f"录屏{c}") for c in clips])
        line += f"{'n/a' if value is None else f'{value:.4f}':>24}"
    print(line)


if __name__ == "__main__":
    main()
