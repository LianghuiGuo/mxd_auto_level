#!/usr/bin/env python3
"""Run cumulative motion-feature ablations for the lie identity ranker.

The existing trajectory dataset remains untouched.  This script augments its
listwise queries with four increasingly expensive, appearance-agnostic cues:

1. a robust global similarity-transform residual;
2. local k-nearest-neighbour graph strain;
3. masked local optical-flow spin;
4. multi-lag polar-descriptor spin.

The green cursor is removed from every source frame before pixel features are
computed.  Results and cached augmented rows are written to a separate output
directory so the online model cannot be overwritten accidentally.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import csv
import json
import math
from itertools import zip_longest
from pathlib import Path
import sys
from typing import Iterable, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.LieDetectorTracker import (  # noqa: E402
    LieDetectorTracker,
    ShapeDetection,
)

SIMILARITY_FEATURES = (
    "global_similarity_residual_norm",
    "global_similarity_residual_peer_z",
    "history_mean_global_similarity_residual_norm",
    "history_std_global_similarity_residual_norm",
)
GRAPH_FEATURES = (
    "local_graph_strain_norm",
    "local_graph_strain_peer_z",
    "history_mean_local_graph_strain_norm",
    "history_std_local_graph_strain_norm",
)
FLOW_FEATURES = (
    "optical_spin_abs_norm",
    "optical_spin_confidence",
    "optical_spin_evidence",
)
_CURSOR_BLIND_RADIUS_FRACTION = 0.44
_FLOW_OUTER_RADIUS_FRACTION = 0.86
_POLAR_BLIND_RADIAL_BINS = 18
MULTILAG_FEATURES = (
    "multilag_spin_abs_norm",
    "multilag_spin_confidence",
    "multilag_spin_consistency",
)

STAGES = (
    ("baseline", ()),
    ("similarity", SIMILARITY_FEATURES),
    ("graph", (*SIMILARITY_FEATURES, *GRAPH_FEATURES)),
    ("optical_flow", (*SIMILARITY_FEATURES, *GRAPH_FEATURES, *FLOW_FEATURES)),
    (
        "multilag_rotation",
        (
            *SIMILARITY_FEATURES,
            *GRAPH_FEATURES,
            *FLOW_FEATURES,
            *MULTILAG_FEATURES,
        ),
    ),
)
SELECTION_STAGES = (
    ("baseline_optical_angle", (FLOW_FEATURES[0], FLOW_FEATURES[2])),
    ("baseline_optical_flow", FLOW_FEATURES),
    ("baseline_multilag_rotation", MULTILAG_FEATURES),
    ("baseline_optical_multilag", (*FLOW_FEATURES, *MULTILAG_FEATURES)),
    (
        "baseline_optical_angle_multilag",
        (FLOW_FEATURES[0], FLOW_FEATURES[2], *MULTILAG_FEATURES),
    ),
    (
        "similarity_optical_multilag",
        (*SIMILARITY_FEATURES, *FLOW_FEATURES, *MULTILAG_FEATURES),
    ),
    (
        "graph_optical_multilag",
        (*GRAPH_FEATURES, *FLOW_FEATURES, *MULTILAG_FEATURES),
    ),
)
LOVO_STAGES = (
    ("baseline", ()),
    ("baseline_optical_multilag", (*FLOW_FEATURES, *MULTILAG_FEATURES)),
    (
        "baseline_optical_angle_multilag",
        (FLOW_FEATURES[0], FLOW_FEATURES[2], *MULTILAG_FEATURES),
    ),
    (
        "similarity_optical_multilag",
        (*SIMILARITY_FEATURES, *FLOW_FEATURES, *MULTILAG_FEATURES),
    ),
)


def _safe_z(values: np.ndarray) -> np.ndarray:
    if values.size < 2:
        return np.zeros_like(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1e-8, 1.4826 * mad)
    return np.clip((values - median) / scale, -8.0, 8.0)


def _fit_similarity(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit current = translation + complex_scale * (previous - center)."""
    count = len(previous)
    if count < 2:
        center = np.mean(previous, axis=0) if count else np.zeros(2)
        translation = (
            np.median(current - previous, axis=0) if count else np.zeros(2)
        )
        return center, np.asarray([*translation, 1.0, 0.0]), 1.0
    center = np.median(previous, axis=0)
    centered = previous - center
    design = np.zeros((2 * count, 4), dtype=np.float64)
    design[0::2] = np.column_stack(
        [np.ones(count), np.zeros(count), centered[:, 0], -centered[:, 1]]
    )
    design[1::2] = np.column_stack(
        [np.zeros(count), np.ones(count), centered[:, 1], centered[:, 0]]
    )
    target = current.reshape(-1)
    weights = np.ones(count, dtype=np.float64)
    beta = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    for _ in range(iterations):
        row_weights = np.repeat(np.sqrt(weights), 2)
        beta = np.linalg.lstsq(
            design * row_weights[:, None],
            target * row_weights,
            rcond=None,
        )[0]
        residuals = (design @ beta - target).reshape(-1, 2)
        distances = np.linalg.norm(residuals, axis=1)
        median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median)))
        scale = max(0.35, 1.4826 * mad)
        weights = np.minimum(1.0, 1.5 * scale / np.maximum(distances, 1e-8))
    similarity_scale = float(math.hypot(beta[2], beta[3]))
    return center, beta, similarity_scale


def _predict_similarity(
    points: np.ndarray, center: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    relative = points - center
    tx, ty, scale_cos, scale_sin = beta
    return np.column_stack(
        [
            tx + scale_cos * relative[:, 0] - scale_sin * relative[:, 1],
            ty + scale_sin * relative[:, 0] + scale_cos * relative[:, 1],
        ]
    )


def _graph_strain(
    previous: np.ndarray,
    current: np.ndarray,
    scale: float,
    *,
    neighbours: int = 3,
) -> np.ndarray:
    count = len(previous)
    if count < 2:
        return np.zeros(count, dtype=np.float64)
    pairwise = np.linalg.norm(previous[:, None, :] - previous[None, :, :], axis=2)
    result = np.zeros(count, dtype=np.float64)
    for index in range(count):
        order = np.argsort(pairwise[index])
        adjacent = [item for item in order if item != index][: min(neighbours, count - 1)]
        errors = []
        for other in adjacent:
            before = pairwise[index, other]
            after = float(np.linalg.norm(current[index] - current[other]))
            errors.append(abs(after - scale * before))
        result[index] = float(np.median(errors)) if errors else 0.0
    return result


def _candidate_history_positions(
    candidate: dict,
    history_feature_names: Sequence[str],
    width: float,
    height: float,
) -> np.ndarray:
    x_index = history_feature_names.index("x_norm")
    y_index = history_feature_names.index("y_norm")
    return np.asarray(
        [
            [float(item[x_index]) * width, float(item[y_index]) * height]
            for item in candidate["history"]
        ],
        dtype=np.float64,
    )


def constellation_features(
    candidates: Sequence[dict],
    history_feature_names: Sequence[str],
    width: int,
    height: int,
    *,
    history_limit: int = 8,
) -> dict[int, dict[str, float]]:
    """Return robust similarity and local graph features per track."""
    track_ids = [int(item["track_id"]) for item in candidates]
    histories = [
        _candidate_history_positions(item, history_feature_names, width, height)
        for item in candidates
    ]
    diagonal = max(1.0, math.hypot(width, height))
    residual_history: dict[int, list[float]] = defaultdict(list)
    strain_history: dict[int, list[float]] = defaultdict(list)

    for lag in range(history_limit):
        eligible = [
            index for index, history in enumerate(histories) if len(history) >= lag + 2
        ]
        if len(eligible) < 3:
            continue
        previous = np.asarray([histories[i][-lag - 2] for i in eligible])
        current = np.asarray([histories[i][-lag - 1] for i in eligible])
        center, beta, scale = _fit_similarity(previous, current)
        predicted = _predict_similarity(previous, center, beta)
        residuals = np.linalg.norm(current - predicted, axis=1) / diagonal
        strains = _graph_strain(previous, current, scale) / diagonal
        for position, index in enumerate(eligible):
            residual_history[track_ids[index]].append(float(residuals[position]))
            strain_history[track_ids[index]].append(float(strains[position]))

    current_residuals = np.asarray(
        [
            residual_history[track_id][0] if residual_history[track_id] else 0.0
            for track_id in track_ids
        ],
        dtype=np.float64,
    )
    current_strains = np.asarray(
        [
            strain_history[track_id][0] if strain_history[track_id] else 0.0
            for track_id in track_ids
        ],
        dtype=np.float64,
    )
    residual_z = _safe_z(current_residuals)
    strain_z = _safe_z(current_strains)
    output: dict[int, dict[str, float]] = {}
    for index, track_id in enumerate(track_ids):
        residual_values = residual_history[track_id]
        strain_values = strain_history[track_id]
        output[track_id] = {
            "global_similarity_residual_norm": float(current_residuals[index]),
            "global_similarity_residual_peer_z": float(residual_z[index]),
            "history_mean_global_similarity_residual_norm": (
                float(np.mean(residual_values)) if residual_values else 0.0
            ),
            "history_std_global_similarity_residual_norm": (
                float(np.std(residual_values)) if residual_values else 0.0
            ),
            "local_graph_strain_norm": float(current_strains[index]),
            "local_graph_strain_peer_z": float(strain_z[index]),
            "history_mean_local_graph_strain_norm": (
                float(np.mean(strain_values)) if strain_values else 0.0
            ),
            "history_std_local_graph_strain_norm": (
                float(np.std(strain_values)) if strain_values else 0.0
            ),
        }
    return output


def _clean_gray(frame: np.ndarray) -> np.ndarray:
    cleaned, _ = LieDetectorTracker._remove_cursor(frame)
    return cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)


def optical_spin_features(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_center: Sequence[float],
    current_center: Sequence[float],
    radius: float,
) -> tuple[float, float, float]:
    """Estimate local self-spin after allowing arbitrary local translation."""
    height, width = previous_gray.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    center = tuple(int(round(value)) for value in previous_center)
    sample_radius = max(8, int(round(0.78 * radius)))
    # The label cursor is centred on the positive candidate.  A fixed annular
    # sampling region for *every* candidate prevents the green ring, its
    # inpainted texture, or a reduced corner count from becoming a label proxy.
    # The surviving outer shape boundary is also the most useful region for
    # measuring tangential self-spin.
    inner_radius = max(7, int(round(_CURSOR_BLIND_RADIUS_FRACTION * radius)))
    sample_radius = max(
        inner_radius + 4, int(round(_FLOW_OUTER_RADIUS_FRACTION * radius))
    )
    cv2.circle(mask, center, sample_radius, 255, -1)
    cv2.circle(mask, center, inner_radius, 0, -1)
    corners = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=60,
        qualityLevel=0.025,
        minDistance=3.0,
        mask=mask,
        blockSize=5,
    )
    if corners is None or len(corners) < 6:
        return 0.0, 0.0, 0.0
    moved, status, errors = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        corners,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01),
    )
    if moved is None or status is None:
        return 0.0, 0.0, 0.0
    valid = status.reshape(-1) > 0
    if errors is not None:
        valid &= errors.reshape(-1) < 24.0
    source = corners.reshape(-1, 2)[valid]
    target = moved.reshape(-1, 2)[valid]
    if len(source) < 6:
        return 0.0, 0.0, 0.0
    next_center = np.asarray(current_center, dtype=np.float64)
    target_radius = np.linalg.norm(target - next_center, axis=1)
    valid_target = (
        (target_radius >= _CURSOR_BLIND_RADIUS_FRACTION * radius)
        & (target_radius <= 1.02 * radius)
    )
    source = source[valid_target]
    target = target[valid_target]
    if len(source) < 6:
        return 0.0, 0.0, 0.0
    matrix, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=1.7,
        maxIters=200,
        confidence=0.98,
        refineIters=10,
    )
    if matrix is None or inliers is None:
        return 0.0, 0.0, 0.0
    angle = math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
    inlier_ratio = float(np.mean(inliers.reshape(-1) > 0))
    coverage = min(1.0, len(source) / 24.0)
    confidence = inlier_ratio * coverage
    normalized = min(3.0, abs(angle) / 22.0)
    return float(normalized), float(confidence), float(normalized * confidence)


def _polar_descriptor(
    gray: np.ndarray, center: Sequence[float], radius: float
) -> np.ndarray | None:
    detection = ShapeDetection(
        center=(float(center[0]), float(center[1])),
        radius=float(radius),
        bbox=(
            int(round(center[0] - radius)),
            int(round(center[1] - radius)),
            max(2, int(round(2.0 * radius))),
            max(2, int(round(2.0 * radius))),
        ),
    )
    descriptor = LieDetectorTracker._rotation_descriptor(
        LieDetectorTracker._rotation_edge_map(gray), detection
    )
    if descriptor is None:
        return None
    # See optical_spin_features: make the centre cursor-blind for every
    # candidate.  The tracker descriptor's radial axis has 40 bins and the
    # candidate radius occupies roughly 36 of them, so 18 bins conservatively
    # suppress the cursor/inpaint support while retaining the outer boundary.
    descriptor = descriptor.copy()
    descriptor[:, :_POLAR_BLIND_RADIAL_BINS] = 0.0
    norm = float(np.linalg.norm(descriptor))
    return None if norm <= 1e-6 else descriptor / norm


def _match_multilag(
    previous: np.ndarray, current: np.ndarray, *, max_degrees: float
) -> tuple[float | None, float]:
    if previous.shape != current.shape or previous.ndim != 2:
        return None, 0.0
    degrees_per_bin = 360.0 / previous.shape[0]
    max_shift = max(1, int(round(max_degrees / degrees_per_bin)))
    shifts = np.arange(-max_shift, max_shift + 1, dtype=np.int32)
    correlations = np.asarray(
        [float(np.sum(previous * np.roll(current, int(shift), axis=0))) for shift in shifts]
    )
    best = int(np.argmax(correlations))
    peak = float(correlations[best])
    if peak < 0.25:
        return None, 0.0
    exclusion = max(2, int(round(5.0 / degrees_per_bin)))
    alternatives = np.ones(len(correlations), dtype=bool)
    alternatives[max(0, best - exclusion) : min(len(correlations), best + exclusion + 1)] = False
    second = float(np.max(correlations[alternatives])) if np.any(alternatives) else -1.0
    confidence = float(
        np.clip((peak - 0.25) / 0.65, 0.0, 1.0)
        * np.clip((peak - second) / 0.10, 0.0, 1.0)
    )
    if confidence < 0.05:
        return None, 0.0
    return float(-shifts[best] * degrees_per_bin), confidence


def multilag_rotation_features(
    frames: dict[int, np.ndarray],
    frame_index: int,
    history_positions: np.ndarray,
    radius: float,
) -> tuple[float, float, float]:
    current_gray = frames.get(frame_index)
    if current_gray is None or not len(history_positions):
        return 0.0, 0.0, 0.0
    current_descriptor = _polar_descriptor(
        current_gray, history_positions[-1], radius
    )
    if current_descriptor is None:
        return 0.0, 0.0, 0.0
    measurements: list[tuple[float, float]] = []
    for lag in (1, 2, 4):
        if len(history_positions) <= lag or frame_index - lag not in frames:
            continue
        previous_descriptor = _polar_descriptor(
            frames[frame_index - lag], history_positions[-lag - 1], radius
        )
        if previous_descriptor is None:
            continue
        delta, confidence = _match_multilag(
            previous_descriptor,
            current_descriptor,
            max_degrees=min(88.0, 22.0 * lag),
        )
        if delta is not None:
            measurements.append((float(delta) / lag, confidence))
    if not measurements:
        return 0.0, 0.0, 0.0
    values = np.asarray([item[0] for item in measurements], dtype=np.float64)
    weights = np.asarray([item[1] for item in measurements], dtype=np.float64)
    signed = float(np.average(values, weights=np.maximum(weights, 1e-6)))
    sign_consistency = abs(float(np.sum(weights * np.sign(values)))) / max(
        float(np.sum(weights)), 1e-6
    )
    confidence = float(np.mean(weights)) * sign_consistency
    return min(3.0, abs(signed) / 22.0), confidence, sign_consistency


def _load_queries(dataset: Path, manifest: dict) -> tuple[list[dict], list[dict]]:
    base_names = manifest["flat_feature_names"]
    output = []
    for split in ("train", "val"):
        queries = []
        with (dataset / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                query = json.loads(line)
                for candidate in query["candidates"]:
                    candidate["base"] = dict(zip(base_names, candidate["features"]))
                query["split"] = split
                queries.append(query)
        output.append(queries)
    return output[0], output[1]


def _video_specs(config_path: Path) -> dict[str, dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {item["filename"]: item for item in config["recordings"]}


def augment_queries(
    queries: Sequence[dict],
    *,
    videos_dir: Path,
    specs: dict[str, dict],
    history_feature_names: Sequence[str],
) -> list[dict]:
    by_video: dict[str, list[dict]] = defaultdict(list)
    for query in queries:
        by_video[query["video"]].append(query)

    augmented: list[dict] = []
    for video_name, video_queries in sorted(by_video.items()):
        spec = specs[video_name]
        x, y, width, height = (int(value) for value in spec["roi"])
        ordered = sorted(video_queries, key=lambda item: int(item["frame"]))
        wanted = {int(item["frame"]) for item in ordered}
        needed = {max(0, frame - lag) for frame in wanted for lag in range(5)}
        capture = cv2.VideoCapture(str(videos_dir / video_name))
        if not capture.isOpened():
            raise RuntimeError(f"unable to open {video_name}")
        frames: dict[int, np.ndarray] = {}
        frame_index = 0
        last_needed = max(needed)
        while frame_index <= last_needed:
            ok, full = capture.read()
            if not ok:
                break
            if frame_index in needed:
                frames[frame_index] = _clean_gray(full[y : y + height, x : x + width])
            frame_index += 1
        capture.release()

        for query in ordered:
            frame = int(query["frame"])
            constellation = constellation_features(
                query["candidates"], history_feature_names, width, height
            )
            for candidate in query["candidates"]:
                track_id = int(candidate["track_id"])
                candidate["extra"] = dict(constellation[track_id])
                history = _candidate_history_positions(
                    candidate, history_feature_names, width, height
                )
                radius = float(candidate["base"]["radius_norm"]) * min(width, height)
                if len(history) >= 2 and frame - 1 in frames and frame in frames:
                    optical = optical_spin_features(
                        frames[frame - 1],
                        frames[frame],
                        history[-2],
                        history[-1],
                        radius,
                    )
                else:
                    optical = (0.0, 0.0, 0.0)
                candidate["extra"].update(dict(zip(FLOW_FEATURES, optical)))
                multilag = multilag_rotation_features(
                    frames, frame, history, radius
                )
                candidate["extra"].update(dict(zip(MULTILAG_FEATURES, multilag)))
            augmented.append(query)
        print(f"augmented {video_name}: {len(ordered)} queries", flush=True)
    return augmented


def write_cache(queries: Sequence[dict], path: Path, base_names: Sequence[str]) -> None:
    extra_names = [
        *SIMILARITY_FEATURES,
        *GRAPH_FEATURES,
        *FLOW_FEATURES,
        *MULTILAG_FEATURES,
    ]
    metadata = [
        "query_id", "video", "frame", "track_id", "label", "group_size"
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*metadata, *base_names, *extra_names])
        writer.writeheader()
        for query in queries:
            for candidate in query["candidates"]:
                row = {
                    "query_id": query["query_id"],
                    "video": query["video"],
                    "frame": query["frame"],
                    "track_id": candidate["track_id"],
                    "label": candidate["label"],
                    "group_size": len(query["candidates"]),
                }
                row.update(candidate["base"])
                row.update(candidate["extra"])
                writer.writerow(row)


def load_matrix(
    base_path: Path,
    augmented_path: Path,
    feature_names: Sequence[str],
    base_names: Sequence[str],
) -> dict:
    """Load exact base values plus cached extras without quantizing the baseline.

    The JSONL history intentionally stores values rounded to six decimals, while
    the canonical CSV retains more precision.  Re-serializing the JSONL values
    changes LightGBM's split thresholds enough to make the 43-feature baseline
    irreproducible, so base values always come from the canonical CSV.
    """
    rows = []
    labels = []
    groups = []
    videos = []
    last_query = None
    group_size = 0
    base_name_set = set(base_names)
    with (
        base_path.open(encoding="utf-8") as base_handle,
        augmented_path.open(encoding="utf-8") as augmented_handle,
    ):
        base_reader = csv.DictReader(base_handle)
        augmented_reader = csv.DictReader(augmented_handle)
        sentinel = object()
        for item, augmented in zip_longest(
            base_reader, augmented_reader, fillvalue=sentinel
        ):
            if item is sentinel or augmented is sentinel:
                raise ValueError("canonical and augmented CSV row counts differ")
            identity = ("query_id", "video", "frame", "track_id", "label")
            if any(item[name] != augmented[name] for name in identity):
                raise ValueError(
                    f"cache row mismatch: {item['query_id']}/{item['track_id']} != "
                    f"{augmented['query_id']}/{augmented['track_id']}"
                )
            query = item["query_id"]
            if last_query is None:
                last_query = query
                current_video = item["video"]
            elif query != last_query:
                groups.append(group_size)
                videos.append(current_video)
                group_size = 0
                last_query = query
                current_video = item["video"]
            rows.append(
                [
                    float(item[name] if name in base_name_set else augmented[name])
                    for name in feature_names
                ]
            )
            labels.append(int(item["label"]))
            group_size += 1
    if last_query is not None:
        groups.append(group_size)
        videos.append(current_video)
    return {
        "x": np.asarray(rows, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.int32),
        "groups": np.asarray(groups, dtype=np.int32),
        "videos": np.asarray(videos),
    }


def select_queries(data: dict, mask: np.ndarray) -> dict:
    row_mask = np.repeat(mask, data["groups"])
    return {
        "x": data["x"][row_mask],
        "y": data["y"][row_mask],
        "groups": data["groups"][mask],
        "videos": data["videos"][mask],
    }


def concat(left: dict, right: dict) -> dict:
    return {
        "x": np.concatenate([left["x"], right["x"]]),
        "y": np.concatenate([left["y"], right["y"]]),
        "groups": np.concatenate([left["groups"], right["groups"]]),
        "videos": np.concatenate([left["videos"], right["videos"]]),
    }


def score(predictions: np.ndarray, data: dict, current_index: int) -> dict:
    offset = 0
    hits = []
    hard_hits = []
    for size in data["groups"]:
        end = offset + int(size)
        truth = data["y"][offset:end]
        predicted = int(np.argmax(predictions[offset:end]))
        hit = int(truth[predicted] == 1)
        hits.append(hit)
        current = data["x"][offset:end, current_index]
        positive = int(np.argmax(truth))
        if current[positive] < 0.5:
            hard_hits.append(hit)
        offset = end
    return {
        "queries": len(hits),
        "top1": float(np.mean(hits)),
        "hard_queries": len(hard_hits),
        "hard_top1": float(np.mean(hard_hits)) if hard_hits else None,
    }


def train_model(lgb, train: dict, val: dict, feature_names: Sequence[str], seed: int):
    train_set = lgb.Dataset(
        train["x"], label=train["y"], group=train["groups"], feature_name=list(feature_names)
    )
    val_set = lgb.Dataset(
        val["x"], label=val["y"], group=val["groups"], feature_name=list(feature_names), reference=train_set
    )
    return lgb.train(
        {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1],
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 5,
            "min_data_in_leaf": 40,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "seed": seed,
        },
        train_set,
        num_boost_round=200,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )


def run_ablation(
    dataset: Path,
    train_csv: Path,
    val_csv: Path,
    base_names: Sequence[str],
    *,
    seed: int,
    lovo: bool,
    stages: Sequence[tuple[str, Sequence[str]]] = STAGES,
    model_dir: Path | None = None,
) -> dict:
    import lightgbm as lgb

    results = []
    for stage, additions in stages:
        names = [*base_names, *additions]
        train = load_matrix(dataset / "train.csv", train_csv, names, base_names)
        val = load_matrix(dataset / "val.csv", val_csv, names, base_names)
        current_index = names.index("is_current_target")
        model = train_model(lgb, train, val, names, seed)
        if model_dir is not None:
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(model_dir / f"{stage}.txt"))
        result = {
            "stage": stage,
            "features": len(names),
            "added_features": list(additions),
            "best_iteration": model.best_iteration,
            "train": score(model.predict(train["x"], num_iteration=model.best_iteration), train, current_index),
            "val": score(model.predict(val["x"], num_iteration=model.best_iteration), val, current_index),
            "feature_gain": dict(
                zip(names, [float(value) for value in model.feature_importance(importance_type="gain")])
            ),
        }
        if lovo:
            pooled = concat(train, val)
            folds = []
            for video in sorted(set(pooled["videos"].tolist())):
                mask = pooled["videos"] == video
                fold_train = select_queries(pooled, ~mask)
                fold_val = select_queries(pooled, mask)
                fold_model = train_model(lgb, fold_train, fold_val, names, seed)
                metrics = score(
                    fold_model.predict(fold_val["x"], num_iteration=fold_model.best_iteration),
                    fold_val,
                    current_index,
                )
                metrics["video"] = video
                metrics["best_iteration"] = fold_model.best_iteration
                folds.append(metrics)
            result["lovo"] = {
                "videos": len(folds),
                "queries": sum(item["queries"] for item in folds),
                "top1": sum(item["top1"] * item["queries"] for item in folds) / sum(item["queries"] for item in folds),
                "hard_queries": sum(item["hard_queries"] for item in folds),
                "hard_top1": sum((item["hard_top1"] or 0.0) * item["hard_queries"] for item in folds) / max(1, sum(item["hard_queries"] for item in folds)),
                "mean_video_top1": float(np.mean([item["top1"] for item in folds])),
                "min_video_top1": float(min(item["top1"] for item in folds)),
                "folds": folds,
            }
        results.append(result)
        print(
            f"{stage}: val={result['val']['top1']:.4f} "
            f"hard={result['val']['hard_top1']:.4f}"
            + (
                f" lovo={result['lovo']['top1']:.4f} "
                f"lovo_hard={result['lovo']['hard_top1']:.4f}"
                if lovo else ""
            ),
            flush=True,
        )
    return {"seed": seed, "stages": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("ml/lie_identity_dataset"))
    parser.add_argument("--videos", type=Path, default=Path("ml/videos"))
    parser.add_argument("--config", type=Path, default=Path("ml/lie_videos_config.json"))
    parser.add_argument("--output", type=Path, default=Path("ml/lie_motion_ablation"))
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--lovo", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    base_names = list(manifest["flat_feature_names"])
    history_names = list(manifest["history_feature_names"])
    train_cache = args.output / "train_augmented.csv"
    val_cache = args.output / "val_augmented.csv"
    if args.rebuild_cache or not (train_cache.is_file() and val_cache.is_file()):
        train_queries, val_queries = _load_queries(args.dataset, manifest)
        specs = _video_specs(args.config)
        train_augmented = augment_queries(
            train_queries, videos_dir=args.videos, specs=specs, history_feature_names=history_names
        )
        val_augmented = augment_queries(
            val_queries, videos_dir=args.videos, specs=specs, history_feature_names=history_names
        )
        write_cache(train_augmented, train_cache, base_names)
        write_cache(val_augmented, val_cache, base_names)
    report = run_ablation(
        args.dataset,
        train_cache,
        val_cache,
        base_names,
        seed=args.seed,
        lovo=False,
        model_dir=args.output / "models",
    )
    report["selection"] = run_ablation(
        args.dataset,
        train_cache,
        val_cache,
        base_names,
        seed=args.seed,
        lovo=False,
        stages=SELECTION_STAGES,
        model_dir=args.output / "models",
    )["stages"]
    if args.lovo:
        report["lovo_selection"] = run_ablation(
            args.dataset,
            train_cache,
            val_cache,
            base_names,
            seed=args.seed,
            lovo=True,
            stages=LOVO_STAGES,
        )["stages"]
    report["base_features"] = base_names
    report["feature_groups"] = {
        "similarity": list(SIMILARITY_FEATURES),
        "graph": list(GRAPH_FEATURES),
        "optical_flow": list(FLOW_FEATURES),
        "multilag_rotation": list(MULTILAG_FEATURES),
    }
    output = args.output / "ablation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
