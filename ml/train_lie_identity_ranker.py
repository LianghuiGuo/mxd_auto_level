#!/usr/bin/env python3
"""Train a LightGBM LambdaMART REAL-track ranker from the identity dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_split(path: Path, feature_names: list[str]):
    import csv

    queries: list[str] = []
    videos: list[str] = []
    groups: list[int] = []
    labels: list[int] = []
    rows: list[list[float]] = []
    last_query = None
    last_video = None
    group_size = 0
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for item in reader:
            query_id = item["query_id"]
            if last_query is None:
                last_query = query_id
                last_video = item["video"]
            elif query_id != last_query:
                queries.append(last_query)
                videos.append(last_video or "")
                groups.append(group_size)
                last_query = query_id
                last_video = item["video"]
                group_size = 0
            labels.append(int(item["label"]))
            rows.append([float(item[name]) for name in feature_names])
            group_size += 1
    if last_query is not None:
        queries.append(last_query)
        videos.append(last_video or "")
        groups.append(group_size)
    return {
        "x": np.asarray(rows, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.int32),
        "groups": np.asarray(groups, dtype=np.int32),
        "queries": queries,
        "videos": np.asarray(videos),
    }


def _concat_splits(left: dict, right: dict) -> dict:
    if left["x"].size == 0:
        return right
    if right["x"].size == 0:
        return left
    return {
        "x": np.concatenate([left["x"], right["x"]], axis=0),
        "y": np.concatenate([left["y"], right["y"]], axis=0),
        "groups": np.concatenate([left["groups"], right["groups"]], axis=0),
        "queries": [*left["queries"], *right["queries"]],
        "videos": np.concatenate([left["videos"], right["videos"]], axis=0),
    }


def _select_queries(split: dict, mask: np.ndarray) -> dict:
    row_mask = np.repeat(mask, split["groups"])
    return {
        "x": split["x"][row_mask],
        "y": split["y"][row_mask],
        "groups": split["groups"][mask],
        "queries": [query for query, keep in zip(split["queries"], mask) if keep],
        "videos": split["videos"][mask],
    }


def _top1_hits(preds: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    offset = 0
    hits = []
    for size in groups:
        chunk = preds[offset : offset + int(size)]
        truth = labels[offset : offset + int(size)]
        hits.append(int(truth[int(np.argmax(chunk))] == 1))
        offset += int(size)
    return np.asarray(hits, dtype=np.int32)


def _top1_accuracy(preds: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    hits = _top1_hits(preds, labels, groups)
    return float(hits.mean()) if hits.size else 0.0


def _current_target_preds(features: np.ndarray, feature_index: int, groups: np.ndarray) -> np.ndarray:
    scores = features[:, feature_index].copy()
    offset = 0
    for size in groups:
        start = offset
        end = offset + int(size)
        if not np.any(scores[start:end] > 0.5):
            scores[start:end] = 0.0
            scores[start] = 1.0
        offset = end
    return scores


def _hard_query_mask(labels: np.ndarray, current_target: np.ndarray, groups: np.ndarray) -> np.ndarray:
    offset = 0
    mask = []
    for size in groups:
        truth = labels[offset : offset + int(size)]
        current = current_target[offset : offset + int(size)]
        positive = int(np.argmax(truth))
        mask.append(int(current[positive] < 0.5))
        offset += int(size)
    return np.asarray(mask, dtype=bool)


def _lgb_params(args: argparse.Namespace) -> dict:
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1],
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": args.seed,
    }


def _train_booster(lgb, args: argparse.Namespace, train: dict, val: dict, feature_names: list[str]):
    train_set = lgb.Dataset(
        train["x"],
        label=train["y"],
        group=train["groups"],
        feature_name=feature_names,
        free_raw_data=False,
    )
    callbacks = [lgb.log_evaluation(0)]
    valid_sets = [train_set]
    valid_names = ["train"]
    if val["groups"].size:
        val_set = lgb.Dataset(
            val["x"],
            label=val["y"],
            group=val["groups"],
            feature_name=feature_names,
            reference=train_set,
            free_raw_data=False,
        )
        valid_sets.append(val_set)
        valid_names.append("val")
        callbacks.insert(0, lgb.early_stopping(args.early_stopping, verbose=False))
    return lgb.train(
        _lgb_params(args),
        train_set,
        num_boost_round=args.n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )


def _score_split(preds: np.ndarray, split: dict, current_index: int) -> dict:
    hits = _top1_hits(preds, split["y"], split["groups"])
    current_preds = _current_target_preds(split["x"], current_index, split["groups"])
    current_hits = _top1_hits(current_preds, split["y"], split["groups"])
    hard_mask = _hard_query_mask(
        split["y"],
        split["x"][:, current_index],
        split["groups"],
    )
    hard_total = int(hard_mask.sum())
    return {
        "queries": int(split["groups"].size),
        "hits": int(hits.sum()),
        "top1": float(hits.mean()) if hits.size else 0.0,
        "current_target_top1": (
            float(current_hits.mean()) if current_hits.size else 0.0
        ),
        "hard_queries": hard_total,
        "hard_hits": int(hits[hard_mask].sum()) if hard_total else 0,
        "hard_top1": (
            float(hits[hard_mask].mean()) if hard_total else None
        ),
        "hard_current_target_top1": (
            float(current_hits[hard_mask].mean()) if hard_total else None
        ),
    }


def _run_lovo(lgb, args: argparse.Namespace, pooled: dict, feature_names: list[str]) -> dict:
    current_index = feature_names.index("is_current_target")
    videos = sorted(set(pooled["videos"].tolist()))
    folds = []
    total_hits = 0
    total_queries = 0
    total_hard_hits = 0
    total_hard = 0
    for video in videos:
        val_mask = pooled["videos"] == video
        train_mask = ~val_mask
        train = _select_queries(pooled, train_mask)
        val = _select_queries(pooled, val_mask)
        if train["groups"].size == 0 or val["groups"].size == 0:
            continue
        booster = _train_booster(lgb, args, train, val, feature_names)
        preds = booster.predict(val["x"], num_iteration=booster.best_iteration)
        metrics = _score_split(preds, val, current_index)
        metrics["video"] = video
        metrics["best_iteration"] = booster.best_iteration
        folds.append(metrics)
        total_hits += metrics["hits"]
        total_queries += metrics["queries"]
        total_hard_hits += metrics["hard_hits"]
        total_hard += metrics["hard_queries"]
        print(
            f"{video}: queries={metrics['queries']} "
            f"top1={metrics['top1']:.3f} "
            f"current={metrics['current_target_top1']:.3f} "
            f"hard={metrics['hard_queries']}"
            + (
                f" hard_top1={metrics['hard_top1']:.3f}"
                if metrics["hard_top1"] is not None
                else ""
            ),
            flush=True,
        )
    return {
        "folds": folds,
        "videos": len(folds),
        "queries": total_queries,
        "top1": total_hits / max(1, total_queries),
        "hard_queries": total_hard,
        "hard_top1": (
            total_hard_hits / total_hard if total_hard else None
        ),
        "mean_video_top1": (
            float(np.mean([fold["top1"] for fold in folds])) if folds else 0.0
        ),
        "min_video_top1": (
            float(min(fold["top1"] for fold in folds)) if folds else 0.0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("ml/lie_identity_dataset"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/lie_identity_ranker.txt"),
    )
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-data-in-leaf", type=int, default=40)
    parser.add_argument("--early-stopping", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument(
        "--lovo",
        action="store_true",
        help="run leave-one-video-out after the default split train",
    )
    parser.add_argument(
        "--lovo-only",
        action="store_true",
        help="skip default split training and only run leave-one-video-out",
    )
    return parser.parse_args()


def main() -> int:
    try:
        import lightgbm as lgb
    except ImportError:
        print("Install LightGBM first: python3 -m pip install lightgbm")
        return 1

    args = parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    feature_names = list(manifest["flat_feature_names"])
    train = _load_split(args.data / "train.csv", feature_names)
    val = _load_split(args.data / "val.csv", feature_names)
    current_index = feature_names.index("is_current_target")
    metrics: dict[str, object] = {}

    if not args.lovo_only:
        booster = _train_booster(lgb, args, train, val, feature_names)
        train_preds = booster.predict(train["x"], num_iteration=booster.best_iteration)
        val_preds = booster.predict(val["x"], num_iteration=booster.best_iteration)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(args.out))
        metrics.update(
            {
                "model": str(args.out),
                "best_iteration": booster.best_iteration,
                "train": _score_split(train_preds, train, current_index),
                "val": _score_split(val_preds, val, current_index),
                "feature_importance": dict(
                    zip(
                        feature_names,
                        [int(value) for value in booster.feature_importance()],
                    )
                ),
            }
        )

    if args.lovo or args.lovo_only:
        pooled = _concat_splits(train, val)
        metrics["lovo"] = _run_lovo(lgb, args, pooled, feature_names)

    metrics_path = args.out.with_suffix(".metrics.json")
    if args.lovo_only:
        metrics_path = args.out.with_suffix(".lovo.json")
    elif args.lovo:
        lovo_path = args.out.with_suffix(".lovo.json")
        lovo_path.write_text(
            json.dumps(metrics.get("lovo", {}), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {lovo_path}")
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if not args.lovo_only:
        print(f"wrote {args.out}")
    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
