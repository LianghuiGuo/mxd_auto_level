#!/usr/bin/env python3
"""Train a switch/keep gate from state-aware on-policy replay events."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.LieSwitchEventModel import (  # noqa: E402
    SWITCH_EVENT_BASE_FEATURE_NAMES,
    SWITCH_EVENT_FEATURE_NAMES,
    SWITCH_TRACK_BASE_FEATURE_NAMES,
    SWITCH_TRACK_PREFLOW_FEATURE_NAMES,
    SWITCH_TRACK_ROTATION_FEATURE_NAMES,
    build_switch_event_features,
)
from src.engine.LieIdentityRanker import (  # noqa: E402
    FLOW_FEATURE_NAMES,
    MULTILAG_FEATURE_NAMES,
)


def _event_feature_names(track_features: tuple[str, ...]) -> tuple[str, ...]:
    return (
        *SWITCH_EVENT_BASE_FEATURE_NAMES,
        *(
            f"{side}_{name}"
            for side in ("current", "challenger")
            for name in track_features
        ),
        *(f"delta_{name}" for name in track_features),
    )


FEATURE_SETS = {
    "legacy": _event_feature_names(SWITCH_TRACK_BASE_FEATURE_NAMES),
    "rotation": _event_feature_names(
        (*SWITCH_TRACK_BASE_FEATURE_NAMES, *SWITCH_TRACK_ROTATION_FEATURE_NAMES)
    ),
    "preflow": _event_feature_names(
        (
            *SWITCH_TRACK_BASE_FEATURE_NAMES,
            *SWITCH_TRACK_ROTATION_FEATURE_NAMES,
            *SWITCH_TRACK_PREFLOW_FEATURE_NAMES,
        )
    ),
    "optical": _event_feature_names(
        (
            *SWITCH_TRACK_BASE_FEATURE_NAMES,
            *SWITCH_TRACK_ROTATION_FEATURE_NAMES,
            *FLOW_FEATURE_NAMES,
        )
    ),
    "optical-multilag": _event_feature_names(
        (
            *SWITCH_TRACK_BASE_FEATURE_NAMES,
            *SWITCH_TRACK_ROTATION_FEATURE_NAMES,
            *FLOW_FEATURE_NAMES,
            *MULTILAG_FEATURE_NAMES,
            "spin_estimator_agreement",
            "spin_joint_confidence",
        )
    ),
    "preflow-optical-multilag": SWITCH_EVENT_FEATURE_NAMES,
}


def _video_from_path(path: Path) -> str:
    prefix = path.stem.split("_switch_events", 1)[0]
    return prefix.replace("录屏", "测谎录屏") + ".mp4"


def _load_events(
    directory: Path,
    *,
    good_radius: float,
    bad_radius: float,
    feature_names,
    label_policy: str,
    future_min_frames: int,
    future_good_ratio: float,
    future_wrong_ratio: float,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    examples: list[dict[str, object]] = []
    categories: Counter[str] = Counter()
    for path in sorted(directory.glob("*_switch_events.jsonl")):
        fallback_video = _video_from_path(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if label_policy == "future-window":
                    window_frames = int(event.get("future_window_frames") or 0)
                    if window_frames < future_min_frames:
                        categories["future_too_short"] += 1
                        continue
                    current_good = float(event.get("future_current_good_ratio") or 0.0)
                    challenger_good = float(
                        event.get("future_challenger_good_ratio") or 0.0
                    )
                    if (
                        current_good >= future_good_ratio
                        and challenger_good <= future_wrong_ratio
                    ):
                        label = 0
                        category = "keep"
                    elif (
                        challenger_good >= future_good_ratio
                        and current_good <= future_wrong_ratio
                    ):
                        label = 1
                        category = "switch"
                    else:
                        categories["ambiguous"] += 1
                        continue
                else:
                    current_error = event.get("current_error_px")
                    challenger_error = event.get("challenger_error_px")
                    if current_error is None or challenger_error is None:
                        categories["no_ground_truth"] += 1
                        continue
                    if (
                        float(current_error) <= good_radius
                        and float(challenger_error) >= bad_radius
                    ):
                        label = 0
                        category = "keep"
                    elif (
                        float(challenger_error) <= good_radius
                        and float(current_error) >= bad_radius
                    ):
                        label = 1
                        category = "switch"
                    else:
                        categories["ambiguous"] += 1
                        continue
                categories[category] += 1
                if not bool(event.get("basic_qualifies", 0)):
                    categories[f"{category}_basic_rejected"] += 1
                    continue
                feature_map = build_switch_event_features(event)
                examples.append(
                    {
                        "video": str(event.get("video") or fallback_video),
                        "frame": int(event["frame"]),
                        "current_id": int(event["current_id"]),
                        "challenger_id": int(event["challenger_id"]),
                        "label": label,
                        "proposed": bool(event.get("proposed", 0)),
                        "x": np.asarray(
                            [feature_map[name] for name in feature_names],
                            dtype=np.float32,
                        ),
                    }
                )
    return examples, dict(categories)


def _episode_weights(examples: list[dict[str, object]]) -> np.ndarray:
    episode_ids: list[tuple[str, int]] = []
    episode = -1
    previous: dict[str, object] | None = None
    for item in examples:
        continuous = bool(
            previous is not None
            and previous["video"] == item["video"]
            and previous["current_id"] == item["current_id"]
            and previous["challenger_id"] == item["challenger_id"]
            and int(item["frame"]) - int(previous["frame"]) <= 6
        )
        if not continuous:
            episode += 1
        episode_ids.append((str(item["video"]), episode))
        previous = item
    counts = Counter(episode_ids)
    # Long consecutive runs are highly correlated. Give each run roughly one
    # vote while retaining a small amount of duration information.
    return np.asarray(
        [1.0 / math.sqrt(counts[episode_id]) for episode_id in episode_ids],
        dtype=np.float32,
    )


def _arrays(
    examples: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.stack([item["x"] for item in examples]),
        np.asarray([item["label"] for item in examples], dtype=np.int32),
        _episode_weights(examples),
    )


def _params(args: argparse.Namespace) -> dict[str, object]:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 6.0,
        "verbosity": -1,
        "seed": args.seed,
        "num_threads": args.num_threads,
    }


def _lovo_predictions(lgb, args, examples, feature_names):
    predictions = np.full(len(examples), np.nan, dtype=np.float64)
    folds = []
    for video in sorted({str(item["video"]) for item in examples}):
        train_indices = [i for i, item in enumerate(examples) if item["video"] != video]
        val_indices = [i for i, item in enumerate(examples) if item["video"] == video]
        train = [examples[i] for i in train_indices]
        val = [examples[i] for i in val_indices]
        x_train, y_train, weights = _arrays(train)
        booster = lgb.train(
            _params(args),
            lgb.Dataset(
                x_train,
                label=y_train,
                weight=weights,
                feature_name=list(feature_names),
            ),
            num_boost_round=args.n_estimators,
            callbacks=[lgb.log_evaluation(0)],
        )
        x_val, y_val, _ = _arrays(val)
        predictions[val_indices] = booster.predict(x_val)
        folds.append(
            {
                "video": video,
                "examples": len(val),
                "switch": int(y_val.sum()),
                "keep": int((y_val == 0).sum()),
                "proposed": sum(bool(item["proposed"]) for item in val),
            }
        )
    if np.isnan(predictions).any():
        raise RuntimeError("LOVO predictions are incomplete")
    return predictions, folds


def _episode_count(examples, mask):
    selected = [item for item, use in zip(examples, mask) if use]
    selected.sort(key=lambda item: (item["video"], int(item["frame"])))
    count = 0
    previous = None
    for item in selected:
        continuous = bool(
            previous is not None
            and previous["video"] == item["video"]
            and previous["current_id"] == item["current_id"]
            and previous["challenger_id"] == item["challenger_id"]
            and int(item["frame"]) - int(previous["frame"]) <= 6
        )
        if not continuous:
            count += 1
        previous = item
    return count


def _metrics(examples, predictions, threshold, *, proposed_only):
    labels = np.asarray([item["label"] for item in examples], dtype=np.int32)
    scope = np.asarray(
        [bool(item["proposed"]) or not proposed_only for item in examples],
        dtype=bool,
    )
    selected = predictions >= threshold
    tp = int(np.sum(scope & selected & (labels == 1)))
    fp = int(np.sum(scope & selected & (labels == 0)))
    fn = int(np.sum(scope & ~selected & (labels == 1)))
    tn = int(np.sum(scope & ~selected & (labels == 0)))
    per_video = []
    for video in sorted({str(item["video"]) for item in examples}):
        mask = np.asarray([item["video"] == video for item in examples]) & scope
        if not mask.any():
            continue
        video_labels = labels[mask]
        video_selected = selected[mask]
        keep = int(np.sum(video_labels == 0))
        switch = int(np.sum(video_labels == 1))
        video_fp = int(np.sum(video_selected & (video_labels == 0)))
        video_tp = int(np.sum(video_selected & (video_labels == 1)))
        per_video.append(
            {
                "video": video,
                "examples": int(mask.sum()),
                "switch": switch,
                "keep": keep,
                "true_switch": video_tp,
                "false_switch": video_fp,
                "switch_recall": video_tp / switch if switch else None,
                "keep_accuracy": 1.0 - video_fp / keep if keep else None,
            }
        )
    scoped_examples = [item for item, use in zip(examples, scope) if use]
    scoped_selected = selected[scope]
    return {
        "threshold": float(threshold),
        "scope": "proposed" if proposed_only else "all_basic",
        "examples": int(scope.sum()),
        "true_switch": tp,
        "false_switch": fp,
        "missed_switch": fn,
        "true_keep": tn,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "switch_recall": tp / (tp + fn) if tp + fn else 0.0,
        "keep_accuracy": tn / (tn + fp) if tn + fp else 1.0,
        "accepted_episodes": _episode_count(scoped_examples, scoped_selected),
        "per_video": per_video,
    }


def _select_threshold(examples, predictions, args):
    candidates = sorted(
        {
            *np.arange(0.50, 0.976, 0.025).tolist(),
            *np.quantile(predictions, np.linspace(0.50, 0.995, 60)).tolist(),
        }
    )
    sweep = []
    feasible = []
    for threshold in candidates:
        proposed = _metrics(examples, predictions, threshold, proposed_only=True)
        all_basic = _metrics(examples, predictions, threshold, proposed_only=False)
        item = {
            "threshold": float(threshold),
            "proposed_precision": proposed["precision"],
            "proposed_switch_recall": proposed["switch_recall"],
            "proposed_false_switch": proposed["false_switch"],
            "all_precision": all_basic["precision"],
            "all_switch_recall": all_basic["switch_recall"],
            "all_keep_accuracy": all_basic["keep_accuracy"],
        }
        sweep.append(item)
        if (
            float(proposed["precision"]) >= args.min_proposed_precision
            and int(proposed["false_switch"]) <= args.max_proposed_false_switch
            and float(all_basic["keep_accuracy"]) >= args.min_keep_accuracy
        ):
            feasible.append((item, proposed, all_basic))
    if not feasible:
        raise RuntimeError("no threshold meets the conservative switch constraints")
    return max(
        feasible,
        key=lambda value: (
            float(value[1]["switch_recall"]),
            float(value[2]["switch_recall"]),
            float(value[1]["precision"]),
        ),
    ), sweep


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("log/lie_switch_event_20260912/raw"))
    parser.add_argument("--out", type=Path, default=Path("models/lie_switch_event_model.txt"))
    parser.add_argument("--good-radius", type=float, default=40.0)
    parser.add_argument("--bad-radius", type=float, default=60.0)
    parser.add_argument("--n-estimators", type=int, default=70)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--num-leaves", type=int, default=7)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-data-in-leaf", type=int, default=24)
    parser.add_argument("--min-proposed-precision", type=float, default=0.90)
    parser.add_argument("--max-proposed-false-switch", type=int, default=2)
    parser.add_argument("--min-keep-accuracy", type=float, default=0.90)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument(
        "--label-policy",
        choices=("instant", "future-window"),
        default="instant",
    )
    parser.add_argument("--future-min-frames", type=int, default=5)
    parser.add_argument("--future-good-ratio", type=float, default=0.60)
    parser.add_argument("--future-wrong-ratio", type=float, default=0.25)
    parser.add_argument(
        "--feature-set",
        choices=tuple(FEATURE_SETS),
        default="optical-multilag",
        help="cumulative switch-event feature ablation stage",
    )
    return parser.parse_args()


def main() -> int:
    try:
        import lightgbm as lgb
    except (ImportError, OSError) as error:
        print(f"LightGBM unavailable: {error}")
        return 1
    args = parse_args()
    feature_names = FEATURE_SETS[args.feature_set]
    examples, categories = _load_events(
        args.events,
        good_radius=args.good_radius,
        bad_radius=args.bad_radius,
        feature_names=feature_names,
        label_policy=args.label_policy,
        future_min_frames=args.future_min_frames,
        future_good_ratio=args.future_good_ratio,
        future_wrong_ratio=args.future_wrong_ratio,
    )
    examples.sort(key=lambda item: (item["video"], int(item["frame"])))
    if not examples or len({item["label"] for item in examples}) != 2:
        raise RuntimeError("event dataset needs both switch and keep examples")
    predictions, folds = _lovo_predictions(lgb, args, examples, feature_names)
    (selection, proposed, all_basic), sweep = _select_threshold(
        examples, predictions, args
    )
    x, y, weights = _arrays(examples)
    booster = lgb.train(
        _params(args),
        lgb.Dataset(
            x,
            label=y,
            weight=weights,
            feature_name=list(feature_names),
        ),
        num_boost_round=args.n_estimators,
        callbacks=[lgb.log_evaluation(0)],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(args.out))
    threshold = float(selection["threshold"])
    metrics = {
        "format_version": 1,
        "task": "on_policy_state_aware_switch_gate",
        "feature_set": args.feature_set,
        "feature_names": list(feature_names),
        "label_policy_name": args.label_policy,
        "model": str(args.out),
        "events": str(args.events),
        "label_policy": {
            "keep": (
                f"future current good >= {args.future_good_ratio:.2f} and "
                f"challenger good <= {args.future_wrong_ratio:.2f}"
                if args.label_policy == "future-window"
                else f"current <= {args.good_radius}px and challenger >= {args.bad_radius}px"
            ),
            "switch": (
                f"future challenger good >= {args.future_good_ratio:.2f} and "
                f"current good <= {args.future_wrong_ratio:.2f}"
                if args.label_policy == "future-window"
                else f"challenger <= {args.good_radius}px and current >= {args.bad_radius}px"
            ),
            "ambiguous": "excluded",
            "green_cursor_features": False,
        },
        "categories": categories,
        "training_examples": len(examples),
        "training_switch": int(y.sum()),
        "training_keep": int((y == 0).sum()),
        "videos": sorted({str(item["video"]) for item in examples}),
        "independent_end_to_end_videos": ["测谎录屏22.mp4", "测谎录屏23.mp4"],
        "selected_probability_threshold": threshold,
        "selected_raw_score_threshold": math.log(threshold / (1.0 - threshold)),
        "selection": selection,
        "lovo_proposed": proposed,
        "lovo_all_basic": all_basic,
        "lovo_folds": folds,
        "threshold_sweep": sweep,
        "feature_importance_gain": dict(
            zip(
                feature_names,
                [float(value) for value in booster.feature_importance(importance_type="gain")],
            )
        ),
    }
    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"selection": selection, "proposed": proposed, "all_basic": all_basic}, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
