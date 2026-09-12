"""Features and lightweight inference for identity switch-event gating."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import numpy as np

from src.engine.LieIdentityRanker import LightGbmTextModel


SWITCH_TRACK_FEATURE_NAMES = (
    "radius_norm",
    "speed_norm",
    "translation_residual_norm",
    "rotation_score_norm",
    "direction_score_norm",
    "speed_score_norm",
    "rigidity_score_norm",
    "motion_reliability",
    "visible_streak_norm",
    "lost_frames_norm",
    "predicted_only",
    "appearance_distance",
    "association_quality",
    "yolo_confidence",
    "bg_certified",
    "position_uncertainty_norm",
    "looks_background",
)

SWITCH_EVENT_BASE_FEATURE_NAMES = (
    "ranker_switch_delta",
    "ranker_top_margin",
    "ranker_current_score",
    "ranker_current_rank_norm",
    "candidate_count_norm",
    "ranker_disagreement_streak_norm",
    "ranker_evidence_before",
    "ranker_evidence_after",
    "path_support",
    "motion_corroborated",
    "recovery",
    "short_coast",
    "current_suspicious",
)

SWITCH_EVENT_FEATURE_NAMES = (
    *SWITCH_EVENT_BASE_FEATURE_NAMES,
    *(
        f"{side}_{name}"
        for side in ("current", "challenger")
        for name in SWITCH_TRACK_FEATURE_NAMES
    ),
    *(f"delta_{name}" for name in SWITCH_TRACK_FEATURE_NAMES),
)


def build_switch_event_features(event: Mapping[str, object]) -> dict[str, float]:
    """Convert a pre-commit tracker snapshot into model input features."""

    def number(name: str, default: float = 0.0) -> float:
        value = event.get(name, default)
        if value is None:
            return default
        return float(value)

    features = {
        "ranker_switch_delta": float(
            np.clip(number("ranker_switch_delta"), -12.0, 12.0)
        ),
        "ranker_top_margin": float(
            np.clip(number("ranker_top_margin"), 0.0, 12.0)
        ),
        "ranker_current_score": float(
            np.clip(number("ranker_current_score"), -12.0, 12.0)
        ),
        "ranker_current_rank_norm": float(
            np.clip(number("ranker_current_rank") / 5.0, 0.0, 1.0)
        ),
        "candidate_count_norm": float(
            np.clip(number("candidate_count") / 20.0, 0.0, 1.0)
        ),
        "ranker_disagreement_streak_norm": float(
            np.clip(number("ranker_disagreement_streak") / 15.0, 0.0, 1.0)
        ),
        "ranker_evidence_before": float(
            np.clip(number("ranker_evidence_before"), 0.0, 8.0)
        ),
        "ranker_evidence_after": float(
            np.clip(number("ranker_evidence_after"), 0.0, 8.0)
        ),
        "path_support": float(np.clip(number("path_support"), 0.0, 1.0)),
        "motion_corroborated": number("motion_corroborated"),
        "recovery": number("recovery"),
        "short_coast": number("short_coast"),
        "current_suspicious": number("current_suspicious"),
    }
    for name in SWITCH_TRACK_FEATURE_NAMES:
        current = number(f"current_{name}")
        challenger = number(f"challenger_{name}")
        features[f"current_{name}"] = current
        features[f"challenger_{name}"] = challenger
        features[f"delta_{name}"] = challenger - current
    return features


class SwitchEventModel:
    """Binary switch gate backed by the portable LightGBM text reader."""

    def __init__(self, model_path: str | Path, *, min_probability: float = 0.5):
        self.model = LightGbmTextModel.load(model_path)
        missing = set(self.model.feature_names) - set(SWITCH_EVENT_FEATURE_NAMES)
        if missing:
            raise ValueError(
                f"switch-event model uses unavailable features: {sorted(missing)}"
            )
        self.min_probability = float(np.clip(min_probability, 0.0, 1.0))

    def predict_raw(self, event: Mapping[str, object]) -> float:
        features = build_switch_event_features(event)
        return self.model.predict([features[name] for name in self.model.feature_names])

    def predict_probability(self, event: Mapping[str, object]) -> float:
        raw = self.predict_raw(event)
        if raw >= 0.0:
            return 1.0 / (1.0 + math.exp(-raw))
        exp_raw = math.exp(raw)
        return exp_raw / (1.0 + exp_raw)

    def approves(self, event: Mapping[str, object]) -> bool:
        return self.predict_probability(event) >= self.min_probability
