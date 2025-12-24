from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ubrain.types import EpisodeResult


@dataclass
class RewardWeights:
    """Weights for computing reward."""

    difficulty: float = 1.0
    cost_scale: float = 1.0
    idk_penalty: float = 0.2
    wrong_confident_penalty: float = 2.0


def compute_reward(
    result: EpisodeResult,
    difficulty: float,
    weights: RewardWeights,
    correct: Optional[bool],
) -> float:
    """Compute final reward using quality adjusted by difficulty minus cost."""
    quality = 0.0
    if correct is True:
        quality = result.confidence * weights.difficulty
    elif correct is False:
        quality = -weights.wrong_confident_penalty * result.confidence
    else:
        quality = 0.0

    if result.decision == "idk":
        quality -= weights.idk_penalty

    cost = weights.cost_scale * result.budget_used
    return quality * difficulty - cost
