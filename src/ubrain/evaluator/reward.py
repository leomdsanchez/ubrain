from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ubrain.types import EpisodeResult


@dataclass
class RewardWeights:
    """Weights for computing reward."""

    base_correct: float = 1.0
    cost_scale: float = 1.0
    idk_penalty: float = 0.2
    wrong_confident_penalty: float = 2.0


def compute_reward(
    result: EpisodeResult,
    weights: RewardWeights,
    correct: Optional[bool],
) -> float:
    """Compute final reward using fixed base quality minus cost.

    External reward is simple; most shaping comes from internal satisfaction.
    """
    quality = 0.0
    if correct is True:
        quality = weights.base_correct
    elif correct is False:
        quality = -weights.wrong_confident_penalty
    else:
        quality = 0.0

    if result.decision == "idk":
        quality -= weights.idk_penalty

    cost = weights.cost_scale * result.budget_used
    return quality - cost
