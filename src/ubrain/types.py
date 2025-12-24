from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional


Decision = Literal["continue", "conclude", "idk"]


@dataclass
class Challenge:
    """Canonical challenge definition."""

    id: str
    prompt: Any  # tokenized input; can be text/earcon/icon tokens
    output_format: str  # closed format description (options/indices/patches)
    reward: float = 1.0  # base reward for correct
    ground_truth: Optional[Any] = None  # used only in train/eval


@dataclass
class LoopSignals:
    """Signals measured at a given step."""

    confidence: float
    entropy: float
    stability: float
    satisfaction: float
    budget: float


@dataclass
class LoopState:
    """State tracked across the cognitive loop."""

    step: int
    candidate: Any
    best_candidate: Any
    signals: LoopSignals


@dataclass
class EpisodeResult:
    """Final outcome of an episode."""

    decision: Decision
    candidate: Any
    confidence: float
    budget_used: float
    reward: Optional[float] = None
    correct: Optional[bool] = None
