from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ubrain.types import Decision, LoopSignals, LoopState


class DecisionPolicy(Protocol):
    """Policy that selects the next action given current signals/state."""

    def choose(self, state: LoopState) -> Decision:
        ...


@dataclass
class HeuristicPolicy:
    """Bootstrap heuristic; replace with learned policy after training.

    Keeps running while satisfaction is rising and budget is acceptable,
    stops when confidence is high and stability is present, yields IDK
    when satisfaction is dropping and confidence is low.
    """

    min_confidence: float = 0.6
    min_stability: float = 0.5
    min_satisfaction_delta: float = 0.0

    def choose(self, state: LoopState) -> Decision:
        signals: LoopSignals = state.signals
        if signals.confidence >= self.min_confidence and signals.stability >= self.min_stability:
            return "conclude"

        if signals.satisfaction < self.min_satisfaction_delta and signals.confidence < self.min_confidence:
            return "idk"

        return "continue"
