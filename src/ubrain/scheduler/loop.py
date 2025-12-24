from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ubrain.decision.policy import DecisionPolicy
from ubrain.diffusion.model import DiffusionModel
from ubrain.types import Challenge, Decision, EpisodeResult, LoopSignals, LoopState


@dataclass
class CognitiveLoop:
    """Explicit cognitive loop orchestrating diffusion steps and decisions."""

    model: DiffusionModel
    policy: DecisionPolicy
    max_steps: int = 16
    budget_per_step: float = 1.0

    def run_episode(self, challenge: Challenge) -> EpisodeResult:
        noisy_state = self.model.init_noisy(challenge)
        best_candidate: Optional[str] = None
        signals = LoopSignals(
            confidence=0.0,
            entropy=1.0,
            stability=0.0,
            satisfaction=0.0,
            budget=0.0,
        )

        decision: Decision = "continue"
        candidate = None

        for step in range(self.max_steps):
            denoised_state, _ = self.model.step(noisy_state, challenge, step)
            candidate = self._extract_candidate(denoised_state, challenge)

            signals = self._estimate_signals(candidate, best_candidate, signals)
            budget_used = (step + 1) * self.budget_per_step
            signals.budget = -budget_used

            if best_candidate is None or self._is_better(candidate, best_candidate, signals):
                best_candidate = candidate

            loop_state = LoopState(
                step=step,
                candidate=candidate,
                best_candidate=best_candidate,
                signals=signals,
            )
            decision = self.policy.choose(loop_state)

            noisy_state = denoised_state  # feed next step
            if decision in ("conclude", "idk"):
                break

        final_candidate = best_candidate if best_candidate is not None else candidate
        final_confidence = signals.confidence
        budget_used = abs(signals.budget)

        return EpisodeResult(
            decision=decision,
            candidate=final_candidate,
            confidence=final_confidence,
            budget_used=budget_used,
        )

    def _extract_candidate(self, denoised_state, challenge: Challenge):
        """Map model state to task-specific candidate. To be implemented."""
        raise NotImplementedError

    def _estimate_signals(
        self,
        candidate,
        best_candidate,
        prev_signals: LoopSignals,
    ) -> LoopSignals:
        """Estimate confidence/entropy/stability/satisfaction without ground truth.

        Replace with model-specific instrumentation; this placeholder keeps the API explicit.
        """
        raise NotImplementedError

    def _is_better(self, candidate, best_candidate, signals: LoopSignals) -> bool:
        """Decide if the new candidate is an improvement. Placeholder for scoring logic."""
        raise NotImplementedError
