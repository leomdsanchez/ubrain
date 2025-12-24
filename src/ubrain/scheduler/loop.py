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
        """Map model state to task-specific candidate.

        Stub logic for tokens:
        - For numeric sequences, extrapolates using last difference.
        - For hex strings, increments in hex.
        - For single-letter sequences, extrapolates using last difference.
        - For generic string cycles, repeats the observed pattern.
        - Fallback: repeat last token.
        """
        seq = denoised_state
        if not isinstance(seq, list):
            return seq

        tokens = [t for t in seq if t != "?"]
        if not tokens:
            return "?"

        last = tokens[-1]
        # Numeric difference
        if len(tokens) >= 2 and all(isinstance(t, (int, float)) for t in tokens[-2:]):
            diff = tokens[-1] - tokens[-2]
            return tokens[-1] + diff

        # Hex increment
        if len(tokens) >= 2 and all(isinstance(t, str) and t.startswith("0x") for t in tokens[-2:]):
            v1, v2 = int(tokens[-1], 16), int(tokens[-2], 16)
            diff = v1 - v2
            return hex(v1 + diff)

        # Single-letter increment
        if len(tokens) >= 2 and all(isinstance(t, str) and len(t) == 1 for t in tokens[-2:]):
            diff = ord(tokens[-1]) - ord(tokens[-2])
            return chr(ord(tokens[-1]) + diff)

        # Cycle pattern for strings
        if all(isinstance(t, str) for t in tokens):
            pattern = []
            for t in tokens:
                if t not in pattern:
                    pattern.append(t)
            idx = len(tokens) % len(pattern)
            return pattern[idx % len(pattern)]

        return last

    def _estimate_signals(
        self,
        candidate,
        best_candidate,
        prev_signals: LoopSignals,
    ) -> LoopSignals:
        """Estimate confidence/entropy/stability/satisfaction without ground truth."""
        entropy = max(0.0, prev_signals.entropy - 0.1)
        stability = 1.0 if best_candidate is not None and candidate == best_candidate else 0.0
        confidence = min(1.0, max(0.0, 0.5 * (1 - entropy) + 0.5 * stability))
        progress = prev_signals.entropy - entropy
        satisfaction = prev_signals.satisfaction + progress + (0.2 if stability else -0.05)

        return LoopSignals(
            confidence=confidence,
            entropy=entropy,
            stability=stability,
            satisfaction=satisfaction,
            budget=prev_signals.budget,
        )

    def _is_better(self, candidate, best_candidate, signals: LoopSignals) -> bool:
        """Decide if the new candidate is an improvement."""
        if best_candidate is None:
            return True
        if candidate == best_candidate:
            return False
        # Prefer candidates with non-zero stability/confidence.
        return signals.confidence >= 0.5
