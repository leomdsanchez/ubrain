from __future__ import annotations

from dataclasses import dataclass
import math
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
        dist = self.model.init_noisy(challenge)
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
        prev_dist = None

        for step in range(self.max_steps):
            dist, logits = self.model.step(dist, challenge, step)
            candidate = self._extract_candidate(dist, challenge)

            signals = self._estimate_signals(
                candidate=candidate,
                best_candidate=best_candidate,
                prev_signals=signals,
                dist=dist,
                prev_dist=prev_dist,
            )
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

            prev_dist = dist
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

    def _extract_candidate(self, dist, challenge: Challenge):
        """Decode candidato a partir da distribuicao e do output_format."""
        if not isinstance(dist, dict):
            return dist
        if not dist:
            return "?"
        # argmax
        token = max(dist.items(), key=lambda kv: kv[1])[0]

        fmt = challenge.output_format
        if fmt == "boolean":
            if isinstance(token, bool):
                return token
            if isinstance(token, str):
                low = token.lower()
                if low in ("true", "t", "1"):
                    return True
                if low in ("false", "f", "0"):
                    return False
        # default token
        return token

    def _estimate_signals(
        self,
        candidate,
        best_candidate,
        prev_signals: LoopSignals,
        dist,
        prev_dist,
    ) -> LoopSignals:
        """Estimate confidence/entropy/stability/satisfaction using distributions."""
        eps = 1e-8
        probs = dist if isinstance(dist, dict) else {}
        entropy = 0.0
        for p in probs.values():
            entropy -= p * math.log(max(p, eps))
        max_entropy = math.log(max(len(probs), 1))

        if prev_dist is None:
            stability = 0.0
        else:
            stability = 0.0
            for token, p in probs.items():
                q = prev_dist.get(token, eps)
                stability += p * math.log(max(p, eps) / max(q, eps))
            stability = max(0.0, 1.0 - stability)

        confidence = 0.6 * (1 - entropy / max_entropy) + 0.4 * stability if max_entropy > 0 else 0.0

        prev_entropy = prev_signals.entropy
        delta_entropy = prev_entropy - entropy
        delta_stability = stability - prev_signals.stability
        satisfaction = prev_signals.satisfaction + delta_entropy + 0.5 * delta_stability
        if abs(delta_entropy) < 1e-3 and abs(delta_stability) < 1e-3:
            satisfaction -= 0.05  # penaliza estagnacao leve

        return LoopSignals(
            confidence=max(0.0, min(1.0, confidence)),
            entropy=max(entropy, 0.0),
            stability=max(0.0, min(1.0, stability)),
            satisfaction=satisfaction,
            budget=prev_signals.budget,
        )

    def _is_better(self, candidate, best_candidate, signals: LoopSignals) -> bool:
        """Decide se o novo candidato melhora o best."""
        if best_candidate is None:
            return True
        if candidate == best_candidate:
            return False
        return signals.confidence > 0.5 and signals.stability >= 0.3
