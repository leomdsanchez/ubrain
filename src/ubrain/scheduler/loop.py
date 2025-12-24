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

            if best_candidate is None or self._is_better(candidate, best_candidate, signals, dist):
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
        norm_entropy = (entropy / max_entropy) if max_entropy > 0 else 0.0

        if prev_dist is None:
            stability = 0.0
        else:
            # estabilidade via distancia total variacao (1 - TV)
            all_tokens = set(probs.keys()) | set(prev_dist.keys())
            tv = 0.0
            for tok in all_tokens:
                p = probs.get(tok, 0.0)
                q = prev_dist.get(tok, 0.0)
                tv += abs(p - q)
            tv *= 0.5
            stability = max(0.0, 1.0 - tv)

        confidence = 0.6 * (1 - norm_entropy) + 0.4 * stability

        prev_entropy = prev_signals.entropy
        delta_entropy = prev_entropy - norm_entropy
        delta_stability = stability - prev_signals.stability
        progress = delta_entropy + 0.5 * delta_stability
        satisfaction = prev_signals.satisfaction + progress
        if abs(progress) < 1e-3:
            satisfaction -= 0.05  # penaliza estagnacao leve

        return LoopSignals(
            confidence=max(0.0, min(1.0, confidence)),
            entropy=max(norm_entropy, 0.0),
            stability=max(0.0, min(1.0, stability)),
            satisfaction=satisfaction,
            budget=prev_signals.budget,
        )

    def _is_better(self, candidate, best_candidate, signals: LoopSignals, dist) -> bool:
        """Decide se o novo candidato melhora o best."""
        if best_candidate is None:
            return True
        if candidate == best_candidate:
            return False
        prob_new = dist.get(candidate, 0.0) if isinstance(dist, dict) else 0.0
        prob_best = dist.get(best_candidate, 0.0) if isinstance(dist, dict) else 0.0
        return (prob_new > prob_best + 1e-6) and signals.confidence > 0.5 and signals.stability >= 0.2
