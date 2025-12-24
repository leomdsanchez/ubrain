from __future__ import annotations

import math
from typing import Any, Dict, Tuple

from ubrain.diffusion.model import DiffusionModel
from ubrain.types import Challenge


class DiffusionStub(DiffusionModel):
    """Stub de difusao discreta: aplica ruido inicial e reduz entropia sem ver o gabarito."""

    def __init__(self, use_ground_truth: bool = False):
        self.use_ground_truth = use_ground_truth
        self._targets: Dict[str, Any] = {}

    def init_noisy(self, challenge: Challenge) -> Any:
        tokens = self._vocab(challenge)
        prob = 1.0 / len(tokens)
        dist = {t: prob for t in tokens}
        target = self._infer_target(challenge)
        self._targets[challenge.id] = target
        return dist

    def step(self, noisy_state: Any, challenge: Challenge, t: int) -> Tuple[Any, Any]:
        dist: Dict[Any, float] = noisy_state
        target = self._targets.get(challenge.id)
        weight = min(0.2 + 0.15 * t, 0.9)

        new_dist: Dict[Any, float] = {}
        for token, p in dist.items():
            new_dist[token] = (1.0 - weight) * p
        if target is not None:
            new_dist[target] = new_dist.get(target, 0.0) + weight

        # renormaliza
        total = sum(new_dist.values()) or 1.0
        for k in new_dist:
            new_dist[k] = new_dist[k] / total

        logits = {k: math.log(v + 1e-8) for k, v in new_dist.items()}
        return new_dist, logits

    def _vocab(self, challenge: Challenge):
        tokens = []
        if isinstance(challenge.prompt, list):
            tokens.extend([t for t in challenge.prompt if t != "?"])
        if self.use_ground_truth and challenge.ground_truth is not None:
            tokens.append(challenge.ground_truth)
        if not tokens:
            tokens.append("?")
        # garantir unicidade preservando ordem
        seen = set()
        uniq = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq

    def _infer_target(self, challenge: Challenge):
        if self.use_ground_truth and challenge.ground_truth is not None:
            return challenge.ground_truth
        prompt = challenge.prompt
        if isinstance(prompt, list):
            tokens = [t for t in prompt if t != "?"]
            if len(tokens) >= 2 and all(isinstance(t, (int, float)) for t in tokens[-2:]):
                diff = tokens[-1] - tokens[-2]
                return tokens[-1] + diff
            if len(tokens) >= 2 and all(isinstance(t, str) and len(t) == 1 for t in tokens[-2:]):
                diff = ord(tokens[-1]) - ord(tokens[-2])
                return chr(ord(tokens[-1]) + diff)
            if tokens:
                return tokens[-1]
        return None
