from __future__ import annotations

import math
from typing import Any, Dict, Tuple

from ubrain.diffusion.model import DiffusionModel
from ubrain.types import Challenge


class DiffusionStub(DiffusionModel):
    """Stub de difusao discreta: adiciona ruido inicial e reduz entropia por passo.

    Usa o ground_truth (quando fornecido) como "alvo" para concentrar probabilidade
    ao longo dos passos. Se nao houver ground_truth, distribui de forma quase uniforme.
    """

    def init_noisy(self, challenge: Challenge) -> Any:
        # Estado ruidoso: prompt com tokens observados; distribuicao inicial uniforme.
        tokens = self._vocab(challenge)
        prob = 1.0 / len(tokens)
        dist = {t: prob for t in tokens}
        return dist

    def step(self, noisy_state: Any, challenge: Challenge, t: int) -> Tuple[Any, Any]:
        # Ajusta pesos para concentrar probabilidade no ground_truth (se existir).
        dist: Dict[Any, float] = noisy_state
        tokens = list(dist.keys())
        gt = challenge.ground_truth

        logits = {token: 0.0 for token in tokens}
        base = 1.0
        gain = 1.0 + t * 0.5

        for token in tokens:
            if gt is not None and token == gt:
                logits[token] = base + gain
            else:
                logits[token] = base

        # softmax manual
        max_logit = max(logits.values())
        exp_sum = sum(math.exp(v - max_logit) for v in logits.values())
        new_dist = {k: math.exp(v - max_logit) / exp_sum for k, v in logits.items()}

        return new_dist, logits

    def _vocab(self, challenge: Challenge):
        tokens = []
        if isinstance(challenge.prompt, list):
            tokens.extend([t for t in challenge.prompt if t != "?"])
        if challenge.ground_truth is not None:
            tokens.append(challenge.ground_truth)
        # fallback para evitar vocab vazio
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
