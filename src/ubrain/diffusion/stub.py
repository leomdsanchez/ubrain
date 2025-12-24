from __future__ import annotations

from typing import Any, Tuple

from ubrain.diffusion.model import DiffusionModel
from ubrain.types import Challenge


class EchoSequenceModel(DiffusionModel):
    """Stub model that echoes the prompt and proposes a next token.

    This is not a diffusion model; it exists to exercise the cognitive loop.
    It uses a simple heuristic: for a prompt ending with '?', it attempts to
    predict the next token by repeating the last non-'?' token or by following
    a numeric/letter increment. It also emits a dummy score (None).
    """

    def init_noisy(self, challenge: Challenge) -> Any:
        # Noisy state is just the prompt for the stub.
        return challenge.prompt

    def step(self, noisy_state: Any, challenge: Challenge, t: int) -> Tuple[Any, Any]:
        # For the stub, we keep the state unchanged and propose a candidate.
        return noisy_state, None
