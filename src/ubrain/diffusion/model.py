from __future__ import annotations

from typing import Any, Protocol, Tuple

from ubrain.types import Challenge


class DiffusionModel(Protocol):
    """Abstract interface for discrete diffusion backbone."""

    def step(self, noisy_state: Any, challenge: Challenge, t: int) -> Tuple[Any, Any]:
        """Perform one refinement step.

        Returns a tuple (denoised_state, auxiliary_logits_or_score).
        """
        ...

    def init_noisy(self, challenge: Challenge) -> Any:
        """Create initial noisy latent/tokens for a challenge."""
        ...
