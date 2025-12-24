from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from ubrain.types import Challenge


@dataclass
class ChallengeBatch:
    """Batch container for multiple challenges."""

    items: List[Challenge]

    def __iter__(self) -> Iterable[Challenge]:
        return iter(self.items)
