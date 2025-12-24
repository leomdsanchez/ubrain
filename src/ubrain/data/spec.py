from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import yaml

from ubrain.types import Challenge


@dataclass
class ChallengeBatch:
    """Batch container for multiple challenges."""

    items: List[Challenge]

    def __iter__(self) -> Iterable[Challenge]:
        return iter(self.items)


def load_challenges(path: str) -> ChallengeBatch:
    """Load challenges from a YAML file into a ChallengeBatch."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    items: List[Challenge] = []
    for entry in raw:
        challenge = Challenge(
            id=entry["id"],
            prompt=entry["prompt"],
            output_format=entry["output_format"],
            difficulty=float(entry["difficulty"]),
            ground_truth=entry.get("ground_truth"),
        )
        items.append(challenge)

    return ChallengeBatch(items)
