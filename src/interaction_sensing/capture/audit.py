"""Random audit sampling for unbiased estimates of missed and false events."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random


@dataclass(slots=True)
class AuditSampler:
    """Bernoulli sampler with explicit inclusion probability.

    Audit clips must be sampled independently of the event trigger; otherwise
    the system cannot estimate what it failed to record.
    """

    probability: float
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.probability <= 1.0:
            raise ValueError("probability must be in (0, 1]")
        self._rng = Random(self.seed)

    def should_capture(self) -> bool:
        return self._rng.random() < self.probability
