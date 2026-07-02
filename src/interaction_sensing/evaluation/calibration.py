"""Transparent audit-based correction for binary observation windows.

This is intentionally a small estimator rather than a final hierarchical model.
It demonstrates why independently sampled audit windows are useful: detection and
false-positive rates can be estimated separately from trigger-selected clips.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class AuditCalibration:
    audit_windows: int
    truth_positive_windows: int
    system_positive_windows: int
    detection_probability: float | None
    false_positive_probability: float | None

    def corrected_truth_count(self, *, total_windows: int, observed_positive_windows: int) -> float | None:
        """Estimate true positive windows under a two-error observation model.

        Expected observed positives are:

        ``Y = p * N + phi * (M - N)``

        where `p` is detection probability, `phi` is false-positive probability,
        `M` the number of known observation windows, and `N` the unknown number
        of truth-positive windows. The estimate is clamped to `[0, M]`.
        """

        if total_windows < 0 or observed_positive_windows < 0:
            raise ValueError("window counts must be non-negative")
        if self.detection_probability is None or self.false_positive_probability is None:
            return None
        denominator = self.detection_probability - self.false_positive_probability
        if denominator <= 1e-9:
            return None
        estimate = (observed_positive_windows - self.false_positive_probability * total_windows) / denominator
        return min(float(total_windows), max(0.0, estimate))

    def to_dict(self) -> dict:
        return asdict(self)


def fit_audit_calibration(rows: Iterable[tuple[bool, bool]]) -> AuditCalibration:
    """Fit `p` and `phi` from `(truth_focal, system_focal)` audit windows."""

    observed = list(rows)
    truth_positive = sum(1 for truth, _ in observed if truth)
    truth_negative = len(observed) - truth_positive
    detected_true = sum(1 for truth, system in observed if truth and system)
    false_positive = sum(1 for truth, system in observed if not truth and system)
    system_positive = detected_true + false_positive
    return AuditCalibration(
        audit_windows=len(observed),
        truth_positive_windows=truth_positive,
        system_positive_windows=system_positive,
        detection_probability=None if truth_positive == 0 else detected_true / truth_positive,
        false_positive_probability=None if truth_negative == 0 else false_positive / truth_negative,
    )
