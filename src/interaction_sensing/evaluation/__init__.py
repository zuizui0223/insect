"""Truth matching, error decomposition, and conditional observability summaries."""

from .calibration import AuditCalibration, fit_audit_calibration
from .errors import ErrorSummary, summarise_errors
from .matching import EventMatch, match_events
from .observability import ObservabilityCell, summarise_observability

__all__ = [
    "AuditCalibration",
    "ErrorSummary",
    "EventMatch",
    "ObservabilityCell",
    "fit_audit_calibration",
    "match_events",
    "summarise_errors",
    "summarise_observability",
]
