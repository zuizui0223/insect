"""Truth matching, error decomposition, and conditional observability summaries."""

from .errors import ErrorSummary, summarise_errors
from .matching import EventMatch, match_events
from .observability import ObservabilityCell, summarise_observability

__all__ = [
    "ErrorSummary",
    "EventMatch",
    "ObservabilityCell",
    "match_events",
    "summarise_errors",
    "summarise_observability",
]
