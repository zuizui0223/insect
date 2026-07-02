"""Settings for the motion-only baseline.

The configuration intentionally describes only sensing and capture behaviour.
Deployment-specific source and target geometry belong on the command line or in
a future field-site manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 runtime
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True, slots=True)
class MotionOnlySettings:
    pipeline_id: str = "motion_only_v1"
    context_expand_ratio: float = 0.20
    history: int = 120
    var_threshold: float = 40.0
    min_area: int = 80
    resize_width: int = 320
    resize_height: int = 240
    foreground_ratio_threshold: float = 0.008
    quiet_seconds: float = 2.0
    pre_event_seconds: float = 3.0
    max_event_seconds: float = 60.0
    audit_probability_per_window: float = 0.05
    audit_window_seconds: float = 60.0
    audit_clip_seconds: float = 10.0
    scene_state_interval_seconds: float = 1.0
    codec: str = "mp4v"
    fallback_fps: float = 20.0

    def __post_init__(self) -> None:
        if not self.pipeline_id.strip():
            raise ValueError("pipeline_id cannot be empty")
        if self.context_expand_ratio < 0:
            raise ValueError("context_expand_ratio must be non-negative")
        if self.history <= 0 or self.min_area <= 0:
            raise ValueError("history and min_area must be positive")
        if self.resize_width <= 0 or self.resize_height <= 0:
            raise ValueError("resize dimensions must be positive")
        if self.foreground_ratio_threshold < 0:
            raise ValueError("foreground_ratio_threshold must be non-negative")
        if self.quiet_seconds <= 0 or self.pre_event_seconds < 0:
            raise ValueError("quiet_seconds must be positive and pre_event_seconds non-negative")
        if self.max_event_seconds <= 0:
            raise ValueError("max_event_seconds must be positive")
        if not 0.0 <= self.audit_probability_per_window <= 1.0:
            raise ValueError("audit_probability_per_window must be in [0, 1]")
        if self.audit_window_seconds <= 0 or self.audit_clip_seconds <= 0:
            raise ValueError("audit windows and clips must have positive duration")
        if self.scene_state_interval_seconds <= 0 or self.fallback_fps <= 0:
            raise ValueError("scene_state_interval_seconds and fallback_fps must be positive")
        if len(self.codec) != 4:
            raise ValueError("codec must be a four-character OpenCV code")

    @classmethod
    def from_toml(cls, path: str | Path) -> "MotionOnlySettings":
        with Path(path).open("rb") as handle:
            raw = tomllib.load(handle)
        target = _mapping(raw.get("target"))
        motion = _mapping(raw.get("motion"))
        capture = _mapping(raw.get("capture"))
        audit = _mapping(raw.get("audit"))
        runtime = _mapping(raw.get("runtime"))
        return cls(
            pipeline_id=str(raw.get("pipeline_id", cls.pipeline_id)),
            context_expand_ratio=float(target.get("context_expand_ratio", cls.context_expand_ratio)),
            history=int(motion.get("history", cls.history)),
            var_threshold=float(motion.get("var_threshold", cls.var_threshold)),
            min_area=int(motion.get("min_area", cls.min_area)),
            resize_width=int(motion.get("resize_width", cls.resize_width)),
            resize_height=int(motion.get("resize_height", cls.resize_height)),
            foreground_ratio_threshold=float(
                motion.get("foreground_ratio_threshold", cls.foreground_ratio_threshold)
            ),
            quiet_seconds=float(capture.get("quiet_seconds", cls.quiet_seconds)),
            pre_event_seconds=float(capture.get("pre_event_seconds", cls.pre_event_seconds)),
            max_event_seconds=float(capture.get("max_event_seconds", cls.max_event_seconds)),
            audit_probability_per_window=float(
                audit.get("probability_per_audit_window", cls.audit_probability_per_window)
            ),
            audit_window_seconds=float(audit.get("window_seconds", cls.audit_window_seconds)),
            audit_clip_seconds=float(audit.get("clip_seconds", cls.audit_clip_seconds)),
            scene_state_interval_seconds=float(
                runtime.get("scene_state_interval_seconds", cls.scene_state_interval_seconds)
            ),
            codec=str(runtime.get("codec", cls.codec)),
            fallback_fps=float(runtime.get("fallback_fps", cls.fallback_fps)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
