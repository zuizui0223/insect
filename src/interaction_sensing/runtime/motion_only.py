"""Runnable motion-only baseline with ledger, audit clips, and raw event clips.

This is intentionally the first bridge between the historical motion recorder
and the new error-aware architecture. It does not claim that a motion event is
a biological interaction. It records enough state to test that proposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from interaction_sensing.capture.audit import AuditSampler
from interaction_sensing.capture.opencv_writer import OpenCVClipWriter
from interaction_sensing.capture.ring_buffer import FrameRingBuffer
from interaction_sensing.config.baseline import MotionOnlySettings
from interaction_sensing.data.ledger import EventLedger
from interaction_sensing.domain import AuditRecord, BBox, InteractionEvent, SceneState, TargetSpec
from interaction_sensing.interaction.segment import EventSegmenter
from interaction_sensing.sensing.motion import MOG2MotionExtractor
from interaction_sensing.sensing.scene_state import SceneStateEstimator
from interaction_sensing.targets.manual import build_target_from_boxes
from interaction_sensing.targets.zones import classify_candidate_state
from interaction_sensing.runtime.clock import SourceClock


@dataclass(slots=True)
class MotionOnlyRunConfig:
    source: str | int
    output_dir: Path
    target_id: str
    target_type: str
    core_zone: BBox
    access_zone: BBox | None = None
    settings: MotionOnlySettings = MotionOnlySettings()
    display: bool = False
    max_frames: int | None = None
    source_label: str | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if not self.target_id.strip() or not self.target_type.strip():
            raise ValueError("target_id and target_type cannot be empty")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive when supplied")


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    output_dir: Path
    frames_processed: int
    events_started: int
    audits_started: int
    stopped_by_user: bool


@dataclass(slots=True)
class _EventCapture:
    event: InteractionEvent
    writer: OpenCVClipWriter


@dataclass(slots=True)
class _AuditCapture:
    record: AuditRecord
    writer: OpenCVClipWriter
    end_source_seconds: float


class MotionOnlyRuntime:
    """Run a fixed manually specified target through the motion-only baseline."""

    def __init__(self, config: MotionOnlyRunConfig) -> None:
        self.config = config

    def run(self) -> RunSummary:
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime extra
            raise ImportError("Install interaction-sensing[runtime] to run the motion baseline") from exc

        config = self.config
        settings = config.settings
        output_dir = config.output_dir
        events_dir = output_dir / "events"
        audits_dir = output_dir / "audits"
        output_dir.mkdir(parents=True, exist_ok=True)
        events_dir.mkdir(exist_ok=True)
        audits_dir.mkdir(exist_ok=True)

        capture = cv2.VideoCapture(config.source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open source: {config.source!r}")

        event_capture: _EventCapture | None = None
        audit_capture: _AuditCapture | None = None
        ledger: EventLedger | None = None
        stopped_by_user = False
        run_started_at = datetime.now(timezone.utc)
        run_id = run_started_at.strftime("run_%Y%m%dT%H%M%SZ")
        frames_processed = 0
        events_started = 0
        audits_started = 0
        last_timestamp = run_started_at
        last_source_seconds = 0.0

        try:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read the first frame from source")
            height, width = frame.shape[:2]
            _validate_box_in_frame(config.core_zone, width=width, height=height, name="core_zone")
            if config.access_zone is not None:
                _validate_box_in_frame(config.access_zone, width=width, height=height, name="access_zone")

            target = build_target_from_boxes(
                target_id=config.target_id,
                target_type=config.target_type,
                core_zone=config.core_zone,
                access_zone=config.access_zone,
                context_expand_ratio=settings.context_expand_ratio,
                frame_shape=(height, width),
                metadata={"target_source": "manual", "run_id": run_id},
            )
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps < 1.0:
                fps = settings.fallback_fps
            clock = SourceClock(started_at=run_started_at, fps=fps)
            ring = FrameRingBuffer(max(1, math.ceil(settings.pre_event_seconds * fps)))
            extractor = MOG2MotionExtractor(
                history=settings.history,
                var_threshold=settings.var_threshold,
                min_area=settings.min_area,
                resize_to=(settings.resize_width, settings.resize_height),
            )
            scene_estimator = SceneStateEstimator()
            segmenter = EventSegmenter(quiet_seconds=settings.quiet_seconds)
            audit_sampler = (
                AuditSampler(settings.audit_probability_per_window, seed=_seed_from_run_id(run_id))
                if settings.audit_probability_per_window > 0
                else None
            )
            next_audit_window = 0.0
            last_scene_state_seconds = -float("inf")

            ledger = EventLedger(output_dir / "events.sqlite")
            ledger.register_target(target)
            _write_manifest(
                output_dir,
                {
                    "run_id": run_id,
                    "status": "running",
                    "started_at": run_started_at.isoformat(),
                    "source": config.source_label or str(config.source),
                    "frame_shape": {"height": height, "width": width},
                    "fps": fps,
                    "target": target.to_dict(),
                    "settings": settings.to_dict(),
                },
            )

            frame_index = 0
            while True:
                position_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                timestamp, source_seconds = clock.timestamp(
                    frame_index=frame_index,
                    position_msec=position_msec,
                )
                last_timestamp, last_source_seconds = timestamp, source_seconds
                raw_frame = frame.copy()
                ring.append(raw_frame, timestamp=timestamp)

                # End a quiet event before allowing a later, independent event
                # to start on the current frame.
                for ended in segmenter.close_quiet(now=timestamp):
                    if event_capture is not None and event_capture.event.event_id == ended.event_id:
                        event_capture.writer.write(raw_frame)
                        _finish_event(
                            ended,
                            event_capture,
                            ledger,
                            source_seconds=source_seconds,
                            reason="quiet_timeout",
                        )
                        event_capture = None
                    else:
                        ledger.write_event(ended)

                context_crop = _crop(raw_frame, target.context_zone or target.core_zone)
                if (
                    source_seconds - last_scene_state_seconds >= settings.scene_state_interval_seconds
                    or frame_index == 0
                ):
                    state = scene_estimator.update(context_crop, timestamp=timestamp, target_id=target.target_id)
                    state.metadata.update(
                        {
                            "run_id": run_id,
                            "pipeline_id": settings.pipeline_id,
                            "source_seconds": source_seconds,
                            "frame_index": frame_index,
                        }
                    )
                    ledger.write_scene_state(state)
                    last_scene_state_seconds = source_seconds

                audit_wrote_current = False
                if audit_capture is None and audit_sampler is not None:
                    while source_seconds >= next_audit_window:
                        selected = audit_sampler.should_capture()
                        scheduled_at = next_audit_window
                        next_audit_window += settings.audit_window_seconds
                        if not selected:
                            continue
                        clip_name = f"audit_{run_id}_{frame_index:08d}.mp4"
                        clip_id = str(Path("audits") / clip_name)
                        record = AuditRecord(
                            clip_id=clip_id,
                            sampled_at=timestamp,
                            sampling_probability=settings.audit_probability_per_window,
                            metadata={
                                "run_id": run_id,
                                "pipeline_id": settings.pipeline_id,
                                "target_id": target.target_id,
                                "source_start_seconds": source_seconds,
                                "scheduled_window_seconds": scheduled_at,
                                "review_status": "unreviewed",
                            },
                        )
                        writer = OpenCVClipWriter(
                            output_dir / clip_id,
                            fps=fps,
                            frame_shape=(height, width),
                            codec=settings.codec,
                        )
                        writer.write(raw_frame)
                        audit_capture = _AuditCapture(
                            record=record,
                            writer=writer,
                            end_source_seconds=source_seconds + settings.audit_clip_seconds,
                        )
                        ledger.write_audit(record)
                        audits_started += 1
                        audit_wrote_current = True
                        break
                if audit_capture is not None and not audit_wrote_current:
                    audit_capture.writer.write(raw_frame)
                if audit_capture is not None and source_seconds >= audit_capture.end_source_seconds:
                    _finish_audit(audit_capture, ledger, source_seconds=source_seconds)
                    audit_capture = None

                motion = extractor.extract(raw_frame, target.context_zone or target.core_zone, timestamp=timestamp)
                is_trigger = (
                    frame_index >= settings.history
                    and motion.foreground_ratio >= settings.foreground_ratio_threshold
                    and bool(motion.candidates)
                )
                event_wrote_current = False
                if is_trigger:
                    active_event: InteractionEvent | None = None
                    event_started_this_frame = False
                    for candidate in motion.candidates:
                        state = classify_candidate_state(candidate, target)
                        if state.value == "outside":
                            continue
                        update = segmenter.observe(
                            target_id=target.target_id,
                            actor_track_id=None,
                            timestamp=timestamp,
                            state=state,
                            attribution_score=_state_attribution_score(state.value),
                            verification_score=None,
                            pipeline_id=settings.pipeline_id,
                        )
                        active_event = update.active
                        event_started_this_frame = event_started_this_frame or update.started is not None

                    if active_event is not None:
                        _update_event_metadata(
                            active_event,
                            run_id=run_id,
                            frame_index=frame_index,
                            source_seconds=source_seconds,
                            foreground_ratio=motion.foreground_ratio,
                            candidate_count=len(motion.candidates),
                        )
                        if event_capture is None:
                            clip_name = f"event_{active_event.event_id}.mp4"
                            active_event.clip_id = str(Path("events") / clip_name)
                            event_capture = _EventCapture(
                                event=active_event,
                                writer=OpenCVClipWriter(
                                    output_dir / active_event.clip_id,
                                    fps=fps,
                                    frame_shape=(height, width),
                                    codec=settings.codec,
                                ),
                            )
                            event_capture.writer.write_many(item.frame for item in ring.snapshot())
                            event_wrote_current = True
                            events_started += 1
                        elif event_capture.event.event_id != active_event.event_id:
                            raise RuntimeError("More than one motion-only event capture is active")
                        if event_capture is not None and not event_wrote_current:
                            event_capture.writer.write(raw_frame)
                            event_wrote_current = True
                        ledger.write_event(active_event)
                elif event_capture is not None:
                    event_capture.writer.write(raw_frame)
                    event_wrote_current = True

                if (
                    event_capture is not None
                    and (timestamp - event_capture.event.start_time).total_seconds() >= settings.max_event_seconds
                ):
                    ended = segmenter.close_key(
                        target_id=target.target_id,
                        actor_track_id=None,
                        end_time=timestamp,
                    )
                    if ended is not None:
                        _finish_event(
                            ended,
                            event_capture,
                            ledger,
                            source_seconds=source_seconds,
                            reason="maximum_duration",
                        )
                        event_capture = None

                if config.display:
                    _show_preview(
                        cv2,
                        raw_frame,
                        target=target,
                        motion_ratio=motion.foreground_ratio,
                        candidates=motion.candidates,
                        event_active=event_capture is not None,
                        audit_active=audit_capture is not None,
                    )
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        stopped_by_user = True
                        break

                frames_processed += 1
                frame_index += 1
                if config.max_frames is not None and frames_processed >= config.max_frames:
                    break
                ok, frame = capture.read()
                if not ok:
                    break

            for ended in segmenter.close_all(now=last_timestamp):
                if event_capture is not None and event_capture.event.event_id == ended.event_id:
                    _finish_event(
                        ended,
                        event_capture,
                        ledger,
                        source_seconds=last_source_seconds,
                        reason="source_end",
                    )
                    event_capture = None
                else:
                    ledger.write_event(ended)
            if audit_capture is not None:
                _finish_audit(audit_capture, ledger, source_seconds=last_source_seconds)
                audit_capture = None

            _write_manifest(
                output_dir,
                {
                    "run_id": run_id,
                    "status": "complete",
                    "started_at": run_started_at.isoformat(),
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                    "source": config.source_label or str(config.source),
                    "target": target.to_dict(),
                    "settings": settings.to_dict(),
                    "fps": fps,
                    "frames_processed": frames_processed,
                    "events_started": events_started,
                    "audits_started": audits_started,
                    "stopped_by_user": stopped_by_user,
                },
            )
            return RunSummary(
                run_id=run_id,
                output_dir=output_dir,
                frames_processed=frames_processed,
                events_started=events_started,
                audits_started=audits_started,
                stopped_by_user=stopped_by_user,
            )
        finally:
            if event_capture is not None:
                event_capture.writer.close()
            if audit_capture is not None:
                audit_capture.writer.close()
            if ledger is not None:
                ledger.close()
            capture.release()
            if config.display:
                cv2.destroyAllWindows()


def _validate_box_in_frame(box: BBox, *, width: int, height: int, name: str) -> None:
    if box.left < 0 or box.top < 0 or box.right > width or box.bottom > height:
        raise ValueError(f"{name} must lie inside the first frame ({width} x {height})")


def _crop(frame: Any, box: BBox) -> Any:
    height, width = frame.shape[:2]
    left = max(0, int(box.left))
    top = max(0, int(box.top))
    right = min(width, int(box.right))
    bottom = min(height, int(box.bottom))
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        raise RuntimeError("Target context crop is empty")
    return crop


def _state_attribution_score(state: str) -> float:
    return {
        "access_zone_entry": 1.0,
        "target_contact": 0.8,
        "context_entry": 0.45,
        "approach": 0.25,
    }.get(state, 0.0)


def _update_event_metadata(
    event: InteractionEvent,
    *,
    run_id: str,
    frame_index: int,
    source_seconds: float,
    foreground_ratio: float,
    candidate_count: int,
) -> None:
    metadata = event.metadata
    metadata.setdefault("run_id", run_id)
    metadata.setdefault("source_start_seconds", source_seconds)
    metadata.setdefault("trigger_frame_index", frame_index)
    metadata["source_last_seconds"] = source_seconds
    metadata["last_frame_index"] = frame_index
    metadata["last_candidate_count"] = candidate_count
    metadata["max_candidate_count"] = max(int(metadata.get("max_candidate_count", 0)), candidate_count)
    metadata["last_foreground_ratio"] = foreground_ratio
    metadata["max_foreground_ratio"] = max(float(metadata.get("max_foreground_ratio", 0.0)), foreground_ratio)
    metadata["event_frames_with_trigger"] = int(metadata.get("event_frames_with_trigger", 0)) + 1


def _finish_event(
    event: InteractionEvent,
    capture: _EventCapture,
    ledger: EventLedger,
    *,
    source_seconds: float,
    reason: str,
) -> None:
    capture.writer.close()
    event.metadata.update(
        {
            "source_end_seconds": source_seconds,
            "end_reason": reason,
            "clip_frames_written": capture.writer.frames_written,
        }
    )
    ledger.write_event(event)


def _finish_audit(capture: _AuditCapture, ledger: EventLedger, *, source_seconds: float) -> None:
    capture.writer.close()
    capture.record.metadata.update(
        {
            "source_end_seconds": source_seconds,
            "clip_frames_written": capture.writer.frames_written,
            "review_status": "unreviewed",
        }
    )
    ledger.write_audit(capture.record)


def _write_manifest(output_dir: Path, payload: dict[str, Any]) -> None:
    path = output_dir / "run_manifest.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _seed_from_run_id(run_id: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(run_id))


def _show_preview(
    cv2: Any,
    frame: Any,
    *,
    target: TargetSpec,
    motion_ratio: float,
    candidates: list[Any],
    event_active: bool,
    audit_active: bool,
) -> None:
    preview = frame.copy()
    core = target.core_zone
    context = target.context_zone or target.core_zone
    cv2.rectangle(preview, (int(context.left), int(context.top)), (int(context.right), int(context.bottom)), (0, 0, 255), 2)
    cv2.rectangle(preview, (int(core.left), int(core.top)), (int(core.right), int(core.bottom)), (0, 255, 0), 2)
    if target.access_zone is not None:
        access = target.access_zone
        cv2.rectangle(preview, (int(access.left), int(access.top)), (int(access.right), int(access.bottom)), (255, 0, 255), 2)
    for candidate in candidates:
        box = candidate.bbox
        cv2.rectangle(preview, (int(box.left), int(box.top)), (int(box.right), int(box.bottom)), (0, 255, 255), 1)
    text = f"motion={motion_ratio:.4f} event={event_active} audit={audit_active}"
    cv2.putText(preview, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("interaction-sensing motion baseline", preview)
