"""Run a taxon-agnostic motion-only baseline with structured event logging."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from interaction_sensing.capture import AuditSampler, FrameRingBuffer, VideoClipRecorder
from interaction_sensing.data import EventLedger
from interaction_sensing.domain import AuditRecord, BBox, InteractionState
from interaction_sensing.interaction import EventSegmenter, assign_target, state_rank
from interaction_sensing.sensing import MOG2MotionExtractor
from interaction_sensing.targets import build_target_from_boxes, classify_candidate_state


def _parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _parse_bbox(values: list[float] | None) -> BBox | None:
    if values is None:
        return None
    if len(values) != 4:
        raise argparse.ArgumentTypeError("bbox requires four numbers: left top right bottom")
    return BBox(*values)


def _crop(frame: Any, bbox: BBox) -> Any:
    height, width = frame.shape[:2]
    left = max(0, int(bbox.left))
    top = max(0, int(bbox.top))
    right = min(width, int(bbox.right))
    bottom = min(height, int(bbox.bottom))
    return frame[top:bottom, left:right]


def _frame_shape(frame: Any) -> tuple[int, int]:
    height, width = frame.shape[:2]
    return height, width


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--target-id", default="target-001")
    parser.add_argument("--target-type", default="target")
    parser.add_argument(
        "--target-bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="Focal target box in pixel coordinates",
    )
    parser.add_argument(
        "--access-bbox",
        nargs=4,
        type=float,
        default=None,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="Optional nested access zone, for example a corolla opening",
    )
    parser.add_argument("--context-expand-ratio", type=float, default=0.2)
    parser.add_argument("--ledger", type=Path, default=Path("runs/motion_baseline/events.sqlite"))
    parser.add_argument("--clips-dir", type=Path, default=Path("runs/motion_baseline/clips"))
    parser.add_argument("--pipeline-id", default="motion_only_v1")
    parser.add_argument("--motion-threshold", type=float, default=0.008)
    parser.add_argument("--quiet-seconds", type=float, default=2.0)
    parser.add_argument("--pre-event-frames", type=int, default=60)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until source ends or q is pressed")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--write-clips", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--audit-probability", type=float, default=0.0)
    parser.add_argument("--audit-window-seconds", type=float, default=60.0)
    parser.add_argument("--audit-seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install interaction-sensing[runtime] to run the motion baseline") from exc

    cap = cv2.VideoCapture(_parse_source(str(args.source)))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read the first frame")

    frame_shape = _frame_shape(first_frame)
    core_zone = BBox(*args.target_bbox)
    access_zone = _parse_bbox(args.access_bbox)
    target = build_target_from_boxes(
        target_id=args.target_id,
        target_type=args.target_type,
        core_zone=core_zone,
        access_zone=access_zone,
        context_expand_ratio=args.context_expand_ratio,
        frame_shape=frame_shape,
        metadata={"source": str(args.source), "pipeline_id": args.pipeline_id},
    )

    ledger = EventLedger(args.ledger)
    ledger.register_target(target)
    motion = MOG2MotionExtractor()
    segmenter = EventSegmenter(quiet_seconds=args.quiet_seconds)
    ring = FrameRingBuffer(max_frames=args.pre_event_frames)
    recorder = VideoClipRecorder(args.clips_dir, fps=args.fps)
    sampler = AuditSampler(args.audit_probability, seed=args.audit_seed) if args.audit_probability > 0 else None

    open_clips: dict[str, Any] = {}
    active_clip_by_event: dict[str, str] = {}
    next_audit_time = datetime.now() + timedelta(seconds=args.audit_window_seconds)

    def start_event_clip(event_id: str, frame: Any) -> str:
        clip_id = f"event_{event_id}"
        clip = recorder.open(clip_id=clip_id, frame_shape=frame_shape)
        for buffered in ring.snapshot():
            clip.write(buffered.frame)
        clip.write(frame)
        open_clips[event_id] = clip
        active_clip_by_event[event_id] = clip_id
        return clip_id

    def close_event(event) -> None:
        clip = open_clips.pop(event.event_id, None)
        if clip is not None:
            clip.close()
        event.clip_id = active_clip_by_event.get(event.event_id)
        ledger.write_event(event)

    frame_index = 0
    current_frame = first_frame
    try:
        while True:
            timestamp = datetime.now()
            frame_index += 1
            frame = current_frame
            ring.append(frame.copy(), timestamp=timestamp)

            result = motion.extract(frame, target.context_zone or target.core_zone, timestamp=timestamp)
            any_event_candidate = False
            best_state = InteractionState.OUTSIDE
            best_score = 0.0
            best_candidate_id: str | None = None

            if result.foreground_ratio >= args.motion_threshold:
                for candidate in result.candidates:
                    state = classify_candidate_state(candidate, target)
                    attribution = assign_target(candidate, [target])
                    if attribution.target_id != target.target_id:
                        continue
                    if state_rank(state) > state_rank(best_state):
                        best_state = state
                        best_score = attribution.score
                        best_candidate_id = candidate.candidate_id
                    if state_rank(state) >= state_rank(InteractionState.CONTEXT_ENTRY):
                        any_event_candidate = True

            if any_event_candidate:
                update = segmenter.observe(
                    target_id=target.target_id,
                    actor_track_id=None,
                    timestamp=timestamp,
                    state=best_state,
                    attribution_score=best_score,
                    verification_score=None,
                    pipeline_id=args.pipeline_id,
                )
                event = update.active
                if event is not None:
                    event.metadata.update(
                        {
                            "foreground_ratio": result.foreground_ratio,
                            "candidate_id": best_candidate_id,
                            "n_motion_candidates": len(result.candidates),
                            "source_frame_index": frame_index,
                        }
                    )
                    if update.started is not None and args.write_clips:
                        event.clip_id = start_event_clip(event.event_id, frame)
                    elif args.write_clips and event.event_id in open_clips:
                        open_clips[event.event_id].write(frame)

            for ended in segmenter.close_quiet(now=timestamp):
                close_event(ended)

            if sampler is not None and timestamp >= next_audit_time:
                next_audit_time = timestamp + timedelta(seconds=args.audit_window_seconds)
                if sampler.should_capture():
                    audit_clip_id = f"audit_{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
                    if args.write_clips:
                        recorder.write_clip(clip_id=audit_clip_id, frames=ring.snapshot(), frame_shape=frame_shape)
                    ledger.write_audit(
                        AuditRecord(
                            clip_id=audit_clip_id,
                            sampled_at=timestamp,
                            sampling_probability=args.audit_probability,
                            metadata={"pipeline_id": args.pipeline_id, "source_frame_index": frame_index},
                        )
                    )

            if args.show:
                vis = frame.copy()
                target_box = target.core_zone
                context_box = target.context_zone or target.core_zone
                cv2.rectangle(vis, (int(context_box.left), int(context_box.top)), (int(context_box.right), int(context_box.bottom)), (0, 0, 255), 2)
                cv2.rectangle(vis, (int(target_box.left), int(target_box.top)), (int(target_box.right), int(target_box.bottom)), (0, 255, 0), 2)
                cv2.putText(vis, f"motion {result.foreground_ratio:.4f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("interaction-sensing motion baseline", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q")}:
                    break

            if args.max_frames and frame_index >= args.max_frames:
                break

            ok, current_frame = cap.read()
            if not ok:
                break
    finally:
        for ended in segmenter.close_all(now=datetime.now()):
            close_event(ended)
        for clip in list(open_clips.values()):
            clip.close()
        ledger.close()
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
