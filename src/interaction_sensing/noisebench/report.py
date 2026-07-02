"""NoiseBench manifest and protocol-report writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .protocol import NoiseBenchPlan, PerturbationKind


def write_noisebench_plan(output_dir: str | Path, plan: NoiseBenchPlan) -> dict[str, Path]:
    """Write a run manifest, a window-level truth table, and a human protocol.

    These outputs are deliberately created before any camera recording. The
    manifest is therefore a preregistered schedule of perturbation truth rather
    than a label reconstructed after seeing model outputs.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "noisebench_manifest.csv"
    _write_csv(manifest_path, plan.manifest_rows())

    windows_path = output / "noisebench_windows.csv"
    _write_csv(windows_path, _window_truth_rows(plan))

    assumptions_path = output / "noisebench_config.json"
    assumptions_path.write_text(json.dumps(plan.config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    protocol_path = output / "noisebench_protocol.md"
    protocol_path.write_text(_render_protocol(plan), encoding="utf-8")
    return {
        "manifest": manifest_path,
        "windows": windows_path,
        "config": assumptions_path,
        "protocol": protocol_path,
    }


def _window_truth_rows(plan: NoiseBenchPlan) -> list[dict[str, Any]]:
    """Expand each run to one-second windows with explicit active perturbation truth."""

    rows: list[dict[str, Any]] = []
    for recording_order, scenario in enumerate(plan.scenarios, start=1):
        for second in range(int(scenario.duration_seconds)):
            start = float(second)
            end = float(second + 1)
            active = [
                perturbation
                for perturbation in scenario.perturbations
                if perturbation.start_seconds < end and perturbation.end_seconds > start
            ]
            sources = sorted({p.kind.primary_noise_source.value for p in active})
            kinds = sorted({p.kind.value for p in active})
            channels = sorted({channel for p in active for channel in p.kind.expected_error_channels})
            rows.append(
                {
                    "recording_order": recording_order,
                    "scenario_id": scenario.scenario_id,
                    "replicate": scenario.replicate,
                    "window_start_seconds": start,
                    "window_end_seconds": end,
                    "active": bool(active),
                    "active_kinds": "|".join(kinds) if kinds else PerturbationKind.STABLE_CONTROL.value,
                    "active_noise_sources": "|".join(sources) if sources else "stable_scene",
                    "max_intensity": 0.0 if not active else max(p.intensity for p in active),
                    "expected_error_channels": "|".join(channels),
                    "target_agnostic": True,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _render_protocol(plan: NoiseBenchPlan) -> str:
    config = plan.config
    single_conditions = 9 * len(config.intensities) * config.replicates
    mixed_conditions = 3 * len(config.intensities) * config.replicates if config.include_mixed_disturbance else 0
    controls = config.replicates
    lines = [
        "# NoiseBench recording protocol",
        "",
        "## Purpose",
        "",
        "NoiseBench is a target-agnostic controlled perturbation benchmark. Its primary outcome is whether a sensing system can identify *when and why an observation becomes unreliable* — not whether it can recognise a particular organism.",
        "",
        "## Design",
        "",
        f"- Randomised scenarios: {plan.scenario_count}",
        f"- Stable controls: {controls}",
        f"- Single-disturbance scenarios: {single_conditions}",
        f"- Mixed-disturbance scenarios: {mixed_conditions}",
        f"- Replicates: {config.replicates}",
        f"- Recording duration: {config.duration_seconds:.1f} seconds",
        f"- Nominal frame rate: {config.frame_rate:.1f} fps",
        f"- Intensity levels: {', '.join(f'{value:.2f}' for value in config.intensities)}",
        "",
        "## Target independence",
        "",
        "The required scene contains no focal flower, insect, animal, or species label. Natural-looking backgrounds and physical surrogates are allowed, but every run must retain its perturbation truth independently of any later biological video analysis.",
        "",
        "## Required records per run",
        "",
        "```text",
        "raw low-cost stream",
        "IMX500 inference records and KPI metadata",
        "Pi-side temporal features",
        "camera settings and battery/storage state",
        "manifest scenario ID",
        "perturbation timing and intensity truth",
        "noise-state prediction",
        "observability decision",
        "whether high-resolution context or audit capture occurred",
        "```",
        "",
        "## Evaluation endpoints",
        "",
        "```text",
        "noise-source classification / multi-label detection",
        "intensity ordering or calibration",
        "false-event-risk calibration",
        "missed-event-risk calibration",
        "attribution-risk calibration",
        "audit yield per GB and per Wh",
        "fraction of unobservable windows correctly retained as such",
        "performance under mixed disturbances and unknown conditions",
        "```",
        "",
        "## Interpretation constraint",
        "",
        "A successful NoiseBench result does not demonstrate organism detection or ecological inference. It demonstrates a prerequisite: the sensing system can make its own observation conditions auditable before downstream target-specific models are trusted.",
    ]
    return "\n".join(lines) + "\n"
