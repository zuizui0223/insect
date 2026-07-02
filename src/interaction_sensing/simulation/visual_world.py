"""Rendered visual worlds for latent-disturbance sensing benchmarks.

This module deliberately renders *image sequences*, not precomputed motion
scores.  Ground truth is retained for evaluation only.  Feature extraction in
``visual_benchmark`` receives rendered pixels and never the hidden camera,
illumination, or vegetation-motion variables.

The world contains no taxon or object category.  It contains an independent
local pulse embedded in a natural-scene-like texture plus external disturbances:

* camera displacement;
* spatially coherent but non-rigid vegetation-like sway;
* moving shadow / global illumination change;
* sensor noise.

The purpose is to test whether shared visual signatures can be estimated from
pixels before interpreting a biological target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi, sin
from random import Random
from typing import Iterator

import numpy as np


Slice2D = tuple[slice, slice]


@dataclass(frozen=True, slots=True)
class VisualWorldConfig:
    """Configuration for one reproducible, truth-labelled rendered sequence."""

    name: str = "visual-default"
    frames: int = 240
    height: int = 72
    width: int = 96
    event_start_rate: float = 0.055
    event_amplitude: float = 0.78
    event_radius: float = 2.2
    camera_motion_sd: float = 1.0
    camera_motion_persistence: float = 0.72
    max_camera_shift: int = 3
    sway_amplitude: float = 2.4
    shadow_amplitude: float = 0.42
    illumination_sd: float = 0.10
    illumination_persistence: float = 0.86
    sensor_noise_sd: float = 0.012
    leaf_groups: int = 4
    leaves_per_group: int = 11
    seed: int = 1

    def __post_init__(self) -> None:
        if self.frames < 3:
            raise ValueError("frames must be at least 3")
        if self.height < 40 or self.width < 48:
            raise ValueError("render dimensions are too small")
        if not 0.0 < self.event_start_rate < 1.0:
            raise ValueError("event_start_rate must lie in (0, 1)")
        if self.event_amplitude <= 0.0 or self.event_radius <= 0.0:
            raise ValueError("event parameters must be positive")
        if self.max_camera_shift < 0:
            raise ValueError("max_camera_shift must be non-negative")
        if self.leaf_groups < 1 or self.leaves_per_group < 1:
            raise ValueError("leaf group counts must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RenderedFrame:
    """Rendered image plus hidden truth, retained only for evaluation."""

    frame_index: int
    image: np.ndarray
    true_local_event: bool
    camera_shift: tuple[int, int]
    latent_illumination: float
    latent_sway: float


class VisualDisturbanceWorld:
    """Target-agnostic natural-scene-like renderer with hidden causal truth."""

    def __init__(self, config: VisualWorldConfig) -> None:
        self.config = config
        self._rng = Random(config.seed)
        self._np_rng = np.random.default_rng(config.seed)
        self._y, self._x = np.mgrid[: config.height, : config.width]
        self._local_region = self._make_local_region()
        self._reference_regions = self._make_reference_regions()
        self._mismatched_region = self._make_mismatched_region()
        self._base, self._leaf_layers = self._make_scene_layers()
        self._frames = tuple(self._render())

    @property
    def local_region(self) -> Slice2D:
        return self._local_region

    @property
    def reference_regions(self) -> tuple[Slice2D, ...]:
        return self._reference_regions

    @property
    def mismatched_region(self) -> Slice2D:
        """A visible panel intentionally insulated from illumination/sway.

        It is a *physical visual mismatch*, not a hidden-value shortcut.  It
        moves with the camera but does not share the external photometric and
        vegetation mechanisms that affect the scene.
        """

        return self._mismatched_region

    @property
    def frames_data(self) -> tuple[RenderedFrame, ...]:
        return self._frames

    def iter_frames(self) -> Iterator[RenderedFrame]:
        yield from self._frames

    def _make_local_region(self) -> Slice2D:
        half_height = 8
        half_width = 8
        cy = self.config.height // 2
        cx = self.config.width // 2
        return slice(cy - half_height, cy + half_height + 1), slice(cx - half_width, cx + half_width + 1)

    def _make_reference_regions(self) -> tuple[Slice2D, ...]:
        h, w = self.config.height, self.config.width
        region_h, region_w, margin = 14, 20, 6
        return (
            (slice(margin, margin + region_h), slice(margin, margin + region_w)),
            (slice(margin, margin + region_h), slice(w - margin - region_w, w - margin)),
            (slice(h - margin - region_h, h - margin), slice(margin, margin + region_w)),
            (slice(h - margin - region_h, h - margin), slice(w - margin - region_w, w - margin)),
        )

    def _make_mismatched_region(self) -> Slice2D:
        h, w = self.config.height, self.config.width
        return slice(1, 7), slice(w // 2 - 12, w // 2 + 12)

    def _make_scene_layers(self) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        h, w = self.config.height, self.config.width
        texture = (
            0.33
            + 0.050 * np.sin(self._x * 0.22)
            + 0.040 * np.sin(self._y * 0.36)
            + self._np_rng.normal(0.0, 0.020, size=(h, w))
        )
        layers: list[np.ndarray] = []
        for _ in range(self.config.leaf_groups):
            layer = np.zeros((h, w), dtype=np.float32)
            for _ in range(self.config.leaves_per_group):
                layer += _gaussian_blob(
                    self._x,
                    self._y,
                    center_x=self._rng.uniform(3.0, w - 3.0),
                    center_y=self._rng.uniform(3.0, h - 3.0),
                    sigma_x=self._rng.uniform(1.0, 3.0),
                    sigma_y=self._rng.uniform(1.0, 4.0),
                    amplitude=self._rng.uniform(0.045, 0.150),
                )
            layers.append(layer)
        return texture.astype(np.float32), tuple(layers)

    def _render(self) -> Iterator[RenderedFrame]:
        cfg = self.config
        camera_state = np.zeros(2, dtype=float)
        illumination = 0.0
        for frame_index in range(cfg.frames):
            camera_state = (
                cfg.camera_motion_persistence * camera_state
                + self._np_rng.normal(0.0, cfg.camera_motion_sd, size=2)
            )
            shift_y, shift_x = np.clip(
                np.rint(camera_state),
                -cfg.max_camera_shift,
                cfg.max_camera_shift,
            ).astype(int)
            illumination = (
                cfg.illumination_persistence * illumination
                + float(self._np_rng.normal(0.0, cfg.illumination_sd))
            )
            image = self._base.astype(np.float64).copy()
            sway_state = 0.0
            for group_index, layer in enumerate(self._leaf_layers):
                period = 55.0 + 7.0 * group_index
                phase = 2.0 * pi * frame_index / period + group_index
                displacement = int(
                    round(
                        cfg.sway_amplitude
                        * sin(phase)
                        * (0.52 + 0.17 * group_index)
                    )
                )
                image += np.roll(layer, displacement, axis=1)
                sway_state += abs(displacement)

            image += 0.40 * illumination
            image += self._moving_shadow(frame_index)
            true_event = self._rng.random() < cfg.event_start_rate
            if true_event:
                image += _gaussian_blob(
                    self._x,
                    self._y,
                    center_x=self.config.width / 2.0 + self._rng.uniform(-3.0, 3.0),
                    center_y=self.config.height / 2.0 + self._rng.uniform(-3.0, 3.0),
                    sigma_x=cfg.event_radius,
                    sigma_y=cfg.event_radius,
                    amplitude=cfg.event_amplitude,
                )

            # This visible panel is deliberately not affected by illumination,
            # shadow, or vegetation sway. It supplies a spatial-mismatch control
            # after global camera motion is corrected from the image itself.
            panel_y, panel_x = self._mismatched_region
            image[panel_y, panel_x] = 0.48 + 0.01 * np.sin(self._x[panel_y, panel_x] * 0.60)
            image = _shift_image(image, int(shift_y), int(shift_x))
            image += self._np_rng.normal(0.0, cfg.sensor_noise_sd, size=image.shape)
            yield RenderedFrame(
                frame_index=frame_index,
                image=image.astype(np.float32),
                true_local_event=true_event,
                camera_shift=(int(shift_y), int(shift_x)),
                latent_illumination=float(illumination),
                latent_sway=float(sway_state),
            )

    def _moving_shadow(self, frame_index: int) -> np.ndarray:
        cfg = self.config
        shadow_center_x = cfg.width * (0.50 + 0.35 * sin(2.0 * pi * frame_index / 110.0))
        shadow_center_y = cfg.height * 0.50
        return -cfg.shadow_amplitude * _gaussian_blob(
            self._x,
            self._y,
            center_x=shadow_center_x,
            center_y=shadow_center_y,
            sigma_x=cfg.width * 0.28,
            sigma_y=cfg.height * 0.70,
            amplitude=1.0,
        )


def _gaussian_blob(
    x: np.ndarray,
    y: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    sigma_x: float,
    sigma_y: float,
    amplitude: float,
) -> np.ndarray:
    return amplitude * np.exp(
        -(
            ((x - center_x) ** 2) / (2.0 * sigma_x**2)
            + ((y - center_y) ** 2) / (2.0 * sigma_y**2)
        )
    )


def _shift_image(image: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    """Integer-pixel camera displacement; wraparound is cropped by reference ROIs."""

    return np.roll(np.roll(image, shift_y, axis=0), shift_x, axis=1)
