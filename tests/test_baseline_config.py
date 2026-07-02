from interaction_sensing.config import MotionOnlySettings


def test_motion_settings_read_audit_windows(tmp_path) -> None:
    config = tmp_path / "motion.toml"
    config.write_text(
        """
pipeline_id = "test"
[target]
context_expand_ratio = 0.3
[motion]
foreground_ratio_threshold = 0.02
[capture]
quiet_seconds = 4
[audit]
probability_per_audit_window = 0.2
window_seconds = 30
clip_seconds = 8
[runtime]
scene_state_interval_seconds = 2
""".strip(),
        encoding="utf-8",
    )
    settings = MotionOnlySettings.from_toml(config)

    assert settings.pipeline_id == "test"
    assert settings.context_expand_ratio == 0.3
    assert settings.foreground_ratio_threshold == 0.02
    assert settings.quiet_seconds == 4
    assert settings.audit_probability_per_window == 0.2
    assert settings.audit_window_seconds == 30
    assert settings.audit_clip_seconds == 8
    assert settings.scene_state_interval_seconds == 2
