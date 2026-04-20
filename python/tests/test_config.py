import pytest

from anima_host.config import GenerationConfig


def test_generation_config_accepts_supported_resolution():
    config = GenerationConfig(
        width=1024,
        height=1024,
        steps=12,
        cfg=5.0,
        max_tokens=256,
    )

    assert config.width == 1024
    assert config.height == 1024
    assert config.steps == 12


def test_generation_config_rejects_unsupported_resolution():
    with pytest.raises(ValueError, match="Unsupported resolution"):
        GenerationConfig(
            width=640,
            height=640,
            steps=12,
            cfg=5.0,
            max_tokens=256,
        )


def test_generation_config_rejects_invalid_step_count():
    with pytest.raises(ValueError, match="Unsupported steps"):
        GenerationConfig(
            width=1024,
            height=1024,
            steps=30,
            cfg=5.0,
            max_tokens=256,
        )


def test_generation_config_rejects_cfg_out_of_range():
    with pytest.raises(ValueError, match="cfg must be between 3.0 and 7.0"):
        GenerationConfig(
            width=1024,
            height=1024,
            steps=12,
            cfg=8.0,
            max_tokens=256,
        )
