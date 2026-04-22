from pathlib import Path

import numpy as np

from anima_host.calibration_data import CalibrationDataGenerator, CalibrationSpec


def test_synthetic_calibration_generates_npz():
    output = Path("/tmp/test_anima_calib.npz")
    spec = CalibrationSpec(
        output_path=output,
        num_samples=10,
        seed=42,
        resolutions=[(1024, 1024)],
    )
    gen = CalibrationDataGenerator(spec)
    result = gen.generate()

    assert result == output
    assert output.exists()

    data = np.load(output)
    assert "latent" in data
    assert "timestep" in data
    assert "cond" in data
    assert "uncond" in data

    assert data["latent"].shape == (10, 4, 128, 128)
    assert data["timestep"].shape == (10, 1)
    assert data["cond"].shape == (10, 256, 16)
    assert data["uncond"].shape == (10, 256, 16)

    output.unlink()


def test_calibration_with_different_sample_count():
    output = Path("/tmp/test_anima_calib_50.npz")
    spec = CalibrationSpec(
        output_path=output,
        num_samples=50,
        seed=123,
        resolutions=[(1024, 1024)],
    )
    gen = CalibrationDataGenerator(spec)
    gen.generate()

    data = np.load(output)
    assert data["latent"].shape[0] == 50

    output.unlink()
