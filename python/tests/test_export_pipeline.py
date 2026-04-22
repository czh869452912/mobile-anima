from pathlib import Path

from anima_host.config import ExportConfig, ModelConfig
from anima_host.export_pipeline import ExportPipeline, SubmodelExportResult


def test_pipeline_dry_run_succeeds_for_all_submodels():
    config = ExportConfig(
        output_dir=Path("/tmp/test_anima_export"),
        model_config=ModelConfig(),
        dry_run=True,
        resolutions=[(1024, 1024)],
    )
    pipeline = ExportPipeline(config)
    report = pipeline.export_all()

    assert report.dry_run is True
    assert report.model_id == "circlestone-labs/Anima"
    assert report.denoiser.success is True
    assert report.text_encoder.success is True
    assert report.vae.success is True
    assert report.total_duration_ms >= 0


def test_pipeline_report_to_dict_structure():
    config = ExportConfig(
        output_dir=Path("/tmp/test_anima_export"),
        dry_run=True,
        resolutions=[(1024, 1024)],
    )
    pipeline = ExportPipeline(config)
    report = pipeline.export_all()
    d = report.to_dict()

    assert d["model_id"] == "circlestone-labs/Anima"
    assert d["dry_run"] is True
    assert "total_duration_ms" in d
    assert "results" in d
    assert "denoiser" in d["results"]
    assert "text_encoder" in d["results"]
    assert "vae" in d["results"]


def test_submodel_result_fields():
    result = SubmodelExportResult(
        submodel="denoiser",
        success=True,
        output_path=Path("build/denoiser.onnx"),
        duration_ms=100,
    )
    assert result.submodel == "denoiser"
    assert result.success is True
    assert result.output_path == Path("build/denoiser.onnx")
    assert result.duration_ms == 100
    assert result.error is None
