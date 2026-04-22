from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anima_host.config import ExportConfig, ModelConfig, SUPPORTED_RESOLUTIONS
from anima_host.export_denoiser import ExportSpec as DenoiserExportSpec, export_denoiser, dry_run_export as dry_run_denoiser
from anima_host.export_text_encoder import TextEncoderExportSpec, export_text_encoder, dry_run_export as dry_run_text_encoder
from anima_host.export_vae import VaeExportSpec, export_vae, dry_run_export as dry_run_vae
from anima_host.model_loader import load_anima_bundle


@dataclass(frozen=True)
class SubmodelExportResult:
    submodel: str
    success: bool
    output_path: Path | None
    duration_ms: int
    error: str | None = None


@dataclass(frozen=True)
class ExportReport:
    model_id: str
    dry_run: bool
    total_duration_ms: int
    denoiser: SubmodelExportResult
    text_encoder: SubmodelExportResult
    vae: SubmodelExportResult

    def to_dict(self) -> dict[str, Any]:
        def result_dict(r: SubmodelExportResult) -> dict[str, Any]:
            return {
                "submodel": r.submodel,
                "success": r.success,
                "output_path": str(r.output_path) if r.output_path else None,
                "duration_ms": r.duration_ms,
                "error": r.error,
            }
        return {
            "model_id": self.model_id,
            "dry_run": self.dry_run,
            "total_duration_ms": self.total_duration_ms,
            "results": {
                "denoiser": result_dict(self.denoiser),
                "text_encoder": result_dict(self.text_encoder),
                "vae": result_dict(self.vae),
            },
        }


class ExportPipeline:
    """End-to-end pipeline that exports all Anima submodels to ONNX.

    When ``dry_run=True`` the pipeline validates all specs and reports
    expected graph topologies without performing real ``torch.onnx.export``
    calls.
    """

    def __init__(self, config: ExportConfig) -> None:
        self.config = config
        self.bundle = load_anima_bundle(config.model_config)

    def export_all(self) -> ExportReport:
        started = time.perf_counter()
        config = self.config

        denoiser_result = self._export_denoiser()
        text_encoder_result = self._export_text_encoder()
        vae_result = self._export_vae()

        ended = time.perf_counter()
        total_ms = int((ended - started) * 1000)

        return ExportReport(
            model_id=config.model_config.model_id,
            dry_run=config.dry_run,
            total_duration_ms=total_ms,
            denoiser=denoiser_result,
            text_encoder=text_encoder_result,
            vae=vae_result,
        )

    def _export_denoiser(self) -> SubmodelExportResult:
        started = time.perf_counter()
        try:
            for width, height in self.config.resolutions:
                spec = DenoiserExportSpec(
                    bundle_dir=Path("."),
                    output_path=self.config.denoiser_onnx_path,
                    width=width,
                    height=height,
                    max_tokens=self.config.max_tokens,
                )
                if self.config.dry_run:
                    dry_run_denoiser(spec)
                else:
                    export_denoiser(spec, denoiser_module=self.bundle.denoiser if self.bundle.available else None)
            ended = time.perf_counter()
            return SubmodelExportResult(
                submodel="denoiser",
                success=True,
                output_path=self.config.denoiser_onnx_path,
                duration_ms=int((ended - started) * 1000),
            )
        except Exception as exc:
            ended = time.perf_counter()
            return SubmodelExportResult(
                submodel="denoiser",
                success=False,
                output_path=None,
                duration_ms=int((ended - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )

    def _export_text_encoder(self) -> SubmodelExportResult:
        started = time.perf_counter()
        try:
            spec = TextEncoderExportSpec(
                bundle_dir=Path("."),
                output_path=self.config.text_encoder_onnx_path,
                max_tokens=self.config.max_tokens,
            )
            if self.config.dry_run:
                dry_run_text_encoder(spec)
            else:
                export_text_encoder(spec, text_encoder_module=self.bundle.text_encoder if self.bundle.available else None)
            ended = time.perf_counter()
            return SubmodelExportResult(
                submodel="text_encoder",
                success=True,
                output_path=self.config.text_encoder_onnx_path,
                duration_ms=int((ended - started) * 1000),
            )
        except Exception as exc:
            ended = time.perf_counter()
            return SubmodelExportResult(
                submodel="text_encoder",
                success=False,
                output_path=None,
                duration_ms=int((ended - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )

    def _export_vae(self) -> SubmodelExportResult:
        started = time.perf_counter()
        try:
            for width, height in self.config.resolutions:
                spec = VaeExportSpec(
                    bundle_dir=Path("."),
                    output_path=self.config.vae_onnx_path,
                    width=width,
                    height=height,
                )
                if self.config.dry_run:
                    dry_run_vae(spec)
                else:
                    export_vae(spec, vae_decoder_module=self.bundle.vae_decoder if self.bundle.available else None)
            ended = time.perf_counter()
            return SubmodelExportResult(
                submodel="vae",
                success=True,
                output_path=self.config.vae_onnx_path,
                duration_ms=int((ended - started) * 1000),
            )
        except Exception as exc:
            ended = time.perf_counter()
            return SubmodelExportResult(
                submodel="vae",
                success=False,
                output_path=None,
                duration_ms=int((ended - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end Anima ONNX export pipeline")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="circlestone-labs/Anima")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path, help="Path to write JSON report")
    args = parser.parse_args()

    config = ExportConfig(
        output_dir=args.output_dir,
        model_config=ModelConfig(model_id=args.model_id),
        dry_run=args.dry_run,
    )

    pipeline = ExportPipeline(config)
    report = pipeline.export_all()

    print(f"Export complete (dry_run={report.dry_run})")
    print(f"  Total duration: {report.total_duration_ms}ms")
    print(f"  Denoiser: {'OK' if report.denoiser.success else 'FAIL'} ({report.denoiser.duration_ms}ms)")
    print(f"  TextEncoder: {'OK' if report.text_encoder.success else 'FAIL'} ({report.text_encoder.duration_ms}ms)")
    print(f"  VAE: {'OK' if report.vae.success else 'FAIL'} ({report.vae.duration_ms}ms)")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
