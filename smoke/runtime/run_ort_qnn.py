import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OrtQnnRunSpec:
    model_path: Path
    backend_path: Path
    profiling_path: Path
    disable_cpu_fallback: bool


def qnn_provider_options(spec: OrtQnnRunSpec) -> dict[str, str]:
    return {
        "backend_path": str(spec.backend_path),
        "profiling_level": "detailed",
        "profiling_file_path": str(spec.profiling_path),
        "ep.context_enable": "1",
        "ep.context_embed_mode": "0",
        "session.disable_cpu_ep_fallback": "1" if spec.disable_cpu_fallback else "0",
    }


def parse_qnn_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--profile", required=True)
    return parser.parse_args(argv)
