import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompileSpec:
    sdk_root: Path
    onnx_model: Path
    output_dir: Path
    profiling_level: str


def build_compile_command(spec: CompileSpec) -> list[str]:
    return [
        str(spec.sdk_root / "bin" / "x86_64-linux-clang" / "qnn-context-binary-generator"),
        "--model",
        str(spec.onnx_model),
        "--backend",
        str(spec.sdk_root / "lib" / "x86_64-linux-clang" / "libQnnHtp.so"),
        "--output_dir",
        str(spec.output_dir),
        "--profiling_level",
        spec.profiling_level,
    ]


def parse_compile_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-name", required=True)
    parser.add_argument("--onnx-model", required=True)
    parser.add_argument("--sdk-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profiling-level", default="detailed")
    return parser.parse_args(argv)
