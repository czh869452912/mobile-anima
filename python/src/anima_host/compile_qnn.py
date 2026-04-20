import argparse
from pathlib import Path


def build_qnn_context_command(
    onnx_model: Path,
    output_dir: Path,
    sdk_root: Path,
    profiling_level: str,
) -> list[str]:
    binary = sdk_root / "bin" / "x86_64-linux-clang" / "qnn-context-binary-generator"
    return [
        str(binary),
        "--model",
        str(onnx_model),
        "--backend",
        str(sdk_root / "lib" / "x86_64-linux-clang" / "libQnnHtp.so"),
        "--output_dir",
        str(output_dir),
        "--profiling_level",
        profiling_level,
    ]


def validate_qnn_command(command: list[str]) -> None:
    joined = " ".join(command)
    assert "qnn-context-binary-generator" in joined
    assert "--profiling_level" in joined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--profiling-level", default="detailed")
    args = parser.parse_args()
    command = build_qnn_context_command(
        onnx_model=args.onnx_model,
        output_dir=args.output_dir,
        sdk_root=args.sdk_root,
        profiling_level=args.profiling_level,
    )
    validate_qnn_command(command)
    print(" ".join(command))


if __name__ == "__main__":
    main()
