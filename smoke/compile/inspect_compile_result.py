from pathlib import Path


def expected_qnn_outputs(output_dir: Path, graph_name: str) -> dict[str, Path]:
    return {
        "context": output_dir / f"{graph_name}_ctx.onnx",
        "binary": output_dir / f"{graph_name}_qnn.bin",
        "profile": output_dir / f"{graph_name}_profile.csv",
    }
