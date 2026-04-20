import argparse
from pathlib import Path

import numpy as np

from ort_qnn.runtime.run_common import RunSpec, run_once

def parse_mini_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--profile", required=True)
    return parser.parse_args(argv)


def mini_block_input_feed() -> dict[str, np.ndarray]:
    return {"input": (np.arange(32, dtype=np.float32).reshape(1, 4, 8) / 32.0)}


def run_mini_block_once(ort_module, model: str, backend: str, profile: str):
    return run_once(
        ort_module,
        RunSpec(
            model_path=Path(model),
            backend_path=Path(backend),
            profiling_path=Path(profile),
            input_feed=mini_block_input_feed(),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_mini_args(argv or [])

    import onnxruntime as ort

    outcome = run_mini_block_once(ort, args.model, args.backend, args.profile)
    return 0 if outcome.execute_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
