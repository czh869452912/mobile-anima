import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OrtCpuRunSpec:
    model_path: Path


def parse_cpu_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    return parser.parse_args(argv)
