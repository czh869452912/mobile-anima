from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OrtCpuRunSpec:
    model_path: Path
