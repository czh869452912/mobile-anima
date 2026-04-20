import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QuantizeSpec:
    input_model: Path
    output_model: Path
    per_channel: bool = True
    activation_type: str = "QInt8"
    weight_type: str = "QInt8"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    args = parser.parse_args()
    spec = QuantizeSpec(
        input_model=args.input_model,
        output_model=args.output_model,
    )
    print(spec)


if __name__ == "__main__":
    main()
