from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToyGraphSpec:
    batch: int
    in_features: int
    out_features: int


def build_toy_inputs(spec: ToyGraphSpec):
    left = np.zeros((spec.batch, spec.in_features), dtype=np.float32)
    weight = np.zeros((spec.in_features, spec.out_features), dtype=np.float32)
    bias = np.zeros((spec.out_features,), dtype=np.float32)
    return left, weight, bias
