from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MiniBlockSpec:
    batch: int
    tokens: int
    hidden_size: int


def build_mini_block_inputs(spec: MiniBlockSpec):
    hidden = np.zeros((spec.batch, spec.tokens, spec.hidden_size), dtype=np.float32)
    residual = np.zeros((spec.batch, spec.tokens, spec.hidden_size), dtype=np.float32)
    weight = np.zeros((spec.hidden_size, spec.hidden_size), dtype=np.float32)
    return hidden, residual, weight
