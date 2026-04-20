import numpy as np

from smoke.models.mini_block import MiniBlockSpec, build_mini_block_inputs


def test_build_mini_block_inputs_returns_expected_shapes():
    spec = MiniBlockSpec(batch=1, tokens=4, hidden_size=8)

    hidden, residual, weight = build_mini_block_inputs(spec)

    assert hidden.shape == (1, 4, 8)
    assert residual.shape == (1, 4, 8)
    assert weight.shape == (8, 8)
    assert hidden.dtype == np.float32
