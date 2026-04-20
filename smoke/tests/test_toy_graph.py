import numpy as np

from smoke.models.toy_graph import ToyGraphSpec, build_toy_inputs


def test_build_toy_inputs_returns_expected_shapes():
    spec = ToyGraphSpec(batch=1, in_features=8, out_features=4)

    left, weight, bias = build_toy_inputs(spec)

    assert left.shape == (1, 8)
    assert weight.shape == (8, 4)
    assert bias.shape == (4,)
    assert left.dtype == np.float32
