from pathlib import Path

import pytest

try:
    import onnx
    from onnx import helper, TensorProto
    _ONNX_AVAILABLE = True
except Exception:
    _ONNX_AVAILABLE = False

from anima_host.operator_gap import (
    analyze_onnx_for_qnn,
    COMMONLY_UNSUPPORTED_OPS,
)

pytestmark = pytest.mark.skipif(not _ONNX_AVAILABLE, reason="onnx not installed")


def _build_minimal_onnx(op_types, path):
    """Build a minimal ONNX model containing only the specified op types."""
    inputs = []
    outputs = []
    nodes = []
    value_info = []
    initializer = []

    for idx, op_type in enumerate(op_types):
        input_name = f"input_{idx}"
        output_name = f"output_{idx}"

        if op_type in ("Add", "Mul", "Sub", "Div", "Pow"):
            nodes.append(helper.make_node(op_type, [input_name, input_name], [output_name], name=f"node_{idx}"))
        elif op_type == "MatMul":
            w = helper.make_tensor(f"w_{idx}", TensorProto.FLOAT, [1, 1], [1.0])
            initializer.append(w)
            nodes.append(helper.make_node(op_type, [input_name, f"w_{idx}"], [output_name], name=f"node_{idx}"))
        elif op_type == "Constant":
            nodes.append(helper.make_node(op_type, [], [output_name], name=f"node_{idx}", value_float=1.0))
        else:
            nodes.append(helper.make_node(op_type, [input_name], [output_name], name=f"node_{idx}"))

        inputs.append(helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 1]))
        outputs.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 1]))

    graph = helper.make_graph(nodes, "test_graph", inputs, outputs, initializer, value_info)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def test_analyze_all_supported_ops():
    tmp_path = Path("/tmp/test_anima_all_supported.onnx")
    sample_ops = ["Relu", "Add", "MatMul", "Conv"]
    _build_minimal_onnx(sample_ops, tmp_path)

    report = analyze_onnx_for_qnn(tmp_path)
    assert report.risk_level == "low"
    assert len(report.unsupported_ops) == 0
    assert report.unsupported_node_count == 0


def test_analyze_with_unsupported_ops():
    tmp_path = Path("/tmp/test_anima_unsupported.onnx")
    sample_ops = ["Relu", "GroupNormalization", "ScaledDotProductAttention"]
    _build_minimal_onnx(sample_ops, tmp_path)

    report = analyze_onnx_for_qnn(tmp_path)
    assert report.risk_level in ("medium", "high", "critical")
    assert len(report.unsupported_ops) > 0
    assert "GroupNormalization" in report.unsupported_ops or "ScaledDotProductAttention" in report.unsupported_ops


def test_report_to_dict_structure():
    tmp_path = Path("/tmp/test_anima_report.onnx")
    _build_minimal_onnx(["Add", "Relu"], tmp_path)

    report = analyze_onnx_for_qnn(tmp_path)
    d = report.to_dict()
    assert d["model_path"] == str(tmp_path)
    assert d["total_nodes"] == 2
    assert "supported_ops" in d
    assert "unsupported_ops" in d
    assert "risk_level" in d
    assert "recommendations" in d


def test_known_unsupported_ops_are_detected():
    tmp_path = Path("/tmp/test_anima_known_unsupported.onnx")
    _build_minimal_onnx(["If"], tmp_path)

    report = analyze_onnx_for_qnn(tmp_path)
    assert "If" in report.unsupported_ops
    assert report.unsupported_node_count > 0


def test_qdq_recommendation_when_no_quantization():
    tmp_path = Path("/tmp/test_anima_no_qdq.onnx")
    _build_minimal_onnx(["Add", "Relu", "Conv"], tmp_path)

    report = analyze_onnx_for_qnn(tmp_path)
    recs = [r for r in report.recommendations if "QDQ" in r or "quantized" in r]
    assert len(recs) > 0
