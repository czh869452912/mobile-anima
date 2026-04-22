import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import onnx
    _ONNX_AVAILABLE = True
except Exception:
    _ONNX_AVAILABLE = False


# Conservative QNN HTP operator support list for operator gap analysis.
# Based on Qualcomm QNN SDK documentation; treat anything not in this list as potentially unsupported.
QNN_HTP_SUPPORTED_OPS: set[str] = {
    "Abs",
    "Add",
    "And",
    "ArgMax",
    "ArgMin",
    "AveragePool",
    "BatchNormalization",
    "Cast",
    "Ceil",
    "Clip",
    "Concat",
    "Constant",
    "Conv",
    "ConvTranspose",
    "Cos",
    "CumSum",
    "DepthToSpace",
    "DequantizeLinear",
    "Div",
    "Dropout",
    "Elu",
    "Equal",
    "Erf",
    "Exp",
    "Expand",
    "Flatten",
    "Floor",
    "Gather",
    "Gemm",
    "GlobalAveragePool",
    "Greater",
    "GreaterOrEqual",
    "HardSigmoid",
    "HardSwish",
    "Identity",
    "InstanceNormalization",
    "LSTM",
    "LayerNormalization",
    "LeakyRelu",
    "Less",
    "LessOrEqual",
    "Log",
    "LogSoftmax",
    "Loop",
    "MatMul",
    "Max",
    "MaxPool",
    "Min",
    "Mul",
    "Neg",
    "NonMaxSuppression",
    "NonZero",
    "Not",
    "Or",
    "PRelu",
    "Pad",
    "Pow",
    "QuantizeLinear",
    "QLinearConv",
    "QLinearMatMul",
    "QLinearAdd",
    "QLinearAveragePool",
    "QLinearGlobalAveragePool",
    "QLinearConcat",
    "QLinearMul",
    "QLinearLeakyRelu",
    "QLinearSigmoid",
    "QLinearSoftmax",
    "Range",
    "Reciprocal",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "ReduceSumSquare",
    "Relu",
    "Reshape",
    "Resize",
    "RoiAlign",
    "ScatterElements",
    "ScatterND",
    "Shape",
    "Sigmoid",
    "Sign",
    "Sin",
    "Slice",
    "Softmax",
    "Softplus",
    "Softsign",
    "SpaceToDepth",
    "Split",
    "Sqrt",
    "Squeeze",
    "Sub",
    "Sum",
    "Tan",
    "Tanh",
    "Tile",
    "TopK",
    "Transpose",
    "Unsqueeze",
    "Where",
}

# Ops that are commonly problematic on QNN HTP and may require CPU fallback or graph surgery.
COMMONLY_UNSUPPORTED_OPS: set[str] = {
    "GroupNormalization",
    "ScaledDotProductAttention",
    "GridSample",
    "LayerNormalization",  # may be partially supported depending on SDK version
    "InstanceNormalization",
    "Dropout",
    "Loop",
    "Scan",
    "If",
}


@dataclass(frozen=True)
class OperatorGapReport:
    model_path: Path
    total_nodes: int
    unique_ops: set[str] = field(default_factory=set)
    supported_ops: set[str] = field(default_factory=set)
    unsupported_ops: set[str] = field(default_factory=set)
    commonly_unsupported_present: set[str] = field(default_factory=set)
    unsupported_node_count: int = 0
    unsupported_node_percentage: float = 0.0
    risk_level: str = "unknown"
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "total_nodes": self.total_nodes,
            "unique_ops": sorted(self.unique_ops),
            "supported_ops": sorted(self.supported_ops),
            "unsupported_ops": sorted(self.unsupported_ops),
            "commonly_unsupported_present": sorted(self.commonly_unsupported_present),
            "unsupported_node_count": self.unsupported_node_count,
            "unsupported_node_percentage": self.unsupported_node_percentage,
            "risk_level": self.risk_level,
            "recommendations": self.recommendations,
        }


def _extract_ops_from_graph(graph: Any) -> tuple[set[str], dict[str, int]]:
    """Extract unique operators and node counts from an ONNX graph."""
    ops: set[str] = set()
    node_counts: dict[str, int] = {}
    for node in graph.node:
        op = node.op_type
        ops.add(op)
        node_counts[op] = node_counts.get(op, 0) + 1
    return ops, node_counts


def analyze_onnx_for_qnn(model_path: Path) -> OperatorGapReport:
    if not _ONNX_AVAILABLE:
        raise RuntimeError("onnx package is required for operator gap analysis")

    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    graph = model.graph

    unique_ops, node_counts = _extract_ops_from_graph(graph)
    total_nodes = sum(node_counts.values())

    unsupported = unique_ops - QNN_HTP_SUPPORTED_OPS
    supported = unique_ops & QNN_HTP_SUPPORTED_OPS
    commonly_unsupported = unsupported & COMMONLY_UNSUPPORTED_OPS

    unsupported_count = sum(node_counts[op] for op in unsupported)
    percentage = (unsupported_count / total_nodes * 100.0) if total_nodes > 0 else 0.0

    if percentage == 0.0:
        risk = "low"
    elif percentage < 5.0:
        risk = "medium"
    elif percentage < 20.0:
        risk = "high"
    else:
        risk = "critical"

    recommendations: list[str] = []
    if unsupported:
        recommendations.append(
            f"Found {len(unsupported)} potentially unsupported operators: {sorted(unsupported)}"
        )
    if commonly_unsupported:
        recommendations.append(
            f"Commonly problematic ops present: {sorted(commonly_unsupported)}. "
            "Consider graph surgery, CPU fallback, or SDK version upgrade."
        )
    if "QuantizeLinear" not in unique_ops and "DequantizeLinear" not in unique_ops:
        recommendations.append(
            "Model is not QDQ-quantized. QNN HTP strongly prefers QDQ graphs for NPU execution."
        )
    if not unsupported:
        recommendations.append("All operators appear supported. Proceed to QNN compilation.")

    return OperatorGapReport(
        model_path=model_path,
        total_nodes=total_nodes,
        unique_ops=unique_ops,
        supported_ops=supported,
        unsupported_ops=unsupported,
        commonly_unsupported_present=commonly_unsupported,
        unsupported_node_count=unsupported_count,
        unsupported_node_percentage=round(percentage, 2),
        risk_level=risk,
        recommendations=recommendations,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = analyze_onnx_for_qnn(args.model)
    report_dict = report.to_dict()
    print(json.dumps(report_dict, indent=2))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report_dict, indent=2))
        print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
