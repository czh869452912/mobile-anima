from ort_qnn.runtime.run_toy_ort_qnn import parse_toy_args
from ort_qnn.runtime.run_mini_block_ort_qnn import parse_mini_args
from ort_qnn.runtime.run_toy_ort_qnn import toy_input_feed
from ort_qnn.runtime.run_mini_block_ort_qnn import mini_block_input_feed


def test_parse_toy_args_reads_model_and_backend_paths():
    args = parse_toy_args([
        "--model", "smoke/artifacts/onnx/toy.onnx",
        "--backend", "/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/libQnnCpu.so",
        "--profile", "ort_qnn/artifacts/profiles/toy_ort_qnn.csv",
    ])

    assert args.model.endswith("toy.onnx")
    assert args.backend.endswith("libQnnCpu.so")


def test_parse_mini_args_reads_model_and_backend_paths():
    args = parse_mini_args([
        "--model", "smoke/artifacts/onnx/mini_block.onnx",
        "--backend", "/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/libQnnCpu.so",
        "--profile", "ort_qnn/artifacts/profiles/mini_ort_qnn.csv",
    ])

    assert args.model.endswith("mini_block.onnx")
    assert args.backend.endswith("libQnnCpu.so")
from ort_qnn.runtime.run_toy_ort_qnn import provider_available


def test_provider_available_checks_membership():
    assert provider_available("QNNExecutionProvider", ["CPUExecutionProvider", "QNNExecutionProvider"])
    assert not provider_available("QNNExecutionProvider", ["CPUExecutionProvider"])


def test_toy_input_feed_has_expected_shape():
    feed = toy_input_feed()

    assert feed["input"].shape == (1, 8)


def test_mini_block_input_feed_has_expected_shape():
    feed = mini_block_input_feed()

    assert feed["input"].shape == (1, 4, 8)
