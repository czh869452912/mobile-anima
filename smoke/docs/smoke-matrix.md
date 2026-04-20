# Smoke Test Matrix

## Environment

- Host: Ubuntu 24.04 x86_64
- SDK version: QAIRT 2.41.0.251128
- ORT build: `onnxruntime` 1.23.2 CPU-only wheel in `/opt/qcom/qairt-py310`
- Backend path: `/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/libQnnCpu.so`

## Results

| Graph | CPU ORT | QNN Compile | ORT QNN Session | ORT QNN Execute | Profile | Failing Stage |
| --- | --- | --- | --- | --- | --- | --- |
| toy | PASS | PASS | NOT_RUN | NOT_RUN | PASS | ort_qnn_ep_unavailable |
| mini_block | PASS | PASS | NOT_RUN | NOT_RUN | PASS | ort_qnn_ep_unavailable |

## Notes

- `toy` passed `qnn-onnx-converter`, `qnn-model-lib-generator`, `qnn-context-binary-generator`, and host CPU `qnn-net-run`
- `mini_block` passed `qnn-onnx-converter`, `qnn-model-lib-generator`, `qnn-context-binary-generator`, and host CPU `qnn-net-run`
- CPU ORT baseline logs were generated for both graphs
- `ORT + QNN EP` is not yet verified because no Linux `QNNExecutionProvider` build is currently available in the environment
