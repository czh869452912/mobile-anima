# ORT + QNN Validation Matrix

## Environment

- Host: Linux x86_64 local build host
- QNN SDK version: QAIRT/QNN `2.41.0.251128` (`sdk.yaml` reports `2.41.0`)
- Acquisition path: source build
- ORT build: ONNX Runtime `1.23.2` source build under `ort_qnn/artifacts/ort/build/linux_qnn/Release` with direct `PYTHONPATH` import

## Results

| Graph | Provider Visible | Session | Execute | Profile | Failing Stage |
| --- | --- | --- | --- | --- | --- |
| toy | PASS | PASS | PASS | `toy_ort_qnn.csv` | |
| mini_block | PASS | PASS | PASS | `mini_block_ort_qnn.csv` | |
