# Desktop ORT + QNN EP Validation

This subproject validates the last missing desktop layer in the smoke ladder:

1. acquire or build ORT with QNN EP
2. create a `QNN EP` session for `toy.onnx`
3. execute `toy.onnx`
4. repeat for `mini_block.onnx`

## First Manual Run Order

1. Check whether a usable Linux prebuilt `ORT + QNN EP` exists
2. If not, fetch ORT source
3. Build Linux desktop `ORT + QNN EP`
4. Verify provider registration for `QNNExecutionProvider`
5. Run `toy.onnx` session creation
6. Run `toy.onnx` execution
7. Run `mini_block.onnx` session creation
8. Run `mini_block.onnx` execution
9. Update `ort_qnn/docs/ort-qnn-matrix.md`
