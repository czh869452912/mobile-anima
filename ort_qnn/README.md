# Desktop ORT + QNN EP Validation

This subproject validates the last missing desktop layer in the smoke ladder:

1. acquire or build ORT with QNN EP
2. verify `QNNExecutionProvider` visibility
3. create a `QNN EP` session for `toy.onnx`
4. execute `toy.onnx`
5. repeat for `mini_block.onnx`

## First Manual Run Order

1. Check for a usable Linux prebuilt `ORT + QNN EP`
2. If not available, fetch ORT source
3. Build Linux desktop ORT Python wheel with `QNN EP`
4. Confirm `QNNExecutionProvider` is visible
5. Run `toy.onnx` session creation
6. Run `toy.onnx` execution
7. Run `mini_block.onnx` session creation
8. Run `mini_block.onnx` execution
9. Update `ort_qnn/docs/ort-qnn-matrix.md`
