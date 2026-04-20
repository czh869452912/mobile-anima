# Desktop ORT + QNN EP Validation

This subproject validates the last missing desktop layer in the smoke ladder:

1. acquire or build ORT with QNN EP
2. create a `QNN EP` session for `toy.onnx`
3. execute `toy.onnx`
4. repeat for `mini_block.onnx`
