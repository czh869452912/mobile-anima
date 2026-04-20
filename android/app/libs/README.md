Place the custom `onnxruntime-android-qnn.aar` and the Qualcomm runtime `.so` files required for `arm64-v8a` in this directory before building the release candidate.

Required contents for the first device run:

- `onnxruntime-android-qnn.aar`
- `arm64-v8a/libQnnHtp.so`
- `arm64-v8a/libQnnSystem.so`
- `arm64-v8a/libQnnHtpV75Stub.so` or the matching stub for the target SDK release
