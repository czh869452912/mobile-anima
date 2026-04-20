# Android Validation Checklist

## Device Preconditions

- Snapdragon 8 Elite test device connected with `adb`
- Custom `onnxruntime-android-qnn.aar` installed in the Android project
- Qualcomm `.so` files packaged for `arm64-v8a`
- `denoiser_ctx.onnx` and `denoiser_qnn.bin` copied into the app runtime directory

## Validation Run

1. Install the debug app on the phone.
2. Launch the app and enter:
   - Resolution: `1024x1024`
   - Prompt: `cat astronaut, cinematic lighting`
   - Negative prompt: `blurry, low quality`
   - Steps: `8`
   - CFG: `5.0`
3. Start generation.
4. Confirm the app writes a profiling file to the configured runtime directory.
5. Confirm the profiling file contains QNN/HTP activity for the denoiser session.
6. Repeat the run three times.

## Required Acceptance Evidence

- Screenshot of the completed UI run
- The generated image file
- The profiling CSV or QNN log
- A note confirming CPU fallback was disabled for the denoiser validation run
