# Android Denoiser-First Validation Checklist

## Device Preconditions

- Snapdragon 8 Elite test device connected with `adb`
- Custom `onnxruntime-android-qnn.aar` installed in the Android project
- Qualcomm `.so` files packaged for `arm64-v8a`
- `denoiser_ctx.onnx` and `denoiser_qnn.bin` copied into the app runtime directory
- Fixed denoiser inputs copied into the app runtime directory:
  - `inputs/latent.raw`
  - `inputs/timestep.raw`
  - `inputs/cond.raw`
  - `inputs/uncond.raw`

## Validation Run

1. Install the debug app on the phone.
2. Launch the app and enter:
   - Resolution: `1024x1024`
   - Prompt: `cat astronaut, cinematic lighting`
   - Negative prompt: `blurry, low quality`
   - Steps: `8`
   - CFG: `5.0`
3. Start generation.
4. Confirm the app writes an output tensor file to the configured runtime directory.
5. Confirm the app writes a profiling file to the configured runtime directory.
6. Confirm the UI reports:
   - `Session created: true`
   - `QNN active: true`
   - output tensor path
   - profiling path
7. Confirm the profiling file contains QNN/HTP activity for the denoiser session.
8. Repeat the run three times.

## Required Acceptance Evidence

- Screenshot of the completed UI run
- The output tensor/raw file path shown by the app
- The profiling CSV or QNN log
- A note confirming CPU fallback was disabled for the denoiser validation run

## Remaining Gap After This Stage

This checklist only proves the Android denoiser-first local runtime loop.

It does not yet prove full prompt-to-image generation because:

- text encoder execution is still outside the Android runtime loop
- VAE decode is still outside the Android runtime loop
- the app currently uses fixed or precomputed denoiser inputs
