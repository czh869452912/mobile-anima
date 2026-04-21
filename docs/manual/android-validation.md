# Android Denoiser-First Validation Checklist

## 在项目中的位置

- 项目总览：`../project/index.md`
- 项目架构：`../project/architecture.md`
- 模块状态：`../project/status.md`
- 验证体系：`../project/validation.md`

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
2. Launch the app and enter the fixed `denoiser-first` 参数。
3. Start generation.
4. Confirm the app performs one real `denoiser` execute.
5. Confirm the app writes `denoiser_output.raw`.
6. Confirm profiling evidence exists.
7. Confirm malformed runtime inputs surface explicit execute-stage failures.
8. Confirm the UI shows `Session created: true` and `QNN active: true`.

## Acceptance Evidence

- Screenshot of the completed UI run
- The `denoiser_output.raw` path shown by the app
- The profiling CSV or QNN log
- A note confirming CPU fallback was disabled for the validation run

## Remaining Gap

该清单只证明 Android `denoiser-first` 本地 runtime loop。

它仍不证明：

- text encoder 已进入 Android runtime loop
- VAE decode 已进入 Android runtime loop
- 完整 prompt-to-image 已在目标设备上闭环
