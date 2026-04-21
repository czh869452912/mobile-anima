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
2. Start generation.
3. Confirm the app performs one real denoiser execute.
4. Confirm it writes `denoiser_output.raw`.
5. Confirm profiling evidence exists.
6. Confirm explicit execute-stage failures are surfaced when inputs are malformed.
