Place the custom `onnxruntime-android-qnn.aar` and the Qualcomm runtime `.so` files required for `arm64-v8a` in this directory before building the Android denoiser-first runtime loop.

Required contents for the first device run:

- `onnxruntime-android-qnn.aar`
- `arm64-v8a/libQnnHtp.so`
- `arm64-v8a/libQnnSystem.so`
- `arm64-v8a/libQnnHtpV75Stub.so` or the matching stub for the target SDK release

Required runtime payload outside this directory for the first Android denoiser loop:

- app runtime file: `filesDir/runtime/denoiser_ctx.onnx`
- app runtime file: `filesDir/runtime/denoiser_qnn.bin`
- app runtime file: `filesDir/runtime/inputs/latent.raw`
- app runtime file: `filesDir/runtime/inputs/timestep.raw`
- app runtime file: `filesDir/runtime/inputs/cond.raw`
- app runtime file: `filesDir/runtime/inputs/uncond.raw`

The current Android loop assumes the denoiser artifacts already exist and uses fixed or precomputed inputs. It does not yet run the full text encoder or VAE pipeline on-device.
