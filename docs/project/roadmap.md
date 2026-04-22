# 路线图

## Milestone 1: Smoke Validation

**目标**
- 证明最小图可以完成导出、QNN compile 与 host 运行

**当前状态**
- `Validated`

## Milestone 2: Desktop ORT + QNN

**目标**
- 证明桌面 `ORT + QNNExecutionProvider` 的 provider / session / execute 可用

**当前状态**
- `Validated`

## Milestone 3: Android Denoiser-First Runtime

**目标**
- 证明 Android 可以完成一次 `denoiser-first` 本地 execute

**当前状态**
- `In Progress`
- 代码路径与 JVM 测试已具备，真机证据链待完善

## Milestone 4: Real Anima Export and Gap Analysis

**目标**
- 推进真实 `Anima` 子图导出、量化与 operator gap 审查

**当前状态**
- `In Progress`
- Host 侧全功能导出脚手架已完成（model_loader / export_denoiser / export_text_encoder / export_vae / quantize / calibration / operator_gap / numerical_verify / export_pipeline / android_artifacts）
- 42 个 pytest 在无 GPU/无权重环境下全部通过
- 真实权重 ONNX 导出与 operator gap 报告待环境就绪后执行

## Milestone 5: Device-Proven NPU Path

**目标**
- 在目标 Snapdragon 手机上形成可复验的 NPU/HTP 执行证据

**当前状态**
- `Not Started`
