# 模块状态

| Module | Goal | Current Status | Evidence | Next Step |
| --- | --- | --- | --- | --- |
| `smoke/` | 验证最小图 QNN compile / run | `Validated` | `smoke/README.md`, `smoke/docs/smoke-matrix.md` | 作为真实模型落地前的最低层基线 |
| `ort_qnn/` | 验证桌面 `ORT + QNNExecutionProvider` | `Validated` | `ort_qnn/README.md`, `ort_qnn/docs/ort-qnn-matrix.md` | 作为 Android / real model 排障基线 |
| `android/` | 验证 `denoiser-first` Android runtime | `In Progress` | `docs/manual/android-validation.md`, Android runtime tests | 补齐真机 / `adb` 证据链 |
| `python/` | 支撑参考 pipeline 与导出路径 | `In Progress` | `python/`, existing tests | 推进真实 `Anima` 导出与输入契约整理 |
| `python/model_loader` | Anima 模型加载适配（含无权重 fallback） | `Completed` | `python/src/anima_host/model_loader.py`, `tests/test_model_loader.py` | 待接入 DiffSynth 真实权重 |
| `python/export_denoiser` | Denoiser ONNX 导出（含 dry-run） | `Completed` | `python/src/anima_host/export_denoiser.py`, `tests/test_export_compile.py` | 待真实权重验证 |
| `python/export_text_encoder` | Text Encoder ONNX 导出 | `Completed` | `python/src/anima_host/export_text_encoder.py`, `tests/test_export_text_encoder.py` | 待真实权重验证 |
| `python/export_vae` | VAE Decoder ONNX 导出 | `Completed` | `python/src/anima_host/export_vae.py`, `tests/test_export_vae.py` | 待真实权重验证 |
| `python/quantize_denoiser` | QDQ 量化脚本 | `Completed` | `python/src/anima_host/quantize_denoiser.py`, `tests/test_quantize_denoiser.py` | 待 onnxruntime 环境 |
| `python/operator_gap` | QNN operator gap 分析器 | `Completed` | `python/src/anima_host/operator_gap.py`, `tests/test_operator_gap.py` | 待 onnx 安装后运行 |
| `python/calibration_data` | PTQ 校准数据生成 | `Completed` | `python/src/anima_host/calibration_data.py`, `tests/test_calibration_data.py` | 已可用 |
| `python/numerical_verify` | 数值验证工具 | `Completed` | `python/src/anima_host/numerical_verify.py`, `tests/test_numerical_verify.py` | 已可用 |
| `python/export_pipeline` | 端到端导出流水线 | `Completed` | `python/src/anima_host/export_pipeline.py`, `tests/test_export_pipeline.py` | 已可用 |
| `python/android_artifacts` | Android 产物打包 | `Completed` | `python/src/anima_host/android_artifacts.py`, `tests/test_android_artifacts.py` | 已可用 |
| `real model export` | 导出真实 `Anima` 子图并做 operator gap 审查 | `Not Started` | `docs/project/roadmap.md`, `README.md` | 进行真实模型导出与 gap 审查 |
| `documentation` | 提供稳定的人类主入口与开发跟踪体系 | `In Progress` | `README.md`, `docs/project/` | 完成文档体系重构并保持同步 |
