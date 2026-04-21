# 模块状态

| Module | Goal | Current Status | Evidence | Next Step |
| --- | --- | --- | --- | --- |
| `smoke/` | 验证最小图 QNN compile / run | `Validated` | `smoke/README.md`, `smoke/docs/smoke-matrix.md` | 作为真实模型落地前的最低层基线 |
| `ort_qnn/` | 验证桌面 `ORT + QNNExecutionProvider` | `Validated` | `ort_qnn/README.md`, `ort_qnn/docs/ort-qnn-matrix.md` | 作为 Android / real model 排障基线 |
| `android/` | 验证 `denoiser-first` Android runtime | `In Progress` | `docs/manual/android-validation.md`, Android runtime tests | 补齐真机 / `adb` 证据链 |
| `python/` | 支撑参考 pipeline 与导出路径 | `In Progress` | `python/`, existing tests | 推进真实 `Anima` 导出与输入契约整理 |
| `real model export` | 导出真实 `Anima` 子图并做 operator gap 审查 | `Not Started` | `docs/project/roadmap.md`, `README.md` | 进行真实模型导出与 gap 审查 |
| `documentation` | 提供稳定的人类主入口与开发跟踪体系 | `In Progress` | `README.md`, `docs/project/` | 完成文档体系重构并保持同步 |
