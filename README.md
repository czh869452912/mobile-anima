# Mobile Anima NPU

面向 Qualcomm Snapdragon Android 设备的 `Anima` 本地推理实验项目。

当前项目采用“分层验证、逐步收敛风险”的策略，优先证明最昂贵、最关键的 `denoiser` 路径能够通过 `QNN/HTP` 在 Android 上本地执行，再逐步推进真实 `Anima` 模型导出、真机证据闭环，以及更完整的 prompt-to-image 流程。

## 项目目标

- **最终目标**：在 Android 的 Snapdragon 手机上，以 NPU/HTP 运行 `Anima` 图片生成关键路径
- **当前主线**：验证 `denoiser-first` 的 Android 本地 `ORT + QNNExecutionProvider` 路径
- **当前原则**：先证明最小可行路径，再扩大模型与系统边界

## 当前进展

- `smoke/`：最小图的导出、QNN 编译与 host 运行已验证
- `ort_qnn/`：桌面 `ORT + QNNExecutionProvider` 的 provider / session / execute 已验证
- `android/`：Android 端已具备 `ORT + QNN` session、metadata 驱动执行、严格输入绑定与单次 `denoiser` execute 路径
- `python/`：已有参考 pipeline、导出辅助与 host 侧工具雏形
- 真实 `Anima` 导出、operator gap 审查、真机 NPU 证据链仍是后续关键工作

## 推荐阅读顺序

如果你是第一次进入这个仓库，推荐按以下顺序阅读：

1. `docs/project/index.md` — 项目文档入口与阅读地图
2. `docs/project/overview.md` — 项目背景、目标与当前策略
3. `docs/project/requirements.md` — 当前范围、非目标与阶段边界
4. `docs/project/architecture.md` — 系统架构与模块职责
5. `docs/project/roadmap.md` — 里程碑路线图
6. `docs/project/status.md` — 按模块跟踪当前状态、证据与下一步
7. `docs/project/validation.md` — 验证梯度与每一层证明内容

## 模块文档

- `smoke/README.md` — 最低成本 QNN smoke 验证
- `ort_qnn/README.md` — 桌面 `ORT + QNNExecutionProvider` 验证
- `docs/manual/android-validation.md` — Android `denoiser-first` 手工验证清单

## 仓库结构

- `android/` — Android app 与运行时集成
- `python/` — 导出、参考 pipeline 与 host 辅助逻辑
- `smoke/` — 最小图 smoke 验证
- `ort_qnn/` — 桌面 `ORT + QNNExecutionProvider` 验证与打包
- `docs/project/` — 正式项目文档（人类主入口）
- `docs/manual/` — 操作型 runbook / checklist
- `docs/superpowers/` — 设计 spec 与实现 plan 的历史工作痕迹
- `scripts/` — 主机环境与构建辅助脚本

## 当前主要风险

- 真实 `Anima` 模型导出后可能存在 operator gap
- Android 端虽然已有本地 execute 路径，但真机 NPU 证据链尚未固化
- text encoder / VAE 仍未进入 Android runtime loop

## 下一里程碑

- 推进真实 `Anima` 导出与 operator gap 审查
- 补齐 Android 真机 / `adb` 验证路径与 profiling 证据闭环
- 在当前 `denoiser-first` 基础上继续决定是否扩展更多子模型上机

## 说明

`docs/superpowers/` 中保留了设计与实施过程中的 `spec` / `plan` 文档，它们仍有参考价值，但不再是理解项目现状的主要入口。
