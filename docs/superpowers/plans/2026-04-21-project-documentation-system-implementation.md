# Project Documentation System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current progress-update-style documentation with a formal project documentation system centered on a rewritten `README.md` and a new `docs/project/` hierarchy.

**Architecture:** Keep `docs/superpowers/` as historical agent work artifacts, but introduce `docs/project/` as the human-first documentation system. Rewrite the root `README.md` into a durable project entrypoint, add a module-oriented status tracker with evidence and next steps, and align key module docs (`smoke/README.md`, `ort_qnn/README.md`, `docs/manual/android-validation.md`) to the new navigation model.

**Tech Stack:** Markdown documentation, repository-local navigation, module status matrix, Chinese-first writing with preserved English technical terms and file paths

---

### File Structure

**Formal project docs to create:**
- `docs/project/index.md` — project documentation map and reading order
- `docs/project/overview.md` — project mission, target platform, main constraints
- `docs/project/requirements.md` — current scope, goals, non-goals, milestone boundaries
- `docs/project/architecture.md` — system architecture and major subsystem split
- `docs/project/roadmap.md` — milestone-based path from current state to target
- `docs/project/status.md` — module-based status table with evidence and next step
- `docs/project/validation.md` — validation ladder and what each stage proves

**Existing docs to modify:**
- `README.md`
- `smoke/README.md`
- `ort_qnn/README.md`
- `docs/manual/android-validation.md`

### Task 1: Rewrite the root `README.md` as the human entrypoint

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the new README content in a scratch file**

```markdown
# Mobile Anima NPU

面向 Qualcomm Snapdragon Android 设备的 `Anima` 本地推理实验项目，当前目标是逐步验证：

- 主机侧 `QNN/QAIRT` 工具链可用
- 桌面 `ORT + QNNExecutionProvider` 可用
- Android 端可以以 `denoiser-first` 路径完成本地 `ORT + QNN` 执行
- 在这些基础上，再推进到真实 `Anima` 模型导出、算子差距分析与真机验证

## 项目目标

- 最终目标：在 Android 的 Snapdragon 手机上，以 NPU/HTP 运行 `Anima` 图片生成关键路径
- 当前策略：优先验证最贵的 `denoiser` 路径，而不是一开始就做完整 prompt-to-image
- 当前非目标：产品化 UI、全模型端到端上机、广泛设备兼容性

## 当前进展

- `smoke/`：最小图的导出、编译、host 运行已验证
- `ort_qnn/`：桌面 `ORT + QNNExecutionProvider` 的 provider/session/execute 已验证
- `android/`：Android 端已具备 `ORT + QNN` session、metadata 驱动执行、严格输入绑定与单次 denoiser execute 路径
- 真实 `Anima` 导出与 operator-gap 仍是后续关键风险

## 从哪里开始阅读

- 项目总览：`docs/project/index.md`
- 项目概览：`docs/project/overview.md`
- 当前需求与范围：`docs/project/requirements.md`
- 架构：`docs/project/architecture.md`
- 路线图：`docs/project/roadmap.md`
- 模块状态：`docs/project/status.md`
- 验证体系：`docs/project/validation.md`

## 模块文档

- `smoke/README.md`
- `ort_qnn/README.md`
- `docs/manual/android-validation.md`

## Agent 工作痕迹

- 设计与实施历史保留在 `docs/superpowers/`
- 这些文件仍可参考，但不再是项目主入口
```

- [ ] **Step 2: Replace `README.md` with the new content**

```bash
cat > README.md <<'EOF'
# Mobile Anima NPU

面向 Qualcomm Snapdragon Android 设备的 `Anima` 本地推理实验项目，当前目标是逐步验证：

- 主机侧 `QNN/QAIRT` 工具链可用
- 桌面 `ORT + QNNExecutionProvider` 可用
- Android 端可以以 `denoiser-first` 路径完成本地 `ORT + QNN` 执行
- 在这些基础上，再推进到真实 `Anima` 模型导出、算子差距分析与真机验证

## 项目目标

- 最终目标：在 Android 的 Snapdragon 手机上，以 NPU/HTP 运行 `Anima` 图片生成关键路径
- 当前策略：优先验证最贵的 `denoiser` 路径，而不是一开始就做完整 prompt-to-image
- 当前非目标：产品化 UI、全模型端到端上机、广泛设备兼容性

## 当前进展

- `smoke/`：最小图的导出、编译、host 运行已验证
- `ort_qnn/`：桌面 `ORT + QNNExecutionProvider` 的 provider/session/execute 已验证
- `android/`：Android 端已具备 `ORT + QNN` session、metadata 驱动执行、严格输入绑定与单次 denoiser execute 路径
- 真实 `Anima` 导出与 operator-gap 仍是后续关键风险

## 从哪里开始阅读

- 项目总览：`docs/project/index.md`
- 项目概览：`docs/project/overview.md`
- 当前需求与范围：`docs/project/requirements.md`
- 架构：`docs/project/architecture.md`
- 路线图：`docs/project/roadmap.md`
- 模块状态：`docs/project/status.md`
- 验证体系：`docs/project/validation.md`

## 模块文档

- `smoke/README.md`
- `ort_qnn/README.md`
- `docs/manual/android-validation.md`

## Agent 工作痕迹

- 设计与实施历史保留在 `docs/superpowers/`
- 这些文件仍可参考，但不再是项目主入口
