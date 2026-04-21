# 需求与范围

## 总体目标

- 在 Android Snapdragon 手机上验证 `Anima` 关键推理路径的本地执行能力
- 优先证明 `denoiser-first` 路径的 `ORT + QNNExecutionProvider` 集成
- 用逐层验证结果指导真实模型导出与系统扩展

## 当前范围

当前项目范围包含：

- host 侧 `QNN/QAIRT` smoke 验证
- 桌面 `ORT + QNNExecutionProvider` 验证
- Android `denoiser-first` session / execute 路径
- 参考 pipeline 与导出辅助逻辑
- 真机验证准备与文档体系建设

## 当前非目标

- 产品化 UI
- 广泛设备兼容性
- 一步到位的完整 prompt-to-image Android 闭环
- 所有子模型同时迁移到 Android
- 最终性能调优与发布级工程化

## 当前关键要求

- 保持阶段性验证边界清晰
- 所有关键结论应尽量有证据支撑
- Android 端运行失败要有明确分类
- 文档应支持开发指导与进度追踪

## 当前关键风险

- 真实 `Anima` 导出后可能存在 operator gap
- Android 端真机 NPU/HTP 证据链尚未闭环
- text encoder / VAE 仍未进入 Android runtime loop
