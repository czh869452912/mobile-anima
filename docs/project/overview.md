# 项目概览

## 项目是什么

`Mobile Anima NPU` 是一个面向 Qualcomm Snapdragon Android 设备的本地推理实验项目，目标是逐步把 `Anima` 的关键推理路径带到移动端，并最终形成可验证的 NPU/HTP 执行证据。

## 当前策略

当前采用分层验证策略，以最低成本依次收敛风险：

1. `smoke/`：验证最小图的 QNN compile / run 是否可行
2. `ort_qnn/`：验证桌面 `ORT + QNNExecutionProvider` 是否可行
3. `android/`：验证 Android `denoiser-first` 的本地 runtime 路径
4. 在以上路径足够稳定后，再进入真实 `Anima` 模型导出、算子差距分析与真机闭环

## 为什么不是一开始就做完整出图

完整 prompt-to-image 同时包含：

- text encoder
- denoiser / transformer
- scheduler / orchestration
- VAE decode
- Android runtime / packaging / provider / profiling

如果这些部分一起推进，失败面会过大。当前策略优先证明最贵、最关键、最可能决定 NPU 价值的 `denoiser` 路径。

## 当前项目状态

项目已经从“探索型脚手架”演进到“多模块协同验证”的阶段，需要正式文档来统一目标、架构、状态与路线图。
