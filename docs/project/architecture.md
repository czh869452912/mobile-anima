# 架构

## 总体模块划分

当前项目可按五个主要部分理解：

1. `smoke/`
   - 负责最低成本 QNN smoke 验证
2. `ort_qnn/`
   - 负责桌面 `ORT + QNNExecutionProvider` 验证与打包
3. `android/`
   - 负责 Android 端运行时集成与 `denoiser-first` 执行路径
4. `python/`
   - 负责参考 pipeline、导出辅助与 host 侧逻辑
5. `docs/`
   - 负责项目文档、手工验证说明、设计与计划痕迹

## Android 当前架构

Android 当前 runtime 主要由以下部分组成：

- `OrtJavaBridge`
  - 负责连接 `ORT Java API`
- `OrtSessionFactory`
  - 负责 provider / session / execute 边界
- `OrtQnnDenoiserEngine`
  - 负责 Android runtime 编排
- `DenoiserInputBinding`
  - 负责 metadata-first 的严格输入绑定
- `ArtifactManager`
  - 负责运行时文件路径与输入输出位置管理

## 当前运行路径

当前 Android 端已经具备一条 `denoiser-first` 路径：

- session 创建
- input metadata 读取
- raw → tensor 绑定
- 一次 `denoiser` execute
- output raw 落盘

## 当前未闭合部分

- text encoder 未进入 Android runtime loop
- VAE decode 未进入 Android runtime loop
- prompt 到完整图像的链路仍未闭合
- 真机验证证据链仍未正式固化
