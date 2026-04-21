# 验证体系

## Validation Ladder

1. ONNX export 与 CPU runtime 正确性
2. QAIRT / QNN compile 与 host runtime 正确性
3. 桌面 `ORT + QNNExecutionProvider` provider / session / execute 正确性
4. Android `denoiser-first` session + execute 正确性
5. 真机侧 profiling / HTP 证据闭环
6. 真实 `Anima` 模型导出与 operator gap 审查

## 每一层证明什么

- `smoke/`
  - 证明最小图和工具链基本可用
- `ort_qnn/`
  - 证明 `ORT + QNNExecutionProvider` 的桌面路径可用
- `android/`
  - 证明 Android 本地 runtime 路径存在且可以执行 `denoiser-first`
- 真机验证
  - 证明目标 Snapdragon 设备上的 NPU/HTP 实际参与推理
- 真实模型导出与 gap 审查
  - 证明项目是否有机会从验证样例推进到真实 `Anima`

## 当前定位

当前项目已经完成前 3 层，并在第 4 层取得明显进展，但距离第 5 层和第 6 层仍有关键缺口。
