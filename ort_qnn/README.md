# Desktop ORT + QNN EP Validation

该目录负责验证桌面 `ONNX Runtime + QNNExecutionProvider` 路径，是 Android 端 `ORT + QNN` 集成之前的上层基线。

## 在项目中的位置

- 项目总览：`../docs/project/index.md`
- 项目架构：`../docs/project/architecture.md`
- 路线图：`../docs/project/roadmap.md`
- 模块状态：`../docs/project/status.md`
- 验证体系：`../docs/project/validation.md`

## 目标

验证：

1. `QNNExecutionProvider` 可见
2. `toy` 可创建 session 并 execute
3. `mini_block` 可创建 session 并 execute

## First Manual Run Order

1. Check or build a usable Linux `ORT + QNN EP`
2. Confirm `QNNExecutionProvider` visibility
3. Validate `toy`
4. Validate `mini_block`
5. Update `ort_qnn/docs/ort-qnn-matrix.md`
