# QNN Smoke Tests

该目录负责以最低成本验证 Qualcomm `QNN/QAIRT` 工具链的可用性，是整个项目验证梯度的最底层。

## 在项目中的位置

- 项目总览：`../docs/project/index.md`
- 项目架构：`../docs/project/architecture.md`
- 模块状态：`../docs/project/status.md`
- 验证体系：`../docs/project/validation.md`

## 目标

按成本从低到高验证：

1. `toy` graph
2. `mini_block`
3. 只有在前两者足够稳定后，才进入真实 `Anima` 子图

## First Manual Run Order

1. Export `toy.onnx`
2. Run CPU `ORT` on `toy.onnx`
3. Compile `toy.onnx` with `QNN`
4. Run host-side baseline verification
5. Repeat for `mini_block.onnx`
6. Update `smoke/docs/smoke-matrix.md`
