# QNN Smoke Tests

This subproject validates Qualcomm compile and runtime feasibility in the cheapest order:

1. Toy graph
2. Mini transformer-like block
3. Real `Anima` denoiser only after the first two are understood

## Preflight Checklist

- Qualcomm `QNN/QAIRT SDK` is installed or download access is confirmed
- `QPM` login / entitlement / license activation completed if required
- Backend library path is known
- `ONNX Runtime` with `QNN EP` is available on the host
- `smoke/docs/smoke-matrix.md` is updated after every run
