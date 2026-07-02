# Trained model artifacts

Weights produced by the legacy Cirsium YOLO target-detection baseline
(`legacy/target_detection/train_cirsium_yolo.py`), used by the legacy runtime
scripts under `legacy/runtime/` and `legacy/target_detection/`.

| File | Role |
|---|---|
| `best.pt` | best-checkpoint YOLO weights (PyTorch) |
| `last.pt` | final-epoch YOLO weights (PyTorch) |
| `best.onnx` | `best.pt` exported to ONNX for edge / IMX500 deployment (`legacy/target_detection/export_cirsium_imx.py`) |

These are baseline artifacts, not the output of the current
`interaction_sensing` package. Legacy scripts reference them by absolute local
paths (e.g. `C:\Users\zuizui\mc\runs\cirsium_detect\weights\best.pt` or
`/home/zuizui0223/visit_detect/cirsium_best.pt`); point those paths at the
copies here, or vice versa, when running a legacy script.
