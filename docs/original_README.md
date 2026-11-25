# Steel surface defect inspection on Jetson Xavier NX: YOLOv8n + Llama 3.2 (ExecuTorch)

Offline pipeline: **detect** (YOLOv8n, ONNX Runtime) -> **severity** (Low/Medium/High from type, size, location) -> **hand-off JSON** -> **advise + log** (Llama 3.2 1B Instruct on ExecuTorch) -> local log + optional Flask tracker.

Run the notebooks in order (open them from the `notebooks/` folder):

| # | Notebook | Where | What |
|---|---|---|---|
| 00 | `00_save_best_and_export_yolo` | Colab | save the best model, metrics, deploy confidence, ONNX export |
| 01 | `01_export_llama_executorch` | Colab | Llama 3.2 1B Instruct -> 4-bit `.pte` |
| 02 | `02_jetson_setup_and_check` | Jetson | build `llama_main`, install packages, health check |
| 03 | `03_severity_and_handoff` | anywhere | severity rule, coordinate grid, hand-off JSON, tests |
| 04 | `04_llama_advisor` | Jetson | prompt, JSON validation, fallback, Llama metrics |
| 05 | `05_pipeline_demo` | Jetson | folder of images -> annotated images + log |
| 06 | `06_benchmark` | Jetson | FPS, latency, RAM, power, P/R/F1, TTFT, tokens/s |

Tracker (separate): `cd tracker && python3 app.py --media ../runs` then open http://127.0.0.1:5000

Layout after setup: `deploy/` (from notebook 00), `llama_deploy/` (notebook 01), `bin/llama_main` (notebook 02), `runs/` (outputs), `severity_config.json`, `settings.json`.
Severity values are design assumptions: see notebook 03 and tune `severity_config.json`.
