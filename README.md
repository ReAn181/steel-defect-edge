# EdgeSteel — Bharat AI SoC 2026–27

Offline steel surface defect detection and intelligent inspection on NVIDIA Jetson Orin Nano Super. YOLOv8n ONNX detection → rule-based severity → Llama 3.2 1B Instruct advisory or template fallback → local inspection tracker.

## Repository layout
- `edge_inspect/`: core Python modules
- `notebooks/`: notebooks 00–06, from export through benchmarking
- `deploy/`: trained ONNX and PyTorch models, validation metrics, plots and demo images
- `external_dataset/`: phone photographs, preprocessing notebooks, processed crops and review records; not an independently validated test set
- `scripts/`: runner build script
- `tracker/`: Flask application
- `docs/`: original setup references

## Execution
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


## Deployment notes
The original setup was written for Xavier NX; the demonstrated target was Orin Nano Super. Review notebook 02 and hardware dependencies for your installed JetPack before running. The original hardware PDF describes a TensorRT experiment, not the measured ONNX pipeline. Llama weights, tokenizer and compiled runner are not included; generate them using notebooks 01–02. Final Jetson run logs and benchmark JSON, full training notebooks, report and video link must be added separately. Verify dataset and model redistribution permissions before publishing. Full NEU-DET archive is not included. Original notebooks and code are preserved and have not been executed or verified on hardware in this packaging step.
