"""settings.py - one place for file locations. Edit settings.json (created next to the notebooks' parent folder)."""
from __future__ import annotations

import json
from pathlib import Path

DEFAULTS = {
    "onnx": "deploy/yolov8n_neu_combined.onnx",
    "meta": "deploy/model_meta.json",
    "demo_images": "deploy/demo_images",
    "severity_config": "severity_config.json",
    "llama_main": "bin/llama_main",                       # copied there by scripts/build_llama_runner_jetson.sh
    "llama_model": "llama_deploy/llama32_1b_instruct_8da4w.pte",
    "llama_tokenizer": "llama_deploy/tokenizer.model",
    "runs_dir": "runs",
    "tracker_url": "http://127.0.0.1:5000",
}


def repo_root() -> Path:
    cwd = Path.cwd().resolve()
    return cwd.parent if cwd.name == "notebooks" else cwd


def load(root: Path = None) -> dict:
    root = Path(root) if root else repo_root()
    cfg = dict(DEFAULTS)
    f = root / "settings.json"
    if f.exists():
        cfg.update(json.loads(f.read_text()))
    out = {}
    for k, v in cfg.items():
        if k == "tracker_url":
            out[k] = v
        else:
            p = Path(v).expanduser()
            out[k] = str(p if p.is_absolute() else root / p)
    out["root"] = str(root)
    return out
