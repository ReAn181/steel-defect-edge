"""pipeline.py - image folder -> detector -> severity -> hand-off JSON -> Llama advice -> local log (+ tracker).

Everything runs locally/offline. The only network call is an OPTIONAL POST to the tracker on localhost.
Python 3.8+ compatible.
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from .llama_advisor import advise
from .severity import build_handoff

SEV_BGR = {"Low": (107, 124, 62), "Medium": (27, 138, 201), "High": (46, 64, 179)}   # teal / amber / oxide red (BGR)


def annotate(img, handoff: dict):
    out = img.copy()
    for d in handoff["all_defects"]:
        x1, y1, x2, y2 = [int(v) for v in d["bbox_xyxy"]]
        col = SEV_BGR[d["severity"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, "%s %s %.2f" % (d["class"], d["severity"][0], d["confidence"]), (x1, max(10, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return out


def post_json(url: str, payload: dict, timeout: float = 1.5) -> bool:
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=timeout).read()
        return True
    except Exception:
        return False        # tracker is optional; never break inspection because of it


def process_image(path: Path, detector, cfg: dict, backend=None, use_llama: bool = True,
                  conf_min: Optional[float] = None) -> dict:
    t_all = time.perf_counter()
    img = cv2.imread(str(path))
    if img is None:
        raise IOError("cannot read %s" % path)
    t_read = time.perf_counter()
    dets, tm = detector.infer(img)
    t_sev0 = time.perf_counter()
    h, w = img.shape[:2]
    handoff = build_handoff(Path(path).name, dets, w, h, cfg, conf_min=conf_min if conf_min is not None else detector.conf)
    t_sev1 = time.perf_counter()
    advice, llama_ms = None, 0.0
    if handoff["needs_advice"]:
        t0 = time.perf_counter()
        advice = advise(handoff, backend if use_llama else None)
        llama_ms = (time.perf_counter() - t0) * 1e3
    elif handoff["num_defects"] == 0:
        advice = advise(handoff, None)
    t_end = time.perf_counter()
    vision_ms = tm["pre_ms"] + tm["infer_ms"] + tm["post_ms"]
    return {
        "handoff": handoff, "advice": advice, "image": img,
        "timing_ms": {"read": round((t_read - t_all) * 1e3, 2), "pre": round(tm["pre_ms"], 2),
                      "infer": round(tm["infer_ms"], 2), "post": round(tm["post_ms"], 2),
                      "severity": round((t_sev1 - t_sev0) * 1e3, 3), "vision_total": round(vision_ms, 2),
                      "llama_total": round(llama_ms, 1), "end_to_end": round((t_end - t_all) * 1e3, 1)},
    }


def run_folder(folder: str, detector, cfg: dict, out_root: str = "runs", backend=None, use_llama: bool = True,
               tracker_url: Optional[str] = None, limit: Optional[int] = None, save_images: bool = True) -> dict:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in exts)
    if limit:
        files = files[:limit]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_root) / run_id
    (out_dir / "annotated").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "inspections.jsonl"
    records: List[dict] = []
    with open(log_path, "a") as log:
        for p in files:
            r = process_image(p, detector, cfg, backend, use_llama)
            ann_rel = None
            if save_images:
                ann_path = out_dir / "annotated" / (p.stem + "_ann.jpg")
                cv2.imwrite(str(ann_path), annotate(r["image"], r["handoff"]))
                ann_rel = str(ann_path.relative_to(out_root))
            rec = {"run_id": run_id, "handoff": r["handoff"], "advice": r["advice"],
                   "timing_ms": r["timing_ms"], "annotated_image": ann_rel, "source_image": str(p)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if tracker_url:
                post_json(tracker_url.rstrip("/") + "/api/inspections", rec)
            records.append(rec)
    return {"run_id": run_id, "out_dir": str(out_dir), "log": str(log_path), "records": records}
