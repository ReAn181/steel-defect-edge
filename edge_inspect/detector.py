"""detector.py - YOLOv8n ONNX detector (ONNX Runtime; GPU via CUDA provider if present, else CPU).

Python 3.8+ compatible. Needs: numpy, opencv-python (cv2), onnxruntime(-gpu).
Per-stage timings (pre / inference / post) are returned for the benchmark.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def letterbox(img, size: int = 416, pad_value: int = 114):
    """Resize keeping aspect ratio onto a size x size canvas. Returns (canvas, ratio, (pad_x, pad_y))."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, r, (px, py)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def postprocess(raw: np.ndarray, n_classes: int, conf: float, iou: float,
                ratio: float, pad: Tuple[int, int], orig_hw: Tuple[int, int]) -> List[dict]:
    """raw: YOLOv8 head output (1, 4+nc, N) or (1, N, 4+nc). Returns detections in ORIGINAL image pixels."""
    p = raw[0]
    if p.shape[0] < p.shape[1]:       # (4+nc, N) -> (N, 4+nc)
        p = p.T
    boxes_cxcywh, scores = p[:, :4], p[:, 4:4 + n_classes]
    cls = scores.argmax(1)
    conf_all = scores.max(1)
    m = conf_all >= conf
    if not m.any():
        return []
    b, c, s = boxes_cxcywh[m], cls[m], conf_all[m]
    xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
    keep = nms_numpy(xyxy + c[:, None] * 4096.0, s, iou)      # class-aware NMS via coordinate offset
    h0, w0 = orig_hw
    out = []
    for i in keep:
        x1 = (xyxy[i, 0] - pad[0]) / ratio
        y1 = (xyxy[i, 1] - pad[1]) / ratio
        x2 = (xyxy[i, 2] - pad[0]) / ratio
        y2 = (xyxy[i, 3] - pad[1]) / ratio
        out.append({"class_id": int(c[i]), "confidence": float(s[i]),
                    "xyxy": [float(np.clip(x1, 0, w0)), float(np.clip(y1, 0, h0)),
                             float(np.clip(x2, 0, w0)), float(np.clip(y2, 0, h0))]})
    return out


class OnnxYoloDetector:
    def __init__(self, onnx_path: str, class_names: Sequence[str], imgsz: int = 416,
                 conf: float = 0.25, iou: float = 0.5, prefer_gpu: bool = True,
                 intra_threads: Optional[int] = None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if intra_threads:
            so.intra_op_num_threads = intra_threads
        avail = ort.get_available_providers()
        providers = [p for p in (["CUDAExecutionProvider"] if prefer_gpu else []) if p in avail] + ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = list(class_names)
        self.imgsz, self.conf, self.iou = imgsz, conf, iou

    def infer(self, bgr: np.ndarray) -> Tuple[List[dict], Dict[str, float]]:
        t0 = time.perf_counter()
        canvas, ratio, pad = letterbox(bgr, self.imgsz)
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        x = np.ascontiguousarray(x)
        t1 = time.perf_counter()
        raw = self.session.run(None, {self.input_name: x})[0]
        t2 = time.perf_counter()
        dets = postprocess(raw, len(self.class_names), self.conf, self.iou, ratio, pad, bgr.shape[:2])
        for d in dets:
            d["class_name"] = self.class_names[d["class_id"]]
        t3 = time.perf_counter()
        return dets, {"pre_ms": (t1 - t0) * 1e3, "infer_ms": (t2 - t1) * 1e3, "post_ms": (t3 - t2) * 1e3}

    def warmup(self, n: int = 5):
        dummy = np.full((200, 200, 3), 128, np.uint8)
        for _ in range(n):
            self.infer(dummy)


def prepare_camera_frame(frame_bgr: np.ndarray, size: int = 200, to_gray: bool = True) -> np.ndarray:
    """Make a live-camera frame look like a NEU-DET crop: centre square crop -> size x size -> grey (3 channels).
    NEU-DET images are grey-looking 200x200 steel crops, so this reduces (but does not remove) the domain gap."""
    h, w = frame_bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = cv2.resize(frame_bgr[y0:y0 + s, x0:x0 + s], (size, size), interpolation=cv2.INTER_AREA)
    if to_gray:
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return crop
