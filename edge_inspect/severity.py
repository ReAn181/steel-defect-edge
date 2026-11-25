"""severity.py - documented, rule-based defect severity + vision->Llama hand-off.

Severity (Low / Medium / High) comes from THREE things only: defect TYPE, SIZE and LOCATION.
Detector confidence is never used in the score ("confidence is not severity").

Coordinate system (same for every image, 200x200 crops of the steel strip):
    x : 0 -> 1 left to right  = ACROSS the strip width (cross direction)
    y : 0 -> 1 top to bottom  = ALONG the rolling direction
    The crop is divided into a 3x3 grid. Zone code = column letter + row number:
        A1 B1 C1
        A2 B2 C2      A and C are the strip-edge columns, B is the centre column.
        A3 B3 C3
Python 3.8+ compatible (runs on JetPack 5 system Python).
"""
from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

SEV_ORDER = ["Low", "Medium", "High"]
COLS = ["A", "B", "C"]

DEFAULT_CONFIG = {
    "version": "1.0",
    "application": "hot-rolled steel strip surface inspection (NEU-DET classes)",
    "coordinate_system": (
        "x runs across the strip width (left to right), y runs along the rolling "
        "direction (top to bottom); both normalised 0-1; zones A1..C3 on a 3x3 grid"
    ),
    # score = w_type*type + w_size*size + w_location*location   (each part is 0..1)
    "weights": {"type": 0.5, "size": 0.3, "location": 0.2},
    # score < medium -> Low ; medium <= score < high -> Medium ; score >= high -> High
    "thresholds": {"medium": 0.45, "high": 0.65},
    # DESIGN ASSUMPTIONS - tune with your supervisor / a domain expert and document the change.
    "classes": {
        "crazing": {"type_score": 0.90, "meaning": "network of fine surface cracks that can spread under forming"},
        "inclusion": {"type_score": 0.80, "meaning": "foreign material embedded in the surface layer"},
        "rolled-in_scale": {"type_score": 0.60, "meaning": "oxide scale pressed into the surface during rolling"},
        "pitted_surface": {"type_score": 0.60, "meaning": "small local pits where surface material is lost"},
        "scratches": {"type_score": 0.40, "meaning": "linear mechanical marks, usually cosmetic unless deep or long"},
        "patches": {"type_score": 0.30, "meaning": "irregular discoloured or uneven areas, mostly cosmetic"},
    },
    # [upper bound of box area as a fraction of the image (exclusive), size score]
    "size_bands": [[0.02, 0.2], [0.08, 0.5], [0.20, 0.8], [1.01, 1.0]],
    # zone_weights[row][col]; edge columns count more than the centre column
    "zone_weights": [[1.0, 0.5, 1.0], [1.0, 0.5, 1.0], [1.0, 0.5, 1.0]],
    "zone_overlap_min": 0.25,   # a box "covers" a zone if it fills >= 25% of that zone
    "edge_margin": 0.05,        # box within 5% of the left/right border = touches the strip edge
    "count_escalation": {"min_defects": 4, "enabled": True},   # many defects -> image severity +1 level
    "advise_min_severity": "Low",   # run Llama when image severity >= this
}


def load_config(path: Optional[str] = None) -> dict:
    """Return DEFAULT_CONFIG, overridden key-by-key by a JSON file if given."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path and Path(path).exists():
        user = json.loads(Path(path).read_text())
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def size_score(area_frac: float, cfg: dict) -> float:
    for upper, score in cfg["size_bands"]:
        if area_frac < upper:
            return float(score)
    return 1.0


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def normalise_box(xyxy, img_w: int, img_h: int):
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return [_clip01(x1 / img_w), _clip01(y1 / img_h), _clip01(x2 / img_w), _clip01(y2 / img_h)]


def zone_code(col: int, row: int) -> str:
    return "%s%d" % (COLS[col], row + 1)


def zones_covered(box_n, cfg: dict) -> List[str]:
    """Zones the box meaningfully covers (>= zone_overlap_min of the zone) + the zone of its centre."""
    x1, y1, x2, y2 = box_n
    out = []
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    centre = (min(int(cx * 3), 2), min(int(cy * 3), 2))
    for r in range(3):
        for c in range(3):
            zx1, zx2, zy1, zy2 = c / 3.0, (c + 1) / 3.0, r / 3.0, (r + 1) / 3.0
            iw = max(0.0, min(x2, zx2) - max(x1, zx1))
            ih = max(0.0, min(y2, zy2) - max(y1, zy1))
            frac = (iw * ih) / ((1 / 3.0) * (1 / 3.0))
            if frac >= cfg["zone_overlap_min"] or (c, r) == centre:
                out.append(zone_code(c, r))
    return out


def location_score(zones: List[str], cfg: dict) -> float:
    w = cfg["zone_weights"]
    return max(float(w[int(z[1]) - 1][COLS.index(z[0])]) for z in zones)


def location_text(zone: str) -> str:
    col = {"A": "left strip edge", "B": "strip centre", "C": "right strip edge"}[zone[0]]
    row = {"1": "upper part", "2": "middle", "3": "lower part"}[zone[1]]
    if zone == "B2":
        return "centre of the crop"
    return "%s, %s of the crop" % (col, row)


def severity_from_score(score: float, cfg: dict) -> str:
    if score >= cfg["thresholds"]["high"]:
        return "High"
    if score >= cfg["thresholds"]["medium"]:
        return "Medium"
    return "Low"


def score_defect(det: dict, img_w: int, img_h: int, cfg: dict) -> dict:
    """det = {'class_name', 'confidence', 'xyxy':[x1,y1,x2,y2] in pixels}. Returns det + severity fields."""
    name = det["class_name"]
    if name not in cfg["classes"]:
        raise KeyError("class %r missing from severity config" % name)
    box_n = normalise_box(det["xyxy"], img_w, img_h)
    area = max(0.0, box_n[2] - box_n[0]) * max(0.0, box_n[3] - box_n[1])
    cx, cy = (box_n[0] + box_n[2]) / 2.0, (box_n[1] + box_n[3]) / 2.0
    zones = zones_covered(box_n, cfg)
    main_zone = zone_code(min(int(cx * 3), 2), min(int(cy * 3), 2))
    t = float(cfg["classes"][name]["type_score"])
    s = size_score(area, cfg)
    loc = location_score(zones, cfg)
    w = cfg["weights"]
    score = w["type"] * t + w["size"] * s + w["location"] * loc
    m = cfg["edge_margin"]
    return {
        "class": name,
        "confidence": round(float(det["confidence"]), 3),
        "severity": severity_from_score(score, cfg),
        "score": round(score, 3),
        "parts": {"type": round(t, 2), "size": round(s, 2), "location": round(loc, 2)},
        "bbox_xyxy": [round(float(v), 1) for v in det["xyxy"]],
        "bbox_norm": [round(v, 3) for v in box_n],
        "centre_norm": [round(cx, 3), round(cy, 3)],
        "area_pct": round(area * 100.0, 2),
        "zone": main_zone,
        "zones_covered": zones,
        "location_text": location_text(main_zone),
        "touches_strip_edge": bool(box_n[0] <= m or box_n[2] >= 1.0 - m),
        "meaning": cfg["classes"][name]["meaning"],
    }


def _bump(sev: str) -> str:
    return SEV_ORDER[min(SEV_ORDER.index(sev) + 1, 2)]


def build_handoff(image_id: str, detections: List[dict], img_w: int, img_h: int,
                  cfg: dict, conf_min: float = 0.0, timestamp: Optional[str] = None) -> dict:
    """The ONE JSON hand-off between vision and Llama (schema_version 1.0)."""
    scored = [score_defect(d, img_w, img_h, cfg) for d in detections if d["confidence"] >= conf_min]
    scored.sort(key=lambda d: (d["score"], d["confidence"]), reverse=True)
    image_sev, escalated = "None", False
    primary, others = None, []
    if scored:
        primary = scored[0]
        image_sev = max((d["severity"] for d in scored), key=SEV_ORDER.index)
        ce = cfg["count_escalation"]
        if ce["enabled"] and len(scored) >= ce["min_defects"]:
            bumped = _bump(image_sev)
            escalated = bumped != image_sev
            image_sev = bumped
        counts: Dict[str, int] = {}
        for d in scored[1:]:
            counts[d["class"]] = counts.get(d["class"], 0) + 1
        others = [{"class": k, "count": v} for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
    needs = bool(scored) and SEV_ORDER.index(image_sev) >= SEV_ORDER.index(cfg["advise_min_severity"])
    return {
        "schema_version": "1.0",
        "image_id": image_id,
        "timestamp": timestamp or datetime.now().isoformat(timespec="seconds"),
        "image_size": {"w": int(img_w), "h": int(img_h)},
        "num_defects": len(scored),
        "image_severity": image_sev,
        "severity_escalated_by_count": escalated,
        "primary_defect": primary,          # only the major one goes to Llama
        "other_defects": others,            # compact summary of the rest
        "all_defects": scored,              # full list stays in the local log
        "needs_advice": needs,
        "context": {
            "application": cfg["application"],
            "coordinate_system": cfg["coordinate_system"],
            "severity_rule_version": cfg["version"],
        },
    }
