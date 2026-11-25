"""llama_advisor.py - Llama 3.2 Instruct (ExecuTorch `llama_main` runner) explains the major defect.

Design (matches the competition hint): Llama NEVER sees pixels. It receives structured findings
(class, severity, size, location, confidence) and returns SHORT JSON: explanation, actions, log_entry.
Safety nets: JSON repair, a grounding check (no invented defect classes) and a deterministic template
fallback, so the pipeline always produces a log entry even if Llama fails.

Python 3.8+ compatible; stdlib only (psutil optional, used for peak-RAM sampling).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

ALL_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_ALIASES = {
    "crazing": ["crazing"], "inclusion": ["inclusion"], "patches": ["patch"],
    "pitted_surface": ["pitted", "pitting"], "rolled-in_scale": ["rolled-in", "rolled in"],
    "scratches": ["scratch"],
}
PREFILL = '{"explanation": "'

SYSTEM_PROMPT = (
    "You are an on-device assistant for steel strip surface inspection. "
    "You receive structured findings from a vision model; you never see the image. "
    "Use ONLY the findings given. Do not invent defects, numbers or locations. "
    "Reply with ONE JSON object and nothing else, with exactly these keys: "
    "\"explanation\" (at most 2 short sentences: what this defect is and why it matters at this severity and location), "
    "\"actions\" (list of 1 or 2 imperative steps, at most 12 words each), "
    "\"log_entry\" (one line, at most 25 words, starting with the image id)."
)

TEMPLATES = {
    "crazing": ("A network of fine surface cracks that can spread when the strip is formed.", ["Hold the strip section for crack inspection.", "Check rolling and cooling settings."]),
    "inclusion": ("Foreign material is embedded in the surface layer.", ["Mark the section for surface grinding or rejection.", "Check the upstream material source."]),
    "rolled-in_scale": ("Oxide scale was pressed into the surface during rolling.", ["Inspect descaling water pressure.", "Re-check the section after the next pass."]),
    "pitted_surface": ("Small pits show local loss of surface material.", ["Measure pit density on the section.", "Check roll surface condition."]),
    "scratches": ("Linear mechanical marks on the surface.", ["Inspect guides and rolls for contact points.", "Log length and direction."]),
    "patches": ("Irregular patches of uneven surface appearance.", ["Review surface finish at the next checkpoint.", "Log and monitor the trend."]),
}
SEV_ACTION = {"High": "Hold the section for manual inspection.", "Medium": "Flag for review at the next checkpoint.", "Low": "Log and continue; monitor the trend."}


# --------------------------------------------------------------------------- prompt
def findings_for_prompt(handoff: dict) -> dict:
    p = handoff["primary_defect"]
    return {
        "image_id": handoff["image_id"],
        "image_severity": handoff["image_severity"],
        "defect_count": handoff["num_defects"],
        "major_defect": {
            "class": p["class"], "meaning": p["meaning"], "severity": p["severity"],
            "confidence": p["confidence"], "size_pct_of_image": p["area_pct"],
            "location": p["location_text"], "zone": p["zone"],
            "touches_strip_edge": p["touches_strip_edge"],
        },
        "other_defects": handoff["other_defects"],
        "context": "hot-rolled steel strip crop; x across strip width, y along rolling direction",
    }


def build_prompt(handoff: dict) -> str:
    """Llama-3 chat format WITHOUT <|begin_of_text|> (the runner adds BOS), ending in a JSON prefill."""
    user = "Findings:\n%s\nWrite the JSON now." % json.dumps(findings_for_prompt(handoff), separators=(",", ":"))
    return ("<|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n%s" % (SYSTEM_PROMPT, user, PREFILL))


# --------------------------------------------------------------------------- parsing / validation
def extract_json(text: str) -> Optional[dict]:
    s = text.find("{")
    if s < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[s:])
        return obj if isinstance(obj, dict) else None
    except ValueError:
        pass
    out = {}
    m = re.search(r'"explanation"\s*:\s*"(.*?)"\s*(?:,\s*"actions"|\}|$)', text, re.S)
    if m:
        out["explanation"] = m.group(1)
    m = re.search(r'"actions"\s*:\s*\[(.*?)\]', text, re.S)
    if m:
        out["actions"] = re.findall(r'"(.*?)"', m.group(1))
    m = re.search(r'"log_entry"\s*:\s*"(.*?)"\s*(?:\}|$)', text, re.S)
    if m:
        out["log_entry"] = m.group(1)
    return out if "explanation" in out else None


def _clean(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "."


def validate(obj: Optional[dict], handoff: dict):
    """Return (clean_dict or None, problem or None)."""
    if not obj or not obj.get("explanation"):
        return None, "unparseable_or_missing_fields"
    acts = obj.get("actions") or []
    if isinstance(acts, str):
        acts = [acts]
    clean = {
        "explanation": _clean(obj["explanation"], 300),
        "actions": [_clean(a, 100) for a in acts[:2] if str(a).strip()],
        "log_entry": _clean(obj.get("log_entry") or "", 220),
    }
    if not clean["actions"]:
        return None, "no_actions"
    if not clean["log_entry"]:
        clean["log_entry"] = template_advice(handoff)["log_entry"]
    allowed = {handoff["primary_defect"]["class"]} | {o["class"] for o in handoff["other_defects"]}
    blob = (clean["explanation"] + " " + " ".join(clean["actions"]) + " " + clean["log_entry"]).lower()
    for cls, aliases in CLASS_ALIASES.items():
        if cls not in allowed and any(a in blob for a in aliases):
            return None, "ungrounded_class:%s" % cls
    return clean, None


def template_advice(handoff: dict) -> dict:
    p = handoff["primary_defect"]
    expl, acts = TEMPLATES[p["class"]]
    log = "%s: %s (%s, %.1f%% of image, %s), image severity %s." % (
        handoff["image_id"], p["class"], p["severity"], p["area_pct"], p["location_text"], handoff["image_severity"])
    return {"explanation": expl, "actions": [SEV_ACTION[p["severity"]], acts[0]], "log_entry": log}


# --------------------------------------------------------------------------- backend: ExecuTorch llama_main
class RunnerBackend:
    """Runs the ExecuTorch C++ `llama_main` binary as a subprocess (no Python/ExecuTorch version clash)."""

    def __init__(self, llama_main: str, model_path: str, tokenizer_path: str,
                 temperature: float = 0.2, timeout_s: int = 180, threads: Optional[int] = None):
        self.llama_main, self.model, self.tok = str(llama_main), str(model_path), str(tokenizer_path)
        self.temperature, self.timeout_s, self.threads = temperature, timeout_s, threads
        self._help: Optional[str] = None

    def available(self) -> bool:
        return all(Path(p).exists() for p in (self.llama_main, self.model, self.tok))

    def _env(self):
        env = dict(os.environ)
        d = str(Path(self.llama_main).resolve().parent)
        env["LD_LIBRARY_PATH"] = d + ":" + env.get("LD_LIBRARY_PATH", "")
        return env

    def _supports(self, flag: str) -> bool:
        if self._help is None:
            try:
                r = subprocess.run([self.llama_main, "--help"], capture_output=True, text=True, timeout=30, env=self._env())
                self._help = (r.stdout or "") + (r.stderr or "")
            except Exception:
                self._help = ""
        return flag in self._help

    def generate(self, prompt: str, max_new_tokens: int = 110) -> dict:
        est_prompt = int(len(prompt) / 2.8) + 16                       # deliberately over-estimates
        cmd = [self.llama_main, "--model_path=" + self.model, "--tokenizer_path=" + self.tok,
               "--prompt=" + prompt, "--seq_len=%d" % (est_prompt + max_new_tokens),
               "--temperature=%s" % self.temperature]
        echo_off = self._supports("echo")
        if echo_off:
            cmd.append("--echo=false")
        if self.threads and self._supports("num_threads"):
            cmd.append("--num_threads=%d" % self.threads)
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=self._env())
        first = {"t": None}
        buf: Dict[str, List[str]] = {"out": [], "err": []}

        def read(stream, key):
            while True:
                ch = stream.read(1)
                if not ch:
                    break
                if key == "out" and first["t"] is None:
                    first["t"] = time.perf_counter() - t0
                buf[key].append(ch)

        th = [threading.Thread(target=read, args=(proc.stdout, "out"), daemon=True),
              threading.Thread(target=read, args=(proc.stderr, "err"), daemon=True)]
        for t in th:
            t.start()
        peak, ps = 0.0, None
        if psutil:
            try:
                ps = psutil.Process(proc.pid)
            except Exception:
                ps = None
        timed_out = False
        while proc.poll() is None:
            if ps is not None:
                try:
                    peak = max(peak, ps.memory_info().rss / 1e6)
                except Exception:
                    pass
            if time.perf_counter() - t0 > self.timeout_s:
                proc.kill()
                timed_out = True
                break
            time.sleep(0.05)
        for t in th:
            t.join(timeout=2)
        total = time.perf_counter() - t0
        out, err = "".join(buf["out"]), "".join(buf["err"])
        stats = parse_stats(out + "\n" + err)
        text = re.sub(r"PyTorchObserver.*", "", out)
        text = re.sub(r"(?m)^[IWEDF] \d\d:\d\d:\d\d.*$", "", text)
        if not echo_off and prompt in text:
            text = text.split(prompt, 1)[1]
        for tok in ("<|eot_id|>", "<|end_of_text|>", "<|begin_of_text|>"):
            text = text.replace(tok, "")
        return {"text": text, "total_s": total, "first_output_s": first["t"], "peak_rss_mb": round(peak, 1) or None,
                "returncode": proc.returncode, "timed_out": timed_out, "stats": stats,
                "stderr_tail": err[-400:] if proc.returncode else ""}


def parse_stats(blob: str) -> dict:
    """ExecuTorch prints one `PyTorchObserver {json}` line with ms timestamps. Parse defensively."""
    m = re.search(r"PyTorchObserver\s+(\{.*?\})\s*$", blob, re.M)
    if not m:
        return {}
    try:
        d = json.loads(m.group(1))
    except ValueError:
        return {}
    sc = float(d.get("SCALING_FACTOR_UNITS_PER_SECOND", 1000.0))

    def dt(a, b):
        return (d[b] - d[a]) / sc if a in d and b in d else None

    gen, pro = d.get("generated_tokens"), d.get("prompt_tokens")
    decode_s = dt("prompt_eval_end_ms", "inference_end_ms") or dt("first_token_ms", "inference_end_ms")
    prefill_s = dt("inference_start_ms", "prompt_eval_end_ms")
    return {
        "prompt_tokens": pro, "generated_tokens": gen,
        "model_load_s": dt("model_load_start_ms", "model_load_end_ms"),
        "ttft_s": dt("inference_start_ms", "first_token_ms"),
        "prefill_tps": (pro / prefill_s) if pro and prefill_s else None,
        "decode_tps": (gen / decode_s) if gen and decode_s else None,
    }


# --------------------------------------------------------------------------- public API
def advise(handoff: dict, backend: Optional[object] = None, max_new_tokens: int = 110) -> dict:
    """Advice for the MAJOR defect only. Always returns a usable log_entry."""
    if not handoff.get("primary_defect"):
        return {"source": "none", "explanation": "No defect detected.", "actions": ["Continue production."],
                "log_entry": "%s: no defect detected." % handoff["image_id"], "llm": None, "error": None}
    if backend is None:
        return dict(template_advice(handoff), source="template", llm=None, error=None)
    prompt = build_prompt(handoff)
    try:
        g = backend.generate(prompt, max_new_tokens=max_new_tokens)
    except Exception as e:  # noqa: BLE001
        return dict(template_advice(handoff), source="template_fallback", llm=None, error="backend_error:%s" % e)
    obj = extract_json(PREFILL + g["text"])
    clean, problem = validate(obj, handoff)
    st = g.get("stats") or {}
    llm = {"backend": "executorch_llama_main", "total_s": round(g["total_s"], 3),
           "first_output_s": g["first_output_s"], "ttft_s": st.get("ttft_s"), "decode_tps": st.get("decode_tps"),
           "prefill_tps": st.get("prefill_tps"), "prompt_tokens": st.get("prompt_tokens"),
           "generated_tokens": st.get("generated_tokens"), "model_load_s": st.get("model_load_s"),
           "peak_rss_mb": g["peak_rss_mb"], "returncode": g["returncode"]}
    if g["returncode"] not in (0, None) or g["timed_out"]:
        problem = "runner_failed:%s %s" % (g["returncode"], g.get("stderr_tail", ""))
        clean = None
    if clean is None:
        return dict(template_advice(handoff), source="template_fallback", llm=llm, error=problem, raw_output=g["text"][:600])
    return dict(clean, source="llama", llm=llm, error=None, raw_output=g["text"][:600])


class MockBackend:
    """For tests: returns a fixed continuation after the JSON prefill."""
    def __init__(self, text: str):
        self.text = text

    def generate(self, prompt: str, max_new_tokens: int = 110) -> dict:
        return {"text": self.text, "total_s": 0.01, "first_output_s": 0.005, "peak_rss_mb": None,
                "returncode": 0, "timed_out": False, "stats": {}, "stderr_tail": ""}
