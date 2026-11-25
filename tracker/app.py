"""Steel Defect Tracker - small offline Flask dashboard for the inspection log.

Run:      python app.py --media ../runs            (then open http://127.0.0.1:5000)
Import:   python app.py --import ../runs/<run>/inspections.jsonl
The pipeline notebook POSTs every inspection to /api/inspections; the benchmark notebook POSTs to /api/benchmark.
No internet, CDN or web fonts are used, so it works for the offline demo. Python 3.8+, Flask only.
"""
import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, g, jsonify, redirect, render_template, request, send_from_directory, url_for
from jinja2 import DictLoader

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "data" / "tracker.db"
MEDIA_ROOT = HERE.parent / "runs"
SEVS = ["High", "Medium", "Low"]
CHECKLIST = [
    ("app", "Chosen application, dataset and defect classes"),
    ("detector", "Working detector with precision and recall"),
    ("llama", "Llama running via ExecuTorch on the device"),
    ("handoff", "Vision to Llama hand-off and local log"),
    ("offline", "Offline run with FPS, latency and memory"),
    ("repo", "GitHub repo, demo video and report draft"),
]

app = Flask(__name__)
SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections(
 id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, image_id TEXT, ts TEXT, image_severity TEXT, num_defects INTEGER,
 primary_class TEXT, primary_severity TEXT, primary_conf REAL, area_pct REAL, zone TEXT, advice_source TEXT,
 explanation TEXT, actions TEXT, log_entry TEXT, vision_ms REAL, llama_ms REAL, e2e_ms REAL, annotated_image TEXT,
 raw TEXT, UNIQUE(run_id, image_id));
CREATE TABLE IF NOT EXISTS benchmarks(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, report TEXT);
CREATE TABLE IF NOT EXISTS checklist(key TEXT PRIMARY KEY, done INTEGER DEFAULT 0, note TEXT DEFAULT '');
"""


def db():
    if "db" not in g:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(str(DB_PATH))
        g.db.row_factory = sqlite3.Row
        g.db.executescript(SCHEMA)
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    d = g.pop("db", None)
    if d is not None:
        d.close()


def insert_record(conn, rec):
    h, a, t = rec["handoff"], rec.get("advice") or {}, rec.get("timing_ms") or {}
    p = h.get("primary_defect") or {}
    conn.execute(
        "INSERT OR REPLACE INTO inspections(run_id,image_id,ts,image_severity,num_defects,primary_class,primary_severity,"
        "primary_conf,area_pct,zone,advice_source,explanation,actions,log_entry,vision_ms,llama_ms,e2e_ms,annotated_image,raw)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (rec.get("run_id", "manual"), h["image_id"], h["timestamp"], h["image_severity"], h["num_defects"],
         p.get("class"), p.get("severity"), p.get("confidence"), p.get("area_pct"), p.get("zone"), a.get("source"),
         a.get("explanation"), " | ".join(a.get("actions") or []), a.get("log_entry"), t.get("vision_total"),
         t.get("llama_total"), t.get("end_to_end"), rec.get("annotated_image"), json.dumps(rec)))
    conn.commit()


def flatten(d, prefix=""):
    rows = []
    for k, v in d.items():
        key = prefix + str(k)
        if isinstance(v, dict):
            rows += flatten(v, key + ".")
        elif isinstance(v, list):
            rows.append((key, json.dumps(v)[:300]))
        else:
            rows.append((key, v))
    return rows


@app.route("/")
def dashboard():
    c = db()
    total = c.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
    sev = {r[0]: r[1] for r in c.execute("SELECT image_severity, COUNT(*) FROM inspections GROUP BY 1")}
    cls = c.execute("SELECT primary_class, COUNT(*) n FROM inspections WHERE primary_class IS NOT NULL GROUP BY 1 ORDER BY n DESC").fetchall()
    strip = c.execute("SELECT id,image_id,image_severity,primary_class FROM inspections ORDER BY id DESC LIMIT 60").fetchall()[::-1]
    latest = c.execute("SELECT * FROM inspections ORDER BY id DESC LIMIT 8").fetchall()
    done = c.execute("SELECT COUNT(*) FROM checklist WHERE done=1").fetchone()[0]
    return render_template("dash.html", total=total, sev=sev, cls=cls, strip=strip, latest=latest, done=done, n_items=len(CHECKLIST), sevs=SEVS)


@app.route("/inspections")
def inspections():
    sev, cl = request.args.get("severity", ""), request.args.get("cls", "")
    q, args = "SELECT * FROM inspections WHERE 1=1", []
    if sev:
        q += " AND image_severity=?"; args.append(sev)
    if cl:
        q += " AND primary_class=?"; args.append(cl)
    rows = db().execute(q + " ORDER BY id DESC LIMIT 500", args).fetchall()
    classes = [r[0] for r in db().execute("SELECT DISTINCT primary_class FROM inspections WHERE primary_class IS NOT NULL")]
    return render_template("list.html", rows=rows, sev=sev, cl=cl, sevs=SEVS, classes=classes)


@app.route("/inspection/<int:iid>")
def inspection(iid):
    r = db().execute("SELECT * FROM inspections WHERE id=?", (iid,)).fetchone()
    if r is None:
        abort(404)
    raw = json.loads(r["raw"])
    return render_template("detail.html", r=r, defects=raw["handoff"].get("all_defects", []), raw=json.dumps(raw["handoff"], indent=2))


@app.route("/media/<path:rel>")
def media(rel):
    return send_from_directory(str(MEDIA_ROOT), rel)       # send_from_directory blocks path traversal


@app.route("/benchmark")
def benchmark():
    row = db().execute("SELECT * FROM benchmarks ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        return render_template("bench.html", sections=None, ts=None)
    rep = json.loads(row["report"])
    return render_template("bench.html", sections=[(k, flatten(v) if isinstance(v, dict) else [(k, v)]) for k, v in rep.items()], ts=row["ts"])


@app.route("/checklist", methods=["GET", "POST"])
def checklist():
    c = db()
    if request.method == "POST":
        for key, _ in CHECKLIST:
            c.execute("INSERT INTO checklist(key,done,note) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET done=excluded.done, note=excluded.note",
                      (key, 1 if request.form.get(key) else 0, request.form.get("note_" + key, "")))
        c.commit()
        return redirect(url_for("checklist"))
    st = {r["key"]: r for r in c.execute("SELECT * FROM checklist")}
    return render_template("check.html", items=[(k, label, st.get(k)) for k, label in CHECKLIST])


@app.route("/api/inspections", methods=["POST"])
def api_inspections():
    rec = request.get_json(force=True, silent=True)
    if not rec or "handoff" not in rec:
        return jsonify(error="expected a pipeline record with a 'handoff' key"), 400
    insert_record(db(), rec)
    return jsonify(ok=True)


@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    rep = request.get_json(force=True, silent=True)
    if not isinstance(rep, dict):
        return jsonify(error="expected a JSON object"), 400
    db().execute("INSERT INTO benchmarks(ts,report) VALUES(?,?)", (datetime.now().isoformat(timespec="seconds"), json.dumps(rep)))
    db().commit()
    return jsonify(ok=True)


@app.route("/api/health")
def health():
    return jsonify(ok=True)


CSS = """
:root{--ink:#1f2a33;--muted:#5d6b76;--paper:#edeff1;--panel:#fbfbfc;--rule:#c9cfd4;--low:#3f7d6e;--medium:#b9791a;--high:#a83a2a;--none:#9aa5ad}
@media (prefers-color-scheme:dark){:root{--ink:#e6eaed;--muted:#9fadb8;--paper:#161c21;--panel:#1e262c;--rule:#35414a;--low:#5fb09d;--medium:#e0a23c;--high:#e0705d;--none:#6b7882}}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 "Segoe UI","Helvetica Neue","DejaVu Sans",Arial,sans-serif}
header{display:flex;gap:24px;align-items:baseline;padding:14px 28px;border-bottom:2px solid var(--ink);background:var(--panel);flex-wrap:wrap}
header b{font-size:18px;letter-spacing:.2px;margin-right:12px}nav a{color:var(--ink);text-decoration:none;margin-right:18px;padding:4px 0;border-bottom:2px solid transparent}
nav a:hover,nav a:focus-visible{border-color:var(--ink);outline:none}main{max-width:1100px;margin:0 auto;padding:24px 28px 60px}
h1{font-size:26px;margin:.2em 0 .6em}h2{font-size:17px;margin:1.6em 0 .5em}
.strip{display:flex;gap:2px;height:46px;margin:6px 0 4px}.strip a{flex:1;min-width:4px;display:block;border-radius:1px}
.strip a:hover,.strip a:focus-visible{outline:2px solid var(--ink);outline-offset:1px}
.legend{display:flex;gap:16px;color:var(--muted);font-size:13px;margin-bottom:8px}.legend i{display:inline-block;width:10px;height:10px;margin-right:5px;border-radius:2px}
.s-High{background:var(--high)}.s-Medium{background:var(--medium)}.s-Low{background:var(--low)}.s-None{background:var(--none)}
.facts{display:flex;gap:28px;flex-wrap:wrap;margin:14px 0 4px}.facts div{min-width:90px}.facts strong{display:block;font-size:28px;font-variant-numeric:tabular-nums}.facts span{color:var(--muted);font-size:13px}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--rule)}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--rule);vertical-align:top;font-variant-numeric:tabular-nums}
th{font-weight:600;background:var(--paper)}tr:hover td{background:var(--paper)}td a{color:inherit}
.tag{display:inline-block;padding:1px 9px;border-radius:10px;color:#fff;font-size:12.5px}
.box{background:var(--panel);border:1px solid var(--rule);padding:14px 18px;margin:10px 0}.muted{color:var(--muted)}
.two{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px}@media(max-width:760px){.two{grid-template-columns:1fr}}
img{max-width:100%;image-rendering:pixelated;border:1px solid var(--rule)}pre{background:var(--panel);border:1px solid var(--rule);padding:12px;overflow:auto;font-size:12.5px}
select,button,input[type=text]{font:inherit;padding:5px 9px;border:1px solid var(--ink);background:var(--panel);color:var(--ink)}button{cursor:pointer;background:var(--ink);color:var(--panel)}
.empty{padding:28px;border:1px dashed var(--rule);color:var(--muted);background:var(--panel)}code{background:var(--paper);padding:1px 5px}
"""
BASE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{% block title %}Steel Defect Tracker{% endblock %}</title><style>""" + CSS + """</style></head><body>
<header><b>Steel Defect Tracker</b><nav><a href="{{ url_for('dashboard') }}">Overview</a><a href="{{ url_for('inspections') }}">Inspections</a>
<a href="{{ url_for('benchmark') }}">Benchmark</a><a href="{{ url_for('checklist') }}">Checklist</a></nav></header><main>{% block body %}{% endblock %}</main></body></html>"""
MACROS = """{% macro tag(s) %}<span class="tag s-{{ s or 'None' }}">{{ s or 'Clean' }}</span>{% endmacro %}"""
TEMPLATES = {
    "base.html": BASE,
    "macros.html": MACROS,
    "dash.html": """{% extends 'base.html' %}{% from 'macros.html' import tag %}{% block body %}
<h1>Inspection overview</h1>
{% if not total %}<div class="empty">No inspections logged yet. Run the pipeline notebook with the tracker URL set to <code>http://127.0.0.1:5000</code>, or import a log with <code>python app.py --import runs/&lt;run&gt;/inspections.jsonl</code>.</div>
{% else %}
<div class="facts"><div><strong>{{ total }}</strong><span>images inspected</span></div>
{% for s in sevs %}<div><strong>{{ sev.get(s, 0) }}</strong><span>{{ s }} severity</span></div>{% endfor %}
<div><strong>{{ sev.get('None', 0) }}</strong><span>clean</span></div><div><strong>{{ done }}/{{ n_items }}</strong><span>checklist items done</span></div></div>
<h2>Latest {{ strip|length }} images, oldest to newest</h2>
<div class="strip">{% for r in strip %}<a class="s-{{ r.image_severity }}" href="{{ url_for('inspection', iid=r.id) }}" title="{{ r.image_id }}: {{ r.primary_class or 'clean' }}, {{ r.image_severity }}" aria-label="{{ r.image_id }} {{ r.image_severity }}"></a>{% endfor %}</div>
<div class="legend"><span><i class="s-High"></i>High</span><span><i class="s-Medium"></i>Medium</span><span><i class="s-Low"></i>Low</span><span><i class="s-None"></i>Clean</span></div>
<h2>Major defect by class</h2><table><tr><th>Class</th><th>Images</th></tr>{% for c in cls %}<tr><td><a href="{{ url_for('inspections', cls=c.primary_class) }}">{{ c.primary_class }}</a></td><td>{{ c.n }}</td></tr>{% endfor %}</table>
<h2>Latest log entries</h2><table><tr><th>Image</th><th>Severity</th><th>Log entry</th></tr>
{% for r in latest %}<tr><td><a href="{{ url_for('inspection', iid=r.id) }}">{{ r.image_id }}</a></td><td>{{ tag(r.image_severity) }}</td><td>{{ r.log_entry }}</td></tr>{% endfor %}</table>
{% endif %}{% endblock %}""",
    "list.html": """{% extends 'base.html' %}{% from 'macros.html' import tag %}{% block body %}
<h1>Inspections</h1><form method="get" class="box"><label>Severity <select name="severity"><option value="">all</option>{% for s in sevs %}<option {{ 'selected' if s==sev }}>{{ s }}</option>{% endfor %}</select></label>
&nbsp;<label>Major class <select name="cls"><option value="">all</option>{% for c in classes %}<option {{ 'selected' if c==cl }}>{{ c }}</option>{% endfor %}</select></label>&nbsp;<button type="submit">Apply filter</button></form>
{% if not rows %}<div class="empty">No inspections match this filter.</div>{% else %}
<table><tr><th>Image</th><th>Time</th><th>Severity</th><th>Major defect</th><th>Defects</th><th>Advice</th><th>End to end (ms)</th></tr>
{% for r in rows %}<tr><td><a href="{{ url_for('inspection', iid=r.id) }}">{{ r.image_id }}</a></td><td>{{ r.ts }}</td><td>{{ tag(r.image_severity) }}</td><td>{{ r.primary_class or 'none' }}</td><td>{{ r.num_defects }}</td><td>{{ r.advice_source or '' }}</td><td>{{ r.e2e_ms }}</td></tr>{% endfor %}</table>{% endif %}{% endblock %}""",
    "detail.html": """{% extends 'base.html' %}{% from 'macros.html' import tag %}{% block body %}
<h1>{{ r.image_id }} {{ tag(r.image_severity) }}</h1>
<div class="two"><div>{% if r.annotated_image %}<img src="{{ url_for('media', rel=r.annotated_image) }}" alt="Annotated image {{ r.image_id }}" width="400">{% else %}<div class="empty">No annotated image stored.</div>{% endif %}</div>
<div><div class="box"><b>Log entry</b><p>{{ r.log_entry }}</p><b>Explanation</b><p>{{ r.explanation }}</p><b>Suggested actions</b><p>{{ r.actions }}</p>
<p class="muted">Advice source: {{ r.advice_source or 'none' }}. Vision {{ r.vision_ms }} ms, Llama {{ r.llama_ms }} ms, end to end {{ r.e2e_ms }} ms.</p></div></div></div>
<h2>Defects found</h2>{% if defects %}<table><tr><th>Class</th><th>Severity</th><th>Score</th><th>Confidence</th><th>Area %</th><th>Zone</th><th>Type / size / location parts</th></tr>
{% for d in defects %}<tr><td>{{ d['class'] }}</td><td>{{ tag(d.severity) }}</td><td>{{ d.score }}</td><td>{{ d.confidence }}</td><td>{{ d.area_pct }}</td><td>{{ d.zone }}</td><td>{{ d.parts.type }} / {{ d.parts.size }} / {{ d.parts.location }}</td></tr>{% endfor %}</table>
{% else %}<div class="empty">No defects detected above the confidence threshold.</div>{% endif %}
<h2>Hand-off JSON</h2><pre>{{ raw }}</pre>{% endblock %}""",
    "bench.html": """{% extends 'base.html' %}{% block body %}<h1>Benchmark</h1>
{% if not sections %}<div class="empty">No benchmark report yet. Run the benchmark notebook; it posts its report here.</div>{% else %}<p class="muted">Latest report saved {{ ts }}.</p>
{% for name, rows in sections %}<h2>{{ name }}</h2><table><tr><th>Metric</th><th>Value</th></tr>{% for k, v in rows %}<tr><td>{{ k }}</td><td>{{ v }}</td></tr>{% endfor %}</table>{% endfor %}{% endif %}{% endblock %}""",
    "check.html": """{% extends 'base.html' %}{% block body %}<h1>Checklist for 23 November</h1>
<form method="post">{% for key, label, st in items %}<div class="box"><label><input type="checkbox" name="{{ key }}" {{ 'checked' if st and st.done }}> {{ label }}</label>
<div><input type="text" name="note_{{ key }}" value="{{ st.note if st else '' }}" placeholder="Note (optional)" size="60" aria-label="Note for {{ label }}"></div></div>{% endfor %}
<button type="submit">Save checklist</button></form>{% endblock %}""",
}
app.jinja_loader = DictLoader(TEMPLATES)


def main():
    global MEDIA_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--media", default=str(MEDIA_ROOT), help="folder that holds the run folders with annotated images")
    ap.add_argument("--import", dest="imp", help="import an inspections.jsonl file and exit")
    a = ap.parse_args()
    MEDIA_ROOT = Path(a.media).resolve()
    if a.imp:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH)); conn.executescript(SCHEMA)
        n = 0
        for line in open(a.imp):
            if line.strip():
                insert_record(conn, json.loads(line)); n += 1
        print("imported %d records" % n)
        return
    app.run(host=a.host, port=a.port, debug=False)


if __name__ == "__main__":
    main()
