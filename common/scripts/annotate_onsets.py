#!/usr/bin/env python3
"""Local web UI for annotating t_onset_frame in a scenarios YAML.

  python scripts/annotate_onsets.py --scenarios configs/scenarios-scvd-template.yaml
  python scripts/annotate_onsets.py --scenarios configs/scenarios-ucf.yaml   # verify

Opens http://127.0.0.1:8099 — step through frames and mark, per clip, the
first frame in which the weapon is visible:

  ←/→ ±1 frame · Shift+←/→ ±10 · ↑/↓ ±30 · Home/End
  O  set onset = current frame        (marks onset_verified: true)
  U  unusable (weapon visible from frame 0 / no pre-onset baseline)
  N/P next/previous clip

Every decision is saved immediately back into the YAML (a .bak copy of the
original is kept). `active_scenarios` is rebuilt to contain only clips with a
verified onset that leaves enough pre-onset lead-in (>= --min-onset-frame).
No GUI toolkit needed (works with opencv-python-headless); stdlib http.server.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import yaml

REPO = Path(__file__).resolve().parent.parent

state_lock = threading.Lock()
STATE: dict = {}  # scenarios yaml dict
YAML_PATH: Path
DATA_ROOT: Path = Path(".")
MIN_ONSET = 50

_caps: dict[str, cv2.VideoCapture] = {}
_frame_counts: dict[str, int] = {}


def scenario(sid: str) -> dict:
    for s in STATE["scenarios"]:
        if s["id"] == sid:
            return s
    raise KeyError(sid)


def video_path(s: dict) -> Path:
    p = Path(s["file"])
    return p if p.is_absolute() else DATA_ROOT / p


def get_cap(sid: str) -> cv2.VideoCapture:
    if sid not in _caps:
        _caps[sid] = cv2.VideoCapture(str(video_path(scenario(sid))))
    return _caps[sid]


def frame_count(sid: str) -> int:
    if sid not in _frame_counts:
        s = scenario(sid)
        n = s.get("n_frames_decodable")
        if not n:
            n = int(get_cap(sid).get(cv2.CAP_PROP_FRAME_COUNT))
        _frame_counts[sid] = int(n)
    return _frame_counts[sid]


def jpeg_frame(sid: str, idx: int) -> bytes | None:
    cap = get_cap(sid)
    idx = max(0, min(idx, frame_count(sid) - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return buf.tobytes() if ok else None


def save_yaml() -> None:
    active = [
        s["id"] for s in STATE["scenarios"]
        if s.get("tags", {}).get("onset_verified")
        and not s.get("tags", {}).get("unusable")
        and (s.get("t_onset_frame") or 0) >= MIN_ONSET
    ]
    STATE["active_scenarios"] = active
    YAML_PATH.write_text(yaml.safe_dump(STATE, sort_keys=False, allow_unicode=True))


PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>TTD onset annotator</title>
<style>
 body{font-family:-apple-system,sans-serif;margin:0;display:flex;height:100vh;background:#1e1e1e;color:#ddd}
 #side{width:270px;overflow-y:auto;border-right:1px solid #444;padding:8px;flex-shrink:0}
 #main{flex:1;display:flex;flex-direction:column;align-items:center;padding:12px;overflow-y:auto}
 .clip{padding:4px 6px;cursor:pointer;border-radius:4px;font-size:13px;white-space:nowrap}
 .clip:hover{background:#333}.clip.cur{background:#0a4d8c}
 .done{color:#7c7}.unusable{color:#c77;text-decoration:line-through}.pending{color:#aaa}
 img{max-width:100%;max-height:62vh;border:1px solid #555}
 #bar{width:90%;margin:10px 0}
 button{margin:4px;padding:8px 14px;border:none;border-radius:5px;cursor:pointer;font-size:14px}
 #setBtn{background:#2d7d2d;color:#fff}#unBtn{background:#8c3b3b;color:#fff}
 .nav{background:#444;color:#ddd}
 #info{font-size:15px;margin:6px}
 #onsetInfo{color:#7c7;font-size:14px;min-height:18px}
 kbd{background:#333;border-radius:3px;padding:1px 5px;font-size:12px}
 #help{font-size:12px;color:#888;margin-top:6px;text-align:center}
</style></head><body>
<div id="side"></div>
<div id="main">
 <div id="info"></div>
 <img id="frame" src="">
 <input type="range" id="bar" min="0" value="0">
 <div id="onsetInfo"></div>
 <div>
  <button class="nav" onclick="step(-10)">−10</button>
  <button class="nav" onclick="step(-1)">−1</button>
  <button class="nav" onclick="step(1)">+1</button>
  <button class="nav" onclick="step(10)">+10</button>
  <button id="setBtn" onclick="setOnset()">O — onset = tento snímek</button>
  <button id="unBtn" onclick="setUnusable()">U — nepoužitelný klip</button>
 </div>
 <div id="help">←/→ ±1 · Shift ±10 · ↑/↓ ±30 · Home/End · <kbd>O</kbd> onset ·
 <kbd>U</kbd> nepoužitelný · <kbd>N</kbd>/<kbd>P</kbd> další/předchozí klip</div>
</div>
<script>
let clips=[],cur=0,idx=0,nframes=1;
async function load(){
 clips=await (await fetch('/meta')).json();
 render(); show(0);
}
function render(){
 const side=document.getElementById('side'); side.innerHTML='';
 clips.forEach((c,i)=>{
  const d=document.createElement('div');
  let cls='pending', mark='';
  if(c.unusable){cls='unusable'; mark=' ✗';}
  else if(c.verified){cls='done'; mark=' ✓'+c.onset;}
  d.className='clip '+cls+(i===cur?' cur':'');
  d.textContent=c.id+mark;
  d.onclick=()=>show(i);
  side.appendChild(d);
 });
}
function show(i){
 cur=i; const c=clips[cur]; nframes=c.n_frames;
 idx=(c.onset!=null&&c.onset>=0)?c.onset:0;
 document.getElementById('bar').max=nframes-1;
 update(); render();
}
function update(){
 idx=Math.max(0,Math.min(idx,nframes-1));
 const c=clips[cur];
 document.getElementById('frame').src='/frame?vid='+encodeURIComponent(c.id)+'&idx='+idx;
 document.getElementById('bar').value=idx;
 document.getElementById('info').textContent=
   c.id+'  ·  snímek '+idx+' / '+(nframes-1)+'  ·  '+(idx/c.fps).toFixed(2)+' s  ·  '+c.fps+' fps';
 document.getElementById('onsetInfo').textContent=
   c.unusable?'označen jako nepoužitelný':(c.verified?('onset = '+c.onset+' (ověřeno)'):'onset zatím neanotován');
}
function step(d){idx+=d; update();}
document.getElementById('bar').oninput=e=>{idx=+e.target.value; update();};
async function post(action,val){
 const c=clips[cur];
 await fetch('/mark',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({id:c.id,action:action,value:val})});
 clips=await (await fetch('/meta')).json(); update(); render();
}
function setOnset(){post('onset',idx).then(()=>next());}
function setUnusable(){post('unusable',true).then(()=>next());}
function next(){if(cur<clips.length-1)show(cur+1);}
function prev(){if(cur>0)show(cur-1);}
document.addEventListener('keydown',e=>{
 if(e.key==='ArrowRight')step(e.shiftKey?10:1);
 else if(e.key==='ArrowLeft')step(e.shiftKey?-10:-1);
 else if(e.key==='ArrowUp')step(30);
 else if(e.key==='ArrowDown')step(-30);
 else if(e.key==='Home'){idx=0;update();}
 else if(e.key==='End'){idx=nframes-1;update();}
 else if(e.key==='o'||e.key==='O')setOnset();
 else if(e.key==='u'||e.key==='U')setUnusable();
 else if(e.key==='n'||e.key==='N')next();
 else if(e.key==='p'||e.key==='P')prev();
 else return;
 e.preventDefault();
});
load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif u.path == "/meta":
            with state_lock:
                meta = [
                    {
                        "id": s["id"],
                        "n_frames": frame_count(s["id"]),
                        "fps": s["fps"],
                        "onset": s.get("t_onset_frame"),
                        "verified": bool(s.get("tags", {}).get("onset_verified")),
                        "unusable": bool(s.get("tags", {}).get("unusable")),
                    }
                    for s in STATE["scenarios"]
                ]
            self._send(200, json.dumps(meta).encode(), "application/json")
        elif u.path == "/frame":
            q = parse_qs(u.query)
            sid = q["vid"][0]
            idx = int(q.get("idx", ["0"])[0])
            with state_lock:
                data = jpeg_frame(sid, idx)
            if data is None:
                self._send(404, b"no frame", "text/plain")
            else:
                self._send(200, data, "image/jpeg")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path != "/mark":
            self._send(404, b"not found", "text/plain")
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with state_lock:
            s = scenario(body["id"])
            tags = s.setdefault("tags", {})
            if body["action"] == "onset":
                s["t_onset_frame"] = int(body["value"])
                tags["onset_verified"] = True
                tags["unusable"] = False
            elif body["action"] == "unusable":
                tags["unusable"] = True
                tags["onset_verified"] = False
                s["t_onset_frame"] = None
            save_yaml()
        self._send(200, b"{}", "application/json")


def main() -> int:
    global STATE, YAML_PATH, MIN_ONSET
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default=str(REPO / "configs" / "scenarios-scvd-template.yaml"))
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--data-root", default=".",
                    help="base directory for RELATIVE video paths in the YAML")
    ap.add_argument("--min-onset-frame", type=int, default=50,
                    help="clips with verified onset below this are left out of "
                         "active_scenarios (ingest needs pre-onset lead-in)")
    args = ap.parse_args()

    global DATA_ROOT
    YAML_PATH = Path(args.scenarios)
    DATA_ROOT = Path(args.data_root).expanduser().resolve()
    MIN_ONSET = args.min_onset_frame
    STATE = yaml.safe_load(YAML_PATH.read_text())
    if "scenarios" not in STATE:
        print("not a scenarios YAML", file=sys.stderr)
        return 1

    bak = YAML_PATH.with_suffix(YAML_PATH.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(YAML_PATH, bak)
        print(f"backup: {bak}")

    missing = [s["id"] for s in STATE["scenarios"] if not video_path(s).exists()]
    if missing:
        print(f"WARNING: {len(missing)}/{len(STATE['scenarios'])} video files "
              f"missing (e.g. {video_path(scenario(missing[0]))}) — check "
              "--data-root", file=sys.stderr)

    url = f"http://127.0.0.1:{args.port}"
    print(f"annotator UI: {url}   (Ctrl+C to quit; every mark is saved to "
          f"{YAML_PATH.name} immediately)")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
