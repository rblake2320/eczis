# UNFILED-IP
"""Deflectometry pattern for the monitor: phase-shifted sinusoidal stripes a shiny surface will reflect.

    uv run python tools/stripes.py                 # writes artifacts/deflectometry/stripes.html and opens it
    uv run python tools/stripes.py --period 120 --cycle 2.0

Open it full screen (F11) on the big monitor. It cycles four phases (0, 90, 180, 270 deg) every
--cycle seconds, alternating horizontal and vertical stripes after each set, and shows a small
index in a corner ("H1".."H4", "V1".."V4") so every recorded frame can be tied to the pattern
it saw. A mirror-like tube reflects the stripes: the stripes' apparent position encodes the
surface slope (phase-measuring deflectometry), and the stripe texture itself gives the trackers
something to lock onto. Keys: space pauses, h/v force orientation, +/- change the period.
Record with the phone bridge or the endoscope while the tube sits in front of the screen.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "deflectometry"

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>ECZIS stripes</title>
<style>html,body{margin:0;height:100%;background:#000;overflow:hidden}canvas{display:block}
#idx{position:fixed;right:14px;bottom:10px;color:#444;font:16px monospace}</style></head><body>
<canvas id="c"></canvas><div id="idx"></div><script>
const c=document.getElementById('c'),ctx=c.getContext('2d'),idx=document.getElementById('idx');
let period=__PERIOD__,cycle=__CYCLE__,phase=0,orient='H',paused=false,force=null,step=0;
function draw(){ c.width=innerWidth;c.height=innerHeight; const w=c.width,h=c.height,img=ctx.createImageData(w,h),d=img.data;
  for(let y=0;y<h;y++){for(let x=0;x<w;x++){const u=orient==='H'?y:x;const v=0.5+0.5*Math.cos(2*Math.PI*u/period+phase);const g=Math.round(255*v);const i=4*(y*w+x);d[i]=g;d[i+1]=g;d[i+2]=g;d[i+3]=255;}}
  ctx.putImageData(img,0,0); idx.textContent=orient+(Math.round(phase/(Math.PI/2))%4+1)+' p'+period; }
function tick(){ if(!paused){ phase=(step%4)*Math.PI/2; if(force){orient=force;} else {orient=(Math.floor(step/4)%2===0)?'H':'V';} step++; draw(); } }
addEventListener('keydown',e=>{ if(e.key===' ')paused=!paused; if(e.key==='h')force='H'; if(e.key==='v')force='V'; if(e.key==='a')force=null;
  if(e.key==='+'||e.key==='=')period=Math.min(600,period+10); if(e.key==='-')period=Math.max(10,period-10); draw(); });
addEventListener('resize',draw); draw(); setInterval(tick,cycle*1000);
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=int, default=80, help="stripe period in screen pixels")
    ap.add_argument("--cycle", type=float, default=2.0, help="seconds per phase step")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "stripes.html"
    path.write_text(HTML.replace("__PERIOD__", str(a.period)).replace("__CYCLE__", str(a.cycle)), encoding="utf-8")
    print(f"wrote {path}  (F11 for full screen; corner index ties frames to phases)")
    if not a.no_open:
        os.startfile(path)  # noqa: S606 - local file in the default browser
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
