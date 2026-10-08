# UNFILED-IP
"""Use a phone's camera as a live frame source over the local network, no app install.

The PC serves one HTTPS page on the LAN (self-signed certificate, generated locally). The phone
opens it, grants camera access, and posts JPEG frames to the PC over HTTPS (plain requests: iOS Safari
refuses secure WebSockets on a self-signed certificate but accepts fetch); the page also
shows the coach caption and speaks it with the phone's own voice, so the operator holding the
phone sees and hears the agent. Frames never leave the two devices.

    uv run python tools/phone_cam.py                 # prints the URL to open on the phone
    uv run python -m eczis.lab.scan_app --phone      # scan window on the phone's camera

`PhoneCamera` has the same surface as OpenCVCamera (read / read_display / capture / wait_for_frame /
set_mode / describe / geometry / close), so every tool downstream is unchanged. Live video from a
phone browser is capped by the browser (iOS Safari ~1280x720); the page's Still button sends a
full-resolution photo through the same channel and it lands as a snapshot. EXPERIMENTAL.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CERT_DIR = ROOT / "artifacts" / "phone_cam"
PHONE_COMMAND = ROOT / "artifacts" / "coach" / "phone_command.json"
PHONE_STATUS = ROOT / "artifacts" / "coach" / "phone_status.json"  # what the page reports back: version, last command result
AVATAR_DIR = ROOT / "artifacts" / "phone_cam" / "avatar"  # downscaled face frames (operator-owned art, not committed)
SPEECH_WAV = ROOT / "artifacts" / "coach" / "speech_last.wav"  # latest coach line rendered on the PC, played by the phone too
PORT = 8443

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ECZIS phone camera</title><style>
body{margin:0;background:#000;color:#fff;font-family:system-ui;display:flex;flex-direction:column;height:100vh}
video{width:100%;flex:1;object-fit:contain;background:#111}
#cap{font-size:22px;padding:12px;text-align:center;min-height:1.4em;background:#000}
#bar{display:flex;gap:8px;padding:8px}button{flex:1;font-size:18px;padding:12px;border:0;border-radius:8px;background:#2563eb;color:#fff}
#st{font-size:13px;color:#9ca3af;text-align:center;padding:4px}</style></head><body>
<canvas id="pat" style="display:none;position:fixed;inset:0;z-index:9"></canvas>
<audio id="spk" playsinline preload="auto"></audio>
<img id="ava" src="/avatar/neutral.jpg" style="position:fixed;top:10px;right:10px;width:34vw;max-width:220px;border-radius:14px;box-shadow:0 2px 14px #000;z-index:8;display:block">
<video id="v" autoplay playsinline muted></video><div id="cap"></div>
<div id="bar"><button id="still">Still (full res)</button><button id="flip">Flip</button><button id="mute">Voice on</button></div>
<select id="cam" style="font-size:16px;margin:0 8px 6px 8px;padding:8px;background:#111;color:#fff;border:1px solid #444"></select>
<div id="st">connecting...</div>
<script>
const st=document.getElementById('st'),cap=document.getElementById('cap'),v=document.getElementById('v');
let facing='environment',stream,ws,voice=true,lastCap='',sending=false,deviceId=null;
const sel=document.getElementById('cam');
async function listCams(){ const devs=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput'); sel.innerHTML='';
 for(const d of devs){const o=document.createElement('option');o.value=d.deviceId;o.textContent=d.label||('camera '+(sel.length+1));sel.appendChild(o);}
 const single=devs.find(d=>/^back camera$/i.test(d.label))||devs.find(d=>/back/i.test(d.label)&&!/dual|triple|wide/i.test(d.label));
 if(single&&!deviceId){deviceId=single.deviceId;sel.value=deviceId;} }
async function start(){ if(stream)stream.getTracks().forEach(t=>t.stop());
 const cons=deviceId?{deviceId:{exact:deviceId},width:{ideal:1920},height:{ideal:1080}}:{facingMode:facing,width:{ideal:1920},height:{ideal:1080}};
 stream=await navigator.mediaDevices.getUserMedia({video:cons,audio:false});
 v.srcObject=stream; await listCams(); const tr=stream.getVideoTracks()[0]; const s=tr.getSettings();
 st.textContent=`camera ${s.width}x${s.height} ${tr.label}`; tr.onmute=()=>{st.textContent='camera paused by iOS (lens switch?) - restarting';setTimeout(start,500);}; }
sel.onchange=()=>{deviceId=sel.value;start();};
let linked=false; async function poll(){ try{ const r=await fetch('/caption',{cache:'no-store'}); const m=await r.json(); if(!linked){linked=true;st.textContent=st.textContent.replace(' | link lost','')+' | linked';}
   if(m.caption!==undefined&&m.caption!==lastCap){lastCap=m.caption;cap.textContent=m.caption;
   if(voice&&m.caption){ if(m.audio&&audioOk){ const el=document.getElementById('spk'); el.src='/speech/'+m.audio+'.wav'; el.onplay=startMouth; el.onended=stopMouth; el.onerror=()=>{stopMouth();sayLocal(m.caption);}; el.play().catch(()=>sayLocal(m.caption)); } else sayLocal(m.caption); }}
   if(m.command)handle(m.command,m.args||{}); }catch(e){ if(linked){linked=false;st.textContent+=' | link lost';} } setTimeout(poll,500); }
const pat=document.getElementById('pat'),ava=document.getElementById('ava');
let audioOk=false,ac=null;
function sayLocal(text){ if('speechSynthesis'in window){speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(text);u.rate=1.0;u.onstart=startMouth;u.onend=stopMouth;u.onerror=stopMouth;speechSynthesis.speak(u);} }
function unlock(){ try{ ac=ac||new (window.AudioContext||window.webkitAudioContext)(); ac.resume(); const el=document.getElementById('spk'); el.muted=false; el.play().catch(()=>{}); audioOk=true; st.textContent=st.textContent.replace(' | tap to enable voice','')+' | voice unlocked'; }catch(e){} }
document.addEventListener('touchend',unlock,{once:true}); document.addEventListener('click',unlock,{once:true});
let avaState='neutral',mouthTimer=null,blinkTimer=null;
function showFrame(n){ava.src='/avatar/'+n+'.jpg';}
function setAvatar(s){ avaState=s; if(!mouthTimer)showFrame(s); lastResult='avatar:'+s; }
function startMouth(){ if(mouthTimer)return; const fr=['speak-small','speak-mid','speak-wide','speak-mid']; let i=0; mouthTimer=setInterval(()=>{showFrame(fr[i++%fr.length]);},140); }
function stopMouth(){ if(mouthTimer){clearInterval(mouthTimer);mouthTimer=null;} showFrame(avaState); }
blinkTimer=setInterval(()=>{ if(!mouthTimer&&avaState==='neutral'){showFrame('blink');setTimeout(()=>{if(!mouthTimer)showFrame(avaState);},160);} },4500+Math.random()*2000);
async function handle(cmd,a){ lastResult=cmd+':start'; try{
  if(cmd==='still')sendStill();
  else if(cmd==='zoom'){const tr=stream.getVideoTracks()[0];const cap=tr.getCapabilities?tr.getCapabilities():{};
    if(cap.zoom){const z=Math.min(cap.zoom.max,Math.max(cap.zoom.min,a.value||1));await tr.applyConstraints({advanced:[{zoom:z}]});st.textContent='zoom '+z+'x';lastResult='zoom:ok '+z;}else{st.textContent='zoom not supported here';lastResult='zoom:unsupported caps='+Object.keys(cap).join(',');}}
  else if(cmd==='camera'){const o=[...sel.options].find(o=>o.textContent.toLowerCase().includes((a.name||'back').toLowerCase()));if(o){deviceId=o.value;sel.value=deviceId;start();}}
  else if(cmd==='stripes'){pat.style.display='block';drawStripes(a.period||60,a.phase||0,a.orient||'V');lastResult='stripes:shown '+pat.width+'x'+pat.height;}
  else if(cmd==='clear'){pat.style.display='none';lastResult='clear:ok';}
  else if(cmd==='avatar'){ if(a.state==='off'){ava.style.display='none';lastResult='avatar:off';} else {ava.style.display='block';setAvatar(a.state||'neutral');} }
  else if(cmd==='beep'){ac=ac||new (window.AudioContext||window.webkitAudioContext)();const o=ac.createOscillator();o.frequency.value=880;o.connect(ac.destination);o.start();setTimeout(()=>o.stop(),250);lastResult='beep:'+ac.state;}
 }catch(e){st.textContent='command failed: '+e;lastResult=cmd+':error '+e} }
function drawStripes(period,phase,orient){ pat.width=innerWidth;pat.height=innerHeight;const ctx2=pat.getContext('2d'),w=pat.width,h=pat.height,img=ctx2.createImageData(w,h),d=img.data;
  for(let y=0;y<h;y++){for(let x=0;x<w;x++){const u=orient==='H'?y:x;const g=Math.round(255*(0.5+0.5*Math.cos(2*Math.PI*u/period+phase)));const i=4*(y*w+x);d[i]=g;d[i+1]=g;d[i+2]=g;d[i+3]=255;}} ctx2.putImageData(img,0,0); }
function connect(){ poll(); }
const c=document.createElement('canvas');
const PAGE_VERSION='cmd-2'; let lastResult='';
async function post(kind,blob){ const r=await fetch('/frame?kind='+kind,{method:'POST',body:blob,headers:{'Content-Type':'image/jpeg','X-Page':PAGE_VERSION,'X-Last':lastResult.slice(0,120)}}); return r.ok; }
function frame(){ if(!sending&&v.videoWidth){sending=true;c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0);
  c.toBlob(b=>{post(0,b).catch(()=>{}).finally(()=>{sending=false;});},'image/jpeg',0.85);}
  setTimeout(frame,80); }
async function sendStill(){ try{ const tr=stream.getVideoTracks()[0]; let blob;
  if('ImageCapture'in window){ blob=await new ImageCapture(tr).takePhoto(); } else { c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0); blob=await new Promise(r=>c.toBlob(r,'image/jpeg',0.95)); }
  await post(1,blob); st.textContent='still sent '+Math.round(blob.size/1024)+' kB'; }catch(e){st.textContent='still failed: '+e} }
document.getElementById('still').onclick=sendStill;
document.getElementById('flip').onclick=()=>{facing=facing==='environment'?'user':'environment';deviceId=null;start();};
document.getElementById('mute').onclick=e=>{voice=!voice;e.target.textContent=voice?'Voice on':'Voice off';};
start().then(()=>{connect();frame();}).catch(e=>{st.textContent='camera denied: '+e});
</script></body></html>"""


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return str(s.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_cert() -> tuple[Path, Path]:
    """Self-signed certificate for the LAN address (the phone accepts it once); generated locally."""
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    crt, key = CERT_DIR / "cert.pem", CERT_DIR / "key.pem"
    if crt.exists() and key.exists():
        return crt, key
    import datetime as dt
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "eczis-phone-cam")])
    ip = ipaddress.ip_address(lan_ip())
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(dt.datetime.now(dt.UTC) - dt.timedelta(days=1))
            .not_valid_after(dt.datetime.now(dt.UTC) + dt.timedelta(days=3650))
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ip), x509.DNSName("localhost")]), critical=False)
            .sign(k, hashes.SHA256()))
    key.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
    crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return crt, key


class PhoneCamera:
    """Frame source fed by the phone page; same surface as OpenCVCamera."""

    def __init__(self, port: int = PORT, caption_path: Path | None = None) -> None:
        self.port = port
        self._lock = threading.Lock()
        self._grey: np.ndarray | None = None
        self._rgb: np.ndarray | None = None
        self._t = 0.0
        self._still: np.ndarray | None = None
        self._still_t = 0.0
        self._size = (0, 0)
        self._clients: set[Any] = set()
        self._frames = 0
        self.caption_path = caption_path
        self._want_still = False
        self._phone_cmd_mtime = 0.0
        self._page_info: dict[str, Any] = {}
        self._thread = threading.Thread(target=self._serve, daemon=True)

    # --- server -----------------------------------------------------------------
    def start(self) -> str:
        self._thread.start()
        return f"https://{lan_ip()}:{self.port}/"

    def _serve(self) -> None:
        import uvicorn
        from starlette.applications import Starlette
        from starlette.responses import HTMLResponse, JSONResponse
        from starlette.routing import Route

        async def page(_: Any) -> HTMLResponse:
            return HTMLResponse(PAGE, headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"})

        async def frame(req: Any) -> JSONResponse:
            kind = 1 if req.query_params.get("kind") == "1" else 0
            body = await req.body()
            self._on_frame(bytes([kind]) + body)
            info = {"page": req.headers.get("x-page", ""), "last_command_result": req.headers.get("x-last", ""), "frames": self._frames, "t": time.time()}
            if info["page"] != self._page_info.get("page") or info["last_command_result"] != self._page_info.get("last_command_result"):
                self._page_info = info
                try:
                    PHONE_STATUS.write_text(json.dumps(info), encoding="utf-8")
                except OSError:
                    pass
            return JSONResponse({"ok": True, "n": self._frames})

        async def caption(_: Any) -> JSONResponse:
            cmd, args = ("still", {}) if self._want_still else self._pending_phone_command()
            self._want_still = False
            audio = ""
            try:
                if SPEECH_WAV.exists():
                    audio = str(int(SPEECH_WAV.stat().st_mtime * 1000))
            except OSError:
                pass
            return JSONResponse({"caption": self._current_caption(), "command": cmd, "args": args, "audio": audio})

        async def speech(_: Any) -> Any:
            from starlette.responses import FileResponse, PlainTextResponse

            return FileResponse(str(SPEECH_WAV), media_type="audio/wav", headers={"Cache-Control": "no-store"}) if SPEECH_WAV.exists() else PlainTextResponse("none", status_code=404)

        async def avatar(req: Any) -> Any:
            from starlette.responses import FileResponse, PlainTextResponse

            name = str(req.path_params.get("name", "")).replace("/", "").replace("\\", "")
            f = AVATAR_DIR / name
            return FileResponse(str(f)) if f.exists() and f.suffix == ".jpg" else PlainTextResponse("no such frame", status_code=404)

        crt, key = ensure_cert()
        app = Starlette(routes=[Route("/", page), Route("/frame", frame, methods=["POST"]), Route("/caption", caption),
                                Route("/avatar/{name}", avatar), Route("/speech/{sid}", speech)])
        uvicorn.run(app, host="0.0.0.0", port=self.port, ssl_certfile=str(crt), ssl_keyfile=str(key), log_level="warning")  # noqa: S104 - LAN page by design

    def _pending_phone_command(self) -> tuple[str, dict[str, Any]]:
        """A phone-side command (zoom / camera / stripes / clear / beep / still) written by tools/say.py --phone, once."""
        p = PHONE_COMMAND
        try:
            m = p.stat().st_mtime
        except OSError:
            return "", {}
        if m <= self._phone_cmd_mtime:
            return "", {}
        self._phone_cmd_mtime = m
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return str(d.get("command", "")), dict(d.get("args", {}))
        except (OSError, ValueError):
            return "", {}

    def _current_caption(self) -> str:
        p = self.caption_path
        if p is None or not p.exists():
            return ""
        try:
            if time.time() - p.stat().st_mtime > 8.0:
                return ""
            return p.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def _on_frame(self, data: bytes) -> None:
        import cv2

        kind, payload = data[0], data[1:]
        arr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        grey = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        with self._lock:
            if kind == 1:
                self._still, self._still_t = rgb, time.monotonic()
            else:
                self._grey, self._rgb, self._t = grey, rgb, time.monotonic()
                self._size = (arr.shape[1], arr.shape[0])
                self._frames += 1

    # --- OpenCVCamera-compatible surface -------------------------------------------
    def wait_for_frame(self, timeout: float = 30.0) -> None:
        t0 = time.monotonic()
        while self._grey is None:
            if time.monotonic() - t0 > timeout:
                raise RuntimeError("no frames from the phone yet: open the page on the phone and allow the camera")
            time.sleep(0.1)

    def read(self, max_age: float = 2.0) -> np.ndarray:
        with self._lock:
            if self._grey is None or time.monotonic() - self._t > max_age:
                raise RuntimeError("phone stream stalled")
            return self._grey.copy()

    def read_display(self) -> np.ndarray | None:
        with self._lock:
            return None if self._rgb is None else self._rgb.copy()

    def capture(self, count: int, timeout_per_frame: float = 2.0) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        last = -1
        while len(out) < count:
            t0 = time.monotonic()
            while self._frames == last:
                if time.monotonic() - t0 > timeout_per_frame:
                    raise RuntimeError("phone stream stalled during capture")
                time.sleep(0.005)
            with self._lock:
                last = self._frames
                if self._grey is not None:
                    out.append(self._grey.astype(np.float64))
        return out

    def request_still(self) -> np.ndarray | None:
        """Ask the phone for a full-resolution photo; returns it (RGB) or None after 8 s."""
        before = self._still_t
        self._want_still = True
        t0 = time.monotonic()
        while self._still_t == before:
            if time.monotonic() - t0 > 8.0:
                return None
            time.sleep(0.05)
        with self._lock:
            return None if self._still is None else self._still.copy()

    def set_mode(self, width: int, height: int, fourcc: str = "") -> None:  # the browser decides; recorded only
        return None

    def describe(self) -> str:
        return f"phone browser camera {self._size[0]}x{self._size[1]} over https://{lan_ip()}:{self.port}"

    def geometry(self) -> Any:
        from eczis.lab.sources import Geometry

        return Geometry(scale=1, origin_hr=(0.0, 0.0))

    def close(self) -> None:
        return None
