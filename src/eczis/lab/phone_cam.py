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
CHANNEL_HEARD = ROOT / "artifacts" / "coach" / "heard.jsonl"  # same log the microphone listener appends to
LATEST_JPG = ROOT / "artifacts" / "coach" / "latest.jpg"  # the phone's live view, refreshed ~1 Hz, for the responder
PORT = 8443

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ECZIS phone camera</title><style>
body{margin:0;background:#000;color:#fff;font-family:system-ui;display:flex;flex-direction:column;height:100vh;height:100dvh;overflow:hidden}
video{width:100%;max-height:52vh;flex:0 1 auto;object-fit:contain;background:#111}
#bar{flex:0 0 auto}#bar button{min-height:48px}
#cap{font-size:22px;padding:12px;text-align:center;min-height:1.4em;background:#000}
#bar{display:flex;gap:8px;padding:8px}button{flex:1;font-size:18px;padding:12px;border:0;border-radius:8px;background:#2563eb;color:#fff}
#st{font-size:13px;color:#9ca3af;text-align:center;padding:4px}</style></head><body>
<canvas id="pat" style="display:none;position:fixed;inset:0;z-index:9"></canvas>
<audio id="spk" playsinline preload="auto"></audio>
<img id="ava" src="/avatar/neutral.jpg" style="position:fixed;top:10px;right:10px;width:34vw;max-width:220px;border-radius:14px;box-shadow:0 2px 14px #000;z-index:8;display:block">
<video id="v" autoplay playsinline muted></video><div id="cap"></div>
<div id="bar"><button id="still">Still (full res)</button><button id="flip">Flip</button><button id="mute" style="background:#15803d">&#128266; Voice: ON</button><button id="mic">Auto-listen off</button><button id="ptt" style="background:#b91c1c">Hold to talk</button></div>
<select id="cam" style="font-size:16px;margin:0 8px 6px 8px;padding:8px;background:#111;color:#fff;border:1px solid #444"></select>
<div id="st">connecting...</div>
<script>
const st=document.getElementById('st'),cap=document.getElementById('cap'),v=document.getElementById('v');
let facing='environment',stream,ws,voice=true,lastCap='',sending=false,deviceId=null,micOn=false,rec=null,micStream=null,ana=null,ac=null;
const sel=document.getElementById('cam');
async function listCams(){ const devs=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput'); sel.innerHTML='';
 for(const d of devs){const o=document.createElement('option');o.value=d.deviceId;o.textContent=d.label||('camera '+(sel.length+1));sel.appendChild(o);}
 const single=devs.find(d=>/^back camera$/i.test(d.label))||devs.find(d=>/back/i.test(d.label)&&!/dual|triple|wide/i.test(d.label));
 if(single&&!deviceId){deviceId=single.deviceId;sel.value=deviceId;} }
let wantMic=true;  // one permission prompt covers camera and microphone
async function start(){ if(stream)stream.getTracks().forEach(t=>t.stop());
 const cons=deviceId?{deviceId:{exact:deviceId},width:{ideal:1920},height:{ideal:1080}}:{facingMode:facing,width:{ideal:1920},height:{ideal:1080}};
 stream=await navigator.mediaDevices.getUserMedia({video:cons,audio:wantMic?{echoCancellation:true,noiseSuppression:true}:false});
 if(wantMic&&stream.getAudioTracks().length){ micStream=new MediaStream(stream.getAudioTracks()); ac=ac||new (window.AudioContext||window.webkitAudioContext)(); const src=ac.createMediaStreamSource(micStream); ana=ac.createAnalyser(); ana.fftSize=512; src.connect(ana); if(!micOn){micOn=true;} if(autoListen)loopMic(); }
 v.srcObject=stream; await listCams(); const tr=stream.getVideoTracks()[0]; const s=tr.getSettings();
 st.textContent=`camera ${s.width}x${s.height} ${tr.label}`; tr.onmute=()=>{st.textContent='camera paused by iOS (lens switch?) - restarting';setTimeout(start,500);}; }
sel.onchange=()=>{deviceId=sel.value;start();};
let linked=false; async function poll(){ try{ const r=await fetch('/caption',{cache:'no-store'}); const m=await r.json(); if(!linked){linked=true;st.textContent=st.textContent.replace(' | link lost','')+' | linked';}
   if(m.caption!==undefined&&m.caption!==lastCap){lastCap=m.caption;cap.textContent=m.caption;
   if(voice&&m.caption){ vd('caption audio='+(m.audio?'yes':'no')+' unlocked='+audioOk); if(m.audio&&audioOk){ playPc(m.audio,m.caption); } else sayLocal(m.caption); } else vd('caption but voice '+(voice?'on':'OFF')+' empty='+(!m.caption)); }
   if(m.command)handle(m.command,m.args||{}); }catch(e){ if(linked){linked=false;st.textContent+=' | link lost';} } setTimeout(poll,500); }
const pat=document.getElementById('pat'),ava=document.getElementById('ava');
let audioOk=false;
let curSrc=null;
async function playPc(id,text){ try{ ac=ac||new (window.AudioContext||window.webkitAudioContext)(); if(ac.state!=='running')await ac.resume(); vd('fetching ac='+ac.state);
  const r=await fetch('/speech/'+id+'.wav',{cache:'no-store'}); const ab=await r.arrayBuffer(); vd('fetched '+ab.byteLength+'B'); const buf=await ac.decodeAudioData(ab); vd('decoded '+buf.duration.toFixed(1)+'s playing');
  if(curSrc){try{curSrc.stop();}catch(e){}} const s=ac.createBufferSource(); s.buffer=buf; s.connect(ac.destination); curSrc=s; startMouth(); s.onended=()=>{stopMouth(); vd('played '+buf.duration.toFixed(1)+'s'); st.textContent=st.textContent.replace(/ [|] played.*$/,'')+' | played '+buf.duration.toFixed(1)+'s'; }; s.start(); }
 catch(e){ stopMouth(); vd('FAILED '+String(e).slice(0,60)+' -> phone voice'); st.textContent='pc voice failed: '+e+' - using phone voice'; sayLocal(text); } }
function sayLocal(text){ if('speechSynthesis'in window){speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(text);u.rate=1.0;u.onstart=()=>{startMouth();vd('phone voice speaking');};u.onend=()=>{stopMouth();vd('phone voice done');};u.onerror=e=>{stopMouth();vd('phone voice error '+e.error);};speechSynthesis.speak(u);} else vd('no speechSynthesis'); }
function unlock(){ try{ ac=ac||new (window.AudioContext||window.webkitAudioContext)(); ac.resume(); const el=document.getElementById('spk'); el.muted=false; el.play().catch(()=>{}); audioOk=true; st.textContent=st.textContent.replace(' | tap to enable voice','')+' | voice unlocked'; }catch(e){} }
document.addEventListener('touchend',unlock,{once:true}); document.addEventListener('click',unlock,{once:true});

async function micStart(){ if(micOn)return; wantMic=true; try{ ac=ac||new (window.AudioContext||window.webkitAudioContext)(); await ac.resume(); await start(); }catch(e){st.textContent='mic denied: '+e; wantMic=false;} }
let micDiag='mic:off',voiceDiag='voice:idle';
function vd(s){ voiceDiag='voice:'+s; }
function micReport(){ try{ const lvl=(micOn&&ana)?micLevel().toFixed(3):'-'; micDiag='mic:'+(micOn?'on':'off')+' auto:'+(autoListen?'on':'off')+' tracks:'+(stream?stream.getAudioTracks().length:0)+' ac:'+(ac?ac.state:'none')+' lvl:'+lvl+' rec:'+(rec?rec.state:'none')+' MR:'+(('MediaRecorder' in window)?'yes':'no'); }catch(e){micDiag='mic:err '+e;} }
setInterval(micReport,500);
let pttRec=null,pttChunks=[];
const ptt=document.getElementById('ptt');
async function pttStart(ev){ ev.preventDefault(); if(!micOn){ await micStart(); } if(!micStream){st.textContent='no mic stream';return;} try{ pttChunks=[]; pttRec=new MediaRecorder(micStream); pttRec.ondataavailable=e=>{if(e.data.size)pttChunks.push(e.data);}; pttRec.onstop=async()=>{ const blob=new Blob(pttChunks,{type:pttRec.mimeType||'audio/mp4'}); st.textContent='sending '+Math.round(blob.size/1024)+' kB'; try{ await fetch('/hear',{method:'POST',body:blob,headers:{'Content-Type':blob.type}}); st.textContent='sent '+Math.round(blob.size/1024)+' kB of speech'; }catch(e){st.textContent='hear failed: '+e;} }; pttRec.start(); ptt.style.background='#15803d'; ptt.textContent='Talking... release to send'; }catch(e){st.textContent='recorder: '+e;} }
function pttStop(ev){ if(ev)ev.preventDefault(); if(pttRec&&pttRec.state==='recording'){pttRec.stop();} ptt.style.background='#b91c1c'; ptt.textContent='Hold to talk'; }
ptt.addEventListener('touchstart',pttStart,{passive:false}); ptt.addEventListener('touchend',pttStop,{passive:false}); ptt.addEventListener('touchcancel',pttStop,{passive:false});
ptt.addEventListener('mousedown',pttStart); ptt.addEventListener('mouseup',pttStop);
function micLevel(){ const d=new Uint8Array(ana.fftSize); ana.getByteTimeDomainData(d); let s=0; for(const v of d){const x=(v-128)/128; s+=x*x;} return Math.sqrt(s/d.length); }
let talking=false,quietN=0,chunks=[];
function loopMic(){ if(!micOn||!autoListen)return; if(ac&&ac.state!=='running'){ac.resume().catch(()=>{});} const lvl=micLevel(); if(lvl>0.02&&!talking&&!mouthTimer){ talking=true;quietN=0;chunks=[]; try{ rec=new MediaRecorder(micStream); rec.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);}; rec.onstop=sendChunk; recStart=Date.now(); rec.start(); }catch(e){st.textContent='recorder: '+e;talking=false;} }
  else if(talking){ if(lvl<0.012)quietN++; else quietN=0; if(quietN>=12||(rec&&rec.state==='recording'&&chunksSec()>12)){ talking=false; try{rec.stop();}catch(e){} } }
  setTimeout(loopMic,80); }
let recStart=0; function chunksSec(){return (Date.now()-recStart)/1000;}
async function sendChunk(){ const blob=new Blob(chunks,{type:rec.mimeType||'audio/mp4'}); if(blob.size<2000)return; try{ await fetch('/hear',{method:'POST',body:blob,headers:{'Content-Type':blob.type}}); st.textContent='sent '+Math.round(blob.size/1024)+' kB of speech'; }catch(e){st.textContent='hear failed: '+e;} }
let autoListen=false;
document.getElementById('mic').onclick=async()=>{ autoListen=!autoListen; document.getElementById('mic').textContent=autoListen?'Auto-listen on':'Auto-listen off'; if(autoListen){ if(!micOn)await micStart(); ac&&ac.resume(); loopMic(); } };
let avaState='neutral',mouthTimer=null,blinkTimer=null;
function showFrame(n){ava.src='/avatar/'+n+'.jpg';}
function setAvatar(s){ avaState=s; if(!mouthTimer)showFrame(s); lastResult='avatar:'+s; }
function startMouth(){ if(mouthTimer)return; const fr=['speak-small','speak-mid','speak-wide','speak-mid']; let i=0; mouthTimer=setInterval(()=>{showFrame(fr[i++%fr.length]);},140); }
function stopMouth(){ if(mouthTimer){clearInterval(mouthTimer);mouthTimer=null;} showFrame(avaState); }
blinkTimer=setInterval(()=>{ if(!mouthTimer&&avaState==='neutral'){showFrame('blink');setTimeout(()=>{if(!mouthTimer)showFrame(avaState);},160);} },4500+Math.random()*2000);
async function handle(cmd,a){ lastResult=cmd+':start'; setTimeout(()=>{lastResult='';},5000); try{
  if(cmd==='still')sendStill();
  else if(cmd==='zoom'){const tr=stream.getVideoTracks()[0];const cap=tr.getCapabilities?tr.getCapabilities():{};
    if(cap.zoom){const z=Math.min(cap.zoom.max,Math.max(cap.zoom.min,a.value||1));await tr.applyConstraints({advanced:[{zoom:z}]});st.textContent='zoom '+z+'x';lastResult='zoom:ok '+z;}else{st.textContent='zoom not supported here';lastResult='zoom:unsupported caps='+Object.keys(cap).join(',');}}
  else if(cmd==='camera'){const o=[...sel.options].find(o=>o.textContent.toLowerCase().includes((a.name||'back').toLowerCase()));if(o){deviceId=o.value;sel.value=deviceId;start();}}
  else if(cmd==='stripes'){pat.style.display='block';drawStripes(a.period||60,a.phase||0,a.orient||'V');lastResult='stripes:shown '+pat.width+'x'+pat.height;}
  else if(cmd==='clear'){pat.style.display='none';lastResult='clear:ok';}
  else if(cmd==='voice'){ setVoice(a.on!==false); audioOk=true; if(ac)ac.resume(); lastResult='voice:'+(voice?'on':'off'); }
  else if(cmd==='avatar'){ if(a.state==='off'){ava.style.display='none';lastResult='avatar:off';} else {ava.style.display='block';setAvatar(a.state||'neutral');} }
  else if(cmd==='beep'){ac=ac||new (window.AudioContext||window.webkitAudioContext)();const o=ac.createOscillator();o.frequency.value=880;o.connect(ac.destination);o.start();setTimeout(()=>o.stop(),250);lastResult='beep:'+ac.state;}
 }catch(e){st.textContent='command failed: '+e;lastResult=cmd+':error '+e} }
function drawStripes(period,phase,orient){ pat.width=innerWidth;pat.height=innerHeight;const ctx2=pat.getContext('2d'),w=pat.width,h=pat.height,img=ctx2.createImageData(w,h),d=img.data;
  for(let y=0;y<h;y++){for(let x=0;x<w;x++){const u=orient==='H'?y:x;const g=Math.round(255*(0.5+0.5*Math.cos(2*Math.PI*u/period+phase)));const i=4*(y*w+x);d[i]=g;d[i+1]=g;d[i+2]=g;d[i+3]=255;}} ctx2.putImageData(img,0,0); }
function connect(){ poll(); }
const c=document.createElement('canvas');
const PAGE_VERSION='cmd-11-voicebtn'; let lastResult='';
async function post(kind,blob){ const r=await fetch('/frame?kind='+kind,{method:'POST',body:blob,headers:{'Content-Type':'image/jpeg','X-Page':PAGE_VERSION,'X-Last':(lastResult||(micDiag+' '+voiceDiag)).slice(0,200)}}); return r.ok; }
function frame(){ if(!sending&&v.videoWidth){sending=true;c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0);
  c.toBlob(b=>{post(0,b).catch(()=>{}).finally(()=>{sending=false;});},'image/jpeg',0.85);}
  setTimeout(frame,80); }
async function sendStill(){ try{ const tr=stream.getVideoTracks()[0]; let blob;
  if('ImageCapture'in window){ blob=await new ImageCapture(tr).takePhoto(); } else { c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0); blob=await new Promise(r=>c.toBlob(r,'image/jpeg',0.95)); }
  await post(1,blob); st.textContent='still sent '+Math.round(blob.size/1024)+' kB'; }catch(e){st.textContent='still failed: '+e} }
document.getElementById('still').onclick=sendStill;
document.getElementById('flip').onclick=()=>{facing=facing==='environment'?'user':'environment';deviceId=null;start();};
function setVoice(on){ voice=on; const b=document.getElementById('mute'); b.innerHTML=on?'&#128266; Voice: ON':'&#128263; Voice: OFF'; b.style.background=on?'#15803d':'#6b7280'; vd('set '+(on?'on':'off')); }
document.getElementById('mute').onclick=()=>setVoice(!voice);
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
        self._latest_written = 0.0
        self._speech_fetches = 0
        self._thread = threading.Thread(target=self._serve, daemon=True)

    # --- server -----------------------------------------------------------------
    def start(self) -> str:
        self._thread.start()
        threading.Thread(target=self._serve_ask, daemon=True).start()
        return f"https://{lan_ip()}:{self.port}/"

    _whisper: Any = None

    def _transcribe(self, path: Path) -> None:
        """Phone mic chunk -> text in the heard log (faster-whisper on the GPU; the file is deleted after)."""
        try:
            if PhoneCamera._whisper is None:
                from faster_whisper import WhisperModel

                try:
                    PhoneCamera._whisper = WhisperModel("base.en", device="cuda", compute_type="float16")
                except Exception:  # noqa: BLE001
                    PhoneCamera._whisper = WhisperModel("base.en", device="cpu", compute_type="int8")
            segs, _ = PhoneCamera._whisper.transcribe(str(path), language="en", beam_size=1, vad_filter=True)
            text = " ".join(s.text.strip() for s in segs).strip()
            if text:
                CHANNEL_HEARD.parent.mkdir(parents=True, exist_ok=True)
                with CHANNEL_HEARD.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"t": time.time(), "text": text, "via": "phone-mic"}) + "\n")
        except Exception as exc:  # noqa: BLE001 - transcription is a convenience; log and move on
            try:
                (CHANNEL_HEARD.parent / "phone_mic_errors.log").open("a", encoding="utf-8").write(f"{time.time()} {exc}\n")
            except OSError:
                pass
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _serve_ask(self) -> None:
        """Plain-HTTP text intake for Siri Shortcuts (they reject the self-signed certificate): POST /ask {"text": ...}
        appends the words to the coach's heard log exactly as the microphone listener does, so the agent answers
        through the page (caption, voice, face). Text only, LAN only."""
        import uvicorn
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        heard = CHANNEL_HEARD

        async def ask(req: Any) -> JSONResponse:
            try:
                data = await req.json()
            except ValueError:
                data = {"text": (await req.body()).decode("utf-8", "ignore")}
            text = str(data.get("text", "")).strip()
            if not text:
                return JSONResponse({"ok": False, "error": "no text"}, status_code=400)
            heard.parent.mkdir(parents=True, exist_ok=True)
            with heard.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.time(), "text": text, "via": "siri-shortcut"}) + "\n")
            return JSONResponse({"ok": True, "heard": text})

        app = Starlette(routes=[Route("/ask", ask, methods=["POST"])])
        uvicorn.run(app, host="0.0.0.0", port=self.port + 1, log_level="warning")  # noqa: S104 - LAN text intake by design

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
            info = {**{k: v for k, v in self._page_info.items() if k.startswith("speech") or k.startswith("last_speech")},
                    "page": req.headers.get("x-page", ""), "last_command_result": req.headers.get("x-last", ""), "frames": self._frames, "t": time.time()}
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

        async def speech(req: Any) -> Any:
            from starlette.responses import FileResponse, PlainTextResponse

            self._speech_fetches += 1
            self._page_info = {**self._page_info, "speech_fetches": self._speech_fetches, "last_speech_fetch": time.time(),
                               "last_speech_id": str(req.path_params.get("sid", ""))}
            try:
                PHONE_STATUS.write_text(json.dumps(self._page_info), encoding="utf-8")
            except OSError:
                pass
            return FileResponse(str(SPEECH_WAV), media_type="audio/wav", headers={"Cache-Control": "no-store"}) if SPEECH_WAV.exists() else PlainTextResponse("none", status_code=404)

        async def avatar(req: Any) -> Any:
            from starlette.responses import FileResponse, PlainTextResponse

            name = str(req.path_params.get("name", "")).replace("/", "").replace("\\", "")
            f = AVATAR_DIR / name
            return FileResponse(str(f)) if f.exists() and f.suffix == ".jpg" else PlainTextResponse("no such frame", status_code=404)

        async def hear(req: Any) -> JSONResponse:
            body = await req.body()
            ext = ".mp4" if "mp4" in req.headers.get("content-type", "") else ".webm"
            path = CHANNEL_HEARD.parent / f"phone_mic_{int(time.time() * 1000)}{ext}"
            path.write_bytes(body)
            threading.Thread(target=self._transcribe, args=(path,), daemon=True).start()
            return JSONResponse({"ok": True, "bytes": len(body)})

        crt, key = ensure_cert()
        app = Starlette(routes=[Route("/", page), Route("/frame", frame, methods=["POST"]), Route("/caption", caption), Route("/hear", hear, methods=["POST"]),
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
                if time.monotonic() - self._latest_written > 1.0:  # a 1 Hz copy of the live view for the responder/agents
                    self._latest_written = time.monotonic()
                    try:
                        LATEST_JPG.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(LATEST_JPG), cv2.resize(arr, (arr.shape[1] // 2, arr.shape[0] // 2)))
                    except (OSError, cv2.error):
                        pass

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
