# UNFILED-IP
"""Listen on the microphone and write what is said into the coach channel (local Whisper, nothing leaves the PC).

    python tools/listen.py --list                 # show input devices
    python tools/listen.py --device 3 --model base.en

Runs with the system Python that has faster-whisper + sounddevice (not the project venv). Records in
short chunks, skips silence by an energy threshold, transcribes each utterance and appends
{"t", "text"} lines to artifacts/coach/heard.jsonl; the agent polls that file. Speech-to-text here
is an operator convenience, never evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HEARD = Path(__file__).resolve().parents[1] / "artifacts" / "coach" / "heard.jsonl"
SPEAKING = HEARD.with_name("speaking")  # written by the coach while it talks: drop what the mic hears then (echo)
RATE = 16000


def main() -> int:
    import sounddevice as sd

    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--model", default="base.en")
    ap.add_argument("--chunk-s", type=float, default=3.0)
    ap.add_argument("--threshold", type=float, default=0.01, help="RMS below this is silence")
    ap.add_argument("--compute", default="auto", help="cuda | cpu | auto")
    a = ap.parse_args()
    if a.list:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(i, d["name"], "| default" if i == sd.default.device[0] else "")
        return 0
    from faster_whisper import WhisperModel

    device = a.compute
    if device == "auto":
        try:
            model = WhisperModel(a.model, device="cuda", compute_type="float16")
            device = "cuda"
        except Exception as exc:  # noqa: BLE001 - CUDA/cuDNN missing is the common case; CPU is fine for short commands
            print(f"cuda unavailable ({str(exc)[:80]}); using cpu", flush=True)
            model = WhisperModel(a.model, device="cpu", compute_type="int8")
            device = "cpu"
    else:
        model = WhisperModel(a.model, device=device, compute_type="float16" if device == "cuda" else "int8")
    HEARD.parent.mkdir(parents=True, exist_ok=True)
    print(f"listening on device {a.device if a.device is not None else 'default'} with {a.model} on {device}; Ctrl+C to stop", flush=True)
    buf: list[np.ndarray] = []
    quiet = 0
    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", device=a.device, blocksize=int(RATE * 0.5)) as stream:
        while True:
            block, _ = stream.read(int(RATE * 0.5))
            x = block[:, 0]
            rms = float(np.sqrt(np.mean(x * x)))
            if SPEAKING.exists():  # the coach is talking: do not transcribe its own voice
                buf, quiet = [], 0
                continue
            if rms > a.threshold:
                buf.append(x)
                quiet = 0
            elif buf:
                quiet += 1
                buf.append(x)
            if buf and (quiet >= 2 or len(buf) * 0.5 >= 12):  # ~1 s of silence ends the utterance, or 12 s max
                audio = np.concatenate(buf)
                buf, quiet = [], 0
                if len(audio) < RATE * 0.6:
                    continue
                segs, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=True)
                text = " ".join(s.text.strip() for s in segs).strip()
                if text:
                    with HEARD.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"t": time.time(), "text": text}) + "\n")
                    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(0)
