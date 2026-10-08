# UNFILED-IP
"""Caption / voice channel between a live lab window and an operator or AI coach.

The window calls `Coach.tick(state)` every frame: it writes the current state to STATE (JSON) so an
AI agent can read what the window sees, reads CAPTION (text) so the agent can put words on screen,
and speaks new captions aloud through the Windows speech synthesizer (optional). The window's own
rule-based guidance is spoken too, rate-limited, so the operator hears "tilt more" the moment it
applies; an agent's caption overrides it for CAPTION_HOLD_S seconds.

    uv run python tools/say.py "move closer and hold"      # from the agent side

Local only: nothing leaves the machine. EXPERIMENTAL.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CHANNEL = ROOT / "artifacts" / "coach"
STATE = CHANNEL / "state.json"
CAPTION = CHANNEL / "caption.txt"
COMMAND = CHANNEL / "command.txt"  # one word per write: record | snapshot | zoom_off; consumed once
CAPTION_HOLD_S = 6.0  # an agent caption stays on screen this long before the app's own guidance resumes
GUIDANCE_REPEAT_S = 8.0  # the app repeats the same guidance aloud no more often than this


_speak_lock = threading.Lock()
_pending: list[str] = []  # at most one queued utterance: the newest replaces older ones


TARGET_RMS = 0.22  # loudness of spoken lines (a 0.4-amplitude test tone is ~0.28); SAPI output is far quieter by default


def render_speech(text: str, path: Path) -> None:
    safe = text.replace("'", "")
    subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                    "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$s.SetOutputToWaveFile('{path}'); $s.Rate = 0; $s.Speak('{safe}'); $s.Dispose()"], capture_output=True, timeout=30)


SPEAKING = CHANNEL / "speaking"  # exists while the coach is talking, so the listener ignores the echo
VOICE_DEVICE_PREF = os.environ.get("ECZIS_VOICE_DEVICE", "Speakers (Realtek")  # substring; default output if absent


def output_device() -> int | None:
    """Output device matching the preference (WASAPI first); None = system default.

    On this bench the headset endpoint is mixed down by the vendor's virtual mixer and the voice is
    faint there, while the desk speakers are loud; the preference is an environment variable so the
    choice is explicit, not guessed per run.
    """
    import sounddevice as sd

    apis = [a["name"] for a in sd.query_hostapis()]
    found = [(i, apis[d["hostapi"]]) for i, d in enumerate(sd.query_devices())
             if d["max_output_channels"] > 0 and VOICE_DEVICE_PREF.lower() in d["name"].lower()]
    for want in ("Windows WASAPI", "Windows DirectSound", "MME"):
        for i, api in found:
            if api == want:
                return i
    return None


def play_wav_loud(path: Path, target_rms: float = TARGET_RMS) -> None:
    """Play a wav at a fixed loudness with a soft limiter (speech engines render quietly), on the preferred device."""
    import wave

    import numpy as np
    import sounddevice as sd

    with wave.open(str(path)) as w:
        rate = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        if w.getnchannels() == 2:
            data = data.reshape(-1, 2).mean(axis=1)
    rms = float(np.sqrt(np.mean(data * data))) or 1e-6
    data = np.tanh(data * (target_rms / rms) * 1.2) / 1.2  # gain to target loudness, soft-limit peaks
    n = int(len(data) * 48000 / rate)  # WASAPI endpoints want their native rate
    data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.float32)
    SPEAKING.parent.mkdir(parents=True, exist_ok=True)
    SPEAKING.write_text(str(time.time()), encoding="utf-8")
    try:
        sd.play(data, 48000, device=output_device())
        sd.wait()
    finally:
        time.sleep(0.4)
        SPEAKING.unlink(missing_ok=True)


def _speaker() -> None:
    wav = Path(tempfile.gettempdir()) / "eczis_coach_say.wav"
    while True:
        with _speak_lock:
            text = _pending.pop() if _pending else None
            _pending.clear()
        if text is None:
            time.sleep(0.1)
            continue
        try:
            render_speech(text, wav)
            play_wav_loud(wav)
        except Exception:  # noqa: BLE001 - audio is a convenience; the window must never die because of it
            pass


_thread: threading.Thread | None = None


def say_async(text: str) -> None:
    """Queue text for the single speaker thread: one voice at a time, newest wins if lines pile up."""
    global _thread
    with _speak_lock:
        _pending.clear()
        _pending.append(text)
        if _thread is None:
            _thread = threading.Thread(target=_speaker, daemon=True)
            _thread.start()


def write_caption(text: str, path: Path = CAPTION) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip(), encoding="utf-8")


class Coach:
    def __init__(self, voice: bool = True, state_path: Path = STATE, caption_path: Path = CAPTION, command_path: Path = COMMAND) -> None:
        self.voice = voice
        self.state_path, self.caption_path, self.command_path = state_path, caption_path, command_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._caption_mtime = 0.0
        self._caption, self._caption_at = "", 0.0
        self._last_guidance, self._last_guidance_at = "", 0.0
        self._last_state_write = 0.0
        self._command_mtime = 0.0

    def poll_command(self) -> str:
        """A command written since the last poll, or empty (each command is acted on once)."""
        path = self.command_path
        try:
            m = path.stat().st_mtime
        except OSError:
            return ""
        if m <= self._command_mtime:
            return ""
        self._command_mtime = m
        return path.read_text(encoding="utf-8").strip().lower()

    def tick(self, state: dict[str, Any], guidance: str = "") -> str:
        """Publish state, pick up a new caption, speak what changed; returns the text to draw."""
        now = time.monotonic()
        if now - self._last_state_write > 0.5:
            self._last_state_write = now
            try:
                self.state_path.write_text(json.dumps({**state, "guidance": guidance, "t": time.time()}, indent=1), encoding="utf-8")
            except OSError:
                pass
        try:
            m = self.caption_path.stat().st_mtime
        except OSError:
            m = 0.0
        if m > self._caption_mtime:
            self._caption_mtime = m
            text = self.caption_path.read_text(encoding="utf-8").strip()
            if text and text != self._caption:
                self._caption, self._caption_at = text, now  # spoken by the agent side (tools/say.py), shown here
        if self._caption and now - self._caption_at < CAPTION_HOLD_S:
            return f"COACH: {self._caption}"
        if guidance and self.voice and (guidance != self._last_guidance or now - self._last_guidance_at > GUIDANCE_REPEAT_S):
            self._last_guidance, self._last_guidance_at = guidance, now
            say_async(guidance)
        return guidance
