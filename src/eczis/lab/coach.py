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
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CHANNEL = ROOT / "artifacts" / "coach"
STATE = CHANNEL / "state.json"
CAPTION = CHANNEL / "caption.txt"
CAPTION_HOLD_S = 6.0  # an agent caption stays on screen this long before the app's own guidance resumes
GUIDANCE_REPEAT_S = 8.0  # the app repeats the same guidance aloud no more often than this


_speak_lock = threading.Lock()
_pending: list[str] = []  # at most one queued utterance: the newest replaces older ones


def _speaker() -> None:
    while True:
        with _speak_lock:
            text = _pending.pop() if _pending else None
            _pending.clear()
        if text is None:
            time.sleep(0.1)
            continue
        safe = text.replace("'", "")
        try:
            subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                            "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                            f"$s.Volume = 100; $s.Rate = 0; $s.Speak('{safe}')"], capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
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
    def __init__(self, voice: bool = True, state_path: Path = STATE, caption_path: Path = CAPTION) -> None:
        self.voice = voice
        self.state_path, self.caption_path = state_path, caption_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._caption_mtime = 0.0
        self._caption, self._caption_at = "", 0.0
        self._last_guidance, self._last_guidance_at = "", 0.0
        self._last_state_write = 0.0

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
                self._caption, self._caption_at = text, now
                if self.voice:
                    say_async(text)
        if self._caption and now - self._caption_at < CAPTION_HOLD_S:
            return f"COACH: {self._caption}"
        if guidance and self.voice and (guidance != self._last_guidance or now - self._last_guidance_at > GUIDANCE_REPEAT_S):
            self._last_guidance, self._last_guidance_at = guidance, now
            say_async(guidance)
        return guidance
