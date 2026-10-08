# UNFILED-IP
"""Always-on voice responder: whatever the operator says (phone mic, headset, Siri shortcut) gets an answer.

    uv run python tools/responder.py                      # answer everything new in the coach channel
    uv run python tools/responder.py --wake "hey coach"   # only lines that start with the wake phrase
    uv run python tools/responder.py --dry                # print replies, do not speak

Watches artifacts/coach/heard.jsonl for new lines, asks the local Claude Code CLI for a short spoken
reply (one conversation, resumed each turn), and speaks it through tools/say.py (ElevenLabs if the key
is present, Windows voice otherwise), with the phone avatar switched to listening / thinking / speaking.
Turn-based: a few seconds per exchange. Nothing from the scans is sent anywhere; only the operator's
words go to the model and only the reply text goes to the voice service. EXPERIMENTAL.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
HEARD = ROOT / "artifacts" / "coach" / "heard.jsonl"
SAY = ROOT / "tools" / "say.py"
SYSTEM = ("You are the bench coach for a camera and 3D measurement lab. Reply in one or two short spoken sentences, plain words, "
          "no lists, no markdown. If the operator asks for an action you cannot do, say what you can do instead. "
          "Never claim a measurement you have not been given.")


def say(text: str, avatar: str = "neutral", dry: bool = False) -> None:
    if dry:
        print("WOULD SAY:", text)
        return
    subprocess.run([sys.executable, str(SAY), text, "--phone", f"avatar={avatar}"], capture_output=True, timeout=120)


def avatar(state: str) -> None:
    subprocess.run([sys.executable, str(SAY), "--silent", "", "--phone", f"avatar={state}"], capture_output=True, timeout=30)


def ask_claude(text: str, session: list[str]) -> str:
    """One Claude Code CLI turn, continuing the responder's own conversation; falls back to a plain echo on failure."""
    cmd = ["claude", "-p", "--output-format", "text", "--append-system-prompt", SYSTEM, *session, text]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=90, encoding="utf-8", errors="ignore")
        out = (r.stdout or "").strip()
        if r.returncode == 0 and out:
            if not session:
                session.extend(["--continue"])
            return out.splitlines()[-1] if len(out) > 400 else out
        return "I heard you, but my reply did not come through. Say it again."
    except (OSError, subprocess.SubprocessError):
        return "I heard you, but the model is not answering right now."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wake", default="", help="only answer lines starting with this phrase (case-insensitive)")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--sources", default="phone-mic,siri-shortcut,headset", help="comma list of 'via' values to answer ('headset' = lines without via)")
    a = ap.parse_args()
    sources = {s.strip() for s in a.sources.split(",")}
    session: list[str] = []
    seen = HEARD.stat().st_size if HEARD.exists() else 0
    print(f"responder listening (wake={a.wake!r}, sources={sorted(sources)}); Ctrl+C to stop", flush=True)
    while True:
        try:
            size = HEARD.stat().st_size if HEARD.exists() else 0
        except OSError:
            size = 0
        if size > seen:
            with HEARD.open("r", encoding="utf-8") as f:
                f.seek(seen)
                new = f.read()
            seen = size
            for line in new.splitlines():
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                via = d.get("via", "headset")
                text = str(d.get("text", "")).strip()
                if via not in sources or not text:
                    continue
                if a.wake:
                    if not text.lower().startswith(a.wake.lower()):
                        continue
                    text = text[len(a.wake):].strip(" ,.") or "hello"
                print(f"[{time.strftime('%H:%M:%S')}] {via}: {text}", flush=True)
                avatar("thinking")
                reply = ask_claude(text, session)
                print(f"    -> {reply}", flush=True)
                say(reply, avatar="smile", dry=a.dry)
                avatar("listening")
        elif size < seen:
            seen = size
        time.sleep(0.5)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(0)
