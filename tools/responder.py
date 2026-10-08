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
SYSTEM = ("You are the bench coach for a camera and 3D measurement lab, speaking through a phone with a face on screen. "
          "The operator's words reach you as speech-to-text from the phone microphone and your reply is spoken aloud, so you do "
          "hear them and they hear you. Your reply IS spoken aloud by the phone, so never say you only send text or that you cannot "
          "tell whether the voice works. Answer naturally, as in a conversation. One or two short spoken sentences, plain words, "
          "no lists, no markdown. If asked for an action you cannot do, say what you can do instead. "
          "Never claim a measurement you have not been given.")


def say(text: str, avatar: str = "neutral", dry: bool = False) -> None:
    if dry:
        print("WOULD SAY:", text)
        return
    subprocess.run([sys.executable, str(SAY), text, "--phone", f"avatar={avatar}"], capture_output=True, timeout=120)


def avatar(state: str) -> None:
    subprocess.run([sys.executable, str(SAY), "--silent", "", "--phone", f"avatar={state}"], capture_output=True, timeout=30)


def claude_exe() -> str:
    import shutil

    for name in ("claude", "claude.cmd", "claude.exe"):
        found = shutil.which(name)
        if found:
            return found
    return "claude"


def ask_claude(text: str, session: list[str]) -> str:
    """One Claude Code CLI turn, continuing the responder's own conversation; falls back to a plain echo on failure."""
    cmd = [claude_exe(), "-p", "--output-format", "text", "--append-system-prompt", SYSTEM, *session, text]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=90, encoding="utf-8", errors="ignore",
                           shell=cmd[0].lower().endswith(".cmd"))
        out = (r.stdout or "").strip()
        if r.returncode == 0 and out:
            if not session:
                session.extend(["--continue"])
            return out.splitlines()[-1] if len(out) > 400 else out
        return "I heard you, but my reply did not come through. Say it again."
    except (OSError, subprocess.SubprocessError):
        return "I heard you, but the model is not answering right now."
    except Exception as exc:  # noqa: BLE001 - never let the loop die on an odd shell error
        print("claude call failed:", exc, flush=True)
        return "I heard you, but the model is not answering right now."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wake", default="", help="only answer lines starting with this phrase (case-insensitive)")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--sources", default="phone-mic,siri-shortcut,headset", help="comma list of 'via' values to answer ('headset' = lines without via)")
    a = ap.parse_args()
    sources = {s.strip() for s in a.sources.split(",")}
    session: list[str] = []
    spoken: list[str] = []  # the responder's own recent lines, so hearing itself through a speaker is ignored
    seen = HEARD.stat().st_size if HEARD.exists() else 0

    def is_echo(said: str) -> bool:
        """The responder heard itself through a speaker: the text is nearly the same as one of its recent lines."""
        import difflib

        s = said.lower().strip()
        return any(difflib.SequenceMatcher(None, s, prev.lower().strip()).ratio() >= 0.75 for prev in spoken[-4:])
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
                if is_echo(text):
                    print(f"[{time.strftime('%H:%M:%S')}] echo ignored: {text[:60]}", flush=True)
                    continue
                print(f"[{time.strftime('%H:%M:%S')}] {via}: {text}", flush=True)
                avatar("thinking")
                reply = ask_claude(text, session)
                print(f"    -> {reply}", flush=True)
                spoken.append(reply)
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
