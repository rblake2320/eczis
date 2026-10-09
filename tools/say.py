# UNFILED-IP
"""Put words on the live lab window (and speak them): the agent side of the coach channel.

    uv run python tools/say.py "move closer and hold"
    uv run python tools/say.py --state            # print what the window currently sees
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def phone_linked(max_age_s: float = 6.0) -> bool:
    """True when the phone page posted a frame within max_age_s (its report file is rewritten on change; use mtime)."""
    import time

    from eczis.lab.phone_cam import PHONE_STATUS

    try:
        return time.time() - PHONE_STATUS.stat().st_mtime < max_age_s
    except OSError:
        return False


def main() -> int:
    from eczis.lab.coach import STATE, write_caption

    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="")
    ap.add_argument("--state", action="store_true")
    ap.add_argument("--silent", action="store_true", help="caption only, no voice")
    ap.add_argument("--local", action="store_true", help="play on the PC speakers even when the phone is linked")
    ap.add_argument("--command", choices=["record", "snapshot", "zoom_off"], default=None, help="press a button in the live window")
    ap.add_argument("--phone", default=None, help="phone-side command: still | zoom=2 | camera=back | stripes[=period] | clear | beep | avatar=<state>|off | voice=on|off")
    a = ap.parse_args()
    if a.phone:
        from eczis.lab.phone_cam import PHONE_COMMAND

        name, _, val = a.phone.partition("=")
        args: dict[str, object] = {}
        if name == "zoom":
            args = {"value": float(val or 2)}
        elif name == "camera":
            args = {"name": val or "back"}
        elif name == "stripes":
            args = {"period": int(val or 60), "orient": "V"}
        elif name == "avatar":
            args = {"state": val or "neutral"}
        elif name == "voice":
            args = {"on": (val or "on").lower() != "off"}
        PHONE_COMMAND.parent.mkdir(parents=True, exist_ok=True)
        PHONE_COMMAND.write_text(json.dumps({"command": name, "args": args, "t": __import__("time").time()}), encoding="utf-8")
        print(f"phone: {name} {args}")
        if not a.text and not a.command:
            return 0
    if a.command:
        from eczis.lab.coach import COMMAND

        COMMAND.parent.mkdir(parents=True, exist_ok=True)
        COMMAND.write_text(a.command, encoding="utf-8")
        print(f"command: {a.command}")
        if not a.text:
            return 0
    if a.state or not a.text:
        print(json.dumps(json.loads(STATE.read_text(encoding="utf-8")), indent=1) if STATE.exists() else "no window state yet")
        return 0
    if a.silent:
        write_caption(a.text)
        print(f"caption: {a.text}", flush=True)
        return 0
    import tempfile

    from eczis.lab.coach import play_wav_loud, render_speech

    wav = Path(tempfile.gettempdir()) / "eczis_say.wav"
    render_speech(a.text, wav)  # publishes speech_last.wav for the phone page first ...
    write_caption(a.text)  # ... then the caption, so the page pairs the text with its own audio
    print(f"caption: {a.text}", flush=True)
    if phone_linked() and not a.local:
        print("(phone is the voice; PC speakers quiet)")
    else:
        play_wav_loud(wav)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
