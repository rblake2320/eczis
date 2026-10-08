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


def main() -> int:
    from eczis.lab.coach import STATE, write_caption

    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="")
    ap.add_argument("--state", action="store_true")
    ap.add_argument("--silent", action="store_true", help="caption only, no voice")
    ap.add_argument("--command", choices=["record", "snapshot", "zoom_off"], default=None, help="press a button in the live window")
    a = ap.parse_args()
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
    write_caption(a.text)
    print(f"caption: {a.text}", flush=True)
    if not a.silent:
        import tempfile

        from eczis.lab.coach import play_wav_loud, render_speech

        wav = Path(tempfile.gettempdir()) / "eczis_say.wav"
        render_speech(a.text, wav)
        play_wav_loud(wav)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
