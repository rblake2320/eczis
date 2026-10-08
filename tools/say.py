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
    a = ap.parse_args()
    if a.state or not a.text:
        print(json.dumps(json.loads(STATE.read_text(encoding="utf-8")), indent=1) if STATE.exists() else "no window state yet")
        return 0
    write_caption(a.text)
    print(f"caption: {a.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
