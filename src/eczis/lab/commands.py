# UNFILED-IP
"""Instrument control protocol, control levels and controllers.

Line protocol (ASCII, one command per line, e.g. sent over serial to an Arduino/ESP32):

    FOCUS +12            relative focus steps
    STAGE_X -30          relative X stage steps (STAGE_Y likewise)
    LIGHT SIDE_LEFT 70   illumination channel and level 0..100
    ZOOM 2               optical magnification setting (if the hardware has one)
    CAPTURE 16           capture a burst of N frames

Digital zoom is deliberately NOT an instrument command: it adds no measurement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol


class ControlLevel(IntEnum):
    MANUAL = 0  # app tells the operator what to adjust
    ASSISTED = 1  # app detects when the requested focus/position is reached
    SEMI_AUTOMATIC = 2  # app sends commands to lights / focus motor / stage
    AUTOMATIC = 3  # app executes the selected capture end to end
    AUTONOMOUS = 4  # app chooses and runs the next measurement until certified or exhausted


LIGHT_CHANNELS = ("RING", "SIDE_LEFT", "SIDE_RIGHT", "TOP", "BOTTOM", "POLARIZED")


@dataclass(frozen=True, slots=True)
class Command:
    verb: str
    args: tuple[str | int, ...]

    def encode(self) -> str:
        parts = [self.verb]
        for a in self.args:
            parts.append(f"{a:+d}" if isinstance(a, int) and self.verb in ("FOCUS", "STAGE_X", "STAGE_Y") else str(a))
        return " ".join(parts)


_PATTERNS = {
    "FOCUS": re.compile(r"^FOCUS ([+-]?\d{1,5})$"),
    "STAGE_X": re.compile(r"^STAGE_X ([+-]?\d{1,6})$"),
    "STAGE_Y": re.compile(r"^STAGE_Y ([+-]?\d{1,6})$"),
    "LIGHT": re.compile(r"^LIGHT ([A-Z_]+) (\d{1,3})$"),
    "ZOOM": re.compile(r"^ZOOM (\d{1,2})$"),
    "CAPTURE": re.compile(r"^CAPTURE (\d{1,3})$"),
}


def parse(line: str) -> Command:
    """Parse and validate one protocol line; raises ValueError on anything malformed."""
    text = line.strip().upper()
    verb = text.split(" ", 1)[0]
    pattern = _PATTERNS.get(verb)
    if pattern is None:
        raise ValueError(f"unknown command: {line!r}")
    match = pattern.match(text)
    if match is None:
        raise ValueError(f"malformed command: {line!r}")
    if verb == "LIGHT":
        channel, level = match.group(1), int(match.group(2))
        if channel not in LIGHT_CHANNELS or not 0 <= level <= 100:
            raise ValueError(f"invalid light command: {line!r}")
        return Command(verb, (channel, level))
    value = int(match.group(1))
    if verb == "CAPTURE" and not 1 <= value <= 256:
        raise ValueError("CAPTURE count must be 1..256")
    if verb == "ZOOM" and value < 1:
        raise ValueError("ZOOM must be >= 1")
    return Command(verb, (value,))


class Controller(Protocol):
    def send(self, command: Command) -> str: ...


@dataclass
class LoggingController:
    """Records commands; used for MANUAL/ASSISTED levels (operator performs them)."""

    log: list[str] = field(default_factory=list)

    def send(self, command: Command) -> str:
        self.log.append(command.encode())
        return "QUEUED_FOR_OPERATOR"


class SerialController:
    """Sends protocol lines over a serial port and returns the device's one-line reply."""

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 5.0) -> None:
        import serial  # pyserial, optional 'lab' dependency group

        self._port = serial.Serial(port, baudrate=baudrate, timeout=timeout)

    def send(self, command: Command) -> str:
        self._port.write((command.encode() + "\n").encode("ascii"))
        reply = str(self._port.readline().decode("ascii", errors="replace")).strip()
        if not reply:
            raise TimeoutError(f"no reply to {command.encode()!r}")
        return reply

    def close(self) -> None:
        self._port.close()
