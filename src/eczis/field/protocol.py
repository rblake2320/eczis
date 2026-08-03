# UNFILED-IP
from __future__ import annotations

from typing import Protocol

from eczis.schemas import QueryResult


class ContinuousRenderer(Protocol):
    def query(self, x: float, y: float, scale: float) -> QueryResult: ...
