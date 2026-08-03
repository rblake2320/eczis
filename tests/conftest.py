# UNFILED-IP
from __future__ import annotations

from pathlib import Path

import pytest

from eczis.bench.harness import run_demo
from eczis.config import DemoConfig


@pytest.fixture(scope="session")
def demo_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, object]]:
    output = tmp_path_factory.mktemp("eczis-demo")
    result = run_demo(output, DemoConfig(max_iterations=8))
    return output, result
