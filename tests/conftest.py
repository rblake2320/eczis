# UNFILED-IP
from __future__ import annotations

from pathlib import Path

import pytest

from eczis.bench.harness import run_demo
from eczis.bench.pipeline import CaseResult, run_case
from eczis.config import DemoConfig

# Small, fast configuration for adversarial/unit suites (same physics, smaller grid).
SMALL = DemoConfig(sensor_height=24, sensor_width=32, frame_count=9, max_iterations=12)
PLATFORM_NA = "platform-not-applicable"


@pytest.fixture(scope="session")
def demo_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, object]]:
    output = tmp_path_factory.mktemp("eczis-demo")
    result = run_demo(output, DemoConfig(max_iterations=8))
    return output, result


@pytest.fixture(scope="session")
def small_cfg() -> DemoConfig:
    return SMALL


@pytest.fixture(scope="session")
def coin_case() -> CaseResult:
    return run_case("coin", SMALL)
