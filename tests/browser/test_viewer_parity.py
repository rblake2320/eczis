# UNFILED-IP
"""Visible-browser regression: viewer inspection must match Python rendering.

Run with ECZIS_BROWSER_TESTS=1 (headed by default; ECZIS_HEADLESS=1 for headless).
Requires the `browser` dependency group and `uv run playwright install chromium`.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eczis.export.payload import write_offline_viewer
from eczis.field.tiled_raster import TiledRasterRenderer
from eczis.schemas import ProvenanceClass

pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(os.environ.get("ECZIS_BROWSER_TESTS") != "1", reason="set ECZIS_BROWSER_TESTS=1"),
]

NATIVE = 4
ORDER = ["U", "I", "R", "M"]


def _mixed_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = 12, 16
    rng = np.random.default_rng(20261006)
    image = rng.random((h, w)).astype(np.float32)
    prov = np.full((h, w), int(ProvenanceClass.R), dtype=np.uint8)
    prov[:, 8:] = int(ProvenanceClass.U)  # vertical R/U edge
    prov[6:, :] = np.minimum(prov[6:, :], int(ProvenanceClass.I))  # horizontal edge
    prov[3, 3] = int(ProvenanceClass.I)  # isolated diagonal contributor
    conf = rng.random((h, w)).astype(np.float32)
    return image, prov, conf


@pytest.fixture(scope="module")
def page(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    sync_api = pytest.importorskip("playwright.sync_api")
    image, prov, conf = _mixed_case()
    path = tmp_path_factory.mktemp("viewer") / "viewer.html"
    write_offline_viewer(
        path, reconstruction=image, provenance=prov, confidence=conf,
        residual=np.zeros_like(image), metadata={"synthetic_only": True}, native_scale=NATIVE,
    )
    with sync_api.sync_playwright() as pw:
        browser = pw.chromium.launch(headless=os.environ.get("ECZIS_HEADLESS") == "1")
        pg = browser.new_page(viewport={"width": 1400, "height": 900})
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.goto(path.as_uri())
        pg.wait_for_function("typeof window.eczisQuery === 'function'")
        pg.errors = errors
        pg.renderer = TiledRasterRenderer(image, prov, conf, scale=NATIVE)
        yield pg
        browser.close()


def _points(h: int, w: int) -> list[tuple[float, float]]:
    pts = [(0.0, 0.0), (w - 1.0, h - 1.0), (-3.0, -3.0), (w + 3.0, h + 3.0), (7.0, 5.0), (7.5, 5.0),
           (8.0, 5.5), (7.5, 5.5), (2.5, 2.5), (3.0, 3.0), (3.5, 3.0), (2.0, 2.0), (7.999, 0.0)]
    rng = np.random.default_rng(7)
    pts += [(float(x), float(y)) for x, y in zip(rng.uniform(-1, w, 40), rng.uniform(-1, h, 40), strict=True)]
    return pts


@pytest.mark.parametrize("zoom", [1.0, 1.5, 1.75, 3.0, 3.25, 12.0])
def test_viewer_and_python_agree_on_effective_class_and_confidence(page: Any, zoom: float) -> None:
    r: TiledRasterRenderer = page.renderer
    h, w = r.image.shape
    for px, py in _points(h, w):
        js = page.evaluate("([x, y, z]) => window.eczisQuery(x, y, z)", [px, py, zoom])
        py_result = r.query_pixel(px, py, zoom * NATIVE)
        native = r.query_pixel(px, py, float(NATIVE))
        assert js["effective"] == py_result.provenance.name, (px, py, zoom)
        assert js["native"] == native.provenance.name, (px, py)
        assert abs(js["confidence"] - py_result.confidence) < 1e-9, (px, py, zoom)
    assert page.errors == []


def _set_zoom(page: Any, z: float) -> None:
    page.eval_on_selector(
        "#zoom", "(el, v) => { el.value = v; el.dispatchEvent(new Event('input', {bubbles: true})); }", str(z)
    )


def _click_pixel(page: Any, px: float, py: float) -> tuple[float, float]:
    """Click reconstruction coordinate (px, py); return the position the viewer reports."""
    box = page.locator("#base").bounding_box()
    w, h = page.renderer.image.shape[1], page.renderer.image.shape[0]
    page.mouse.click(box["x"] + (px + 0.5) / w * box["width"], box["y"] + (py + 0.5) / h * box["height"])
    shown = page.inner_text("#inspect").split("Position:", 1)[1].splitlines()[0]
    x, y = (float(v) for v in shown.split(","))
    return x, y


def test_inspector_shows_native_and_effective_class_through_zoom_cycle(page: Any) -> None:
    sealed_before = page.evaluate("P.provenanceBytes")
    _set_zoom(page, 4)  # 4 screen px per reconstruction px: sub-pixel picks are possible
    x, y = _click_pixel(page, 2.0, 2.0)  # interior R pixel
    assert abs(x - 2.0) <= 0.25 and abs(y - 2.0) <= 0.25
    _set_zoom(page, 1)
    assert "R — Reconstructed" in page.inner_text("#nativeClass")
    assert "R — Reconstructed" in page.inner_text("#effectiveClass")
    _set_zoom(page, 2)
    assert "I — Inferred" in page.inner_text("#effectiveClass")
    assert "R — Reconstructed" in page.inner_text("#nativeClass")
    assert "supported scale" in page.inner_text("#reason")
    _set_zoom(page, 12)
    assert "U — Unsupported" in page.inner_text("#effectiveClass")
    assert "CSS enlargement" in page.inner_text("#displayCap")
    page.screenshot(path=str(Path(os.environ.get("ECZIS_SHOT_DIR", tempfile.gettempdir())) / "viewer_zoom12.png"))
    _set_zoom(page, 1)
    assert "R — Reconstructed" in page.inner_text("#effectiveClass")
    assert page.evaluate("P.provenanceBytes") == sealed_before
    assert page.errors == []


def test_inspector_mixed_edge_click_reports_weakest_contributor(page: Any) -> None:
    _set_zoom(page, 4)
    x, _ = _click_pixel(page, 7.5, 2.0)  # between column 7 (R) and column 8 (U)
    assert 7.0 < x < 8.0
    assert "U — Unsupported" in page.inner_text("#nativeClass")
    assert "R — Reconstructed" not in page.inner_text("#nativeClass")
    _set_zoom(page, 1)
    x, _ = _click_pixel(page, 2.0, 2.0)
    assert x == pytest.approx(2.0, abs=0.01)  # at 1x a click selects the clicked pixel centre


def test_policy_panel_matches_python_policy(page: Any) -> None:
    from eczis.field.policy import DEFAULT_POLICY

    shown = json.loads(page.inner_text("#policy"))
    assert shown == DEFAULT_POLICY.to_payload(NATIVE)
