# UNFILED-IP
from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eczis.export.heatmap import provenance_rgba
from eczis.field.policy import DEFAULT_POLICY, ScaleDemotionPolicy
from eczis.schemas import FloatArray, UInt8Array


def _png_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _gray_image(array: FloatArray) -> Image.Image:
    pixels = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="L")


def write_offline_viewer(
    path: Path,
    *,
    reconstruction: FloatArray,
    provenance: UInt8Array,
    confidence: FloatArray,
    residual: FloatArray,
    metadata: dict[str, Any],
    native_scale: int,
    policy: ScaleDemotionPolicy = DEFAULT_POLICY,
) -> None:
    template_path = Path(__file__).resolve().parents[3] / "viewer" / "index.html"
    template = template_path.read_text(encoding="utf-8")
    payload = {
        "width": int(reconstruction.shape[1]),
        "height": int(reconstruction.shape[0]),
        "reconstruction": _png_data_url(_gray_image(reconstruction)),
        "provenanceOverlay": _png_data_url(provenance_rgba(provenance)),
        "residual": _png_data_url(_gray_image(np.clip(residual / max(float(np.percentile(residual, 99)), 1e-6), 0, 1))),
        "provenanceBytes": base64.b64encode(provenance.tobytes()).decode("ascii"),
        # Exact float32 little-endian values so viewer inspection matches Python rendering.
        "confidenceF32": base64.b64encode(confidence.astype("<f4").tobytes()).decode("ascii"),
        "policy": policy.to_payload(native_scale),
        "metadata": metadata,
    }
    rendered = template.replace("__ECZIS_PAYLOAD__", json.dumps(payload, sort_keys=True))
    path.write_text(rendered, encoding="utf-8")
