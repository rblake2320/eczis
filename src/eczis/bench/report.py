# UNFILED-IP
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_FORBIDDEN = ("infinite resolution", "infinite detail", "hallucination-free", "perfect reconstruction")


def validate_claim_language(text: str) -> None:
    lowered = text.lower()
    for phrase in _FORBIDDEN:
        if phrase in lowered:
            raise ValueError(f"Forbidden unqualified claim: {phrase}")


def write_report(output: Path, report: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "benchmark.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    methods = report["methods"]
    lines = [
        "# UNFILED-IP",
        "# ECZIS v0.1R1 Synthetic Demonstration Report",
        "",
        "> Infinitely sampleable output, finite and explicitly classified evidence.",
        "",
        "## Scope",
        "",
        "This report covers synthetic observations only. It does not establish real-world optical gains,",
        "universal false-detail detection, forensic admissibility, patentability, or physical-rig performance.",
        "",
        "## Reconstruction metrics",
        "",
        "| Method | PSNR (dB) | RMSE | Feature F1 |",
        "|---|---:|---:|---:|",
    ]
    for name in ("bicubic", "shift_add", "map"):
        item = methods[name]
        lines.append(
            f"| {name} | {item['psnr_db']:.3f} | {item['rmse']:.6f} | {item['feature_f1']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Evidence gate",
            "",
            f"- Gate threshold (mean normalized squared residual): {report['gate']['threshold']:.4f}",
            f"- Base R-eligible fraction: {report['gate']['base_pass_fraction']:.3f}",
            f"- Detectable injected-detail demotion rate: {report['gate']['challenge_detection_rate']:.3f}",
            f"- False-feature challenge amplitude: {report['gate']['challenge_amplitude']:.3f}",
            "",
            "A low residual establishes observation consistency, not factual uniqueness. Non-identifiable",
            "detail remains Inferred rather than being promoted to Reconstructed.",
            "",
            "## Provenance",
            "",
        ]
    )
    for key, count in report["provenance_counts"].items():
        lines.append(f"- {key}: {count}")
    text = "\n".join(lines) + "\n"
    validate_claim_language(text)
    (output / "REPORT.md").write_text(text, encoding="utf-8")
