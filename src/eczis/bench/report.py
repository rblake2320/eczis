# UNFILED-IP
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eczis.atomic import atomic_write_text

_FORBIDDEN = (
    "infinite resolution",
    "infinite detail",
    "hallucination-free",
    "perfect reconstruction",
    "forensic-grade",
    "ground-truth recovery from one ordinary image",
)

_METHODS = ("bicubic", "single_frame_deconvolution", "shift_add", "map")


def validate_claim_language(text: str) -> None:
    lowered = text.lower()
    for phrase in _FORBIDDEN:
        if phrase in lowered:
            raise ValueError(f"Forbidden unqualified claim: {phrase}")


def _scope_lines() -> list[str]:
    return [
        "> Infinitely sampleable output, finite and explicitly classified evidence.",
        "",
        "## Scope",
        "",
        "This report covers synthetic observations only (synthetic software reduction-to-practice).",
        "It does not establish real-world optical gains, universal false-detail detection, forensic",
        "admissibility, patentability, or physical-rig performance.",
        "",
    ]


def write_report(output: Path, report: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output / "benchmark.json", json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    methods = report["methods"]
    prov = report.get("provenance", {})
    lines = [
        "# UNFILED-IP",
        "# ECZIS v0.1R1 Synthetic Demonstration Report",
        "",
        *_scope_lines(),
        "## Reconstruction metrics",
        "",
        "| Method | PSNR (dB) | RMSE | Feature F1 |",
        "|---|---:|---:|---:|",
    ]
    for name in _METHODS:
        if name not in methods:
            continue
        item = methods[name]
        lines.append(f"| {name} | {item['psnr_db']:.3f} | {item['rmse']:.6f} | {item['feature_f1']:.3f} |")
    gate = report["gate"]
    lines.extend(
        [
            "",
            "## Evidence gate",
            "",
            f"- Gate threshold (region mean normalized squared residual): {gate['threshold']:.4f}",
            f"- Per-frame contradiction threshold: {gate.get('max_frame_threshold')}",
            f"- Gate pass fraction: {gate['base_pass_fraction']:.3f}",
            f"- Detectable injected-detail demotion rate: {gate['challenge_detection_rate']:.3f}",
            f"- False-feature challenge amplitude: {gate['challenge_amplitude']:.3f}",
            "",
            "A low residual establishes observation consistency, not factual uniqueness. Detail outside the",
            "estimated operator's supported band is non-identifiable and remains Inferred, however low its",
            "residual. R asserts the supported-band component only.",
            "",
            "## Provenance",
            "",
        ]
    )
    for key, count in report["provenance_counts"].items():
        lines.append(f"- {key}: {count}")
    if prov:
        lines.extend(
            [
                "",
                f"- Evidence cutoff (cycles / HR pixel): {prov['evidence_cutoff_cycles_per_hr_px']:.4f}",
                f"- False-R rate vs supported-band truth: {prov['false_r_rate']:.4f}",
                f"- False-R rate vs full-band truth (includes unobservable detail): {prov['false_r_rate_full_band']:.4f}",
                f"- Injected pixels labeled R or M: {prov['injected_pixels_labeled_r_or_m']}",
                f"- Expected calibration error (uncalibrated confidence): {prov['ece']:.4f}",
            ]
        )
    text = "\n".join(lines) + "\n"
    validate_claim_language(text)
    atomic_write_text(output / "REPORT.md", text)


def write_benchmark_report(output: Path, result: dict[str, Any]) -> None:
    lines = [
        "# UNFILED-IP",
        "# ECZIS Synthetic Benchmark Matrix",
        "",
        *_scope_lines(),
        f"- Split: `{result['split']}` seeds {result['seeds']}",
        f"- Test-set size: {result['test_set_size']['cases_per_arm']} cases per arm "
        f"({result['test_set_size']['seeds_per_family']} seeds x {len(result['families'])} families)",
        "- Intervals: 95% percentile bootstrap of the mean over seeds (2000 resamples, seed 0).",
        "  With 3 seeds per family the intervals are wide; treat them as dispersion, not precision.",
        "",
    ]
    for arm, families in result["summary"].items():
        lines.extend([f"## Arm: {arm}", ""])
        labels = list(next(iter(families.values())).keys())
        for label in labels:
            lines.append(f"### {label}")
            lines.append("")
            lines.append("| Family | mean | 95% CI |")
            lines.append("|---|---:|---|")
            for family, metrics in families.items():
                m = metrics[label]
                lines.append(f"| {family} | {m['mean']:.4f} | [{m['lo']:.4f}, {m['hi']:.4f}] |")
            lines.append("")
    acc = result["acceptance"]
    lines.extend(
        [
            "## Acceptance summary",
            "",
            f"- MAP beats shift-and-add (mean PSNR) on {acc['map_beats_shift_add_families']} of {acc['family_count']} families.",
            f"- Maximum injected pixels labeled R or M in any case: {acc['max_injected_pixels_labeled_r_or_m']}.",
            f"- Maximum M-labeled pixels in any case: {acc['max_m_count']} (M is never emitted for HR reconstructions).",
            "",
        ]
    )
    text = "\n".join(lines)
    validate_claim_language(text)
    atomic_write_text(output / "BENCHMARK.md", text)
