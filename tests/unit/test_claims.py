# UNFILED-IP
from __future__ import annotations

import pytest

from eczis.bench.report import validate_claim_language


def test_claim_linter_accepts_evidence_bounded_language() -> None:
    validate_claim_language("Infinitely sampleable output, finite and explicitly classified evidence.")


def test_claim_linter_rejects_unqualified_language() -> None:
    with pytest.raises(ValueError, match="Forbidden"):
        validate_claim_language("This provides infinite resolution.")
