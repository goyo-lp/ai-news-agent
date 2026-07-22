"""P8.2 — parity (golden characterization) of the ported deterministic logic.

Locks the quality gate's verdicts + the technical ranker's hype-demotion order
against a committed golden. A regression in either surfaces here as a diff. See
``parity_harness`` for why parity is a golden characterization (no runnable
legacy pipeline exists in-host to diff against) and ``scripts/parity.py`` to
regenerate the golden.
"""
from __future__ import annotations

import json
from pathlib import Path

from parity_harness import build_parity_report

_GOLDEN = Path(__file__).parent / "fixtures" / "parity" / "parity_report.json"


def test_parity_report_matches_golden() -> None:
    """No blocking regressions: the ported logic's behavior on the shared
    fixtures matches the committed golden exactly."""
    report = build_parity_report()
    golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    assert report == golden


def test_ranker_demotes_hype_and_funding_below_technical() -> None:
    """The load-bearing behavioral guarantee of the ported ranker, asserted
    directly (not just via the golden) so its intent survives a golden refresh:
    the technical story outranks both the hype and the funding story."""
    ranking = build_parity_report()["technical_rank"]
    assert ranking["technical_ranks_above_hype"] is True
    assert ranking["technical_ranks_above_funding"] is True
    assert ranking["ranked_order"][0] == "technical"


def test_gate_passes_clean_and_fails_each_defect() -> None:
    """The gate accepts a clean proposal and rejects each defect class — the
    intent behind the gate fixtures, asserted independently of the golden's
    exact reason strings."""
    gate = build_parity_report()["quality_gate"]
    assert gate["clean_pass"]["passed"] is True
    assert all(
        gate[name]["passed"] is False
        for name in (
            "too_short",
            "too_long",
            "hype_present",
            "multi_topic",
            "too_few_hashtags",
            "no_hashtags",
        )
    )
