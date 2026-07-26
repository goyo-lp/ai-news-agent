"""Evidence floor — the minimum verification strength a brief must reach
before it may become a post.

Motivation (2026-07-25 production trace): briefs verified at
``partially_verified`` / confidence 0.3 — single-source, self-reported claims
— sailed straight through to drafting and delivery because the only skip rule
lived in the coordinator's *prompt* ("don't delegate insufficient_evidence")
and nothing enforced it in code. One rule in prose is one bad sampling away
from a fabricated claim shipping under the author's name; this module moves
the rule into deterministic code.

The floor is a pure predicate consumed by four choke points, so no path can
forget it:

  * the deterministic spine — only delegates the writer for floored topics;
  * ``submit_draft`` — refuses to write a draft whose brief is below floor;
  * ``deliver_telegram`` — refuses to send a draft whose brief is below floor
    (defense-in-depth for hand-written/legacy drafts);
  * ``export_report`` — reports the per-draft floor state + skip counts.
"""
from __future__ import annotations

from app.config import Settings
from app.orchestrator.schemas import ResearchBrief


def meets_evidence_floor(brief: ResearchBrief, settings: Settings) -> tuple[bool, str]:
    """Return ``(passes, reason)`` for one brief. ``reason`` is empty on pass
    and a short machine-greppable explanation on fail (surfaced in tool
    summaries + the run report).

    Policy:
      * ``verified``            -> pass (the verifier's strongest verdict).
      * ``partially_verified``  -> pass only with confidence >=
        ``settings.verification_min_confidence`` AND >=
        ``settings.verification_min_citations`` citations — a weak-verdict
        brief needs both a confidence signal and real source breadth.
      * everything else (``unverified`` / ``insufficient_evidence`` /
        ``failed``) -> fail.
    """
    status = brief.verification_status
    if status == "verified":
        return True, ""
    if status == "partially_verified":
        if brief.verification_confidence < settings.verification_min_confidence:
            return False, (
                f"partially_verified confidence {brief.verification_confidence:.2f} "
                f"< floor {settings.verification_min_confidence:.2f}"
            )
        if len(brief.citations) < settings.verification_min_citations:
            return False, (
                f"partially_verified with {len(brief.citations)} citation(s) "
                f"< floor {settings.verification_min_citations}"
            )
        return True, ""
    return False, f"verification_status={status} is below the evidence floor"


__all__ = ["meets_evidence_floor"]
