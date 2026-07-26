"""Draft loading + certification — the single answer to "is this draft fit to
ship" that both deliver_telegram and export_report consume.

The agreement test is the point of the module: before it existed, export
resolved a draft's brief via topics.json membership while delivery resolved it
from the draft's own supporting_topic_ids, so a draft whose topic wasn't in
topics.json was reported below-floor by the run report and shipped by delivery.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.orchestrator.schemas import PostProposal
from app.orchestrator.services.drafts import (
    DraftLoadError,
    certify,
    load_draft,
    load_drafts,
)
from app.orchestrator.services.export_bundle import draft_integrity, load_run_artifacts
from app.orchestrator.tools.draft import write_signed_draft
from support import brief_fields, proposal_fields, settings_for


def _write_draft(settings: Settings, post_id: str, topic_id: str) -> PostProposal:
    proposal = PostProposal.model_validate(proposal_fields(post_id, topic_id))
    write_signed_draft(proposal, settings)
    return proposal


def _write_brief(tmp_path: Path, topic_id: str, **overrides: object) -> None:
    briefs = tmp_path / "briefs"
    briefs.mkdir(parents=True, exist_ok=True)
    (briefs / f"{topic_id}.verified.json").write_text(
        json.dumps(brief_fields(topic_id, **overrides)), encoding="utf-8"
    )


def _write_gate(tmp_path: Path, post_id: str, payload: object) -> None:
    drafts = tmp_path / "drafts"
    drafts.mkdir(parents=True, exist_ok=True)
    (drafts / f"{post_id}.gate.json").write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", ["a/b", "..", ".", "", "/leading", "endslash/"])
def test_load_draft_rejects_traversal_in_post_id(tmp_path: Path, bad: str) -> None:
    with pytest.raises(DraftLoadError) as exc:
        load_draft(bad, tmp_path)
    assert exc.value.reason == "invalid_post_id"


def test_load_draft_reports_missing_and_malformed_distinctly(tmp_path: Path) -> None:
    with pytest.raises(DraftLoadError) as missing:
        load_draft("ghost", tmp_path)
    assert missing.value.reason == "draft_not_found"

    (tmp_path / "drafts").mkdir()
    (tmp_path / "drafts" / "bad.json").write_text("{not json}", encoding="utf-8")
    with pytest.raises(DraftLoadError) as malformed:
        load_draft("bad", tmp_path)
    assert malformed.value.reason == "draft_invalid_json"


def test_gate_status_distinguishes_missing_from_malformed(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    _write_draft(settings, "post-1", "topic-a")

    assert load_draft("post-1", tmp_path).gate_status == "missing"

    (tmp_path / "drafts" / "post-1.gate.json").write_text("{not json}", encoding="utf-8")
    assert load_draft("post-1", tmp_path).gate_status == "malformed"

    _write_gate(tmp_path, "post-1", {"passed": True})
    assert load_draft("post-1", tmp_path).gate_status == "present"


@pytest.mark.parametrize("passed", ["True", 1, ["no"], None, "yes"])
def test_gate_passed_rejects_truthy_non_bools(tmp_path: Path, passed: object) -> None:
    """The verdict file is written by an LLM-driven tool, so a truthy-but-not-
    bool value must read as unknown rather than as a pass."""
    settings = settings_for(tmp_path)
    _write_draft(settings, "post-1", "topic-a")
    _write_gate(tmp_path, "post-1", {"passed": passed})

    assert load_draft("post-1", tmp_path).gate_passed is None


def test_load_drafts_skips_unreadable_and_ignores_gate_files(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    _write_draft(settings, "post-1", "topic-a")
    _write_gate(tmp_path, "post-1", {"passed": True})
    (tmp_path / "drafts" / "broken.json").write_text("{not json}", encoding="utf-8")

    loaded = load_drafts(tmp_path)

    assert [d.post_id for d in loaded] == ["post-1"]


# --------------------------------------------------------------------------- #
# Certification
# --------------------------------------------------------------------------- #


def test_certify_passes_a_signed_gated_floored_draft(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    _write_draft(settings, "post-1", "topic-a")
    _write_brief(tmp_path, "topic-a")
    _write_gate(tmp_path, "post-1", {"passed": True})

    cert = certify(load_draft("post-1", tmp_path), settings)

    assert cert.provenance_ok and cert.floor_ok and cert.gate_passed is True
    assert cert.shippable


def test_certify_fails_closed_without_a_brief(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    _write_draft(settings, "post-1", "topic-a")
    _write_gate(tmp_path, "post-1", {"passed": True})

    cert = certify(load_draft("post-1", tmp_path), settings)

    assert cert.floor_ok is False
    assert "no readable brief" in cert.floor_reason
    assert not cert.shippable


def test_certify_rejects_an_unsigned_draft(tmp_path: Path) -> None:
    """A draft an LLM wrote with write_file carries no signature."""
    settings = settings_for(tmp_path)
    drafts = tmp_path / "drafts"
    drafts.mkdir(parents=True)
    (drafts / "post-1.json").write_text(
        json.dumps(proposal_fields("post-1", "topic-a")), encoding="utf-8"
    )
    _write_brief(tmp_path, "topic-a")
    _write_gate(tmp_path, "post-1", {"passed": True})

    cert = certify(load_draft("post-1", tmp_path), settings)

    assert cert.provenance_ok is False
    assert not cert.shippable


def test_certify_rejects_a_below_floor_brief(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    _write_draft(settings, "post-1", "topic-a")
    _write_brief(
        tmp_path,
        "topic-a",
        verification_status="partially_verified",
        verification_confidence=0.3,
    )
    _write_gate(tmp_path, "post-1", {"passed": True})

    cert = certify(load_draft("post-1", tmp_path), settings)

    assert cert.floor_ok is False
    assert "below evidence floor" in cert.floor_reason


# --------------------------------------------------------------------------- #
# The agreement this module exists to guarantee
# --------------------------------------------------------------------------- #


def test_export_and_delivery_agree_when_topic_is_absent_from_topics_json(
    tmp_path: Path,
) -> None:
    """Regression: a draft whose topic isn't listed in topics.json (a backfill,
    or a run whose topics file was replaced) used to be reported below-floor by
    export while delivery shipped it — export resolved the brief by topics.json
    membership, delivery by the draft's own supporting_topic_ids. Both now use
    certify(), so the run report can't contradict what actually ships."""
    settings = settings_for(tmp_path)
    (tmp_path / "topics.json").write_text(json.dumps([]), encoding="utf-8")
    _write_brief(tmp_path, "topic-b")
    _write_draft(settings, "post-1", "topic-b")
    _write_gate(tmp_path, "post-1", {"passed": True})

    reported = draft_integrity(load_run_artifacts(tmp_path), settings)["post-1"]
    enforced = certify(load_draft("post-1", tmp_path), settings)

    assert reported.floor_ok is enforced.floor_ok is True
    assert reported.floor_reason == enforced.floor_reason
    assert reported.provenance_ok is enforced.provenance_ok
