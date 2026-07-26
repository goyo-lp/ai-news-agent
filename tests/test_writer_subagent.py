"""P4.2 — writer-subagent spec.

No live LLM: pin the spec shape (name, the single gate tool, the Stage-B writer
model, the linkedin-voice skill wiring) and guard the prompt's contract — the
PostProposal field list and the gate constraints — against drift, so the writer
is told exactly what quality_gate will enforce.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import Settings
from app.orchestrator.schemas import PostProposal
from app.orchestrator.services.quality_gate import _MAX_POST_WORDS, _MIN_POST_WORDS
from app.orchestrator.subagents.writer import (
    AUTHORED_POST_FIELDS,
    WRITER_SUBAGENT_NAME,
    build_writer_subagent,
)


def _settings(**kw: object) -> Settings:
    return Settings(_env_file=None, **kw)


def test_subagent_core_shape() -> None:
    sub = build_writer_subagent(_settings())
    assert sub["name"] == WRITER_SUBAGENT_NAME
    assert sub["description"].strip()
    assert sub["system_prompt"].strip()


def test_subagent_carries_only_the_quality_gate_tool() -> None:
    sub = build_writer_subagent(_settings())
    assert {t.name for t in sub["tools"]} == {"submit_draft", "quality_gate"}


def test_subagent_uses_stage_b_writer_model() -> None:
    settings = _settings(openrouter_stage_b_writer_model="anthropic/claude-opus-4.1")
    sub = build_writer_subagent(settings)
    model = sub["model"]
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "anthropic/claude-opus-4.1"


def test_subagent_skill_source_is_absolute_and_holds_linkedin_voice() -> None:
    """The skills source must be an absolute path (a relative one would resolve
    under the coordinator's data-dir backend root, not the repo), and it must
    actually contain the linkedin-voice skill — otherwise step 1 of the prompt
    silently no-ops and the writer loses the voice."""
    from pathlib import Path

    sub = build_writer_subagent(_settings(skills_dir="skills"))
    assert len(sub["skills"]) == 1
    source = Path(sub["skills"][0])
    assert source.is_absolute()
    assert (source / "linkedin-voice" / "SKILL.md").is_file()


def test_prompt_lists_every_post_proposal_field() -> None:
    """The prompt tells the model which fields to author; if PostProposal gains
    a field, the prompt must mention it or drafts will omit it."""
    prompt = build_writer_subagent(_settings())["system_prompt"]
    for field in AUTHORED_POST_FIELDS:
        assert field in prompt, f"prompt omits post field {field!r}"


def test_authored_fields_are_the_full_post_proposal() -> None:
    assert AUTHORED_POST_FIELDS == list(PostProposal.model_fields)


def test_prompt_states_the_gate_constraints() -> None:
    """The writer must design for exactly what quality_gate enforces; pin the
    word window (from the live gate constants) and the other checks."""
    prompt = build_writer_subagent(_settings())["system_prompt"]
    assert str(_MIN_POST_WORDS) in prompt and str(_MAX_POST_WORDS) in prompt
    lowered = prompt.lower()
    assert "quality_gate" in lowered
    assert "hashtag" in lowered
    assert "hype" in lowered
    # exactly-one-topic instruction, pinned to the concrete phrasing (not a bare
    # "one" substring that would match "someone"/"done").
    assert "exactly [topic_id]" in prompt
    assert "supporting_topic_ids" in prompt


def test_prompt_states_the_file_io_contract() -> None:
    prompt = build_writer_subagent(_settings())["system_prompt"]
    assert "briefs/<topic_id>.verified.json" in prompt
    assert "submit_draft" in prompt
    assert "style_profile.json" in prompt
    assert "linkedin-voice" in prompt


def test_prompt_interpolates_absolute_data_dir_into_io_paths(tmp_path) -> None:
    """P8.3 path-semantics fix: the draft write path + brief/style reads are
    interpolated as ABSOLUTE paths under orchestrator_data_dir so LLM-driven
    read_file/write_file resolve on the mounted filesystem."""
    data_dir = str(tmp_path.resolve())
    prompt = build_writer_subagent(_settings(orchestrator_data_dir=data_dir))["system_prompt"]
    assert f"{data_dir}/briefs/<topic_id>.verified.json" in prompt
