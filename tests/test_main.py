"""The `both` CLI command: runs the news digest then the LinkedIn proposal
flow, combining their exit codes. Stubs both lanes so this stays a pure
dispatch/exit-code test, not a re-test of either lane's own behavior."""
from __future__ import annotations

import asyncio

import pytest

from app import main
from app.main import ProposeOptions, RunOptions
from app.orchestrator.schemas import CuratedArticle


@pytest.mark.parametrize(
    ("run_exit", "propose_exit", "expected"),
    [(0, 0, 0), (1, 0, 1), (0, 2, 2), (1, 1, 1)],
)
def test_run_both_combines_exit_codes(
    monkeypatch: pytest.MonkeyPatch, run_exit: int, propose_exit: int, expected: int
) -> None:
    calls: list[str] = []

    async def fake_run_pipeline(
        options: RunOptions,
    ) -> tuple[int, list[CuratedArticle]]:
        calls.append("run")
        return run_exit, []

    async def fake_run_propose(
        options: ProposeOptions, prefetched: list[CuratedArticle] | None = None
    ) -> int:
        calls.append("propose")
        return propose_exit

    monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(main, "run_propose", fake_run_propose)

    result = asyncio.run(main.run_both(RunOptions(), ProposeOptions()))

    assert result == expected
    assert calls == ["run", "propose"]


def _curated(article_id: str) -> CuratedArticle:
    return CuratedArticle(
        id=article_id,
        source_name="TechCrunch (AI)",
        title=f"Title {article_id}",
        url=f"https://example.com/{article_id}",
    )


@pytest.mark.parametrize(
    ("limit", "digest_exit", "digest_articles", "expected_reuse"),
    [
        # The optimization: one curation feeds both lanes.
        (None, 0, 2, True),
        # --limit caps the digest only, so a narrowed set must not be handed on.
        (10, 0, 2, False),
        # A failed digest returns no articles. Handing [] on is NOT the same as
        # handing on nothing: the spine would read it as "no news today" and
        # stop, so one lane's failure would silently take out the other.
        (None, 1, 0, False),
    ],
)
def test_run_both_reuses_the_digest_curation_only_when_it_is_sound(
    monkeypatch: pytest.MonkeyPatch,
    limit: int | None,
    digest_exit: int,
    digest_articles: int,
    expected_reuse: bool,
) -> None:
    """`both` runs one curation, not two — the second pass was ~2min of
    duplicated feed fetching and the direct cause of the 429s. It falls back to
    curating independently whenever reuse would mislead the proposal lane."""
    curated = [_curated(f"a{i}") for i in range(digest_articles)]
    handed_on: list[list[CuratedArticle] | None] = []

    async def fake_run_pipeline(
        options: RunOptions,
    ) -> tuple[int, list[CuratedArticle]]:
        return digest_exit, curated

    async def fake_run_propose(
        options: ProposeOptions, prefetched: list[CuratedArticle] | None = None
    ) -> int:
        handed_on.append(prefetched)
        return 0

    monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(main, "run_propose", fake_run_propose)

    asyncio.run(main.run_both(RunOptions(limit=limit), ProposeOptions()))

    assert handed_on == [curated if expected_reuse else None]


def test_parser_maps_dry_run_to_skip_delivery_on_propose() -> None:
    """`--dry-run` on propose means "don't send", not "don't spend" — the LLM
    still runs. The option name says so at the call site."""
    args = main.build_parser().parse_args(["propose", "--dry-run"])
    options = ProposeOptions(force=args.force, skip_delivery=args.dry_run)
    assert options.skip_delivery is True
    assert options.force is False


def test_both_does_not_thread_the_digest_limit_into_proposals() -> None:
    """`--limit` caps the news digest only; ProposeOptions has no limit field,
    so proposal volume stays governed by max_topics_per_run."""
    args = main.build_parser().parse_args(["both", "--limit", "3"])
    assert RunOptions(dry_run=args.dry_run, limit=args.limit).limit == 3
    assert not hasattr(ProposeOptions(), "limit")
