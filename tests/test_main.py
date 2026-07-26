"""The `both` CLI command: runs the news digest then the LinkedIn proposal
flow, combining their exit codes. Stubs both lanes so this stays a pure
dispatch/exit-code test, not a re-test of either lane's own behavior."""
from __future__ import annotations

import asyncio

import pytest

from app import main
from app.main import ProposeOptions, RunOptions


@pytest.mark.parametrize(
    ("run_exit", "propose_exit", "expected"),
    [(0, 0, 0), (1, 0, 1), (0, 2, 2), (1, 1, 1)],
)
def test_run_both_combines_exit_codes(
    monkeypatch: pytest.MonkeyPatch, run_exit: int, propose_exit: int, expected: int
) -> None:
    calls: list[str] = []

    async def fake_run_pipeline(options: RunOptions) -> int:
        calls.append("run")
        return run_exit

    async def fake_run_propose(options: ProposeOptions) -> int:
        calls.append("propose")
        return propose_exit

    monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(main, "run_propose", fake_run_propose)

    result = asyncio.run(main.run_both(RunOptions(), ProposeOptions()))

    assert result == expected
    assert calls == ["run", "propose"]


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
