"""Parity report CLI — Phase 8.2.

Prints the parity report for the ported deterministic logic (quality gate +
technical ranker hype-demotion) so it can be eyeballed before the P8.3 cutover
and the P8.4 legacy removal — the cutover rule wants the parity harness green
before anything is deleted.

    python scripts/parity.py            # print the report
    python scripts/parity.py --update   # rewrite the committed golden

The report content lives in ``tests/parity_harness.build_parity_report`` — the
same single source the regression test (``tests/test_parity.py``) locks; this
script only formats it and (optionally) refreshes the golden.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
# The harness imports from ``app.*`` (src) and lives under tests/; make both
# importable when run as a plain script (pytest handles this via pythonpath).
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "tests"))

from parity_harness import build_parity_report  # noqa: E402

_GOLDEN = _ROOT / "tests" / "fixtures" / "parity" / "parity_report.json"


def _render(report: dict) -> str:
    gate = report["quality_gate"]
    rank = report["technical_rank"]
    lines = ["# Parity report — ported deterministic logic", "", "## Quality gate"]
    for name, result in gate.items():
        verdict = "PASS" if result["passed"] else "FAIL"
        reasons = "; ".join(result["reasons"]) if result["reasons"] else "—"
        lines.append(f"- {name}: {verdict} (words={result['word_count']}) {reasons}")
    lines += ["", "## Technical ranker (hype demotion)"]
    lines.append(f"- ranked order: {' > '.join(rank['ranked_order'])}")
    lines.append(f"- technical above hype: {rank['technical_ranks_above_hype']}")
    lines.append(f"- technical above funding: {rank['technical_ranks_above_funding']}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    report = build_parity_report()
    if "--update" in argv:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote golden: {_GOLDEN}")
        return 0
    print(_render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
