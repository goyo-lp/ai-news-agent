from __future__ import annotations

import pytest

from app.main import _parse_dates_arg


def test_parse_dates_arg_deduplicates_and_validates() -> None:
    parsed = _parse_dates_arg("2026-03-02, 2026-03-01,2026-03-02")
    assert parsed == ["2026-03-02", "2026-03-01"]


def test_parse_dates_arg_rejects_empty_payload() -> None:
    with pytest.raises(ValueError):
        _parse_dates_arg(" , ")
