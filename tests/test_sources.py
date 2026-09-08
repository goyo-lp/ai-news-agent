from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_news_agent.adapters import adapter_for_source
from ai_news_agent.sources import (
    SourceConfig,
    load_source_configs,
    load_sources_as_records,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sources.yaml"


def test_roster_contains_all_50_sources_with_stable_ids() -> None:
    configs = load_source_configs(CONFIG_PATH)

    assert len(configs) == 50
    assert [config.id for config in configs] == [
        f"source-{n:02d}" for n in range(1, 51)
    ]
    assert len({config.id for config in configs}) == 50


def test_roster_covers_expected_route_mix() -> None:
    configs = load_source_configs(CONFIG_PATH)
    by_type = {route for config in configs for route in [config.route.type.value]}

    assert by_type == {"rss", "atom", "official_page_adapter"}
    rss = [c for c in configs if c.route.type.value == "rss"]
    atom = [c for c in configs if c.route.type.value == "atom"]
    adapters = [c for c in configs if c.route.type.value == "official_page_adapter"]

    assert len(rss) == 34
    assert len(atom) == 4
    assert len(adapters) == 12


def test_roster_records_recheck_work_explicitly() -> None:
    configs = {config.id: config for config in load_source_configs(CONFIG_PATH)}

    assert "backoff" in " ".join(configs["source-20"].editorial_notes).lower()
    assert "timed out" in " ".join(configs["source-28"].editorial_notes).lower()


def test_roster_starts_disabled_pending_validation() -> None:
    for config in load_source_configs(CONFIG_PATH):
        assert config.enabled is False
        assert config.disabled_reason == "pending-pr03-validation"
        assert config.discovery_provenance == "plan-research-2026-09-07"
        assert config.feed_scope.strip()
        assert config.editorial_notes


def test_roster_groups_google_family() -> None:
    configs = {config.id: config for config in load_source_configs(CONFIG_PATH)}

    assert configs["source-03"].publisher_family == "google"
    assert configs["source-14"].publisher_family == "google"


def test_records_convert_to_validated_sources() -> None:
    records = load_sources_as_records(CONFIG_PATH)

    assert len(records) == 50
    assert records[0].id == "source-01"
    assert records[0].route_type.value == "rss"


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    import yaml

    first = load_source_configs(CONFIG_PATH)[0].model_dump(mode="json")
    payload_path = tmp_path / "sources.yaml"
    payload_path.write_text(yaml.safe_dump([first, first]), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate source id"):
        load_source_configs(payload_path)


def test_config_must_be_a_list(tmp_path: Path) -> None:
    payload_path = tmp_path / "sources.yaml"
    payload_path.write_text("id: source-01\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a list"):
        load_source_configs(payload_path)


def test_embedded_credentials_are_rejected(tmp_path: Path) -> None:
    import yaml

    first = load_source_configs(CONFIG_PATH)[0].model_dump(mode="json")
    first["route"]["url"] = "https://user:secret@example.com/feed.xml"
    payload_path = tmp_path / "sources.yaml"
    payload_path.write_text(yaml.safe_dump([first]), encoding="utf-8")

    with pytest.raises(ValueError, match="must not embed credentials"):
        load_source_configs(payload_path)


def test_disabled_source_requires_reason() -> None:
    config = load_source_configs(CONFIG_PATH)[0]
    payload = config.model_dump(mode="json")
    payload["disabled_reason"] = None

    with pytest.raises(ValidationError, match="disabled sources require"):
        SourceConfig.model_validate(payload)


def test_adapter_routes_report_unavailable_explicitly() -> None:
    configs = {config.id: config for config in load_source_configs(CONFIG_PATH)}

    rss_adapter = adapter_for_source(configs["source-01"])
    anthropic_adapter = adapter_for_source(configs["source-02"])

    assert rss_adapter is None
    assert anthropic_adapter is not None
    outcome = anthropic_adapter.fetch()
    assert outcome.available is False
    assert outcome.source_id == "source-02"
    assert "not yet implemented" in outcome.reason


def test_active_adapter_without_implementation_is_unavailable() -> None:
    config = load_source_configs(CONFIG_PATH)[1]
    payload = config.model_dump(mode="json")
    payload["route"]["status"] = "active"
    active_config = SourceConfig.model_validate(payload)

    adapter = adapter_for_source(active_config)

    assert adapter is not None
    assert "no validated implementation" in adapter.fetch().reason
