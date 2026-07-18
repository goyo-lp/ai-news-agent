from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped]


@dataclass(frozen=True)
class SourcePolicy:
    tier1_domains: set[str]
    tier2_domains: set[str]

    @classmethod
    def from_file(cls, path: Path) -> "SourcePolicy":
        if not path.exists():
            return cls(tier1_domains=set(), tier2_domains=set())

        with path.open("r", encoding="utf-8") as source_file:
            payload = yaml.safe_load(source_file) or {}

        tiers = payload.get("tiers") or {}
        tier1 = {
            _normalize_domain(item)
            for item in (tiers.get("tier1") or [])
            if str(item).strip()
        }
        tier2 = {
            _normalize_domain(item)
            for item in (tiers.get("tier2") or [])
            if str(item).strip()
        }
        tier2 -= tier1
        return cls(tier1_domains=tier1, tier2_domains=tier2)

    def tier_for(self, domain: str) -> int:
        normalized = _normalize_domain(domain)
        if normalized in self.tier1_domains:
            return 1
        if normalized in self.tier2_domains:
            return 2
        return 3

    def weight_for(self, domain: str) -> float:
        tier = self.tier_for(domain)
        if tier == 1:
            return 1.0
        if tier == 2:
            return 0.85
        return 0.6

    def cluster_allowed(self, domains: set[str]) -> bool:
        normalized = {_normalize_domain(domain) for domain in domains}
        if not normalized:
            return False

        trusted = {
            domain
            for domain in normalized
            if domain in self.tier1_domains or domain in self.tier2_domains
        }
        if not trusted:
            return False

        # Tiered allowlist: untrusted-only stories are excluded.
        return True



def _normalize_domain(value: str) -> str:
    return value.lower().replace("www.", "").strip()
