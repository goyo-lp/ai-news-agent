from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.schemas import DiscoveredItem, LinkedInPost, RankedTopic, ResearchBrief, RunReport


class OutputWriter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def write_outputs(
        self,
        run_date: str,
        seed_items: list[DiscoveredItem],
        ranked_topics: list[RankedTopic],
        adaptive_briefs: list[ResearchBrief],
        briefs: list[ResearchBrief],
        posts: list[LinkedInPost],
        report: RunReport,
    ) -> Path:
        output_dir = self.write_artifacts(
            run_date=run_date,
            seed_items=seed_items,
            ranked_topics=ranked_topics,
            adaptive_briefs=adaptive_briefs,
            briefs=briefs,
            posts=posts,
        )
        self.write_run_report(report=report, run_date=run_date, output_dir=output_dir)
        return output_dir

    def write_artifacts(
        self,
        run_date: str,
        seed_items: list[DiscoveredItem],
        ranked_topics: list[RankedTopic],
        adaptive_briefs: list[ResearchBrief],
        briefs: list[ResearchBrief],
        posts: list[LinkedInPost],
    ) -> Path:
        output_dir = self.settings.outputs_path / run_date
        output_dir.mkdir(parents=True, exist_ok=True)

        seed_path = output_dir / "top_50_articles.json"
        candidates_path = output_dir / "technical_candidates.json"
        adaptive_path = output_dir / "adaptive_briefs.json"
        briefs_path = output_dir / "research_briefs.json"
        posts_path = output_dir / "linkedin_posts.md"

        seed_path.write_text(
            json.dumps([item.model_dump(mode="json") for item in seed_items], indent=2),
            encoding="utf-8",
        )
        candidates_path.write_text(
            json.dumps([topic.model_dump(mode="json") for topic in ranked_topics], indent=2),
            encoding="utf-8",
        )
        adaptive_path.write_text(
            json.dumps([brief.model_dump(mode="json") for brief in adaptive_briefs], indent=2),
            encoding="utf-8",
        )
        briefs_path.write_text(
            json.dumps([brief.model_dump(mode="json") for brief in briefs], indent=2),
            encoding="utf-8",
        )
        posts_path.write_text(self._render_posts_markdown(posts), encoding="utf-8")

        return output_dir

    def write_run_report(
        self,
        *,
        report: RunReport,
        run_date: str,
        output_dir: Path | None = None,
    ) -> Path:
        resolved_dir = output_dir or (self.settings.outputs_path / run_date)
        resolved_dir.mkdir(parents=True, exist_ok=True)
        report_path = resolved_dir / "run_report.json"
        report_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
        return report_path

    def _render_posts_markdown(self, posts: list[LinkedInPost]) -> str:
        lines: list[str] = [
            "# LinkedIn Posts",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
        ]

        if not posts:
            lines.append("No posts generated.")
            lines.append("")
            return "\n".join(lines)

        for idx, post in enumerate(posts, start=1):
            lines.append(f"## Post {idx}: {post.headline}")
            lines.append("")
            lines.append(f"Angle: `{post.angle}`")
            lines.append("")
            lines.append(post.body)
            lines.append("")
            lines.append("Hashtags: " + " ".join(post.hashtags))
            lines.append("")
            lines.append("Citations:")
            for url in post.citation_urls:
                lines.append(f"- {url}")
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)
