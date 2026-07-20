from pathlib import Path

from app.config import Settings
from app.services.style_profile import StyleProfiler


def test_build_style_profile_from_texts() -> None:
    settings = Settings(_env_file=None)
    profiler = StyleProfiler(settings)

    profile = profiler.build_from_texts(
        [
            "I tested a new AI workflow today. It reduced iteration time by 30%.",
            "What do you think about practical AI adoption in product teams?",
        ]
    )

    assert profile.sample_count == 2
    assert profile.sentence_count >= 3
    assert profile.avg_sentence_words > 0
    assert len(profile.tone_traits) >= 1


def test_build_style_profile_from_directory(tmp_path: Path) -> None:
    sample_file = tmp_path / "sample.md"
    sample_file.write_text("AI systems are improving quickly. Curious what teams are shipping.", encoding="utf-8")

    settings = Settings(_env_file=None)
    profiler = StyleProfiler(settings)
    profile = profiler.build_from_directory(tmp_path)

    assert profile.sample_count == 1
