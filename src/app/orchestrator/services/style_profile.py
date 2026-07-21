"""Style-profile builder ported from the reference LinkedIn agent
(``reference/linkedin-agent/src/app/services/style_profile.py``).

Turns a directory of writing samples into a :class:`StyleProfile` — the
author's voice fingerprint (sentence length, common openers, vocabulary, tone
traits, CTA patterns) that the writer subagent (P4.2) reads alongside the
``linkedin-voice`` skill (P3.1). The sample-analysis logic (sentence splitting,
opener/vocabulary counting, tone/CTA heuristics, JSONL parsing) is verbatim
from the reference.

Intentional divergence from the reference (documented so the port's shape
doesn't drift silently):
  The reference wrapped this in a ``StyleProfiler`` class whose *only* instance
  state was ``self.settings`` — used solely to default the save/load path. Here
  the sample analysis is a set of pure functions and every path is an explicit
  argument, so the service has no dependency on :class:`app.config.Settings`
  and is testable with a tmp directory. Path resolution (Settings ->
  ``style_profile_file`` / ``style_samples_dir``) is the loader tool's job
  (P3.2), keeping the Settings coupling in exactly one place instead of leaking
  it into the service.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from app.orchestrator.schemas import StyleProfile

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-zA-Z']+")
_SAMPLE_SUFFIXES = {".txt", ".md", ".jsonl"}
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "to",
    "with", "this", "you", "your", "we", "our",
}


def build_from_texts(texts: list[str]) -> StyleProfile:
    """Build a :class:`StyleProfile` from raw sample texts. Returns
    :func:`default_profile` when there is no usable text, so the result is
    always a complete, valid profile."""
    cleaned_texts = [" ".join(text.split()) for text in texts if text and text.strip()]
    if not cleaned_texts:
        return default_profile()

    sentences: list[str] = []
    for text in cleaned_texts:
        sentences.extend(_split_sentences(text))

    word_counts = [len(_WORD_RE.findall(sentence)) for sentence in sentences if sentence.strip()]
    avg_sentence_words = round(sum(word_counts) / len(word_counts), 2) if word_counts else 0.0

    openers_counter: Counter[str] = Counter()
    vocab_counter: Counter[str] = Counter()

    for sentence in sentences:
        opener = _sentence_opener(sentence)
        if opener:
            openers_counter[opener] += 1

        for token in _WORD_RE.findall(sentence.lower()):
            if token in _STOPWORDS or len(token) <= 2:
                continue
            vocab_counter[token] += 1

    return StyleProfile(
        sample_count=len(cleaned_texts),
        sentence_count=len(sentences),
        avg_sentence_words=avg_sentence_words,
        common_openers=[item for item, _ in openers_counter.most_common(8)],
        vocabulary=[item for item, _ in vocab_counter.most_common(20)],
        tone_traits=_detect_tone_traits(cleaned_texts, sentences),
        cta_patterns=_detect_cta_patterns(cleaned_texts),
    )


def build_from_directory(samples_dir: Path) -> StyleProfile:
    """Build a profile from every ``.txt`` / ``.md`` / ``.jsonl`` file under
    ``samples_dir`` (recursively). A missing/empty directory yields the default
    profile via :func:`build_from_texts`."""
    return build_from_texts(_load_samples(samples_dir))


def save_profile(profile: StyleProfile, path: Path) -> Path:
    """Write the profile to ``path`` as pretty JSON, creating parent dirs.
    Returns the written path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def load_saved_profile(path: Path) -> StyleProfile | None:
    """Load a previously saved profile from ``path``, or ``None`` if the file
    is absent or fails to parse/validate. A corrupt seed is a recoverable miss
    (the caller falls back to build-or-default), not a raise."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return StyleProfile.model_validate(payload)
    except Exception:
        return None


def default_profile() -> StyleProfile:
    """A sensible non-empty profile for when no samples are available."""
    return StyleProfile(
        sample_count=0,
        sentence_count=0,
        avg_sentence_words=14.0,
        common_openers=["Here is", "What matters", "One thing"],
        vocabulary=["ai", "product", "research", "deployment", "teams"],
        tone_traits=["direct", "analytical", "pragmatic"],
        cta_patterns=["What are you seeing in your team?"],
    )


def _load_samples(samples_dir: Path) -> list[str]:
    if not samples_dir.exists() or not samples_dir.is_dir():
        return []

    texts: list[str] = []
    for file_path in sorted(samples_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in _SAMPLE_SUFFIXES:
            continue

        if file_path.suffix.lower() == ".jsonl":
            texts.extend(_read_jsonl_texts(file_path))
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        if content.strip():
            texts.append(content)

    return texts


def _split_sentences(text: str) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(cleaned) if part.strip()]


def _sentence_opener(sentence: str) -> str | None:
    words = [token for token in _WORD_RE.findall(sentence) if token]
    if not words:
        return None
    return " ".join(words[:2]).lower()


def _detect_tone_traits(texts: list[str], sentences: list[str]) -> list[str]:
    joined = "\n".join(texts).lower()
    traits: list[str] = []

    if "?" in joined:
        traits.append("curious")
    if any(token in joined for token in {"i ", "i'm", "i’ve", "my "}):
        traits.append("personal")
    if any(token in joined for token in {"framework", "system", "tradeoff", "signal"}):
        traits.append("analytical")
    if any(len(sentence.split()) < 14 for sentence in sentences):
        traits.append("concise")

    if not traits:
        traits = ["direct", "analytical"]

    deduped: list[str] = []
    for trait in traits:
        if trait not in deduped:
            deduped.append(trait)
    return deduped[:4]


def _detect_cta_patterns(texts: list[str]) -> list[str]:
    patterns = [
        "What are you seeing in your team?",
        "Curious how others are approaching this.",
        "Would you adopt this in production today?",
    ]

    joined = "\n".join(texts).lower()
    found: list[str] = []
    if "what do you think" in joined:
        found.append("What do you think?")
    if "curious" in joined:
        found.append("Curious to hear your take.")

    return found or patterns[:1]


def _read_jsonl_texts(path: Path) -> list[str]:
    texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        text = payload.get("text") or payload.get("content") or payload.get("post")
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return texts


__all__ = [
    "build_from_directory",
    "build_from_texts",
    "default_profile",
    "load_saved_profile",
    "save_profile",
]
