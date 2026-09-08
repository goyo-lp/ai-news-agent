# Source registry contract

Status: proposed input for PR03; routes are not production-validated

Owner: PR03 implementer

Research snapshot: 2026-09-07

The source registry is the allowlist and operating record for candidate
discovery. It does not imply that every source is reachable, current, or active.
PR03 must revalidate each route from the execution host and record unavailable
sources explicitly.

## Configuration shape

`config/sources.yaml` will contain one entry per source using this shape:

```yaml
- id: source-01
  name: OpenAI
  publisher_family: openai
  category: labs-platforms
  tier: core
  homepage_url: https://openai.com/news/
  route:
    type: rss                 # rss | atom | official_page_adapter
    url: https://openai.com/news/rss.xml
    status: candidate        # candidate | active | unavailable | disabled
  discovery_provenance: plan-research-2026-09-07
  feed_scope: ai-news
  editorial_notes:
    - Attribute vendor claims.
    - Filter routine corporate announcements.
  enabled: false
  disabled_reason: pending-pr03-validation
  last_checked_at: null
  last_success_at: null
```

Required rules:

- `id` is stable and unique; renaming a publisher does not change it.
- `publisher_family` groups related outlets so syndicated or corporate-family
  coverage cannot masquerade as independent corroboration.
- `category` is one of `labs-platforms`, `independent-reporting`,
  `research-academia`, `builders-analysis`, or `policy-public-interest`.
- `tier` is `core` or `breadth`. Core is an operational coverage tier, not an
  automatic editorial advantage.
- `route.type` declares the ingestion mechanism. An adapter must emit the same
  article schema as RSS and Atom ingestion.
- `route.status` becomes `active` only after PR03 verifies reachability,
  freshness, canonical links, dates, and at least one readable article.
- `discovery_provenance` records how the route was found; it is not evidence of
  current health.
- `feed_scope` identifies broad feeds that require topic filtering and partial
  feeds that require a second route.
- `editorial_notes` carry source-specific attribution, access, volume, and
  corroboration rules into later stages.
- `enabled` remains false until validation succeeds. `disabled_reason` is
  required whenever it is false.
- `last_checked_at` and `last_success_at` are timezone-aware timestamps updated
  by ingestion health reporting, not by editorial scoring.

Configuration must not contain credentials, cookies, copied article content, or
claims that a route is active merely because it passed this research snapshot.

## Proposed 50-source roster

The snapshot contains 36 parsed RSS/Atom routes, 12 official-page adapter
candidates, and 2 unresolved feed checks. Those labels are starting evidence
only. `Core` identifies the 20 sources used by run-health thresholds.

| ID | Source | Category | Tier | Proposed route | Research status |
| --- | --- | --- | --- | --- | --- |
| 01 | [OpenAI](https://openai.com/news/) | Labs & platforms | Core | <https://openai.com/news/rss.xml> | RSS parsed |
| 02 | [Anthropic](https://www.anthropic.com/news) | Labs & platforms | Core | Official-page adapter | Adapter candidate |
| 03 | [Google DeepMind](https://deepmind.google/blog/) | Labs & platforms | Core | <https://deepmind.google/blog/rss.xml> | RSS parsed |
| 04 | [Meta AI](https://ai.meta.com/blog/) | Labs & platforms | Core | Official-page adapter | Adapter candidate |
| 05 | [Mistral AI](https://mistral.ai/news) | Labs & platforms | Core | <https://mistral.ai/news/rss> | RSS parsed |
| 06 | [Qwen](https://qwen.ai/blog) | Labs & platforms | Core | Official-page adapter | Adapter candidate |
| 07 | [DeepSeek](https://api-docs.deepseek.com/updates) | Labs & platforms | Core | Official-page adapter | Adapter candidate |
| 08 | [Cohere](https://cohere.com/blog) | Labs & platforms | Breadth | Official-page adapter | Adapter candidate |
| 09 | [Allen Institute for AI (Ai2)](https://allenai.org/blog) | Labs & platforms | Core | <https://allenai.org/rss.xml> | RSS parsed |
| 10 | [Hugging Face](https://huggingface.co/blog) | Labs & platforms | Core | <https://huggingface.co/blog/feed.xml> | RSS parsed |
| 11 | [NVIDIA Developer](https://developer.nvidia.com/blog/) | Labs & platforms | Core | <https://developer.nvidia.com/blog/feed/> | Atom parsed |
| 12 | [Microsoft Research](https://www.microsoft.com/en-us/research/blog/) | Labs & platforms | Breadth | <https://www.microsoft.com/en-us/research/feed/> | RSS parsed |
| 13 | [AWS Machine Learning](https://aws.amazon.com/blogs/machine-learning/) | Labs & platforms | Breadth | <https://aws.amazon.com/blogs/machine-learning/feed/> | RSS parsed |
| 14 | [Google Research](https://research.google/blog/) | Labs & platforms | Breadth | <https://research.google/blog/rss/> | RSS parsed |
| 15 | [TechCrunch AI](https://techcrunch.com/category/artificial-intelligence/) | Independent reporting | Core | <https://techcrunch.com/category/artificial-intelligence/feed/> | RSS parsed |
| 16 | [The Verge AI](https://www.theverge.com/ai-artificial-intelligence) | Independent reporting | Breadth | <https://www.theverge.com/rss/ai-artificial-intelligence/index.xml> | Atom parsed |
| 17 | [Ars Technica](https://arstechnica.com/ai/) | Independent reporting | Core | <https://arstechnica.com/ai/feed/> | RSS parsed |
| 18 | [WIRED AI](https://www.wired.com/tag/artificial-intelligence/) | Independent reporting | Breadth | <https://www.wired.com/feed/tag/ai/latest/rss> | RSS parsed |
| 19 | [MIT Technology Review](https://www.technologyreview.com/topic/artificial-intelligence/) | Independent reporting | Core | <https://www.technologyreview.com/feed/> | RSS parsed |
| 20 | [VentureBeat AI](https://venturebeat.com/category/ai/) | Independent reporting | Breadth | <https://venturebeat.com/category/ai/feed/> | Recheck required |
| 21 | [The Register AI](https://www.theregister.com/software/ai_ml/) | Independent reporting | Core | <https://www.theregister.com/software/ai_ml/headlines.atom> | RSS parsed |
| 22 | [IEEE Spectrum](https://spectrum.ieee.org/artificial-intelligence) | Independent reporting | Breadth | <https://spectrum.ieee.org/feeds/topic/artificial-intelligence.rss> | RSS parsed |
| 23 | [The Decoder](https://the-decoder.com/) | Independent reporting | Breadth | <https://the-decoder.com/feed/> | RSS parsed |
| 24 | [404 Media](https://www.404media.co/) | Independent reporting | Breadth | <https://www.404media.co/rss/> | RSS parsed |
| 25 | [Rest of World](https://restofworld.org/) | Independent reporting | Breadth | <https://restofworld.org/feed/> | RSS parsed |
| 26 | [TechNode](https://technode.com/) | Independent reporting | Breadth | <https://technode.com/feed/> | RSS parsed |
| 27 | [arXiv AI / ML / NLP / vision](https://arxiv.org/) | Research & academia | Core | <https://rss.arxiv.org/rss/cs.AI+cs.LG+cs.CL+cs.CV> | RSS parsed |
| 28 | [Berkeley AI Research (BAIR)](https://bair.berkeley.edu/blog/) | Research & academia | Breadth | <https://bair.berkeley.edu/blog/feed.xml> | Recheck required |
| 29 | [Stanford HAI](https://hai.stanford.edu/news) | Research & academia | Breadth | Official-page adapter | Adapter candidate |
| 30 | [MIT News: AI](https://news.mit.edu/topic/artificial-intelligence2) | Research & academia | Breadth | <https://news.mit.edu/rss/topic/artificial-intelligence2> | RSS parsed |
| 31 | [METR](https://metr.org/blog/) | Research & academia | Core | <https://metr.org/feed.xml> | RSS parsed |
| 32 | [Nature Machine Intelligence](https://www.nature.com/natmachintell/) | Research & academia | Breadth | <https://www.nature.com/natmachintell.rss> | RSS parsed |
| 33 | [ScienceDaily AI](https://www.sciencedaily.com/news/computers_math/artificial_intelligence/) | Research & academia | Breadth | <https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml> | RSS parsed |
| 34 | [Epoch AI](https://epoch.ai/latest) | Research & academia | Core | <https://epochai.substack.com/feed> | RSS parsed; newsletter only |
| 35 | [Simon Willison](https://simonwillison.net/) | Builders & analysis | Core | <https://simonwillison.net/atom/everything/> | Atom parsed |
| 36 | [Artificial Analysis](https://artificialanalysis.ai/articles) | Builders & analysis | Core | Official-page adapter | Adapter candidate |
| 37 | [Ahead of AI — Sebastian Raschka](https://magazine.sebastianraschka.com/) | Builders & analysis | Breadth | <https://magazine.sebastianraschka.com/feed> | RSS parsed |
| 38 | [Grok / xAI (SpaceXAI)](https://x.ai/news) | Labs & platforms | Breadth | Official-page adapter | Adapter candidate |
| 39 | [Interconnects — Nathan Lambert](https://www.interconnects.ai/) | Builders & analysis | Breadth | <https://www.interconnects.ai/feed> | RSS parsed |
| 40 | [Latent Space](https://www.latent.space/) | Builders & analysis | Breadth | <https://www.latent.space/feed> | RSS parsed |
| 41 | [Import AI — Jack Clark](https://importai.substack.com/) | Builders & analysis | Breadth | <https://importai.substack.com/feed> | RSS parsed |
| 42 | [LangChain Blog](https://www.langchain.com/blog) | Builders & analysis | Breadth | <https://www.langchain.com/blog/rss.xml> | RSS parsed |
| 43 | [LlamaIndex Blog](https://www.llamaindex.ai/blog) | Builders & analysis | Breadth | Official-page adapter | Adapter candidate |
| 44 | [NIST AI](https://www.nist.gov/artificial-intelligence) | Policy & public interest | Breadth | <https://www.nist.gov/news-events/news/rss.xml> | RSS parsed |
| 45 | [European Commission: AI](https://digital-strategy.ec.europa.eu/en/policies/artificial-intelligence) | Policy & public interest | Breadth | Official-page adapter | Adapter candidate |
| 46 | [OECD.AI](https://oecd.ai/en/wonk) | Policy & public interest | Breadth | Official-page adapter | Adapter candidate |
| 47 | [UK AI Security Institute](https://www.aisi.gov.uk/blog) | Policy & public interest | Core | Official-page adapter | Adapter candidate |
| 48 | [Electronic Frontier Foundation](https://www.eff.org/issues/ai) | Policy & public interest | Breadth | <https://www.eff.org/rss/updates.xml> | RSS parsed |
| 49 | [AI Now Institute](https://ainowinstitute.org/) | Policy & public interest | Breadth | <https://ainowinstitute.org/feed> | RSS parsed |
| 50 | [Georgetown CSET](https://cset.georgetown.edu/) | Policy & public interest | Breadth | <https://cset.georgetown.edu/feed/> | RSS parsed |

## PR03 acceptance notes

PR03 must preserve all 50 IDs and produce a coverage report that distinguishes
attempted, active, unchanged, failed, and unavailable sources. It must not claim
50-source coverage until every source has a working RSS, Atom, or official-page
adapter route.

Known validation work includes rate-limit/backoff handling for VentureBeat,
reachability and recency checks for BAIR, adapters for the 12 candidates, topic
filtering for broad feeds, partial-coverage handling for Epoch AI, publisher
family grouping for Google Research and Google DeepMind, and bounded streaming
for unusually large TechNode and METR responses.

The editorial rules governing these sources are defined in
[the editorial policy](editorial-policy.md).
