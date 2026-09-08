# Editorial policy

Status: accepted for the first implementation

Owner: project maintainer

Last reviewed: 2026-09-07

This document is the editorial contract for AI News Agent. It defines which
stories qualify, how they are scored, what evidence is required, and when a
daily digest must publish fewer than ten stories or not publish at all.

## Audience and promise

The primary audience is AI builders and technically curious readers who want a
compact, evidence-backed account of material AI developments. Readers should be
able to understand what changed, inspect the supporting article, and decide
whether the development deserves more attention.

The digest favors consequence over volume. It publishes up to ten distinct
stories, but never fills a slot with a weak, stale, repetitive, or poorly
supported item.

## Preferred coverage

In descending order of editorial priority, the digest covers:

1. Material model, agent, developer-tool, infrastructure, and open-source
   releases.
2. Research results that change the evidence about capabilities, safety,
   evaluation, efficiency, or real-world usefulness.
3. Security incidents, demonstrated misuse, significant failures, and credible
   accountability reporting.
4. Adopted policy, court decisions, standards, and regulatory milestones with
   practical consequences.
5. Significant market or organizational changes when they alter who can build,
   deploy, fund, or access AI.
6. Labor, education, civil-liberties, and regional effects supported by concrete
   reporting or data.

Coverage is not limited to the United States or to the largest vendors. A
smaller source or regional development can outrank a widely covered launch when
its consequences are more substantial.

## Exclusions

Exclude an item when any of the following applies:

- It is unrelated to AI or only uses AI as a marketing label.
- It is a rumor, anonymous claim, prediction, or social-media reaction without
  publishable evidence.
- It is an evergreen explainer, tutorial, listicle, event promotion, job post,
  or routine corporate announcement without a material new development.
- It is funding news with no demonstrated effect beyond the transaction.
- It repeats a story already published by the digest and contains no meaningful
  follow-up.
- It is a minor version bump, benchmark claim, or product availability change
  that does not materially affect capability, cost, access, safety, or use.
- The article is inaccessible or too incomplete to support a three-sentence
  summary. A headline, feed excerpt, or search snippet alone is insufficient.
- Its central claims cannot be attributed or supported after the bounded
  verification process.

Paywalled reporting may identify a candidate, but it cannot be the basis of a
summary unless enough of the article is legitimately accessible. Preprints,
vendor posts, advocacy analysis, and institutional press releases are allowed
when their provenance and limitations are explicit; none counts as independent
corroboration of itself.

## Freshness and follow-ups

The editorial day ends at the configured digest cutoff time. A normal candidate
must have been first published during the preceding 24 hours. Collection uses a
36-hour overlapping lookback so late-arriving feeds and temporary failures do
not create silent gaps.

An item first discovered 24 to 36 hours after publication may qualify only when
it was not included previously, the delay is recorded, and the item still clears
the normal score and evidence thresholds. Its timeliness score cannot exceed 3.
Older items do not qualify merely because a source republishes, syndicates, or
updates a timestamp.

A follow-up to an older story is meaningful only if it adds a verified fact that
could change a reader's understanding or action. Examples include general
availability after a preview, a newly released technical report, independently
measured results, a binding policy step, a disclosed incident impact, or a
material correction. Commentary, additional reactions, repeated claims, and
minor rollout expansion are not meaningful follow-ups.

Every candidate must retain its original publication time, latest update time,
first-seen time, and any related previously published story ID. When dates are
missing or contradictory, the candidate is ineligible until the uncertainty is
resolved.

## Scoring rubric

Editors and the future agent score each criterion from 0 to 5 using only the
available evidence. Scores of 1, 3, and 5 are the low, medium, and high anchors;
0 means the criterion is absent, and 2 or 4 means the evidence falls between
anchors. The weighted total is computed in code, not by the model:

```text
total = (relevance / 5 * 30)
      + (impact / 5 * 25)
      + (novelty / 5 * 20)
      + (evidence / 5 * 15)
      + (timeliness / 5 * 10)
```

| Criterion | Weight | 1 — low | 3 — medium | 5 — high |
| --- | ---: | --- | --- | --- |
| Relevance | 30% | AI is incidental or the practical change is unclear. | Directly affects a defined part of the audience or a preferred topic. | Broadly changes how AI is built, evaluated, governed, accessed, or experienced. |
| Impact | 25% | Little observable effect beyond the publisher or announcement. | Meaningful effect on one product, research area, organization, or community. | Large or durable consequences across organizations, a major ecosystem, public policy, safety, or society. |
| Novelty | 20% | Repeats known facts or adds commentary only. | Adds a concrete capability, result, event, or meaningful follow-up. | Establishes a genuinely new capability, finding, incident, or binding change that revises prior understanding. |
| Evidence | 15% | Headline, snippet, rumor, or uncheckable assertion. | Accessible primary article with attributable claims, methods, or documentation; material limitations remain. | Detailed primary evidence plus credible independent reporting, replication, or directly inspectable data. |
| Timeliness | 10% | Outside 24 hours without a justified late-arrival exception. | Within 24 hours or a recorded 24–36 hour late arrival. | Newly published near the cutoff and not previously covered. |

To qualify for final selection, a story must:

- score at least 65 out of 100;
- score at least 3 for relevance and evidence;
- satisfy the freshness or meaningful-follow-up rule; and
- pass the evidence, duplication, and summary checks in this policy.

The threshold is a quality floor, not an entitlement to publication. The final
selection may favor a slightly lower-scoring qualifying story to improve topic,
source, geographic, or institutional diversity. Any such choice must retain both
scores and a short rationale.

## Worked include and exclude decisions

### Include: consequential open model release

A lab releases weights, a technical report, and reproducible evaluations for a
model with a materially improved cost/capability tradeoff. Independent testing
confirms part of the claim. Scores: relevance 5, impact 4, novelty 4, evidence 5,
timeliness 5; weighted total 91. Include because builders can act on the release
and the strongest claims have inspectable support.

### Include: binding policy milestone

A regulator publishes a final rule with an effective date and concrete duties
for model providers. The official text and careful independent reporting are
available. Scores: relevance 4, impact 5, novelty 4, evidence 5, timeliness 5;
weighted total 90. Include, distinguishing adopted requirements from proposals
and identifying who is affected.

### Exclude: heavily promoted feature update

A vendor announces a cosmetic assistant redesign, supported only by its launch
post, with no meaningful capability or access change. Scores: relevance 3,
impact 1, novelty 1, evidence 3, timeliness 5; weighted total 48. Exclude because
it misses the quality floor and its practical impact is small.

### Exclude: repeated benchmark coverage

A newsletter repeats a vendor's week-old benchmark claim without new methods,
data, availability, or independent testing. Scores: relevance 3, impact 2,
novelty 1, evidence 1, timeliness 1; weighted total 36. Exclude because it is
stale, repetitive, and insufficiently supported.

### Exclude now, reconsider as a follow-up

A credible outlet reports that a previewed agent is coming soon, but the company
has not provided access details or documentation. Scores: relevance 4, impact 2,
novelty 2, evidence 2, timeliness 5; weighted total 57. Exclude now. Reconsider
when general availability, documentation, or credible testing creates a
meaningful follow-up with sufficient evidence.

## Evidence requirements

Each selected story has one representative article and may have supporting
sources. The evidence record must preserve stable references to the exact text
used for every factual claim.

- Sentence-level factual claims must be supported by accessible article text or
  linked primary material.
- Vendor, government, author, and advocacy claims must be attributed as claims.
- Numbers, dates, benchmark names, release status, and legal status must match
  the source exactly and include necessary context.
- A source derived from the same press release, corporate family, or syndicated
  copy is not independent corroboration.
- Conflicting credible accounts must be represented or the claim omitted.
- Inference is allowed only in the third sentence, must be labeled as analysis,
  and must follow directly from cited facts.

The 50-source registry supplies candidates. Primary sources linked from those
articles may be used for verification. Broader open-web search is not part of
the first implementation; adding it requires a policy review because it changes
the system's evidence boundary.

## Three-sentence story format

Every published story contains a linked headline, publisher and publication
time, followed by exactly three prose sentences:

1. **What happened:** state the concrete new event and identify the actor.
2. **Supporting detail:** give the most decision-relevant evidence, number,
   scope, availability detail, or limitation.
3. **Why it matters:** explain the consequence for the audience or state a
   material uncertainty without introducing unsupported facts.

Each field must contain one grammatical sentence. Bullets, fragments, semicolon
chains that hide multiple sentences, and promotional language are not allowed.
The summary must describe the linked representative article, not synthesize a
different story from unrelated supporting sources.

Example:

> **Example Lab released Model A under an open license, with weights available
> today.** Its report shows a 20% improvement on Benchmark B, although no
> independent reproduction was available at publication time. The release may
> lower deployment costs for builders, but the vendor-reported result should be
> treated as provisional.

## Selection and diversity

The digest contains no duplicate events. When several sources cover one event,
choose the article with the best accessible evidence as representative and keep
other useful coverage as supporting evidence.

Apply these preferences after the quality floor:

- avoid more than two stories from one publisher family unless omitting another
  would hide a clearly more consequential event;
- prefer independent reporting when it is as informative as a vendor account;
- avoid a digest dominated by one topic, company, country, or source type;
- preserve strong research, public-interest, regional, and smaller-source items
  that high-volume feeds might otherwise crowd out; and
- record the reason whenever diversity changes score order.

These are preferences, not quotas. A day dominated by one genuinely major event
may produce a correspondingly focused digest, but duplicated angles still count
as one story.

## Shortfalls and run states

Publishing fewer than ten stories is correct whenever fewer than ten candidates
pass all policies. A shortfall record must report how many slots were unfilled
and summarize why, such as low relevance, stale coverage, duplication,
insufficient evidence, or verification failure.

Every run has one of these states:

- **Healthy:** at least 80% of active Core 20 sources and 70% of all active
  sources were checked successfully, and every published story passed policy.
- **Degraded:** coverage falls below either healthy threshold, one or more
  collection stages partially fails, or the run finishes with unresolved but
  non-selected candidates. The digest may publish only if at least three stories
  pass all normal checks. It must display a visible degraded-run notice with
  attempted, successful, failed, and unchanged source counts. Failed sources do
  not lower the evidence standard.
- **Failed:** fewer than half of active Core 20 sources were checked, collection
  or verification cannot complete, no final structured digest can be persisted,
  or fewer than three stories qualify during a degraded run. Do not send email
  or publish a dated archive; persist the failure and alert the operator.

Unverified drafts never enter the delivery queue. Retries and repair loops must
be bounded; exhaustion produces a degraded or failed result rather than silent
success.

## Source registry contract

The proposed 50 sources and the configuration fields required for PR03 are in
[the source registry](source-registry.md). Registry status describes research
input, not production coverage. PR03 must validate every route on the execution
host before marking a source active.

## Decision record

Critical PR01 decisions are resolved below. Remaining choices are deliberately
assigned and must not become hidden implementation defaults.

| Decision | Status | Resolution or next action | Owner | Due |
| --- | --- | --- | --- | --- |
| Audience and topic priorities | Accepted | Use the audience and ordered coverage areas in this policy. | Project maintainer | PR01 |
| Freshness and meaningful follow-up | Accepted | Use a 24-hour editorial window, 36-hour collection overlap, and the material-new-fact test. | Project maintainer | PR01 |
| Quality floor and shortfalls | Accepted | Require 65/100 plus relevance and evidence minimums; publish fewer than ten. | Project maintainer | PR01 |
| Evidence boundary | Accepted | Registry sources discover candidates; directly linked primary evidence may verify them; no broader search initially. | Project maintainer | PR01 |
| Diversity behavior | Accepted | Apply documented preferences after the quality floor and retain a rationale for score-order changes. | Project maintainer | PR01 |
| Degraded-run behavior | Accepted | Publish three or more fully verified items with a warning; otherwise fail without delivery. | Project maintainer | PR01 |
| Recipient, verified sender, delivery time, and timezone | Open | Select before scheduled email is enabled. | Project maintainer | PR12 |
| Model provider and per-run spend limit | Open | Benchmark supported models and set a hard budget before live editorial calls. | Project maintainer | PR07 |
| HTML archive hosting and visibility | Open | Keep archives local until privacy, retention, and hosting are decided. | Project maintainer | PR12 |
| Registry route readiness | Open | Revalidate all 50 sources, implement adapters, and explicitly disable unavailable sources. | PR03 implementer | PR03 |

Changes to accepted policy require a pull request that updates examples and, once
available, evaluation fixtures. Temporary operational exceptions must be
recorded in the run rather than silently changing this contract.
