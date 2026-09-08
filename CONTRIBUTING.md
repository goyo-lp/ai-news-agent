# Contributing to ai-news-agent

Thanks for contributing. This repo uses **GitHub Flow**.

## Branches

- Default + protected: `main`
- Create short-lived branches from `main`:
  - `feat/<short-name>` — new capability
  - `fix/<short-name>` — bug fix
  - `docs/<short-name>` — docs only
  - `chore/<short-name>` — tooling, CI, deps

Keep branches focused and rebased on latest `main` before opening a PR.

## Commits

- Small, descriptive messages. Conventional Commits preferred: `feat:`, `fix:`, `docs:`, `chore:`.
- Don't mix unrelated changes in one PR.

## Pull requests

1. Fork or branch, push your branch.
2. Open a PR against `main` using the PR template.
3. Requirements to merge (enforced on `main`):
   - No required approvals (solo maintainer may self-merge; reviews welcome but not blocking)
   - Green `ci` status check
   - All conversations resolved
   - Up-to-date with `main` (strict status checks)
4. Maintainers squash-merge by default to keep history readable.

Checklist before requesting review:

- [ ] Scope is focused, linked issue included (`Closes #NNN` if applicable)
- [ ] Tests/docs updated if behavior changed
- [ ] No secrets or `.env` files committed
- [ ] CI passes locally (or in PR)

## Issues

Use the issue templates (bug / feature). Include repro steps, expected vs actual, and environment (Python version, langchain version).

## Code of Conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
