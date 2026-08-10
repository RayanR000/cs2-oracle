# The frontend is deleted, to be rebuilt from scratch

**Date:** 2026-08-10

`frontend/` is gone from `main`. The decision is a design one — the look was not wanted —
combined with a sequencing one: the model is the priority and the dashboard was competing
for attention with it.

## Why this costs nothing to do now

The frontend had **no production surface**. It was never deployed — no `vercel.json`, no
`netlify.toml`, and `.env.local` pointed at `http://localhost:8000`. No workflow in
`.github/workflows/` referenced `frontend`, `npm`, or `node`, so it gated no CI. Over the 60
days to this date it took 31 commits against backend's 398.

There is also a substantive reason not to have kept polishing it. It rendered
`forecast_low`/`forecast_high` from a model that is statistically indistinguishable from
`−return_1d` on rank IC at all four horizons, and whose band coverage has never been
measured on held-out data. Building presentation for an unestablished signal is backwards.

## Recovery points

Nothing is lost. Two references exist:

- **Tag `frontend-last`** → `5a9e325`, the last commit on `main` containing `frontend/`.
- **Branch `frontend/deep-dive-wip`** → `9222bc6`, which additionally carries the
  uncommitted theme/motion pass that was in the working tree at deletion time (theme tokens,
  `MotionShell.tsx`, `PriceChart.tsx`, `MiniPriceChart.tsx`, `useVirtualList.ts`) plus
  `docs/changelog/2026-08-09-theme-toggle-hydration-and-stale-api.md`. **Not to be merged
  as-is** — it is the design being replaced.

For a rebuild, `lib/api.ts` at either ref is the useful artifact: it encodes the shape of
every API route the dashboard consumed.

## What stayed, deliberately

**`backend/config.py`'s `frontend_url` and its consumers.** `main.py:23` uses it for the
CORS allowlist and `api/routes/auth.py:48,105` for the Steam OpenID realm and the
post-login redirect to `/portfolio?session=`. Deleting these would break the auth flow and
they are what a new frontend plugs into.

**`docs/design.md`.** It describes the deleted design. Kept as rebuild input only —
`AGENTS.md` workflow rule 3 now says explicitly that it is not a spec for anything that
currently runs.

**Historical changelog entries.** The dated records under `docs/changelog/` that reference
`frontend/` paths are accurate accounts of decisions made at the time and were not rewritten.

## Pointers updated

`AGENTS.md` (directory map, Commands, and workflow rules 1–3 — the "update
`frontend/lib/api.ts` in the same change" rule is retired), `CLAUDE.md`, `README.md`,
`opencode.json`, `.claude/rules/serving-policy.md`, `.claude/agents/changelog-writer.md`,
and the four `.opencode/agents/*.md` files that told an agent to run `npm run lint` from a
directory that no longer exists.

## Verification

Full backend suite green. No backend test read a frontend file — the only match in
`backend/tests/` was a prose comment in `test_trend_explanation_copy.py` explaining why
`confidence` stays on the response schema. That field's justification is now weaker (no
consumer exists), but the schema was left alone: removing a response field is a breaking
API change and is not part of this.
