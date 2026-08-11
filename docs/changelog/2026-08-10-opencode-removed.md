# OpenCode configuration removed

**Date:** 2026-08-10

Claude Code is the only agent harness used on this project, so the parallel OpenCode
configuration is gone.

Deleted:

- `opencode.json` — the `$schema`, `instructions: ["AGENTS.md"]`, and the `build`/`plan`
  task permissions that allowed the six subagents below.
- `.opencode/agents/{data,document,explore,review,security,test}.md`
- `.opencode/plugins/chain.js` — a `session.idle` hook that re-dispatched work after edits,
  behind a 5-minute cooldown.
- `.opencode/{package.json,package-lock.json,node_modules/,.gitignore}` (the latter two
  were untracked).
- The stray `.git/opencode` file OpenCode wrote to record a commit SHA.

Nothing unique was lost. `data.md` and `document.md` are covered by
`.claude/agents/archive-analyst.md` and `.claude/agents/changelog-writer.md`;
`explore.md`, `review.md`, and `security.md` duplicate the built-in `Explore` agent and the
`/code-review` and `/security-review` skills. `test.md` only ran `pytest`, which
`backend/AGENTS.md` already documents.

`CLAUDE.md` lost the paragraph pointing at `.opencode/agents/`. The reference in
`2026-08-10-frontend-removed.md` is left alone — it is a dated record of what was true then.
