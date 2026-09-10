@AGENTS.md

<!--
Maintainer note (stripped before this file reaches Claude's context, so it costs no tokens):
keep this file + AGENTS.md under ~200 lines combined. Anything subsystem-specific belongs in
.claude/rules/ with `paths:` frontmatter, not here. Run /doctor to check for bloat.
-->

## Claude Code

Claude Code does not read `AGENTS.md`; this file exists so the import above loads it.
`backend/CLAUDE.md` does the same for its directory and loads on demand when files there
are read.

### Rules

`.claude/rules/*.md` carry the backend subsystem detail, each scoped with `paths:`
frontmatter so it enters context only when you read a matching file. `backend/AGENTS.md`
indexes them and states the four invariants that hold regardless.

Two consequences worth knowing:

- A path-scoped rule fires on a **file read**. Writing a new file from scratch does not
  trigger it, so read the relevant rule yourself before starting work in its area.
- Nested `CLAUDE.md` files and path-scoped rules are **not re-injected after `/compact`** —
  only the root `CLAUDE.md` is. After a compaction mid-task, re-read the rule you were
  working under.

These rules are Claude-only. Agents that read `AGENTS.md` directly see the invariants and
the index, but not the detail.

### Subagents

Project subagents live in `.claude/agents/`. Custom subagents load this CLAUDE.md, so
their prompts hold only what is specific to their job. Each one's `description` and `tools`
are already in context — pick from that listing rather than restating it here.

For codebase search use the built-in `Explore`; for diff review use the `/code-review`
and `/security-review` skills. Don't build project agents that duplicate those.

### Delegation to opencode (coordinator pattern)

Claude Code is the **coordinator**: plan, verify, chat. `opencode` (CLI, installed
globally, free model) handles mechanical execution. Its tool calls and file dumps never
enter this context — only final stdout does. Delegate aggressively to keep the token
budget for judgment.

```
opencode run "<self-contained task; state the exact output format>"
```

No agent definitions needed; the default `build` primary agent already has full tool
permissions. Do **not** pass `--auto` — Claude Code's auto-mode classifier blocks it.

**Delegate by default:**
- **Exploration** — multi-file greps, "find all X", codebase inventories, archive scans.
- **Implementation** — boilerplate, mechanical edits, bulk renames, adding flags/args,
  wiring plumbing that doesn't require modelling judgment.
- **Docs** — changelog entries, docstring updates, README sections, doc scaffolding.

**Keep inline** (don't delegate):
- Single file reads or quick 1-2 line edits — spawning a subprocess is slower.
- Modelling judgment, anything touching the forecast chain or ML parameters.
- Tasks where verifying the weak model's output costs more than doing it yourself.
- Iterative work requiring intermediate inspection (progress is invisible mid-run).

**Rule of thumb:** delegate when the task would put ≥3 tool calls of low-value output
into context. A quick `Read` isn't worth a round-trip; grepping 50 files is.
