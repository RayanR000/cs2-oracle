@AGENTS.md

## Claude Code

Claude Code does not read `AGENTS.md`; this file exists so the import above loads it.
`backend/CLAUDE.md` and `frontend/CLAUDE.md` do the same for their directories and load
on demand when files there are read.

### Subagents

Project subagents live in `.claude/agents/`. Custom subagents load this CLAUDE.md, so
their prompts hold only what is specific to their job.

| Agent | Use it for |
|---|---|
| `archive-analyst` | any number that has to come out of `price-archive/` via DuckDB |
| `backtest-triage` | a red or suspicious Backtest Accuracy run, **before** touching backtest code |
| `pipeline-auditor` | "is the daily chain actually working" — freshness, not badge colour |
| `changelog-writer` | a dated decision record in `docs/changelog/` after work lands |

All four are read-only except `changelog-writer`, which writes only under `docs/`.

For codebase search use the built-in `Explore`; for diff review use the `/code-review`
and `/security-review` skills. Don't build project agents that duplicate those.

The agents in `.opencode/agents/` and the task permissions in `opencode.json` are
**OpenCode-only** and are not invocable here.
