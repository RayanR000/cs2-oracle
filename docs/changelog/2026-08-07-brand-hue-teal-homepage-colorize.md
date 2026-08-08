# Brand accent moved from amber to teal; homepage colorized within the system

**Date:** 2026-08-07
**Change:** `frontend/app/globals.css` `--brand-hue` 55 → 190; `docs/design.md` palette and
copy updated; `frontend/app/page.tsx` color pass; `frontend/AGENTS.md` hue note updated.
**Bears on:** the homepage rewrite (2026-08-07-homepage-rewrite.md) — same surface, no copy or
data changes.

Two changes landed in one pass:

- **Accent hue: amber → teal.** `--brand-hue` is the single derived source, so focus rings,
  selection, links, chart series, and every token-based usage re-hued without further edits.
  Chosen to keep the "analytical instrument" register: cool teal reads as an instrument light
  against the warm-tinted carbon (hue 30) neutrals. The `brand-subtle` selection surface keeps
  the warm hue 30 tint; `data-up` (emerald 155) and `data-down` (rose 25) semantics are
  untouched and remain visually distinct from the accent. Light-theme chroma reduced to 0.1–0.12
  and dark-theme active/light values trimmed so teal does not push toward neon.
- **Homepage colorize** (see also the earlier motion pass): ambient teal glows in the hero and
  capabilities sections (`--brand` at ~10% via `.ambient-glow`), the featured item's corner
  brackets and live-dot switched from gray borders to brand, the coverage cell anchored with a
  `brand-subtle` tint, directional accuracy colored semantically against its stated baseline
  (data-up above, data-down below — the same convention the accuracy page already used), and the
  "published backtests" position card given a fading brand hairline. Trending cards stay
  monochrome: `/items/trending` carries no change data, so up/down color would be fabricated.

Verified: `npm run lint` 0 errors, `tsc --noEmit` clean, Impeccable detector clean.
