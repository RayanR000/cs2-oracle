# Frontend rebuilt to the Specimen Archive world

**Date:** 2026-08-07
**Change:** `frontend/app/globals.css`, all pages, and all components rebuilt to mirror the
normative frontmatter in `docs/design.md` (the "Specimen Archive" world committed
2026-08-07). The old carbon-and-teal "analytical instrument" tokens and motion are gone.

**Tokens** — `globals.css` now carries the archive vocabulary, dark primary, light derived:
case stock on hue 55 (ground/stock/recess/surface family), bone paper on hue 85 (paper
family), ink-blue interaction (ink family, hue 250), specimen amber (hue 75, forecast
only), moss/brick deltas (hue 150/25). Legacy names (`--brand`, `--accent-primary`,
`--text-secondary`, `--data-up`, …) are aliased so nothing silently falls back. Typography
scale classes added (`text-display` 48, `text-headline` 32, `text-title` 22,
`text-data-lg` 20, `text-data-sm` 12, `.specimen-tag` mono micro) plus button classes
(`btn-primary` ink-on-recess, `btn-secondary`, `btn-ghost`, `btn-danger`), `.input-search`,
`.wear-pill`, `.specimen-card` (the only shadowed card), `.specimen-pin` (2px ink corner
bracket), `.ledger-row`.

**Pages**
- Home: placard hero (48px statement + the live backtest figure as headline record with
  cohort and horizon named), finding-aid search with keyboard-first listbox, featured
  exhibit under the curator's lamp sweep, featured specimens, holdings summary. All
  scroll-driven calibration, glow, and framer-motion choreography removed.
- Market: catalog drawers — finding aid + type filter (wear pills), 3-col specimen grid,
  ledger table with sticky recess header, sortable ≤7 columns, paginated, moss/brick
  2px-pill change badges, skeleton rows, centered empty state, brick-tinted error banner.
- Item detail: specimen card (pin, plate caption) holding the strata chart and wear tray
  (2/3) beside the sidebar (1/3) with curator tag, ledger stats, signals, event impacts,
  forecast drivers — no nested widget stacks. Strata chart: bone-ink history line,
  multi-source series in paper-tertiary, hairline grid without verticals, amber q10–q90
  corridor with the q50 inked through (drawn only when the time range contains the
  forecast horizon), specimen-tag range tabs active in ink, card-stock mono tooltip.
  Curator tag: amber label, q50 + q10–q90 at 3/7/14/30d, measured-accuracy line naming
  cohort and horizon; forecastless items get the empty-drawer state ("no curator note on
  record") and no amber anywhere.
- Portfolio: collection case — three ledger stats above the full-width inventory table,
  Steam CTA when unauthenticated.

**Removed:** `CountUpNumber` (number animation — the ledger does not perform) and all
framer-motion imports.

**Verified:** `npm run lint` 0 errors / 7 pre-existing warnings; `npx tsc --noEmit` clean;
`npm run build` green; all five routes return 200 on the built app.
