# Homepage layout pass: orphaned CTA fixed, capabilities weighted, hero scale corrected

**Date:** 2026-08-07
**Change:** `frontend/app/page.tsx` structural fixes only — no copy, data, or motion changes.

- **Trending grid orphan removed.** The CTA card was the 4th item in a 3-column grid on
  desktop (3 cards + 1 dangling card). It is now a full-width closing band
  (`col-span-1 lg:col-span-3`): mobile keeps the clean 2x2 grid, desktop gets a row of three
  item cards then a horizontal band with the item count and a "View market" arrow.
- **Capabilities de-trioed.** Three identical widget cards (a banned "identical feature
  cards" pattern) are now a weighted structure: the Published Backtests card — the
  product's position — leads full-width with the backtest link right-aligned; Quantile
  Forecasts and Multi-Source Archive sit below it in a 2-column row. Grid gap tightened
  `gap-8` → `gap-6`.
- **Hero headline raised to the documented Display scale** (48px on lg, `text-5xl`,
  replacing the off-spec `text-[2.75rem]`).
- Section order and all copy/state/motion preserved; verified against DESIGN.md layout
  patterns for `/`.

Verified at 640 / 768 / 1024 / 1440 / 390 px: no horizontal overflow at any breakpoint,
trending grid resolves to 2-col 2x2 on mobile and 3-col + full-width CTA on lg, capabilities
resolve to 1-col / 2-col+span-2 / stacked correctly. `npm run lint` 0 errors (11
pre-existing warnings), `npx tsc --noEmit` clean, Impeccable detector reports no findings.
