# Homepage rewrite: fabricated data removed, published accuracy becomes the hero, brand unified on CS2 Oracle

**Date:** 2026-08-07
**Change:** `frontend/app/page.tsx` rewritten; `frontend/app/globals.css` contrast tokens
raised to WCAG AA in both themes; `frontend/components/Header.tsx` and the footer rebranded;
`frontend/lib/api.ts` exposes `price_tier` on the accuracy calls; `backend/api/routes/accuracy.py`
sanitises NaN/Inf before JSON serialisation.
**Bears on:** `.impeccable/critique/2026-08-07T18-26-23Z__frontend-app-page-tsx.md` (the critique this addresses).

The homepage shipped hardcoded market figures presented as live: `FALLBACK_ITEMS`, default
stats, and `volume24h = avgPrice * 2800`. That violates the product's binding principle
("never fabricate numbers") on the product's own face. The rewrite:

- **Fabricated data is gone.** No fallback items, no invented stats. The page renders
  skeletons while fetching, and on failure shows a labelled degraded state ("Market data is
  unavailable…") instead of plausible numbers. `getItemsCount()` is the only live number and
  is omitted, not replaced, when it fails.
- **The hero-metric row (a documented anti-pattern) is replaced by the position itself:**
  a live `interval_coverage` figure from `/accuracy/latest?price_tier=-1` (the ≥$1 headline
  cohort), with horizon, cohort, and measurement date named, linking to `/accuracy`. If the
  accuracy API is unreachable the block hides — no placeholder number is ever shown.
- **Capabilities section rewritten** around the product's real differentiators: quantile
  forecasts, published backtests (with the live directional-accuracy-vs-baseline figure when
  available), and the multi-source archive. The generic "Built for traders who need clarity"
  copy is gone.
- **Brand unified on CS2 Oracle** across header and footer; the dead "Terminal" tag is gone
  from the search box; header container aligned to the 1200px main column.
- **Search hardened:** Esc closes, arrow keys move selection, Enter opens the highlighted
  result, listbox semantics added, dropdown height-capped, and a network failure now says so
  instead of rendering "No items match".
- **Contrast fixed in both themes** (`text-muted`/`text-tertiary` raised to ≥4.5:1 on all
  surfaces they sit on — dark tertiary 52→64% L, muted 38→62% L; light tertiary 55→48% L,
  muted 72→50% L), the 9px labels raised to the 10px scale floor, and the
  `widget-block:hover` glow removed per the "no glow" rule.

Backend: `/accuracy/*` returned 500 because the Parquet mirror stores NaN in nullable float
columns (e.g. `evaluation_window_days`), which `json.dumps` rejects. `_json_safe` in
`backend/api/routes/accuracy.py` recursively maps NaN/Inf to `null` in every response path.
Without it, the proof figure (and the whole accuracy surface) could never be served.
