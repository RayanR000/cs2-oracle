# Homepage critique pass: the exhibit shows its strata, search is unified, holdings read as a ledger

**Date:** 2026-08-07
**Change:** homepage (`frontend/app/page.tsx`) reworked per the `$impeccable critique` run on
the specimen-archive homepage; no change to routes, copy claims, or the DESIGN.md world.

- **The exhibit now demonstrates the signature visual.** A live compact strata strip
  (`components/ExhibitStrata.tsx`) mounts under the specimen plate: the item's real 120-day
  history drawn in bone ink, with the 7-day q10–q90 forecast as a translucent specimen-amber
  corridor and the q50 inked through — the first amber on the homepage, gated on an actual
  forecast (Specimen Mark Rule: a forecastless strip renders no amber, just the history
  line; under two points it reads "no strata on record"). Data comes from
  `/items/{id}/price-history` and `/items/{id}/prediction?period=7_days`; the corridor span
  label is computed from the data, never claimed.
- **The status dot no longer borrows a delta color.** The moss pulsing dot beside
  "7 markets · daily" became a neutral `paper-muted` dot, and the claim is anchored with the
  hero item's last recorded close date ("last close Jul 25") when the strip data is present —
  freshness comes from the data, per PRODUCT.md.
- **The holdings section stopped being the banned identical-card grid.** Four identical
  cards became one ledger band using the accuracy card's exact anatomy (divided cells,
  specimen-tag captions, mono values) — the page no longer ends on its most generic block.
- **Search is unified into one shared finding aid.** `components/Search.tsx` now owns the
  full combobox (debounce, arrow-wrap, Enter-opens-first, Esc, aria-activedescendant,
  click-outside + scroll-close, loading/error/empty states, token z-index) and the homepage's
  inline duplicate was deleted. The market page's filter-bar search is a different pattern
  (type chips) and was left untouched.
- **Cold-start skeleton sketches the case.** The exhibit's dead gray square became a
  structured skeleton (caption bar, plate, price row) inside the specimen-card frame; an
  empty-drawer state ("No specimen mounted on record") replaces the vanishing column when
  trending resolves empty.
- **Token hygiene:** header sticky z-index and the search dropdown now use `--z-sticky` /
  `--z-dropdown` instead of raw `z-50`.

**Bears on:** the exhibit fetch adds two requests per homepage load (price history +
7-day prediction for the featured item) — a deliberate depth-vs-latency trade: the plate
renders before the strip resolves, and the strip degrades to an honest unfilled line.
