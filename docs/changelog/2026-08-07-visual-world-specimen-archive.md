# Visual world replaced: the carbon-and-teal "analytical instrument" becomes "The Specimen Archive"

**Date:** 2026-08-07
**Change:** `docs/design.md` rewritten — new visual world committed via the impeccable new-work
workshop (seed `08df685d`, roll index 4). The doc now follows the canonical DESIGN.md format
(machine-readable frontmatter + eight sections) and is normative for the rebuild; no
implementation exists yet.
**Bears on:** the previous design system (`docs/design.md` as of 2026-08-07), which it replaces.

The incumbent world (warm carbon, teal accent, "analytical instrument") was replaced after
three re-rolls and two steers — the final steer being "closer to the CS2 world". The chosen
world, **The Specimen Archive**, reframes the product as a working museum of the market:

- **Survivors kept:** name, routes, all content; soft dark as the primary theme (light derived
  secondary); asset-grounded skin imagery; honest uncertainty (ranges + published accuracy).
- **World:** every item is a specimen on an archive card — skin image as specimen plate, price
  history as strata, q10–q90 forecast corridor as a translucent amber stratum with the q50
  inked through, forecast summary as a hanging curator tag.
- **Palette:** warm umber case stock (hue 55) and bone paper text (hue 85) replace carbon;
  one interaction ink (archive blue, hue 250); specimen amber (hue 75) reserved for forecast
  marks only; moss/brick data colors replace the old teal/green-red pair.
- **Laws:** The Specimen Mark Rule (amber = forecast only), The Ink Rule (ink = interaction
  only, never in charts), The Ledger Rule (all numerals in tabular mono), The Case Rule
  (flat by default), The Empty Drawer Rule (absence shown honestly — the "no forecast"
  state is an unfilled ledger line).
- **Failure modes held off:** no glow/neon, no trading green/red exuberance, no gambling
  energy, no decorative motion, no display fonts — per PRODUCT.md's binding anti-references.

Deferred: `frontend/app/globals.css` still carries the old tokens (brand-hue 190 teal, hue 30
neutrals); the rebuild must mirror the frontmatter in `docs/design.md` (hue 55 stock / hue 85
paper / hue 250 ink / hue 75 specimen). The `frontend/AGENTS.md` design-context block restates
the old token values and goes stale with it.
