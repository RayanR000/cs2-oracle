# Homepage motion pass: "the instrument reads the market" — authored focal sequence, scroll reveals, hover lift

**Date:** 2026-08-07
**Change:** `frontend/app/page.tsx` hero and section motion rebuilt; motion primitives added to
`frontend/app/globals.css`; `frontend/components/CountUpNumber.tsx` now honours
`prefers-reduced-motion`.
**Bears on:** DESIGN.md's "no decorative motion" rule — deliberately relaxed for the homepage
at the user's explicit request ("make the homepage more animated and creative"). The rest of
the design system is untouched; other pages keep their restraint.

One material idea — a thin amber **scan-line** — carries the whole pass, so the motion reads
as an instrument measuring rather than decoration:

- **Focal sequence (hero, on load):** headline lines clip-reveal, copy/search/proof stagger
  in, the measured `interval_coverage` figure counts up via `CountUpNumber`, and the featured
  item settles from a 1.06 zoom while a scan-line sweeps its frame once. A pulsing status dot
  + "Market intelligence" eyebrow opens the column.
- **Ambient:** the same scan-line returns every 12s across the featured frame at low opacity —
  the instrument re-reading the market. Killed by reduced-motion.
- **Continuity:** trending grid and capabilities section now reveal on first scroll-into-view
  (`whileInView`, `once`), staggered 50–60ms per child.
- **Feedback:** `.widget-lift` raises widget cards 2px on hover (border/background transition
  preserved); arrow links ("View all", "Full backtest", "Read the full backtest") nudge their
  arrows 2px.
- **Craft detail:** four corner reticles on the featured frame echo the viewfinder/scan motif.

Motion primitives in `globals.css`: `.oracle-scan` (keyframes `oracle-scan`, 12s loop,
`--brand` gradient via `color-mix`), `.status-dot` (2.4s pulse), `.widget-lift` (transform +
border/background/shadow transition). All CSS animations are already neutered by the global
`prefers-reduced-motion` block; `CountUpNumber` now collapses to a ~0ms count under it.

Frontend checks: `npm run lint` 0 errors (11 pre-existing warnings), `npx tsc --noEmit` clean,
`npm run build` clean, Impeccable detector reports no findings.
