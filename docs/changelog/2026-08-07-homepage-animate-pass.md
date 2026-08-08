# Homepage animate pass: ambient loops stop offscreen

**Date:** 2026-08-07
**Change:** `frontend/app/page.tsx` + `frontend/app/globals.css` — the Specimen Archive
motion contract (lamp sweep, status pulse, card hover shifts) verified implemented on the
home page; added the one missing piece of the motion discipline: the two infinite ambient
loops now pause while their exhibit is offscreen.

The curator's lamp sweep (14s) and the status pulse (2.4s) are the world's only ambient
motion. They ran unconditionally even after the featured specimen scrolled out of view.
An `IntersectionObserver` on the exhibit card now toggles `.exhibit-hidden`, which sets
`animation-play-state: paused` on `.lamp-sweep` and `.status-dot`. Loops resume on
re-entry; reduced-motion handling unchanged (sweep removed, all durations ~0).

Nothing else moves: no entrance choreography, no scroll-triggered decoration, no chart
line animation — per `docs/design.md`.

**Verified:** `npm run lint` 0 errors (7 pre-existing warnings), `npx tsc --noEmit` clean,
`npm run build` green, Impeccable detector no findings.
