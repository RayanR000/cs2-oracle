# 2026-08-09 — Deep Dive committed harder: depth-axis hero, mesophotic forecast zone

Follow-up to the design change: the user found the first Deep Dive implementation read as
"the same site, recolored" — the palette had changed but the composition had not. Three
levers were pulled to make the world's signature visible.

- **Depth-axis composition (home).** The hero no longer splits copy/featured card in two
  columns. A 56px depth ruler now runs down the hero's left edge on desktop (SURFACE /
  THERMOCLINE / MESOPHOTIC / THE FLOOR ticks, vertical mono labels), and the featured dive
  becomes a full-width exhibit: skin image left, name + live price + a 170px profile chart
  right (ExhibitStrata gained a `height` prop, default 68 unchanged for other uses).
- **Chart dive treatment (item page).** The forecast region is now visually the mesophotic
  zone: a right-edge gradient over the chart (`forecastZonePct` from `chartData` — from the
  last real close onward, `var(--recess)` at 100%) darkens exactly where the thermocline
  corridor sits; corridor band fill raised 0.16 → 0.24.
- **Stronger palette contrast.** Dark tokens deepened and separated: ground 14→11.5%,
  stock 18.5→17%, recess 11.5→8.5%, paper 93→94%, borders 30→33%, all with more blue-ink
  chroma. Frontmatter in `docs/design.md` synced to match (spec and code still agree).

`docs/design.md` and `frontend/app/globals.css` remain the normative pair; no other pages
were touched beyond the above.
