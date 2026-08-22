> **The frontend this describes was deleted on 2026-08-10.** There is no Next.js app; the
> `frontend/app/*` paths referenced throughout this document — `layout.tsx`, `globals.css`,
> the six routes — no longer exist. See
> [`changelog/2026-08-10-frontend-removed.md`](changelog/2026-08-10-frontend-removed.md).
> This file is kept as **rebuild input only** — the design system it records is the starting
> point for a rebuild, not a description of anything that currently runs. Read every present
> tense below as past tense.
>
> ⚠️ Two data claims in here have since moved: the served band is now a *signed* two-quantile
> conformal interval scaled by a per-item climatology (2026-08-19/20, see the banner in
> `docs/README.md`), and the range product gained a magnitude signal — `move_odds` /
> `stability_label`, ranked at `GET /items/volatility` — that no component below accounts for.
> The `verdict-strip`'s directional-verdict cell is also no longer a shippable product claim
> (`changelog/2026-08-15-cs2-oracle-is-a-range-forecaster.md`).

<!-- THE MARKET DESK: the category standard, played straight. Chosen on 2026-08-09 via the
standing exit on decision board 308b07a7 (roll seed 5654f592). Deal 7 of the concept-seed
seeded run; after weighing challengers against the assigned Case Stencil Kit, the challenger
won both axes — "more like the thing itself" and "something else entirely" — and became the
build. The world replaces The Deep Dive, which the user marked wrong on every axis. Craft bar:
TradingView, Skinport, CSGOStash. The contract comment below mirrors the one in
frontend/app/layout.tsx; the tokens mirror frontend/app/globals.css exactly — reference them,
don't restate them in components.

THESIS: A market analytics dashboard where a visitor searches, reads a price history with a
forecast band, and checks the published accuracy record beside it — the arrangement this
category ships, at the craft bar of TradingView, Skinport, and CSGOStash, with the brand's
anti-references still binding: no data-vomit density, no neon gaming chrome, no trading
exuberance, no SaaS cookie-cutter.

OWN-WORLD: Neutral slate ground in light and dark; one blue accent owns interaction and the
forecast band; desaturated green/red belong to price deltas alone; Inter for notes, JetBrains
Mono for every numeral; flat bordered panels, hairline grid, 4/6/8px radii, no shadows, no
motion loops.

STORY: The visitor understands a forecast as a measured band beside a price history and
believes the error record because it is published with its cohort and horizon named; every
surface reads in the standard order — item, price, context, chrome.

FIRST VIEWPORT: The buoy header; a statement and a sounding search; the headline accuracy
strip (directional verdict, hit rate vs chance, coverage, cohort — 7d horizon, named); a
featured-item grid with price readouts; the market coverage row.

FORM: The category standard, chosen via the standing exit on decision board 308b07a7 (roll
seed 5654f592); craft bar TradingView / Skinport / CSGOStash.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review,
the verdict, and DESIGN.md. The build passed: npm run lint (0 errors), npx tsc --noEmit,
npm run build; headless review round found and fixed two defects — the accuracy table's
sr-only cells escaped their scroll wrapper on mobile (position:absolute with no containing
block; fixed with `relative` on the table), and fresh sessions fell back to the system
light theme (ThemeProvider now defaults to the brand's dark primary). DOM review: no
horizontal overflow at 1440px or 390px across all six routes, no console errors, both font
families applied, canvas charts render, token vars resolve on every element, no broken
images after full scroll. -->
---
name: CS2 Oracle
description: The market desk — thirteen years of CS2 prices, forecast one item at a time.
# Grounds & surfaces (light first; dark overrides in [data-theme="dark"])
colors:
  ground: "oklch(97.5% 0.006 250) / oklch(12.5% 0.012 255)"
  stock: "oklch(94.5% 0.008 250) / oklch(16.5% 0.014 255)"
  recess: "oklch(91.5% 0.008 250) / oklch(10% 0.01 255)"
  surface: "oklch(99.5% 0.004 250) / oklch(19.5% 0.016 255)"
  surface-hover: "oklch(97% 0.007 250) / oklch(22% 0.018 255)"
  surface-active: "oklch(94% 0.009 250) / oklch(24.5% 0.02 255)"
# Ink: the neutral scale, identical hue family to the grounds
  paper: "oklch(21% 0.02 250) / oklch(93% 0.006 90)"
  paper-secondary: "oklch(34% 0.014 250) / oklch(71% 0.01 240)"
  paper-tertiary: "oklch(43% 0.013 250) / oklch(60% 0.012 240)"
  paper-muted: "oklch(48% 0.012 250) / oklch(56% 0.014 240)"
# Accent: the single blue. Interaction AND the forecast band. Nothing else may use it.
  accent: "oklch(47% 0.13 250) / oklch(63% 0.14 250)"
  accent-hover: "oklch(43% 0.12 250) / oklch(68% 0.14 250)"
  accent-active: "oklch(51% 0.12 250) / oklch(57% 0.13 250)"
  accent-solid: "oklch(44% 0.12 250) / oklch(50% 0.12 250)"
  accent-soft: "oklch(92% 0.03 250) / oklch(24% 0.05 250)"
  accent-line: "oklch(72% 0.06 250) / oklch(45% 0.1 250)"
  on-accent: "oklch(98% 0.004 90) / oklch(97% 0.005 90)"
# Data: deltas only. Desaturated, never neon.
  up: "oklch(42% 0.1 155) / oklch(65% 0.12 155)"
  down: "oklch(42% 0.11 25) / oklch(64% 0.12 25)"
  up-subtle: "oklch(93% 0.03 155) / oklch(19% 0.04 155)"
  down-subtle: "oklch(93% 0.03 25) / oklch(19% 0.04 25)"
# Lines: hairline by default; elevation is a border, never a shadow
  border: "oklch(88% 0.01 250) / oklch(26% 0.02 255)"
  border-light: "oklch(92% 0.008 250) / oklch(22% 0.018 255)"
  border-accent: "var(--accent-line)"
  grid: "oklch(89% 0.01 250) / oklch(24% 0.015 255)"
  divider: "oklch(91% 0.01 250) / oklch(23% 0.018 255)"
radii:
  xs: 4px
  sm: 6px
  md: 8px
type:
  display: "Inter 600 / 40px / 1.1 / -0.03em"
  headline: "Inter 600 / 28px / 1.2 / -0.02em"
  title: "Inter 600 / 20px / 1.3 / -0.01em"
  body: "Inter 400 / 15px / 1.6 / -0.01em"
  col-head: "Inter 600 / 10px / 1.2 / +0.08em, uppercase"
  data-lg: "JetBrains Mono 600 / 20px / 1.2 / tabular-nums"
  data-sm: "JetBrains Mono 500 / 12px / 1.2 / tabular-nums"
motion:
  duration-fast: 150ms
  duration-normal: 200ms
  rule: "No motion loops, no decorative entrance animation. Interaction states
    transition at 200ms; the forecast band may cross-fade. MotionConfig reducedMotion
    wraps the app; the chart honors prefers-reduced-motion."
spacing:
  page: "max-w-6xl, px-6, py-10"
  grid-gap: 16px
  panel-padding: "px-4 py-2.5 header / p-4 body"
components:
  header: "sticky top-0, bg-ground/95, border-b divider, z-sticky. Mark: hand-drawn
    bar-chart glyph in a bordered square. Nav: Market / Portfolio / Accuracy with a
    layoutId underline on the active route. Theme toggle (sun/moon glyphs) right of
    auth; hydration-safe mounted pattern."
  search: "Recessed input on stock, hairline border, accent on focus. Listbox of
    results with a11y (aria-expanded, aria-activedescendant, role=listbox); no custom
    dropdown chrome beyond the border."
  widget-block: "bg-stock + border-border + radius-sm; hover raises border to
    border-accent and bg to surface. All panels use it."
  ledger-row: "row of a ledger table; hover bg-surface."
  stripe-row: "alternate-row striping for compare tables; hover bg-surface."
  filter-chip: "bordered chip, uppercase 10-11px label; active state = accent-soft
    bg + accent border + accent text."
  btn-primary: "bg-accent, text-on-accent; hover accent-hover; no border, no shadow."
  btn-secondary: "transparent, hairline border, paper text; hover surface bg."
  btn-ghost: "text-only; hover paper."
  btn-danger: "border down-subtle, text down; reserved for destructive actions."
  price-chart: "Canvas. Paper primary line 1.5px; dashed tertiary for secondary
    sources; horizontal $ hairlines with mono labels; x-axis date labels on
    non-compact; forecast = translucent accent band (14% fill, 45% edge hairlines)
    with a 2px accent q50 line and endpoint dot; hover = hairline crosshair +
    tooltip card with mono numerals. Chart colors are read from computed CSS
    variables at draw time, so both themes stay correct."
  mini-price-chart: "Compact 68px PriceChart under the lead featured card; header
    row names the history span and flags the 7d forecast in accent."
  stat-card: "col-head label, data-lg value, optional delta pill or annotation."
  item-card: "widget-block; type eyebrow, name, recessed image well (aspect-square,
    object-contain), data-lg price + delta pill; optional footer slot."
  delta-pill: "font-data 11px semibold; bg up-subtle/down-subtle, text up/down;
    sign always shown. Green/red appear nowhere else."
  table: "sticky recessed header row of col-heads; hairline dividers; sort arrows in
    paper-tertiary; empty state with ∅ and a btn-secondary reset."
  verdict-strip: "Home accuracy strip: four cells (verdict with t, hit rate vs
    chance with pp excess, conformal band coverage, cohort with horizon/dates/eval date)
    plus a 'Full Backtest' cell; verdict tone = up/down/neutral only, and
    `no_skill` is deliberately neutral — a completed null, not a failure."
states:
  focus: "outline 2px accent, offset 2px"
  hover: "border-accent + surface bg, 200ms"
  disabled: "opacity-30, cursor-not-allowed"
  loading: "bg-recess pulse skeletons at component shape"
  empty: "∅ glyph + one-line reason + recovery action"
  error: "down-subtle banner with Retry"
icons: "Hand-drawn inline SVGs, 1.5px stroke, round caps; glyph set only — no icon
  library. Header mark is the only filled/branded glyph."
copy-tone: "Market ledger, not dive log. No metaphor vocabulary; the model's claims
  carry their measurement. Deltas always signed; forecasts always published as a
  split-conformal band around the q50 point forecast, with its horizon named. There are
  no q10/q90 models — `forecaster.py` trains `QUANTILES = [0.5]` and the band comes from
  `models/conformal.py`."
