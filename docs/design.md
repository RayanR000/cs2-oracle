<!-- SPECIMEN ARCHIVE: visual world committed with the user on 2026-08-07 via the impeccable new-work workshop (seed 08df685d, roll index 4). The world is decided; no implementation of it exists yet. The frontmatter tokens below are normative for the rebuild — mirror them into frontend/app/globals.css when the rebuild lands, then re-run $impeccable document to capture the implemented system. -->

---
name: CS2 Oracle
description: The specimen archive of the CS2 market — thirteen years of prices, pinned and catalogued.
colors:
  ground: "oklch(16% 0.006 55)"
  stock: "oklch(19.5% 0.007 55)"
  recess: "oklch(13.5% 0.005 55)"
  surface: "oklch(23.5% 0.008 55)"
  surface-hover: "oklch(26.5% 0.009 55)"
  surface-active: "oklch(29.5% 0.01 55)"
  paper: "oklch(92% 0.008 85)"
  paper-secondary: "oklch(71% 0.012 70)"
  paper-tertiary: "oklch(64% 0.012 60)"
  paper-muted: "oklch(62% 0.01 55)"
  ink: "oklch(66% 0.1 250)"
  ink-hover: "oklch(70% 0.11 250)"
  ink-active: "oklch(60% 0.1 250)"
  ink-light: "oklch(74% 0.09 250)"
  ink-subtle: "oklch(26% 0.03 250)"
  specimen: "oklch(70% 0.12 75)"
  specimen-subtle: "oklch(24% 0.04 75)"
  up: "oklch(64% 0.11 150)"
  down: "oklch(64% 0.11 25)"
  up-subtle: "oklch(23% 0.03 150)"
  down-subtle: "oklch(23% 0.03 25)"
  border: "oklch(30% 0.008 55)"
  border-light: "oklch(25.5% 0.007 55)"
  border-accent: "oklch(46% 0.04 250)"
  grid: "oklch(27% 0.008 55)"
  divider: "oklch(25% 0.007 55)"
typography:
  display:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: "48px"
    fontWeight: 700
    letterSpacing: "-0.03em"
  headline:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: "32px"
    fontWeight: 700
    letterSpacing: "-0.025em"
  title:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: "22px"
    fontWeight: 600
    letterSpacing: "-0.02em"
  body:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.6
    letterSpacing: "-0.01em"
  label:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 500
  data:
    fontFamily: "JetBrains Mono, monospace"
    fontSize: "14px"
    fontWeight: 400
    letterSpacing: "-0.02em"
    fontFeature: "tnum"
rounded:
  xs: "2px"
  sm: "4px"
  md: "6px"
spacing:
  base: "4px"
  xs: "8px"
  sm: "12px"
  md: "16px"
  lg: "24px"
  xl: "32px"
  "2xl": "48px"
components:
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.recess}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "10px 20px"
  button-primary-hover:
    backgroundColor: "{colors.ink-hover}"
    textColor: "{colors.recess}"
    rounded: "{rounded.sm}"
    padding: "10px 20px"
  button-primary-active:
    backgroundColor: "{colors.ink-active}"
    textColor: "{colors.recess}"
    rounded: "{rounded.sm}"
    padding: "10px 20px"
  button-secondary:
    backgroundColor: "transparent"
    textColor: "{colors.paper-secondary}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "10px 20px"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.paper-secondary}"
    typography: "{typography.label}"
    padding: "8px 12px"
  input-search:
    backgroundColor: "{colors.stock}"
    textColor: "{colors.paper}"
    typography: "{typography.body}"
    rounded: "{rounded.sm}"
    height: "40px"
  chip-wear:
    backgroundColor: "{colors.ink-subtle}"
    textColor: "{colors.paper}"
    typography: "{typography.label}"
    rounded: "{rounded.xs}"
    padding: "6px 12px"
  tag-specimen:
    backgroundColor: "transparent"
    textColor: "{colors.specimen}"
    typography: "{typography.label}"
    rounded: "{rounded.xs}"
    padding: "2px 6px"
  nav-link:
    textColor: "{colors.paper-secondary}"
    typography: "{typography.label}"
  card-specimen:
    backgroundColor: "{colors.stock}"
    typography: "{typography.body}"
    rounded: "{rounded.sm}"
---

# Design System: CS2 Oracle

## Overview

**Creative North Star: "The Specimen Archive"**

CS2 Oracle is the working museum of the CS2 market. Thirteen years of daily prices are the collection; every item is a specimen pinned to its archive card — the skin image mounted as the specimen plate, price history drawn as strata across the card, and the forecast hanging from the frame as a curator's tag. The product's promise — published, measured forecast accuracy — is the archive's own cataloguing discipline: a collection that records its errors on the same ledger as its findings. The interface is the display case, the data is the exhibit, and honesty is the curator's first rule.

The world replaces the previous "analytical instrument" (carbon + teal terminal chrome). What survives is what the archive itself preserves: soft dark as the primary viewing condition, calm precision, asset-grounded skin imagery, and uncertainty shown as ranges, never as a confident single number. What is banned is anything that would turn a museum into an arcade — neon, glow, trading-exuberance green/red, gambling energy, and decorative motion. The archive is warm where a terminal is cold: bone paper text on umber-dark case stock, one ink-blue interaction color, and one amber reserved exclusively for forecast marks.

**Key Characteristics:**
- One item, one card: the specimen plate leads, the ledger follows.
- Warm dark case stock (hue 55, near-zero chroma), bone paper text (hue 85).
- Ink blue is the only interaction accent; amber belongs to the forecast alone.
- Price history as strata; the q10–q90 corridor as a translucent amber stratum with the q50 inked through it.
- Absence is information: an item without a forecast shows an unfilled ledger line, never a hidden or zero-filled one.
- Motion is the curator's lamp and the settling of a pinned specimen — nothing else moves.

## Colors

A warm, low-chroma dark archive in which bone paper and umber case stock replace the old carbon-and-teal. Three ink roles carry meaning, and each is governed by a scarcity law: **ink** for interaction, **specimen amber** for forecast marks, **moss/brick** for price deltas. The frontmatter holds the normative dark values; the light theme below is the derived secondary.

### Primary (interaction)
- **Archive Ink** (`oklch(66% 0.1 250)` dark): the only interaction accent — primary buttons, active nav, focus rings, links, selected states. Cool blue against warm stock reads as the archive's label ink.
- **Ink-hover / Ink-active** (`oklch(70% 0.11 250)` / `oklch(60% 0.1 250)`): hover brightens, press deepens.

### Secondary (data semantics — never interaction)
- **Specimen Amber** (`oklch(70% 0.12 75)` dark, `oklch(58% 0.11 75)` light): forecast marks only — the curator tag, the q10–q90 corridor, forecast horizons. Nothing else in the interface wears amber.
- **Moss Up / Brick Down** (`oklch(64% 0.11 150)` / `oklch(64% 0.11 25)` dark): price deltas. Muted archive colors, not trading-lite green/red. Subtle tints (`oklch(23% 0.03 …)`) tint badge and band backgrounds.

### Neutral
- **Case Ground / Stock / Recess** (`oklch(16% 0.006 55)` / `oklch(19.5% 0.007 55)` / `oklch(13.5% 0.005 55)`): the floor, the card stock, the recessed frame. The umber undertone keeps the dark warm, never "dead gray".
- **Surface family** (`oklch(23.5% 0.008 55)` → `29.5%`): interactive surfaces and hovered rows, stepping up from stock.
- **Bone Paper** (`oklch(92% 0.008 85)`): primary text. **Paper-secondary/tertiary/muted** (`71%` / `64%` / `62%`): descriptions, labels, placeholders — all ≥4.5:1 on the surfaces they sit on (AA).
- **Border / Border-light / Border-accent / Grid / Divider** (`oklch(30% 0.008 55)` / `25.5%` / `46% 0.04 250` / `27%` / `25%`): card rules, chart hairline grid, and the ink-tinted hover emphasis.

### Light theme (derived, secondary)
Gallery light, not paper white: ground `oklch(97.5% 0.008 80)`, stock `oklch(95% 0.009 80)`, recess `oklch(92% 0.01 80)`, surface `oklch(99.5% 0.006 80)`; text ink `oklch(19% 0.012 60)`, secondary `38%`, tertiary `50%`, muted `55%`; ink `oklch(48% 0.1 250)` (hover `43%`, active `38%`); specimen `oklch(58% 0.11 75)`; up `oklch(46% 0.1 150)`, down `oklch(46% 0.1 25)`; border `oklch(88% 0.012 80)`, border-accent `oklch(75% 0.04 250)`. Dark remains the primary theme; the light set exists to be complete, not preferred.

### Named Rules
**The Specimen Mark Rule.** Amber is reserved for the forecast — the curator tag, the q10–q90 corridor, horizon labels. A forecastless screen contains no amber at all.
**The Ink Rule.** Ink blue is the interaction color and nothing else: buttons, nav, focus, links, selection. Data never borrows it, and it never appears in a chart.
**The Two-Way Rule.** Up and down are always two colors — moss and brick — used together. Never a single green-for-good or red-for-bad; the archive reports change, not judgment.

## Typography

**Display Font:** Inter (with system-ui, sans-serif fallback)
**Body Font:** Inter (with system-ui, sans-serif fallback)
**Label/Mono Font:** JetBrains Mono (with monospace fallback)

**Character:** Inter carries the archive's quiet officiousness — the curator's desk hand, calm at every weight. JetBrains Mono is the ledger ink: every number, price, tick, and horizon in tabular monospace, as if typed into the accession book. Nothing decorative, nothing display; hierarchy comes from weight and size alone.

### Hierarchy
- **Display** (700, 48px, -0.03em): the museum placard — hero headings on the home entrance only.
- **Headline** (700, 32px, -0.025em): page titles.
- **Title** (600, 22px, -0.02em): section headings, card titles.
- **Body** (400, 15px, 1.6, -0.01em): descriptions, primary text. Max line length 65–70ch.
- **Label** (500, 13px): form labels, table body text.
- **Data** (JetBrains Mono, 400, 14px, -0.02em, tabular-nums): prices, numbers, ledger figures. Large values (20px, 500) for stat readouts; compact (12px) inside tables.
- **Specimen tag** (JetBrains Mono, 600, 10px, +0.15em, uppercase): micro labels, captions, nav — the archive's printed plate captions.

### Named Rules
**The Ledger Rule.** Every numeral in the product — price, delta, horizon, accuracy figure, date — is set in mono with tabular-nums via the `.font-data` class. If it is a number, it types itself like a ledger entry.
**The One-Voice Rule.** Two faces, always: Inter for prose, JetBrains Mono for data. Introducing a third face is a design decision that requires a reason no existing face could satisfy.

## Layout

A 4px base grid; the museum's casework. Container max-width 1200px with 24px side padding; 24px minimum between major sections, 32–40px between content blocks. Tables keep 44px rows with generous cell padding (th `px-5 py-4`, td `px-5 py-3`). Numbers right-align, text left-aligns.

### Reading order (the cross-check)
1. **Specimen plate** — the skin image. The "what."
2. **Strata chart / price** — the "value." Prominent, clearly drawn.
3. **Ledger metadata** — wear, volume, signals. Subordinate.
4. **Chrome** — the casework frame. Present, never competing.

### Page patterns
- **Home (`/`)** — the gallery entrance: placard hero (product statement + the live backtest figure as the collection's headline record, cohort and horizon named), featured specimens, a holdings summary. No hero-metric template; the product's accuracy IS the headline exhibit.
- **Market (`/market`)** — the catalog drawers: search + type filter above, specimen grid (3-col), the ledger table below (sortable, paginated, ≤7 columns, sticky header).
- **Item detail (`/items/[id]`)** — the specimen card: primary plate (2/3) holding the strata chart and wear tray; sidebar (1/3) holding ledger stats, signals, and the curator tag. No nested widget stacks.
- **Portfolio (`/portfolio`)** — your collection case: summary row of three ledger stats above a full-width inventory table; Steam CTA when unauthenticated.

### Named Rules
**The Empty Drawer Rule.** An item with no forecast, or a surface with no data, shows its absence honestly: an unfilled ledger line ("no curator note on record") or a labeled degraded state. Never a hidden slot, a zero-filled chart, or a plausible-looking number.

## Elevation & Depth

Depth is tonal, not shaded: the archive is flat casework where layers step by lightness — floor, stock, recess — and the chart grid is a hairline, not a shadow. Shadows exist in exactly two places: mounted specimens (cards lift with a soft `0 4px 16px` under `0 1px 3px` at rest) and the sticky header's separation. Everything else is flat; depth comes from paper-on-stock contrast, never from drop-shadow decoration.

### Named Rules
**The Case Rule.** Flat by default. A shadow appears only on a mounted card or the sticky header — the two things that physically stand off the case floor.
**The No-Glow Rule.** No glows, no bloom, no neon halos, in either theme. The archive is lit by the curator's lamp, not a neon sign.

## Shapes

The pin discipline: corners are `2px` (tags, badges, micro indicators), `4px` (cards, inputs, buttons, table rows), or `6px` (modals, larger containers) — nothing rounder. Radius reads as the crisp edge of a pinned label, never a pillow. The world's signature geometry is the **specimen pin**: a 2px ink-blue corner bracket at the top-left of the primary specimen card, like the museum label plate that identifies the exhibit. Tags and badges stay 2px rectangles; pills are reserved for scrollbar thumbs only.

## Components

### Buttons
- **Shape:** 4px (`--radius-sm`), uppercase labels (11px/600, +0.12em).
- **Primary:** the ink-blue plate (`bg-ink`) with the darkest ground tone as text (`text-recess`). Hover: `ink-hover`; active: pressed (`ink-active`, `scale-[0.97]`).
- **Secondary:** transparent, 1px `border`, paper-secondary text; hover: `border-accent` + paper.
- **Ghost:** transparent, paper-secondary; hover: `surface` background.
- **Danger:** brick tint (`down-subtle`) with brick text and `down/30` border.

### Search (the finding aid)
- **Style:** full-width input on `stock` with 1px `border`, 4px radius; left icon in paper-muted → paper on focus.
- **Focus:** border → `border-accent` (ink-tinted), background → `surface`. The tag-chip inside the field is a specimen tag.
- **States:** keyboard-first (Esc closes, arrows move, Enter opens), listbox semantics, height-capped dropdown.

### Ledger Stats (StatCard)
- **Style:** widget-block on stock; top caption as a specimen tag (10px mono uppercase, paper-tertiary); center value in data-lg mono (20px/500); bottom annotation in 10px paper-muted.
- No hover animation — a ledger entry does not perform.

### Specimen Cards (ItemCard)
- **Style:** widget-block, `overflow-hidden`, 4px; the primary card of a page carries the specimen pin (2px ink corner bracket).
- **Composition:** plate caption (mono micro) + name (13px/600) above the image frame (aspect-square on `recess`); below, data-lg price in mono with a moss/brick change badge.
- **Hover:** border → `border-accent`, image scales 103%. No glow, no fade-ins.

### Ledger Tables
- **Style:** full-width on stock, 1px border, 4px radius, `recess` header row with mono uppercase captions; rows divided by 1px `divider`, hover → `surface`.
- **Cells:** numerals in mono, right-aligned; sortable headers show direction arrows; change badges are 2px pills on `up-subtle`/`down-subtle`.
- **States:** skeleton plate rows while loading; centered empty state with a clear action; red-tinted banner on error.

### Strata Charts (the signature)
- **Composition:** the price history drawn as strata on a hairline grid (no vertical lines). History line in bone ink (1.5px); the q10–q90 corridor is a translucent **specimen-amber stratum** with the q50 inked through it (2px); multi-source series draw as separate strata in paper-tertiary. The forecast region is the amber stratum; nothing else on the chart is amber.
- **Tooltip:** card-stock panel, 1px border, mono numerals.
- **Time ranges:** specimen-tag tabs (24h / 7d / 30d / All), active in ink.
- **Motion:** lines draw statically — data accuracy over decoration. Hover crosshair only.

### Wear Tray (quality selector)
- **Style:** horizontal pill group on the specimen card; each pill: 1px border, 2px radius, label caps + data-sm price in mono.
- **States:** active = `ink-subtle` background + `border-accent`; hover = `surface`.

### Curator Tag (forecast summary — signature)
- **Style:** a hanging tag beside the strata chart: specimen-amber label, data-lg q50 with "q10–q90" range at each horizon (3/7/14/30d), and the measured accuracy line naming its cohort and horizon. No forecast on record → the empty-drawer state.

### Header
- **Style:** sticky, `ground/95` + blur, 1px divider beneath; the CS2 Oracle collection mark left, nav (MARKET, PORTFOLIO) in mono specimen tags, active = ink + bottom rule; right: theme toggle, auth or avatar.
- **States:** loading skeleton, authenticated avatar + logout, unauthenticated AUTHENTICATE button.

### Motion & State
150–250ms transitions; the world's only allowed motions are the curator's lamp sweep on the home exhibit (the slow vertical scan), the status pulse, and card hover border/background shifts. Focus rings: 2px ink, 2px offset, instant. `prefers-reduced-motion: reduce` makes all transitions instant and removes the lamp sweep. No orchestrated page loads, no chart line animation, no scroll-triggered decoration.

## Do's and Don'ts

### Do:
- **Do** set every number in mono with tabular-nums — the ledger never lies in a proportional face.
- **Do** keep amber exclusively on forecast marks; a screen without a forecast is a screen without amber.
- **Do** use ink blue for interaction and moss/brick (always paired) for deltas — the ink never enters the chart, the data never borrows the button color.
- **Do** keep corners to 2/4/6px and tag plates rectangular — radius is a pin, not a pillow.
- **Do** show absence honestly: unfilled ledger lines, labeled degraded states, no fabricated figures. Published accuracy is the product — every accuracy number shown must name its cohort and horizon.
- **Do** let the specimen plate (skin image) anchor every item view; the tool never abstracts into pure numbers.

### Don't:
- **Don't** use glow, bloom, neon, or gradients that don't carry data meaning.
- **Don't** use trading-platform green/red exuberance, candles, or "stonks" chrome — the archive reports in moss and brick.
- **Don't** gamify: no confetti, no hype energy, no unboxing drama. The archive catalogs; it does not celebrate.
- **Don't** add a display font or a third face; two voices are the whole chorus.
- **Don't** put shadow and border on the same element as decoration; pick one.
- **Don't** add decorative motion, orchestrated page loads, or hover effects that perform instead of inform.
- **Don't** ship identical card grids; every card type in this system has a distinct anatomy.
