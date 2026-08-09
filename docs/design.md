<!-- THE DEEP DIVE: visual world committed with the user on 2026-08-09 via the impeccable new-work workshop. Dealt as a challenger by concept-seed (key fa9b3b81) and chosen on decision board 888dd74f over the assigned Nautical Almanac direction and the hand (Ticket Wallet, Step Row). The world is decided; no implementation of it exists yet. The frontmatter tokens below are normative for the rebuild — mirror them into frontend/app/globals.css when the rebuild lands, then re-run $impeccable document to capture the implemented system.

THESIS: The market is a body of water and CS2 Oracle is the dive you take in it. History is depth; the forecast is the thermocline — the one cyan band where the water stops behaving. It refuses the category default of the ticker dashboard and its predictable opposite, the sterile terminal.

OWN-WORLD: Abyssal ink-blue grounds that deepen by dive stage; chalk-white text and chalk interaction plates; one thermocline cyan band reserved for forecast marks; reef moss and vent ember for deltas; marine-snow hairline grids; JetBrains Mono dive-computer digits on Inter quiet-grotesk notes; one vertical sounding rule ruling every line of copy.

STORY: The visitor understands a forecast as a measured band in the water, not a promise; believes the error record because it is published beside every prediction; and descends from the surface search, through the reef of the catalog, to the mesophotic wall of the item — then surfaces with a decision.

FIRST VIEWPORT: The surface zone: the buoy header, a sounding search, and the headline dive-log entry — the live backtest figure with its cohort and horizon named — over featured reef cards whose thermocline bands mark where their forecasts begin.

FORM: The Deep Dive challenger, dealt by concept-seed fa9b3b81 (roll 2 of 6) and fused with the grounded Tide Tables material; the thermocline cyan band is the forecast corridor, the vertical depth axis rules every surface. Seed key fa9b3b81.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, and DESIGN.md -->
---
name: CS2 Oracle
description: The dive log of the CS2 market — thirteen years of prices, measured at depth.
colors:
  ground: "oklch(11.5% 0.03 250)"
  stock: "oklch(17% 0.035 250)"
  recess: "oklch(8.5% 0.025 250)"
  surface: "oklch(23.5% 0.04 250)"
  surface-hover: "oklch(26.5% 0.045 250)"
  surface-active: "oklch(29.5% 0.05 250)"
  paper: "oklch(94% 0.008 90)"
  paper-secondary: "oklch(74% 0.02 230)"
  paper-tertiary: "oklch(66% 0.02 230)"
  paper-muted: "oklch(58% 0.025 230)"
  ink: "oklch(92% 0.01 90)"
  ink-hover: "oklch(97% 0.008 90)"
  ink-active: "oklch(86% 0.012 90)"
  ink-light: "oklch(78% 0.015 90)"
  ink-subtle: "oklch(29% 0.035 230)"
  thermocline: "oklch(74% 0.1 205)"
  thermocline-subtle: "oklch(32% 0.07 205)"
  up: "oklch(66% 0.1 150)"
  down: "oklch(66% 0.1 25)"
  up-subtle: "oklch(24% 0.03 150)"
  down-subtle: "oklch(24% 0.03 25)"
  border: "oklch(33% 0.04 250)"
  border-light: "oklch(28% 0.035 250)"
  border-accent: "oklch(64% 0.03 230)"
  grid: "oklch(31% 0.035 250)"
  divider: "oklch(27% 0.03 250)"
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
  tag-forecast:
    backgroundColor: "transparent"
    textColor: "{colors.thermocline}"
    typography: "{typography.label}"
    rounded: "{rounded.xs}"
    padding: "2px 6px"
  nav-link:
    textColor: "{colors.paper-secondary}"
    typography: "{typography.label}"
  card-reef:
    backgroundColor: "{colors.stock}"
    typography: "{typography.body}"
    rounded: "{rounded.sm}"
---

# Design System: CS2 Oracle

## Overview

**Creative North Star: "The Deep Dive"**

CS2 Oracle is the dive you take into the CS2 market. Thirteen years of daily prices are the water column; every item is a dive site reached on its own profile — the skin image as what the lamp first finds at the reef, price history as the descent itself, and the forecast as the thermocline: the one cyan band in the water where certainty changes and the dive computer's predictions take over. The product's promise — published, measured forecast accuracy — is the dive log: every prediction enters the log with the error it made, so the instrument is honest about the water at every depth.

The world replaces the "Specimen Archive" (warm umber museum casework). What survives is what the dive preserves: soft dark as the primary viewing condition, calm precision, asset-grounded skin imagery, and uncertainty shown as ranges, never as a confident single number. What is banned is anything that would turn a dive into an arcade — neon, bloom, trading-exuberance green/red, gambling energy, and decorative motion. The water is cool where the archive was warm: abyssal ink-blue grounds that deepen by dive stage, chalk-white text and interaction (the diver's slate), and one thermocline cyan band reserved exclusively for forecast marks.

**Key Characteristics:**
- One item, one dive: the plate leads, the profile follows, the log records.
- Abyssal ink-blue grounds (hue 250, low chroma) that deepen by dive stage; chalk-white paper (hue 90).
- Chalk is the only interaction color; thermocline cyan belongs to the forecast alone.
- The chart is a dive profile: history descends into depth, the q10–q90 corridor is the thermocline band with the q50 inked through it.
- Absence is depth: an item without a forecast reads "below working depth — no signal at this depth", never a hidden or zero-filled line.
- Motion is the drift of marine snow and the settling of a dive — nothing else moves.

## Colors

Cool, low-chroma abyssal water in which chalk-white slate and ink-blue depths replace the warm casework. Three color roles carry meaning, each governed by a scarcity law: **chalk** for interaction, **thermocline cyan** for forecast marks, **reef moss/vent ember** for price deltas. The frontmatter holds the normative dark values; the light theme below is the derived secondary.

### Primary (interaction)
- **Chalk** (`oklch(92% 0.01 90)` dark): the only interaction accent — primary buttons, active nav, focus rings, links, selected states. The diver's chalk mark on the dark slate: bright enough to write with at depth, calm enough to never shout.

- **Ink-hover / Ink-active** (`oklch(97% 0.008 90)` / `oklch(86% 0.012 90)`): hover brightens, press deepens.
### Secondary (data semantics — never interaction)
- **Thermocline Cyan** (`oklch(74% 0.1 205)` dark, `oklch(50% 0.09 205)` light): forecast marks only — the dive-log entry, the q10–q90 corridor, forecast horizons. Nothing else in the interface wears cyan. Its subtle tint (`oklch(32% 0.07 205)`) is the band fill.
- **Reef Moss Up / Vent Ember Down** (`oklch(66% 0.1 150)` / `oklch(66% 0.1 25)` dark): price deltas. Muted water colors, not trading-lite green/red. Subtle tints (`oklch(24% 0.03 …)`) tint badge and band backgrounds.

### Neutral
- **Abyssal Ground / Stock / Recess** (`oklch(11.5% 0.03 250)` / `oklch(17% 0.035 250)` / `oklch(8.5% 0.025 250)`): the deep water, the reef ledge, the wall shadow. The blue-ink undertone keeps the dark cool and alive, never "dead gray".
- **Surface family** (`oklch(23.5% 0.04 250)` → `29.5%`): interactive surfaces and hovered rows, lit by the lamp as they step up from stock.
- **Chalk Paper** (`oklch(94% 0.008 90)`): primary text. **Paper-secondary/tertiary/muted** (`74%` / `66%` / `58%`): descriptions, labels, placeholders — all ≥4.5:1 on the surfaces they sit on (AA).
- **Border / Border-light / Border-accent / Grid / Divider** (`oklch(33% 0.04 250)` / `28%` / `64% 0.03 230` / `31%` / `27%`): card rules, chart hairlines, and the chalk-tinted hover emphasis.

### Light theme (derived, secondary)
Surface daylight, not paper white: the same dive seen from the surface. Ground `oklch(97% 0.015 220)`, stock `oklch(94% 0.02 220)`, recess `oklch(91% 0.02 220)`, surface `oklch(99% 0.01 220)`; text ink `oklch(18% 0.03 250)`, secondary `38%`, tertiary `48%`, muted `55%`; ink (interaction) `oklch(32% 0.04 250)` (hover `27%`, active `36%`); thermocline `oklch(50% 0.09 205)`; up `oklch(44% 0.1 150)`, down `oklch(44% 0.1 25)`; border `oklch(88% 0.02 220)`, border-accent `oklch(70% 0.04 230)`. Dark remains the primary theme; the light set exists to be complete, not preferred.

### Named Rules
**The Thermocline Rule.** Cyan is reserved for the forecast — the dive-log entry, the q10–q90 corridor, horizon labels. A forecastless screen contains no cyan at all.
**The Chalk Rule.** Chalk is the interaction color and nothing else: buttons, nav, focus, links, selection. Data never borrows it, and it never appears in a chart.
**The Two-Way Rule.** Up and down are always two colors — reef moss and vent ember — used together. Never a single green-for-good or red-for-bad; the water reports change, not judgment.

## Typography

**Display Font:** Inter (with system-ui, sans-serif fallback)
**Body Font:** Inter (with system-ui, sans-serif fallback)
**Label/Mono Font:** JetBrains Mono (with monospace fallback)

**Character:** Inter carries the dive's quiet notes — the observer's desk hand, calm at every weight. JetBrains Mono is the dive computer: every number, price, tick, horizon, and depth in tabular monospace, read off the instrument's digits. Nothing decorative, nothing display; hierarchy comes from weight and size alone.

### Hierarchy
- **Display** (700, 48px, -0.03em): the surface placard — hero headings on the home entrance only.
- **Headline** (700, 32px, -0.025em): page titles.
- **Title** (600, 22px, -0.02em): section headings, card titles.
- **Body** (400, 15px, 1.6, -0.01em): descriptions, primary text. Max line length 65–70ch.
- **Label** (500, 13px): form labels, table body text.
- **Data** (JetBrains Mono, 400, 14px, -0.02em, tabular-nums): prices, numbers, instrument figures. Large values (20px, 500) for stat readouts; compact (12px) inside tables.
- **Dive tag** (JetBrains Mono, 600, 10px, +0.15em, uppercase): micro labels, captions, nav — the gauge plates on the instrument face.

### Named Rules
**The Ledger Rule.** Every numeral in the product — price, delta, horizon, depth, accuracy figure, date — is set in mono with tabular-nums via the `.font-data` class. If it is a number, it reads off the dive computer.
**The One-Voice Rule.** Two faces, always: Inter for notes, JetBrains Mono for digits. Introducing a third face is a design decision that requires a reason no existing face could satisfy.

## Layout

A 4px base grid; the water column. Container max-width 1200px with 24px side padding; 24px minimum between major sections, 32–40px between content blocks. Tables keep 44px rows with generous cell padding (th `px-5 py-4`, td `px-5 py-3`). Numbers right-align, text left-aligns.

**The Sounding Rule.** Every primary card and every chart carries a vertical depth rule on its left edge — a 2px chalk hairline with tick marks and depth labels (days back). It is the single composition device of the system: the eye descends the page the way the dive descends the water.

### Reading order (the cross-check)
1. **Plate** — the skin image. The "what." What the lamp first finds.
2. **Profile / price** — the "value." The descent itself, prominent and clearly drawn.
3. **Log metadata** — wear, volume, signals. Subordinate instrument readouts.
4. **Chrome** — the buoy and casework frame. Present, never competing.

### Page patterns
- **Home (`/`)** — the surface zone: placard hero (product statement + the live backtest figure as the headline dive-log entry, cohort and horizon named), featured reef cards, a holdings summary. No hero-metric template; the product's accuracy IS the surface reading.
- **Market (`/market`)** — the reef ledge: search + type filter above, reef grid (3-col), the dive-log table below (sortable, paginated, ≤7 columns, sticky header).
- **Item detail (`/items/[id]`)** — the mesophotic wall: primary dive card (2/3) holding the profile chart and wear tray; sidebar (1/3) holding log stats, signals, and the dive-log entry. No nested widget stacks.
- **Portfolio (`/portfolio`)** — the ascent stops: summary row of three log stats above a full-width holdings table; Steam CTA when unauthenticated.

### Named Rules
**The Working-Depth Rule.** An item with no forecast, or a surface with no data, shows its absence honestly: a "below working depth — no signal at this depth" reading, or a labeled degraded state. Never a hidden slot, a zero-filled chart, or a plausible-looking number.

## Elevation & Depth

Depth is tonal, not shaded: the water column steps by dive stage — ground, stock, recess — and the chart grid is a marine-snow hairline, not a shadow. Shadows exist in exactly two places: mounted reef cards (cards lift with a soft `0 4px 16px` under `0 1px 3px` at rest) and the buoy header's separation. Everything else is flat; depth comes from paper-on-water contrast, never from drop-shadow decoration.

### Named Rules
**The Buoyancy Rule.** Flat by default. A shadow appears only on a mounted card or the sticky header — the two things that physically stand off the water column.
**The No-Bloom Rule.** No bioluminescent glow, no bloom, no neon halos, in either theme — except the thermocline band itself and the delta tints, which carry data. The dive is lit by the lamp and the instrument, not a neon sign.

## Shapes

The bezel discipline: corners are `2px` (tags, badges, micro indicators), `4px` (cards, inputs, buttons, table rows), or `6px` (modals, larger containers) — nothing rounder. Radius reads as the machined edge of an instrument bezel, never a pillow. The world's signature geometry is the **sounding rule**: a 2px chalk vertical hairline with tick marks at the left edge of the primary card — the depth axis that rules every line of copy. Tags and badges stay 2px rectangles; pills are reserved for scrollbar thumbs only.

## Components

### Buttons
- **Shape:** 4px (`--radius-sm`), uppercase labels (11px/600, +0.12em).
- **Primary:** the chalk plate (`bg-ink`) with the deepest abyssal tone as text (`text-recess`) — chalk mark on dark slate. Hover: `ink-hover`; active: pressed (`ink-active`, `scale-[0.97]`).
- **Secondary:** transparent, 1px `border`, paper-secondary text; hover: `border-accent` + paper.
- **Ghost:** transparent, paper-secondary; hover: `surface` background.
- **Danger:** ember tint (`down-subtle`) with ember text and `down/30` border.

### Search (the sounding line)
- **Style:** full-width input on `stock` with 1px `border`, 4px radius; left sounding icon in paper-muted → paper on focus.
- **Focus:** border → `border-accent` (chalk-tinted), background → `surface`. The tag-chip inside the field is a dive tag.
- **States:** keyboard-first (Esc closes, arrows move, Enter opens), listbox semantics, height-capped dropdown.

### Log Stats (StatCard)
- **Style:** widget-block on stock; top caption as a dive tag (10px mono uppercase, paper-tertiary); center value in data-lg mono (20px/500); bottom annotation in 10px paper-muted.
- No hover animation — an instrument readout does not perform.

### Reef Cards (ItemCard)
- **Style:** widget-block, `overflow-hidden`, 4px; the primary card of a page carries the sounding rule (2px chalk vertical rule with tick marks).
- **Composition:** gauge caption (mono micro) + name (13px/600) above the image frame (aspect-square on `recess`); below, data-lg price in mono with a moss/ember change badge.
- **Hover:** border → `border-accent`, image scales 103%, and the card's lamp sweep passes once. No glow, no fade-ins.

### Dive Log Tables
- **Style:** full-width on stock, 1px border, 4px radius, `recess` header row with mono uppercase captions; rows divided by 1px `divider`, hover → `surface`.
- **Cells:** numerals in mono, right-aligned; sortable headers show direction arrows; change badges are 2px pills on `up-subtle`/`down-subtle`.
- **States:** skeleton plate rows while loading; centered empty state with a clear action; ember-tinted banner on error.

### Dive Profile Charts (the signature)
- **Composition:** the price history drawn as a descent on a marine-snow hairline grid (no vertical lines), with the sounding rule at the left edge carrying depth ticks (days back). History line in chalk (1.5px); the q10–q90 corridor is the **thermocline band** — a translucent cyan stratum (`thermocline-subtle`) with the q50 inked through it (2px); multi-source series draw as separate profiles in paper-tertiary. The forecast region is the mesophotic zone: the cyan band begins where the water changes, and nothing else on the chart is cyan.
- **Tooltip:** card-stock panel, 1px border, mono numerals.
- **Time ranges:** dive-stage tabs (24h / 7d / 30d / All), active in chalk. Selecting a range re-dives the profile: the deeper the range, the darker the water behind the chart.
- **Motion:** lines draw statically — data accuracy over decoration. Hover crosshair only.

### Wear Tray (quality selector)
- **Style:** horizontal pill group on the dive card; each pill: 1px border, 2px radius, label caps + data-sm price in mono.
- **States:** active = `ink-subtle` background + `border-accent`; hover = `surface`.

### Dive Log Entry (forecast summary — signature)
- **Style:** a log line beside the profile chart: thermocline-cyan label, data-lg q50 with "q10–q90" range at each horizon (3/7/14/30d), and the measured accuracy line naming its cohort and horizon. No forecast on record → the working-depth state ("below working depth — no signal at this depth").

### Header (the buoy)
- **Style:** sticky, `ground/95` + blur, 1px divider beneath; the CS2 Oracle mark left, nav (MARKET, PORTFOLIO) in mono dive tags, active = chalk + bottom rule; right: theme toggle, auth or avatar.
- **States:** loading skeleton, authenticated avatar + logout, unauthenticated AUTHENTICATE button.

### Motion & State
150–250ms transitions; the world's only allowed motions are the drift of marine snow on the home surface (the slow particle fall, 60s+ loop), the card's lamp sweep on hover, the sonar status pulse, and card hover border/background shifts. Focus rings: 2px chalk, 2px offset, instant. `prefers-reduced-motion: reduce` makes all transitions instant, stops the drift and the bubbles, and removes the lamp sweep. No orchestrated page loads, no chart line animation, no scroll-triggered decoration.

## Do's and Don'ts

### Do:
- **Do** set every number in mono with tabular-nums — the dive computer never lies in a proportional face.
- **Do** keep cyan exclusively on forecast marks; a screen without a forecast is a screen without cyan.
- **Do** use chalk for interaction and moss/ember (always paired) for deltas — the chalk never enters the chart, the data never borrows the button color.
- **Do** keep corners to 2/4/6px and tag plates rectangular — radius is a bezel, not a pillow.
- **Do** show absence honestly: working-depth readings, labeled degraded states, no fabricated figures. Published accuracy is the product — every accuracy number shown must name its cohort and horizon.
- **Do** let the plate (skin image) anchor every item view; the tool never abstracts into pure numbers.

### Don't:
- **Don't** use bloom, glow, neon, or gradients that don't carry data meaning — no bioluminescence outside the thermocline band and delta tints.
- **Don't** use trading-platform green/red exuberance, candles, or "stonks" chrome — the water reports in moss and ember.
- **Don't** gamify: no confetti, no hype energy, no unboxing drama. The instrument logs; it does not celebrate.
- **Don't** add a display font or a third face; two voices are the whole chorus.
- **Don't** put shadow and border on the same element as decoration; pick one.
- **Don't** add decorative motion, orchestrated page loads, or hover effects that perform instead of inform.
- **Don't** ship identical card grids; every card type in this system has a distinct anatomy.
