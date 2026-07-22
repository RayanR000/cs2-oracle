# CS2 Oracle — Design System

## Philosophy

**Analytical Instrument.** Not a dashboard, not a terminal — an instrument. Like a precision measuring tool, every element exists to clarify signal from noise. The interface recedes; the data speaks. Carbon surfaces provide deep, restful contrast for extended sessions. Amber accents punctuate only where attention is needed — a selection, a change, a call to action. Restraint is the brand.

## Color Strategy

**Restrained Carbon with Tinted Surface.** A near-monochrome dark palette where the neutral surfaces carry a subtle warm chroma (hue 30) at near-zero saturation. This prevents the "dead gray" feel of pure grayscale while keeping focus on the data. A single accent — warm amber — is used sparingly for primary actions, selection states, and meaningful emphasis.

### Palette (OKLCH)

**Backgrounds:**
- **Primary:** `oklch(18% 0.004 30)` — Soft carbon, main canvas. Not pitch black — restful without harshness.
- **Secondary:** `oklch(22% 0.005 30)` — Elevated panel. Visible separation from canvas.
- **Tertiary:** `oklch(15% 0.003 30)` — Deepest layer. Table headers, asset frames, recessed areas.

**Surfaces:**
- **Surface:** `oklch(26% 0.005 30)` — Interactive surfaces. Cards, inputs, clickable areas. Clearly visible against background.
- **Surface-hover:** `oklch(30% 0.006 30)` — Hovered state. Subtle lift.
- **Surface-active:** `oklch(33% 0.007 30)` — Active/pressed. Clear feedback.

**Text:**
- **Primary:** `oklch(93% 0 0)` — Headings, body text, primary information. Slightly softened white for less harsh contrast.
- **Secondary:** `oklch(70% 0.004 30)` — Descriptions, meta info. Readable but subordinate.
- **Tertiary:** `oklch(52% 0.004 30)` — Labels, captions, secondary metadata.
- **Muted:** `oklch(38% 0.003 30)` — Placeholders, disabled states, decorative markers.
- **Accent:** `oklch(72% 0.14 55)` — Links, emphasis, primary data highlights. Warm amber.

**Structural:**
- **Border:** `oklch(28% 0.005 30)` — Default dividers, card borders, table rules.
- **Border-light:** `oklch(24% 0.004 30)` — Subtle inner dividers, recessed separators.
- **Border-accent:** `oklch(42% 0.012 30)` — Hover/focus emphasis borders.
- **Grid:** `oklch(22% 0.004 30)` — Chart grid lines, minor structural dividers.
- **Divider:** `oklch(24% 0.004 30)` — Row separators, section breaks.

**Data Indicators:**
- **Up:** `oklch(62% 0.14 155)` — Price increase, positive delta. Muted emerald.
- **Down:** `oklch(62% 0.12 25)` — Price decrease, negative delta. Muted rose.
- **Up-subtle:** `oklch(24% 0.04 155)` — Positive delta background tint.
- **Down-subtle:** `oklch(24% 0.04 25)` — Negative delta background tint.

**Brand Accent:**
- **Brand:** `oklch(62% 0.14 55)` — Primary action color. Warm amber at medium lightness.
- **Brand-hover:** `oklch(66% 0.15 55)` — Hovered accent. Slightly brighter.
- **Brand-active:** `oklch(56% 0.13 55)` — Pressed/active accent.
- **Brand-light:** `oklch(72% 0.1 55)` — Muted accent for secondary emphasis.
- **Brand-subtle:** `oklch(26% 0.04 30)` — Accent-tinted surface for selection states.

**Radii & Shadows:**
- **Radius-xs:** `2px` — Tags, badges, micro indicators.
- **Radius-sm:** `4px` — Cards, inputs, buttons, table rows. Crisp, not rounded.
- **Radius-md:** `6px` — Larger containers, modals, dropdowns.
- **Shadow-sm:** `0 1px 3px oklch(0% 0 0 / 0.25)` — Subtle lift.
- **Shadow-md:** `0 4px 16px oklch(0% 0 0 / 0.3)` — Card elevation.

**Theme:** Dark (Default). Soft carbon base at 18% lightness — dark enough to reduce eye strain, light enough to avoid the harsh "pitch black" feel. Warm tint (hue 30) prevents sterile gray.

## Typography

**Inter** for all UI, **JetBrains Mono** for all data. One sans family keeps the interface calm; the monospace sibling handles numeric alignment. No display fonts, no decorative type.

### Scale

| Role | Size | Weight | Tracking | Use |
|------|------|--------|----------|-----|
| Display | 48px | 700 | -0.03em | Hero headings only (home page) |
| H1 | 32px | 700 | -0.025em | Page titles |
| H2 | 22px | 600 | -0.02em | Section headings |
| H3 | 16px | 600 | -0.015em | Subsection headings, card titles |
| Body | 15px | 400 | -0.01em | Paragraphs, descriptions, primary text |
| Small | 13px | 500 | 0 | Form labels, table body text |
| Caption | 11px | 600 | 0.08em | Uppercase labels, metadata, tags |
| Micro | 10px | 600 | 0.15em | Uppercase micro labels, annotations |
| Data | 14px | 400 | -0.02em | Monospace (JetBrains), prices, numbers |
| Data-lg | 20px | 500 | -0.02em | Monospace, hero numbers, large prices |
| Data-sm | 12px | 400 | -0.015em | Monospace, compact data, table numbers |

- Headings use `text-wrap: balance` for even line lengths.
- Body text caps at 65–70ch line length.
- Tabular-nums on all numeric content for alignment.

## Spacing & Layout

- **Base unit:** 4px (8, 12, 16, 20, 24, 32, 40, 48, 64).
- **Container max-width:** 1200px. Side padding: 24px (px-6).
- **Breathing room:** 24px minimum between major sections.
- **Section gap:** 32–40px between content blocks.
- **Tables:** 44px row height. Th: `px-5 py-4`. Td: `px-5 py-3`. Generous but not bloated.
- **Alignment:** Data right-aligned, text left-aligned. Monospace for all numeric columns.

### Layout Patterns

**Home (`/`):** Full-width, centered. Hero with product statement, key stats, featured items, capabilities summary. No hero-metric template — the product IS the metric.

**Market (`/market`):** Top bar: search + type filter. Trending grid (3-col, compact). Data table below: sortable, paginated, 7 columns with clear hierarchy. Sticky table header.

**Item Detail (`/items/[id]`):** Two-panel — primary (2/3) for chart + quality selector, sidebar (1/3) for metrics + signals. Clean separation, no nested widgets.

**Portfolio (`/portfolio`):** Summary row (3 stat cards). Full-width inventory table. Steam login CTA when unauthenticated.

## Motion & State

150–250ms transitions. Motion conveys state, never decoration. No orchestrated page loads.

- **Hover transitions:** 200ms ease-out. Background + border color shifts on interactive surfaces.
- **Focus ring:** 2px solid accent with 2px offset. Instant appearance, no transition.
- **Page content:** 300ms ease-out, staggered children at 30ms intervals. No hero orchestration.
- **Data updates:** CountUpNumber for value changes. Duration proportional to delta magnitude.
- **Table loading:** Skeleton rows with subtle pulse. No full-page spinners over content.
- **Reduced motion:** `prefers-reduced-motion: reduce` → all transitions instant, no scroll triggers, no stagger.

## Components

### Header
- Sticky top, `z-sticky`. `bg-background-primary/95 backdrop-blur-sm`.
- Left: CS logo mark + "DATA TERMINAL" in caption weight.
- Center: Nav links — MARKET, PORTFOLIO. Uppercase, 11px, tracking-[0.15em]. Active state: text-accent + bottom border.
- Right: Theme toggle (sun/moon), auth button or user avatar.
- Bottom: 1px border divider.
- States: loading (pulse skeleton), authenticated (avatar + name + logout), unauthenticated (AUTHENTICATE button).

### Search
- Full-width input, `bg-background-secondary border border-border radius-sm`.
- Left: search icon SVG (text-muted → text-primary on focus).
- Right: "Terminal" tag-tech badge.
- Focus: border → border-accent, `bg-surface`, subtle inner shadow.
- Placeholder: uppercase, micro weight, text-muted.

### StatCard
- Widget-block container.
- Top: caption label (uppercase, 11px, text-tertiary).
- Center: data-lg value (20px, monospace, font-medium).
- Optional: change badge (data-up/data-down tint + colored text).
- Bottom: micro annotation (10px, text-muted).
- No hover animation — the card exists to display data, not perform.

### ItemCard
- Widget-block container, `overflow-hidden`.
- Top: type label (micro, text-muted) + name (small, font-semibold). Rarity indicator (colored dot or thin line).
- Center: asset image in aspect-square frame (`bg-background-tertiary`). Image scales 103% on hover.
- Bottom: data-lg price (monospace) + data change (data-up/data-down).
- Hover: border → border-accent. No glow, no annotation fade-in — just clear interaction feedback.

### Buttons
- Uppercase, caption weight (11px, 600), tracking-[0.12em].
- **Primary:** `bg-accent text-background-primary`. Hover: `bg-brand-hover`. Active: `scale-[0.97]`.
- **Secondary:** `bg-transparent border border-border text-secondary`. Hover: `border-accent text-primary`.
- **Danger:** `bg-data-down-subtle text-data-down border border-data-down/30`. Hover: `bg-data-down/15`.
- **Ghost:** `bg-transparent text-secondary`. Hover: `text-primary bg-surface`.

### Tables
- Full-width, `bg-background-secondary border border-border radius-sm overflow-hidden`.
- **Thead:** `bg-background-tertiary`. Caption weight (11px), text-secondary, uppercase.
- **Rows:** 1px bottom divider. Hover → `bg-surface`.
- **Sortable headers:** Click button, show sort direction arrow (↑/↓).
- **Data cells:** Monospace (font-data). Right-aligned for numbers, left-aligned for text.
- **Change badges:** Inline rounded pill. `bg-data-up-subtle text-data-up` or `bg-data-down-subtle text-data-down`.
- **States:** Loading — skeleton pulse rows. Empty — centered message + clear action. Error — red-tinted banner.

### Charts
- Recharts `LineChart` with `ResponsiveContainer`. 320px height (slightly shorter than current for less visual weight).
- Grid: 1px stroke, `--grid` color. No vertical grid lines.
- Steam line: `oklch(60% 0 0)` stroke, 1.5px. CSFloat line: `--brand` stroke, 1.5px.
- Tooltip: `bg-background-tertiary border border-border radius-sm`. Monospace data, white text.
- Time range tabs: 24h / 7d / 30d / All. Caption weight, active = accent color.
- No animation on chart lines — data accuracy over decoration.

### Quality Selector
- Horizontal pill group. Each pill: border, radius-sm, caption weight, data-sm price.
- Active pill: `bg-brand-subtle border-accent text-primary`.
- Hover: `bg-surface`.
- Compact — no icons, no decorative elements. Just wear name + price.

### Loading States
- **Full-page:** Centered spinner (32px, 2px border, accent color) + caption text.
- **Table:** Skeleton rows (3 per visible page, pulse animation).
- **Header avatar:** 8x8 rounded skeleton, bg-surface, pulse.

### Empty States
- Centered, max-width 400px. Tertiary heading + muted body text + ghost action button.
- No illustrations, no decorative elements.

### Error States
- Banner: `border-data-down/30 bg-data-down/8 radius-sm`. Text-primary. Dismissible.
- Full-page error: Back link + heading + description. Clean, no panic.

## Visual Hierarchy

1. **Asset image** — the "what." Largest visual element on item-focused views.
2. **Price/chart** — the "value." Prominent data, clear trend.
3. **Metadata** — the "context." Wear, volume, signals. Subordinate to the above.
4. **Navigation/chrome** — the "frame." Present but never competing with content.

Use elevation and whitespace to group, not color or borders. Color carries data meaning (up/down/accent), not decoration.

## Anti-patterns to avoid

- **Information Overload:** Max 7 columns in any table. Progressive disclosure for details.
- **Harsh Contrast:** No `#000` with `#fff`. Use tinted darks and neutral whites.
- **Generic Slop:** No shadcn defaults without custom OKLCH tokens.
- **Gamer Cliche:** No neon, no tactical styling, no aggressive gradients.
- **Decorative motion:** Motion conveys state, never performance.
- **Identical cards:** Every card type has a distinct purpose and layout.
- **Display fonts in UI:** One sans (Inter), one mono (JetBrains Mono). Nothing else.
- **Over-rounding:** Cards top out at 6px. Full-pill only for tags/badges.
- **Border + shadow on same element:** Pick one. Never both as decoration.
