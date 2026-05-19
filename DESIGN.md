# CS2 Market Analyzer — Design System

## Philosophy

**Analytical dashboard, not gaming platform.** Bloomberg Terminal meets modern data visualization. Focus: clarity, density, precision. Every element serves information hierarchy. Serious traders, professional appearance.

## Color Strategy

**Restrained:** Neutral grayscale + single accent. Minimal color, maximum data clarity.

### Palette

**Base:** Professional dark background, neutral grays.
- **Background-primary:** `#0f1419` — near-black, professional
- **Background-secondary:** `#131820` — minimal contrast, structural
- **Surface:** `#1a1f2e` — data tables, price panels
- **Text (primary):** `#d1d5db` — light gray (high contrast, readable)
- **Text (secondary):** `#6b7280` — muted gray (secondary info)
- **Border:** `#2d3748` — subtle structural dividers
- **Grid:** `#242d3a` — table striping

**Accent:** Minimal, structural use only.
- **Data-up:** `#10b981` — muted green (gains, volume, positive)
- **Data-down:** `#ef4444` — muted red (losses, negative)
- **Interactive:** `#3b82f6` — restrained blue (links, focus)

**Theme:** Dark mode (traders use 8+ hours, reduce eye strain, professional fintech standard).

## Typography

- **Font stack:** 'IBM Plex Mono' for data (professional, precise), Inter for UI (clean, modern)
- **Headline:** Inter, 600 weight, sentence case (no unnecessary caps)
- **Body:** Inter, 14px, 1.5 line-height, letter-spacing: normal
- **Data:** IBM Plex Mono or system monospace, 13px (prices, floats, numbers)
- **Scale:**
  - H1: 28px, 600 weight
  - H2: 20px, 600 weight
  - H3: 16px, 500 weight
  - Body: 14px, 400 weight
  - Label: 12px, 500 weight (color: text-secondary)
  - Data: 13px, monospace (prices, tickers)

## Spacing & Layout

- **Base unit:** 8px
- **Common intervals:** 8px, 16px, 24px, 32px
- **No unnecessary containers:** Tables flow naturally, minimal padding waste
- **Grid:** 12-column, 16px gaps on desktop; natural flow on mobile
- **Data density:** Tight spacing in tables (8px padding), loose spacing between sections (32px)

## Components & Patterns

- **Buttons:** Subtle. Blue accent, thin border, minimal padding. Hover: opacity 0.9, no scale.
- **Inputs:** 1px border (#2d3748), clean background, thin focus state.
- **Tables:** Minimal styling. Striped rows (#242d3a), mono font for prices, 1px borders only.
- **Charts:** Light grid lines (#2d3748), thin lines, muted accent for data series.
- **Status:** Green (#10b981) for gains/positive, red (#ef4444) for losses/negative. No icons; rely on color + number.
- **Price display:** Monospace, right-aligned, no decoration.
- **Links:** Blue (#3b82f6), underline on hover.

## Visual Hierarchy

- **Emphasis via size + weight:** Not color. Larger = more important.
- **Secondary info:** Gray text (#6b7280), smaller, monospace if numeric.
- **Critical data:** Bold, larger, primary color (#d1d5db).
- **Grouped info:** Whitespace separation, not borders.

## Responsive

- **Desktop:** Full data density, tables with all columns visible.
- **Tablet:** Essential columns (name, price, change, volume).
- **Mobile:** Compact, single column, simplified numbers.

## Accessibility

- **Contrast:** Text ≥4.5:1 on dark (meets WCAG AA)
- **No color alone:** Always pair with text or label
- **Focus:** Subtle blue outline, visible and professional
- **Motion:** Only on interaction (hover, focus), no auto-animation

## Anti-patterns to avoid

- No neon colors (breaks professional tone)
- No gradient text or decorative effects
- No oversized badges or icons
- No unnecessary visual depth or shadows
- No gaming aesthetics (glows, scales, enthusiastic animation)
