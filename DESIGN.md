# CS2 Market Analyzer — Design System

## Philosophy

**Modern Analytical Boutique.** The precision of a high-end trading tool with the visual richness of a premium game showcase. Every data point is grounded by visual context. We prioritize breathing room and clarity over raw data density to ensure a "relaxed" analytical experience.

## Color Strategy

**Soft Charcoal & Navy:** A sophisticated dark theme designed for comfort and long sessions. We use depth and subtle tints rather than harsh blacks.

### Palette (OKLCH)

**Base:**
- **Background-primary:** `oklch(18% 0.02 260)` — Deep charcoal with a hint of navy.
- **Background-secondary:** `oklch(22% 0.02 260)` — Elevation for cards and panels.
- **Surface:** `oklch(26% 0.03 260)` — High-elevation surfaces, active states.
- **Text-primary:** `oklch(90% 0.01 260)` — Off-white for high readability without glare.
- **Text-secondary:** `oklch(70% 0.02 260)` — Muted gray-blue for secondary labels.
- **Border:** `oklch(35% 0.03 260)` — Subtle separators that define space without clutter.

**Accents:**
- **Gain:** `oklch(75% 0.15 150)` — Soft mint green for positive trends.
- **Loss:** `oklch(65% 0.15 20)` — Muted rose for negative trends.
- **Action:** `oklch(70% 0.12 245)` — Soft blue for interactive elements.

**Theme:** Dark (Default). Softer than pitch black to reduce eye strain.

## Typography

- **Font Stack:** 'Inter' for UI (modern, clean), 'JetBrains Mono' or 'IBM Plex Mono' for data (tabular, precise).
- **Headline:** Inter, 500-600 weight, tight tracking.
- **Body:** Inter, 15px, 1.6 line-height for comfort.
- **Data:** Monospace, 14px, for prices and trends.
- **Scale:**
  - H1: 32px, 600 weight (Hero headers)
  - H2: 24px, 500 weight (Section headers)
  - Body: 15px, 400 weight
  - Small: 12px, 500 weight (Labels/Secondary)

## Spacing & Layout

- **Base Unit:** 4px (Scaling to 8, 16, 24, 32, 48, 64).
- **Breathing Room:** Minimum 24px padding for main containers. Avoid "compact" modes.
- **Grid:** Fluid 12-column grid. Large gutters (24px) to prevent information crowding.
- **Alignment:** Data is right-aligned in tables; text is left-aligned.

## Components & Patterns

- **Skin Cards:** Large, high-quality asset displays. Use subtle background gradients or glows based on item rarity (e.g., Covert = subtle red tint).
- **Charts:** Clean, minimalist line charts. Use the Gain/Loss colors for the line. No heavy grids; just essential axes.
- **Search:** Prominent, centered, with large hit areas. 
- **Buttons:** Slightly rounded (8px radius). Flat with subtle hover elevation.
- **Tables:** Generous row height (48px+). Soft striping. No vertical borders.

## Visual Hierarchy

- **Primary:** High-res skin image (The "What").
- **Secondary:** Large price/trend chart (The "Value").
- **Tertiary:** Supporting metadata (Wear, Volume, etc.).
- **Focus:** Use elevation (Background-secondary) and whitespace rather than color to group elements.

## Anti-patterns to avoid

- **Information Overload:** Do not show 20 columns of data at once.
- **Harsh Contrast:** Avoid `#000` background with pure `#fff` text.
- **Generic Slop:** Avoid standard shadcn/tailwind defaults without custom OKLCH adjustments.
- **Gamer Cliche:** No aggressive "tactical" UI or neon glows. Keep it "Boutique."
