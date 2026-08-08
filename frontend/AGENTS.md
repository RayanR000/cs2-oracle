<!-- BEGIN:nextjs-agent-rules -->
# This is NOT the Next.js you know

Next 16.2 with React 19.2 — APIs, conventions, and file structure may all differ from
your training data. Read the relevant guide in `node_modules/next/dist/docs/` (`01-app/`
for the App Router) before writing any code. Heed deprecation notices.
<!-- END:nextjs-agent-rules -->

<!-- BEGIN:design-context -->
# Design Context

Full spec — type scale, spacing, component states — is `docs/design.md`. Token values are
`app/globals.css`. Both are the source of truth; the notes below are only what is easy to
get wrong.

## Product Register
Design SERVES the tool. The interface should disappear into the analysis task.
Analytical, precise, calm. Every element clarifies signal from noise.

- **Clarity over Density** — whitespace and hierarchy guide the eye
- **Asset-Grounded Data** — high-res skin images anchor every analysis
- **Precision Tools** — every component behaves exactly once, predictably
- **Rich but Not Loud** — color only where it signals data or state

## Theming
- **There are two themes.** `:root` holds light, `[data-theme="dark"]` holds dark.
  `app/layout.tsx` sets `data-theme="dark"` on `<html>` and an inline script overrides it
  from `localStorage.theme`. **A new token must be defined in both blocks** or it silently
  falls back to the light value in dark mode.
- Accent is derived, not hardcoded: `--brand-hue: 190` and `--brand-chroma: 0.12` on
  `:root` feed `--brand`, which is `oklch(50% …)` in light and `oklch(62% …)` in dark.
  `--accent-primary` aliases `--brand`; the Tailwind `accent` color maps to it.
- Data colors flip lightness by theme (`--data-up` is `oklch(50% 0.12 155)` light,
  `oklch(62% 0.14 155)` dark). Never write a data color literal.

## Typography
- Inter (`--font-sans`) for UI, JetBrains Mono (`--font-mono`) for data. No display fonts.
- Body is 15px; the rest of the scale is the table in `docs/design.md`.
- Numeric content uses the `.font-data` class — it sets the mono family and `tabular-nums`
  together. Don't reach for `font-variant-numeric` directly.

## Styling
- Use the CSS custom properties from `app/globals.css` — never inline hex or raw `oklch()`.
- Widgets use the existing `.widget-block` class (bg-secondary + border + `--radius-sm`,
  hover → `--border-accent` / `--surface`), not a re-derived set of utilities.
- Radii are only `--radius-xs` 2px / `--radius-sm` 4px / `--radius-md` 6px.
- Primary button: `bg-accent text-background-primary`. No border + shadow on one element.
- No neon, no tactical gaming cliches, no decorative motion on product pages.
<!-- END:design-context -->

<!-- BEGIN:api-server -->
# API Server

FastAPI on port 8000. Start from `backend/`: `venv/bin/uvicorn main:app --port 8000`.

The route list changes often — read `backend/api/routes/` rather than trusting a list
here. The routers are `items`, `opportunities`, `events`, `auth`, `portfolio`, `market`,
`accuracy`, `ab_test`, each mounted at its own prefix, plus `GET /health` on the app
itself. Note that per-item detail lives **under** `/items/{item_id}/` — `price-history`,
`trends`, `prediction`, `prices`, `variants`, `events`, `event-impacts`,
`feature-importance`, `social-sentiment` — not at the top level.

`lib/api.ts` is the only place the frontend talks to the API; it reads
`NEXT_PUBLIC_API_URL` (default `http://localhost:8000`). Adding a backend route means
adding it here in the same change.

When adding migrations or columns, ensure the SQLAlchemy models match the production schema.
<!-- END:api-server -->
