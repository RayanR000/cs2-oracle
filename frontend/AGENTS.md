<!-- BEGIN:nextjs-agent-rules -->
# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.
<!-- END:nextjs-agent-rules -->

<!-- BEGIN:design-context -->
# Design Context

## Product Register
**product** — design SERVES the tool. Familiarity is a feature; the interface should disappear into the analysis task.

## Brand Personality
Analytical, precise, calm. An analytical instrument — every element clarifies signal from noise.

## Key Principles
- **Clarity over Density** — whitespace and hierarchy guide the eye
- **Asset-Grounded Data** — high-res skin images anchor every analysis
- **Comfortable Analysis** — tinted carbon dark mode, warm amber accents, restful for extended sessions
- **Precision Tools** — every component behaves exactly once, predictably
- **Rich but Not Loud** — personality through restraint, color only where it signals data or state

## Palette (OKLCH)
- Bg: `oklch(18% 0.004 30)` (primary), `oklch(22% 0.005 30)` (secondary), `oklch(15% 0.003 30)` (tertiary)
- Text: `oklch(93% 0 0)` (primary), `oklch(70% 0.004 30)` (secondary), `oklch(52% 0.004 30)` (tertiary), `oklch(38% 0.003 30)` (muted)
- Accent: `oklch(62% 0.14 55)` — warm amber for actions, selections, emphasis
- Data: `oklch(62% 0.14 155)` (up), `oklch(62% 0.12 25)` (down)
- Radii: 2px (xs), 4px (sm), 6px (md) — no large rounding on cards

## Typography
- Inter for UI, JetBrains Mono for data. No display fonts.
- Fixed rem scale: Display 48px, H1 32px, H2 22px, H3 16px, Body 15px, Small 13px, Caption 11px, Micro 10px.
- Tabular-nums on all numeric content.

## Styling
- Use CSS custom properties from `app/globals.css` — never inline hex values.
- Widget pattern: `bg-background-secondary border border-border radius-sm` with hover → `border-accent bg-surface`.
- Primary button: `bg-accent text-background-primary`. No border + shadow on same element.
- No neon, no tactical gaming cliches, no decorative motion on product pages.
<!-- END:design-context -->

<!-- BEGIN:api-server -->
# API Server

The FastAPI server runs on port 8000. Start it from `backend/`:
```
source venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8000
```

It connects to Supabase (production DB) and serves all data endpoints:
- `GET /health` — health check
- `GET /items/`, `/items/count`, `/items/search`, `/items/trending`, `/items/{item_id}` — items
- `GET /items/{item_id}/price-history`, `/trends`, `/prediction`, `/events`, `/prices`, `/variants`, `/event-impacts`, `/feature-importance`
- `GET /market/summary` — grouped market view (paginated, cached)
- `GET /opportunities/`, `/undervalued`, `/overheated`, `/momentum`
- `GET /events/`, `/events/recent`
- `GET /accuracy/`, `/accuracy/latest`, `/accuracy/summary`, `/accuracy/outcomes`, `/accuracy/outcomes/stats`
- `GET /ab-test/regime`, `/ab-test/ensemble` — A/B test comparisons
- `GET /auth/me`, `/auth/steam/login`, `POST /auth/logout`
- `GET /portfolio/inventory`

The frontend fetches from `NEXT_PUBLIC_API_URL` (defaults to `http://localhost:8000`).

When adding migrations or columns, ensure the SQLAlchemy models match the production schema — see the schema-alignment gotcha in the root `AGENTS.md`.
<!-- END:api-server -->
