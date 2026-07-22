# CS2 Oracle — Frontend

Next.js 16 dashboard for CS2 market intelligence.

## Pages

- `/` — Home: hero, featured items, capabilities
- `/market` — Market overview: search, trending grid, sortable data table
- `/items/[id]` — Item detail: price chart, quality selector, metrics, signals
- `/portfolio` — User portfolio (requires Steam auth)
- `/accuracy` — Forecast accuracy metrics
- `/ab-test/regime` — Regime vs global-only model comparison

## Tech

Next.js 16, React 19, TypeScript, Tailwind CSS 4, Recharts, Framer Motion.

## Commands

```bash
npm run dev     # Development server
npm run build   # Type-check + production build
npm run start   # Start production server
npm run lint    # Run ESLint
```

## API Client

`lib/api.ts` — typed fetch functions for all backend endpoints. Add new routes here.

## Design Tokens

See `app/globals.css` for OKLCH CSS custom properties. See `docs/design.md` for the full design system.
