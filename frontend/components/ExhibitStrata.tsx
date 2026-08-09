'use client';

import { useMemo } from 'react';
import { Area, ComposedChart, Line, ResponsiveContainer } from 'recharts';
import type { PricePoint } from '@/lib/api';

interface ExhibitStrataProps {
  history: PricePoint[];
  forecast: { low: number; mid: number; high: number } | null;
  loading?: boolean;
  height?: number;
}

const FORECAST_HORIZON_DAYS = 7;

interface StrataRow {
  ts: number;
  price?: number;
  q10?: number;
  q90span?: number;
  q50?: number;
}

export default function ExhibitStrata({
  history,
  forecast,
  loading = false,
  height = 68,
}: ExhibitStrataProps) {
  const rows = useMemo(() => {
    const points = [...history]
      .filter((p) => typeof p.price === 'number' && Number.isFinite(p.price))
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
    if (points.length < 2) return [];
    const rows: StrataRow[] = points.map((p) => ({ ts: new Date(p.timestamp).getTime(), price: p.price }));
    if (forecast) {
      const last = rows[rows.length - 1];
      rows[rows.length - 1] = { ...last, q10: last.price, q90span: 0, q50: last.price };
      rows.push({
        ts: last.ts + FORECAST_HORIZON_DAYS * 24 * 60 * 60 * 1000,
        q10: forecast.low,
        q90span: forecast.high - forecast.low,
        q50: forecast.mid,
      });
    }
    return rows;
  }, [history, forecast]);

  const spanDays = useMemo(() => {
    if (rows.length < 2) return null;
    return Math.max(1, Math.round((rows[rows.length - 1].ts - rows[0].ts) / 86_400_000));
  }, [rows]);

  if (loading) {
    return (
      <div className="px-4 pt-3 pb-3.5 border-t border-divider">
        <div className="rounded-xs bg-recess animate-pulse" style={{ height }} />
      </div>
    );
  }

  return (
    <div className="px-4 pt-2.5 pb-3 border-t border-divider">
      <div className="flex items-center justify-between mb-1">
        <span className="dive-tag text-paper-muted">
          {spanDays != null ? `Profile \u00b7 ${spanDays}d` : 'Profile'}
        </span>
        {forecast && (
          <span className="dive-tag text-thermocline">
            {FORECAST_HORIZON_DAYS}d &middot; q10&ndash;q90
          </span>
        )}
      </div>
      {rows.length > 1 ? (
        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart data={rows} margin={{ top: 2, right: 0, bottom: 0, left: 0 }}>
            <Area
              dataKey="q10"
              stackId="corridor"
              stroke="none"
              fill="transparent"
              connectNulls
              isAnimationActive={false}
            />
            <Area
              dataKey="q90span"
              stackId="corridor"
              stroke="none"
              fill="var(--thermocline-subtle)"
              fillOpacity={0.6}
              connectNulls
              isAnimationActive={false}
            />
            <Line
              dataKey="q50"
              stroke="var(--thermocline)"
              strokeWidth={2}
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
            <Line
              dataKey="price"
              stroke="var(--paper)"
              strokeWidth={1.5}
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      ) : (
        <div className="flex items-center" style={{ height }}>
          <span className="dive-tag text-paper-muted">no profile on record</span>
        </div>
      )}
    </div>
  );
}
