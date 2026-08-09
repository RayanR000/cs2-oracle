'use client';

import { useEffect, useState } from 'react';
import { Header } from '@/components';
import { getRegimeABTest, type RegimeABTestResult, type ABTestHorizonEntry } from '@/lib/api';

function DeltaBadge({ value, suffix, good }: { value: number; suffix?: string; good?: boolean }) {
  const isPositive = value > 0;
  const color = good === true ? 'text-data-up' :
    good === false ? 'text-data-down' :
    isPositive ? 'text-data-up' : 'text-data-down';
  const prefix = isPositive ? '+' : '';
  return (
    <span className={`font-data text-sm tabular-nums ${color}`}>
      {prefix}{value.toFixed(2)}{suffix || ''}
    </span>
  );
}

function HorizonCard({ entry }: { entry: ABTestHorizonEntry }) {
  const r = entry.regime?.metrics;
  const g = entry.global_only?.metrics;
  const d = entry.delta;
  if (!r || !g || !d) return null;

  const regimeWins = d.regime_wins;
  const dirAccGood = d.directional_accuracy_delta_pp > 0;
  const maeGood = d.mae_delta < 0;
  const mapeGood = d.mape_delta < 0;
  const intCovGood = d.interval_coverage_delta_pp > 0;

  return (
    <div className="widget-block p-5 mb-6">
      <div className="flex items-center justify-between mb-5">
        <h3 className="text-sm font-semibold text-primary">{entry.horizon_days}-Day Horizon</h3>
        <span className={`text-[10px] font-bold uppercase tracking-[0.15em] px-2 py-0.5 rounded-sm ${
          regimeWins ? 'bg-data-up-subtle text-data-up' : 'bg-data-down-subtle text-data-down'
        }`}>
          {regimeWins ? 'REGIME' : 'GLOBAL'}
        </span>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="text-[10px] font-semibold uppercase tracking-[0.15em] text-muted">Metric</div>
        <div className="text-[10px] font-semibold uppercase tracking-[0.15em] text-muted text-right">Regime</div>
        <div className="text-[10px] font-semibold uppercase tracking-[0.15em] text-muted text-right">Global</div>
      </div>

      <div className="grid grid-cols-3 gap-4 py-3 stripe-row items-center">
        <span className="text-sm text-secondary">Directional Accuracy</span>
        <span className="font-data tabular-nums text-sm text-primary text-right">{r.directional_accuracy.toFixed(1)}%</span>
        <span className="font-data tabular-nums text-sm text-primary text-right">{g.directional_accuracy.toFixed(1)}%</span>
      </div>
      <div className="grid grid-cols-3 gap-4 py-3 stripe-row items-center">
        <span className="text-sm text-secondary">MAE</span>
        <span className="font-data tabular-nums text-sm text-primary text-right">${r.mae.toFixed(2)}</span>
        <span className="font-data tabular-nums text-sm text-primary text-right">${g.mae.toFixed(2)}</span>
      </div>
      <div className="grid grid-cols-3 gap-4 py-3 stripe-row items-center">
        <span className="text-sm text-secondary">MAPE</span>
        <span className="font-data tabular-nums text-sm text-primary text-right">{r.mape.toFixed(1)}%</span>
        <span className="font-data tabular-nums text-sm text-primary text-right">{g.mape.toFixed(1)}%</span>
      </div>
      <div className="grid grid-cols-3 gap-4 py-3 stripe-row items-center">
        <span className="text-sm text-secondary">Interval Coverage</span>
        <span className="font-data tabular-nums text-sm text-primary text-right">{r.interval_coverage.toFixed(1)}%</span>
        <span className="font-data tabular-nums text-sm text-primary text-right">{g.interval_coverage.toFixed(1)}%</span>
      </div>
      <div className="grid grid-cols-3 gap-4 py-3 stripe-row items-center">
        <span className="text-sm text-secondary">Samples</span>
        <span className="font-data tabular-nums text-sm text-primary text-right">{entry.regime?.sample_count?.toLocaleString() || '—'}</span>
        <span className="font-data tabular-nums text-sm text-primary text-right">{entry.global_only?.sample_count?.toLocaleString() || '—'}</span>
      </div>
      <div className="grid grid-cols-3 gap-4 py-3 stripe-row items-center">
        <span className="text-sm text-secondary">Folds</span>
        <span className="font-data tabular-nums text-sm text-primary text-right">{r.fold_count}</span>
        <span className="font-data tabular-nums text-sm text-primary text-right">{g.fold_count}</span>
      </div>

      <div className="mt-4 pt-4 border-t border-border">
        <div className="text-[10px] font-semibold uppercase tracking-[0.15em] text-muted mb-3">Delta (Regime − Global)</div>
        <div className="grid grid-cols-2 gap-4">
          <div className="widget-block p-3">
            <div className="text-[10px] font-semibold uppercase tracking-[0.15em] text-muted mb-1">Directional Accuracy</div>
            <DeltaBadge value={d.directional_accuracy_delta_pp} suffix="pp" good={dirAccGood} />
          </div>
          <div className="widget-block p-3">
            <div className="text-[10px] font-semibold uppercase tracking-[0.15em] text-muted mb-1">MAE</div>
            <DeltaBadge value={d.mae_delta} suffix="$" good={maeGood} />
          </div>
          <div className="widget-block p-3">
            <div className="text-[10px] font-semibold uppercase tracking-[0.15em] text-muted mb-1">MAPE</div>
            <DeltaBadge value={d.mape_delta} suffix="pp" good={mapeGood} />
          </div>
          <div className="widget-block p-3">
            <div className="text-[10px] font-semibold uppercase tracking-[0.15em] text-muted mb-1">Interval Coverage</div>
            <DeltaBadge value={d.interval_coverage_delta_pp} suffix="pp" good={intCovGood} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default function RegimeABTestPage() {
  const [data, setData] = useState<RegimeABTestResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetch() {
      try {
        const result = await getRegimeABTest();
        if ('status' in result && (result as unknown as Record<string, string>).status === 'no_data') {
          setError((result as unknown as Record<string, string>).message || 'No data');
        } else {
          setData(result as RegimeABTestResult);
        }
      } catch {
        setError('Failed to load A/B test results');
      } finally {
        setLoading(false);
      }
    }
    fetch();
  }, []);

  return (
    <div className="min-h-screen bg-background-primary">
      <Header />
      <main className="max-w-5xl mx-auto px-6 py-12">
        <div className="mb-10">
          <h1 className="text-2xl font-semibold tracking-tight text-primary">A/B Test: Regime-Switching</h1>
          <p className="text-sm text-secondary mt-1 max-w-xl">
            Walk-forward comparison of regime-switching models vs global-only models on historical Parquet data.
            Regime models add ~23 min to training cost.
          </p>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-20">
            <span className="dive-tag text-paper-muted">Reading the dive log...</span>
          </div>
        ) : error ? (
          <div className="widget-block p-10 text-center">
            <p className="text-sm text-secondary">{error}</p>
            <p className="text-[11px] text-tertiary mt-2">
              Run <code className="font-data text-ink">python scripts/ab_test_regime.py</code> to generate A/B test data.
            </p>
          </div>
        ) : data ? (
          <>
            <div className="flex items-center gap-3 mb-6 text-[11px] text-tertiary">
              <span>Test date: {data.test_date || 'N/A'}</span>
              <span className="text-muted">|</span>
              <span className="text-muted">Walk-forward evaluation on historical Parquet data</span>
            </div>

            {data.horizons.map((entry) => (
              <HorizonCard key={entry.horizon_days} entry={entry} />
            ))}
          </>
        ) : null}
      </main>
    </div>
  );
}
