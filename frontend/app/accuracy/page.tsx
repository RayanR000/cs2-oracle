'use client';

import { useEffect, useState } from 'react';
import { Header } from '@/components';
import {
  getAccuracySummary,
  type AccuracyRecord,
} from '@/lib/api';

type MetricValue = string | number | boolean | null | Record<string, unknown>;

interface GroupedAccuracy {
  prediction_type: string;
  horizon_days: number | null;
  evaluation_window_days: number | null;
  records: AccuracyRecord[];
}

function getLatest(records: AccuracyRecord[]): AccuracyRecord | null {
  if (!records.length) return null;
  return records.reduce((a, b) =>
    a.evaluation_date > b.evaluation_date ? a : b
  );
}

function MetricCard({
  label,
  value,
  unit,
  good,
}: {
  label: string;
  value: string | number;
  unit?: string;
  good?: boolean;
}) {
  const color = good === true ? 'text-[oklch(62%_0.14_155)]' :
    good === false ? 'text-[oklch(62%_0.12_25)]' :
    'text-primary';
  return (
    <div className="widget-block p-4">
      <div className="text-[10px] font-semibold uppercase tracking-[0.15em] text-muted mb-1">
        {label}
      </div>
      <div className={`font-data text-xl font-medium tabular-nums ${color}`}>
        {typeof value === 'number' ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : value}
        {unit && <span className="text-sm text-secondary ml-1">{unit}</span>}
      </div>
    </div>
  );
}

function ForecastSection({ records }: { records: AccuracyRecord[] }) {
  const latest = getLatest(records);
  if (!latest) return null;
  const m = latest.metrics as Record<string, MetricValue>;

  return (
    <div className="mb-10">
      <div className="flex items-baseline justify-between mb-4">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-primary">
            {latest.horizon_days}-Day Forecast
          </h3>
          <SourceBadge record={latest} />
        </div>
        <span className="text-[10px] font-data text-muted">
          {latest.sample_count.toLocaleString()} samples &middot; {latest.evaluation_date}
        </span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard label="MAE" value={m.mae as number} unit="$" good={Number(m.mae) < 10} />
        <MetricCard label="RMSE" value={m.rmse as number} unit="$" />
        <MetricCard label="MAPE" value={m.mape as number} unit="%" good={Number(m.mape) < 15} />
        <MetricCard label="Directional Acc" value={m.directional_accuracy as number} unit="%" good={Number(m.directional_accuracy) > 50} />
        <MetricCard label="Interval Coverage" value={m.interval_coverage as number} unit="%" good={Number(m.interval_coverage) > 70} />
        <MetricCard label="Samples" value={latest.sample_count} />
      </div>
      {/* Confidence calibration */}
      <div className="mt-3 grid grid-cols-3 gap-3">
        <MetricCard label="Low Conf Acc" value={m.confidence_accuracy_low as number} unit="%" />
        <MetricCard label="Med Conf Acc" value={m.confidence_accuracy_medium as number} unit="%" />
        <MetricCard label="High Conf Acc" value={m.confidence_accuracy_high as number} unit="%" />
      </div>
    </div>
  );
}

function SourceBadge({ record }: { record: AccuracyRecord }) {
  const isHistorical = record.model_version === 'historical_walkforward';
  return (
    <span className={`text-[9px] font-bold uppercase tracking-[0.15em] px-2 py-0.5 rounded-sm ${
      isHistorical
        ? 'bg-surface text-accent border border-accent/30'
        : 'bg-background-tertiary text-tertiary'
    }`}>
      {isHistorical ? '13-YR WALKFORWARD' : 'LIVE'}
    </span>
  );
}

function TrendSection({ records }: { records: AccuracyRecord[] }) {
  const latest = getLatest(records);
  if (!latest) return null;
  const m = latest.metrics as Record<string, MetricValue>;
  const confusion = m.confusion_matrix as Record<string, number> | undefined;

  return (
    <div className="mb-10">
      <div className="flex items-baseline justify-between mb-4">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-primary">
            {latest.evaluation_window_days}-Day Trend Direction
          </h3>
          <SourceBadge record={latest} />
        </div>
        <span className="text-[10px] font-data text-muted">
          {latest.sample_count.toLocaleString()} samples &middot; {latest.evaluation_date}
        </span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        <MetricCard label="Overall Accuracy" value={m.overall_accuracy as number} unit="%" good={Number(m.overall_accuracy) > 50} />
        <MetricCard label="Avg Return" value={m.avg_subsequent_return_pct as number} unit="%" good={Number(m.avg_subsequent_return_pct) > 0} />
        <MetricCard label="Window" value={m.avg_subsequent_return_days as number} unit="days" />
        <MetricCard label="Samples" value={latest.sample_count} />
      </div>
      {/* Confusion matrix */}
      {confusion && (
        <div className="widget-block p-4">
          <div className="text-[10px] font-semibold uppercase tracking-[0.15em] text-muted mb-3">Confusion Matrix</div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-data">
              <thead>
                <tr className="text-muted">
                  <th className="px-3 py-2 text-left">Predicted \ Actual</th>
                  <th className="px-3 py-2 text-right">Up</th>
                  <th className="px-3 py-2 text-right">Flat</th>
                  <th className="px-3 py-2 text-right">Down</th>
                </tr>
              </thead>
              <tbody className="text-primary">
                {['up', 'flat', 'down'].map((pred) => (
                  <tr key={pred} className="border-t border-border/50">
                    <td className="px-3 py-2 font-semibold text-secondary">{pred}</td>
                    {['up', 'flat', 'down'].map((actual) => (
                      <td key={actual} className="px-3 py-2 text-right tabular-nums">
                        {confusion[`${pred}_${actual}`] ?? 0}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function OpportunitySection({ records }: { records: AccuracyRecord[] }) {
  const latest = getLatest(records);
  if (!latest) return null;
  const m = latest.metrics as Record<string, MetricValue>;
  const undervalued = m.undervalued as Record<string, number> | undefined;
  const overheated = m.overheated as Record<string, number> | undefined;
  const momentum = m.momentum as Record<string, number> | undefined;

  return (
    <div className="mb-10">
      <div className="flex items-baseline justify-between mb-4">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-primary">
            {latest.evaluation_window_days}-Day Opportunity Signals
          </h3>
          <SourceBadge record={latest} />
        </div>
        <span className="text-[10px] font-data text-muted">
          {latest.sample_count.toLocaleString()} samples &middot; {latest.evaluation_date}
        </span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        <MetricCard label="Avg Return" value={m.avg_return_pct as number} unit="%" good={Number(m.avg_return_pct) > 0} />
        <MetricCard label="Total Signals" value={m.total_signals as number} />
        <MetricCard label="Window" value={m.evaluation_window_days as number} unit="days" />
        <MetricCard label="Samples" value={latest.sample_count} />
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {undervalued && (
          <div className="widget-block p-4">
            <div className="text-[10px] font-semibold uppercase tracking-[0.15em] text-muted mb-1">Undervalued</div>
            <div className="font-data text-lg font-medium text-primary tabular-nums">{undervalued.precision}%</div>
            <div className="text-[10px] text-tertiary mt-1">{undervalued.correct}/{undervalued.total} correct</div>
          </div>
        )}
        {overheated && (
          <div className="widget-block p-4">
            <div className="text-[10px] font-semibold uppercase tracking-[0.15em] text-muted mb-1">Overheated</div>
            <div className="font-data text-lg font-medium text-primary tabular-nums">{overheated.precision}%</div>
            <div className="text-[10px] text-tertiary mt-1">{overheated.correct}/{overheated.total} correct</div>
          </div>
        )}
        {momentum && (
          <div className="widget-block p-4">
            <div className="text-[10px] font-semibold uppercase tracking-[0.15em] text-muted mb-1">Momentum</div>
            <div className="font-data text-lg font-medium text-primary tabular-nums">{momentum.precision}%</div>
            <div className="text-[10px] text-tertiary mt-1">{momentum.correct}/{momentum.total} correct</div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function AccuracyPage() {
  const [data, setData] = useState<GroupedAccuracy[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetch() {
      try {
        const summary = await getAccuracySummary();
        setData(summary as GroupedAccuracy[]);
      } catch {
        // empty
      } finally {
        setLoading(false);
      }
    }
    fetch();
  }, []);

  return (
    <div className="min-h-screen bg-background-primary">
      <Header />
      <main className="max-w-6xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-10">
          <h1 className="text-2xl font-semibold tracking-tight text-primary">Prediction Accuracy</h1>
          <p className="text-sm text-secondary mt-1 max-w-lg">
            How well each signal type performs against actual market outcomes.
            <span className="block text-tertiary text-[11px] mt-1">
              <span className="text-accent font-semibold">13-YR WALKFORWARD</span> = retroactively evaluated across 500 items &times; 13 years of parquet archive data.
              <span className="text-tertiary ml-2">LIVE</span> = predictions made by the live system.
            </span>
          </p>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
          </div>
        ) : data.length === 0 ? (
          <div className="widget-block p-10 text-center">
            <p className="text-sm text-secondary">No accuracy data yet.</p>
            <p className="text-[11px] text-tertiary mt-2">
              Run <code className="font-data text-accent">python scripts/backtest_accuracy.py</code> or <code className="font-data text-accent">python scripts/backtest_accuracy.py --type historical</code> to generate metrics.
            </p>
          </div>
        ) : (
          data.map((group) => {
            const type = group.prediction_type;
            const label =
              type === 'forecast' ? `ML Forecast — ${group.horizon_days}d` :
              type === 'trend_direction' ? `Trend Direction — ${group.evaluation_window_days}d Window` :
              type === 'opportunity' ? `Opportunity Signals — ${group.evaluation_window_days}d Window` :
              type;

            return (
              <div key={`${type}-${group.horizon_days ?? ''}-${group.evaluation_window_days ?? ''}`} className="mb-8">
                <div className="flex items-center gap-3 mb-4 border-b border-border pb-3">
                  <span className="text-[10px] font-bold uppercase tracking-[0.15em] text-secondary">{label}</span>
                </div>
                {type === 'forecast' && <ForecastSection records={group.records} />}
                {type === 'trend_direction' && <TrendSection records={group.records} />}
                {type === 'opportunity' && <OpportunitySection records={group.records} />}
              </div>
            );
          })
        )}

        {/* Footer */}
        {data.length > 0 && (
          <div className="mt-12 border-t border-border pt-8">
            <p className="text-[10px] font-data text-muted">
              <span className="font-semibold text-secondary">LIVE</span> metrics compare stored predictions against actual close prices.
              <span className="ml-3 font-semibold text-secondary">13-YR WALKFORWARD</span> simulates MA-crossover and opportunity signals at every 7th day across 500 high-history items from the parquet archive (2013&ndash;2026), then checks outcomes at 7/14/30 day horizons. &mdash; Updated daily/weekly via the backtesting pipeline.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
