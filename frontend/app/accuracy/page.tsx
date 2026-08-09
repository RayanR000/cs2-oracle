'use client';

import { useEffect, useState } from 'react';
import { Header } from '@/components';
import {
  getAccuracySummary,
  getAccuracyHeadline,
  type AccuracyRecord,
  type AccuracyHeadline,
  type AccuracyHeadlineHorizon,
  type HeadlineVerdict,
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

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
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
  const color = good === true ? 'var(--up)' :
    good === false ? 'var(--down)' :
    'var(--paper)';
  return (
    <div className="bg-stock border border-border rounded-sm p-4">
      <div className="dive-tag text-paper-tertiary mb-2">{label}</div>
      <div className="text-data-lg" style={{ color }}>
        {typeof value === 'number' ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : value}
        {unit && <span className="text-sm text-paper-secondary ml-1">{unit}</span>}
      </div>
    </div>
  );
}

const VERDICT_LABEL: Record<HeadlineVerdict, { label: string; tone: 'up' | 'down' | 'neutral' }> = {
  skill: { label: 'Skill', tone: 'up' },
  no_skill: { label: 'No skill', tone: 'neutral' },
  perverse: { label: 'Anti-correlated', tone: 'down' },
  insufficient_dates: { label: 'Untestable', tone: 'neutral' },
  degenerate: { label: 'Untestable', tone: 'neutral' },
  untested: { label: 'Not tested', tone: 'neutral' },
};

function fmtPct(value: number | null | undefined): string {
  return num(value) === null ? '—' : `${(value as number).toFixed(1)}%`;
}

function HeadlineRow({ row, hurdleT }: { row: AccuracyHeadlineHorizon; hurdleT: number }) {
  const v = VERDICT_LABEL[row.verdict] ?? VERDICT_LABEL.untested;
  const color =
    v.tone === 'up' ? 'var(--up)' : v.tone === 'down' ? 'var(--down)' : 'var(--paper)';
  const t = num(row.pt_t_stat);
  const excess = num(row.pt_excess_pp);

  return (
    <tr className="border-t border-divider">
      <td className="py-3 pr-4 font-data text-sm text-paper">
        {row.horizon_days != null ? `${row.horizon_days}d` : '—'}
      </td>
      <td className="py-3 pr-4 font-data text-sm" style={{ color }}>{v.label}</td>
      <td className="py-3 pr-4 font-data text-sm text-paper text-right">
        {excess === null ? '—' : `${excess >= 0 ? '+' : ''}${excess.toFixed(2)}pp`}
      </td>
      <td className="py-3 pr-4 font-data text-sm text-paper text-right">
        {t === null ? '—' : t.toFixed(2)}
      </td>
      <td className="py-3 pr-4 font-data text-sm text-paper-secondary text-right">
        {num(row.pt_p_value) === null ? '—' : (row.pt_p_value as number).toPrecision(3)}
      </td>
      <td className="py-3 pr-4 font-data text-sm text-paper-secondary text-right">
        {fmtPct(row.directional_accuracy)}
      </td>
      <td className="py-3 pr-4 font-data text-sm text-paper-secondary text-right">
        {fmtPct(row.constant_call_accuracy)}
        <span className="text-paper-muted ml-1 text-[10px]">
          {row.constant_call_direction ?? ''}
        </span>
      </td>
      <td className="py-3 pr-4 font-data text-sm text-paper-secondary text-right">
        {fmtPct(row.realised_down_rate)}
      </td>
      <td className="py-3 font-data text-sm text-paper-secondary text-right">
        {row.pt_n_dates ?? 0}
        {(row.pt_n_dates_dropped ?? 0) > 0 && (
          <span className="text-paper-muted ml-1 text-[10px]">
            (+{row.pt_n_dates_dropped} thin)
          </span>
        )}
      </td>
      <td className="sr-only">hurdle t &gt; {hurdleT}</td>
    </tr>
  );
}

/**
 * The published claim, stated as a hypothesis test.
 *
 * This sits above the error metrics on purpose. Directional accuracy is in
 * here, but only inside the triple that makes it readable: on this data the
 * realised down-rate swings between forecast dates while the model's call
 * distribution barely moves, so a hit rate alone reports the market, not the
 * model.
 */
function HeadlineSection({ headline }: { headline: AccuracyHeadline }) {
  return (
    <div className="mb-12">
      <div className="flex items-baseline justify-between gap-4 mb-4 border-b border-divider pb-3">
        <span className="dive-tag text-paper-secondary">
          Directional Verdict &mdash; {headline.cohort} cohort
        </span>
        <span className="font-data text-[10px] text-paper-muted">
          hurdle |t| &gt; {headline.hurdle_t} &middot; {headline.min_forecast_dates}+ dates required
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead>
            <tr>
              {['Horizon', 'Verdict', 'Excess', 't', 'p', 'Hit rate', 'Constant call', 'Down-rate', 'Dates'].map(
                (h, i) => (
                  <th
                    key={h}
                    className={`dive-tag text-paper-tertiary pb-2 font-normal ${i > 1 ? 'text-right pr-4' : 'pr-4'}`}
                  >
                    {h}
                  </th>
                )
              )}
            </tr>
          </thead>
          <tbody>
            {headline.horizons.map((row) => (
              <HeadlineRow key={row.horizon_days ?? 'all'} row={row} hurdleT={headline.hurdle_t} />
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-[11px] text-paper-tertiary mt-4 max-w-3xl leading-relaxed">
        {headline.test}. <span className="text-paper-secondary">Excess</span> is the hit
        rate above the per-date chance null &mdash; the rate the same calls would score if
        they carried no information about the outcome. It is not measured against 50%: an
        always-down call on a falling market scores far above 50% and sits at exactly
        zero excess. A hit rate below the constant call can still be genuine skill, and
        one above it can be none.
      </p>
    </div>
  );
}

function ForecastSection({ records }: { records: AccuracyRecord[] }) {
  const latest = getLatest(records);
  if (!latest) return null;
  const m = latest.metrics as Record<string, MetricValue>;

  return (
    <div className="mb-10">
      <div className="flex items-baseline justify-between gap-4 mb-4">
        <h3 className="dive-tag text-paper-secondary">
          {latest.horizon_days}-Day Forecast
        </h3>
        <span className="font-data text-[10px] text-paper-muted">
          {latest.sample_count.toLocaleString()} samples &middot; {latest.evaluation_date}
        </span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard label="MAE" value={m.mae as number} unit="$" good={num(m.mae) !== null && num(m.mae)! < 10} />
        <MetricCard label="RMSE" value={m.rmse as number} unit="$" />
        <MetricCard label="MAPE" value={m.mape as number} unit="%" good={num(m.mape) !== null && num(m.mape)! < 15} />
        {/* No good/bad colour: >50% is not the bar. The bar is the per-date
            chance null, which the verdict table above tests against. */}
        <MetricCard label="Hit Rate" value={m.directional_accuracy as number} unit="%" />
        <MetricCard label="Interval Coverage" value={m.interval_coverage as number} unit="%" good={num(m.interval_coverage) !== null && num(m.interval_coverage)! > 70} />
        <MetricCard label="Samples" value={latest.sample_count} />
      </div>
      <div className="mt-3 grid grid-cols-3 gap-3">
        <MetricCard label="Low Conf Acc" value={m.confidence_accuracy_low as number} unit="%" />
        <MetricCard label="Med Conf Acc" value={m.confidence_accuracy_medium as number} unit="%" />
        <MetricCard label="High Conf Acc" value={m.confidence_accuracy_high as number} unit="%" />
      </div>
    </div>
  );
}

export default function AccuracyPage() {
  const [data, setData] = useState<GroupedAccuracy[]>([]);
  const [headline, setHeadline] = useState<AccuracyHeadline | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetch() {
      // Settled, not all: the error metrics are still worth showing if the
      // headline endpoint is down, and vice versa.
      const [summaryRes, headlineRes] = await Promise.allSettled([
        getAccuracySummary(),
        getAccuracyHeadline(),
      ]);
      if (summaryRes.status === 'fulfilled') {
        setData(summaryRes.value as GroupedAccuracy[]);
      }
      if (headlineRes.status === 'fulfilled' && headlineRes.value?.horizons?.length) {
        setHeadline(headlineRes.value);
      }
      setLoading(false);
    }
    fetch();
  }, []);

  return (
    <div className="min-h-screen bg-ground">
      <Header />
      <main className="max-w-6xl mx-auto px-6 py-12">
        <div className="mb-10">
          <span className="dive-tag text-paper-tertiary mb-3 block">The Dive Log&rsquo;s Record</span>
          <h1 className="text-headline text-paper mb-2">Prediction Accuracy</h1>
          <p className="text-base text-paper-secondary max-w-xl">
            How well each forecast performed against actual market outcomes — the errors
            logged in the same dive log as the findings.
          </p>
        </div>

        {!loading && headline && <HeadlineSection headline={headline} />}

        {loading ? (
          <div className="flex items-center justify-center py-20">
            <span className="dive-tag text-paper-muted">Reading the dive log...</span>
          </div>
        ) : data.length === 0 ? (
          <div className="bg-stock border border-border rounded-sm p-10 text-center">
            <p className="text-sm text-paper-secondary">No accuracy data yet.</p>
            <p className="text-[11px] text-paper-tertiary mt-2">
              Run <code className="font-data text-ink">python scripts/backtest_accuracy.py --type forecast</code> to generate metrics.
            </p>
          </div>
        ) : (
          data.map((group) => {
            const type = group.prediction_type;
            const label =
              type === 'forecast' ? `ML Forecast \u2014 ${group.horizon_days}d` :
              type;

            return (
              <div key={`${type}-${group.horizon_days ?? ''}-${group.evaluation_window_days ?? ''}`} className="mb-8">
                <div className="flex items-center gap-3 mb-4 border-b border-divider pb-3">
                  <span className="dive-tag text-paper-secondary">{label}</span>
                </div>
                {type === 'forecast' && <ForecastSection records={group.records} />}
              </div>
            );
          })
        )}

        {data.length > 0 && (
          <div className="mt-12 border-t border-divider pt-8">
            <p className="font-data text-[10px] text-paper-muted">
              Metrics compare stored ML forecasts against actual close prices. Updated daily via the backtesting pipeline.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
