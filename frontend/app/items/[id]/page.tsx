'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Header, PriceSourceFilter } from '@/components';
import {
  getItem,
  getItemPrediction,
  getItemTrends,
  getMultiSourcePrices,
  getPriceHistory,
  getItemVariants,
  getItemEventImpacts,
  getItemFeatureImportance,
  getLatestAccuracy,
  MultiSourcePrices,
  PricePoint,
  QualityVariant,
  EventImpact,
  FeatureImportance,
  LatestAccuracyRecord,
} from '@/lib/api';

interface CatalogItem {
  id: number;
  item_id: string;
  name: string;
  type: string;
  release_date?: string;
}

interface TrendResponse {
  item_id: string;
  item_name: string;
  current_price: number | null;
  trend_direction: 'bullish' | 'neutral' | 'bearish' | 'insufficient_data';
  confidence: 'low' | 'medium' | 'high';
  trend_score?: number | null;
  indicators?: {
    sma_7?: number | null;
    sma_30?: number | null;
    volatility?: number | null;
  };
  factors?: string[];
  explanation?: string;
}

interface PredictionResponse {
  item_id: string;
  current_price: number | null;
  forecast?: { low: number; mid: number; high: number };
  period_label?: string;
}

interface ChartRow {
  label: string;
  timestamp: number;
  [key: string]: number | string | null;
}

const TIME_RANGES = ['24h', '7d', '30d', 'all'] as const;
type TimeRange = (typeof TIME_RANGES)[number];

/* Dive-stage depths: the deeper the range, the darker the water behind the chart. */
const RANGE_DEPTH: Record<TimeRange, string> = {
  '24h': 'var(--surface)',
  '7d': 'var(--stock)',
  '30d': 'var(--ground)',
  all: 'var(--recess)',
};

const FORECAST_PERIODS = ['3_days', '7_days', '14_days', '30_days'] as const;
type ForecastPeriod = (typeof FORECAST_PERIODS)[number];
const FORECAST_DAYS: Record<ForecastPeriod, number> = {
  '3_days': 3,
  '7_days': 7,
  '14_days': 14,
  '30_days': 30,
};

const SOURCE_META: Record<string, string> = {
  historical: 'Archive',
  aggregator_sync: 'Live',
  market_csgo: 'Market.CSGO',
  steam_historical: 'Steam (weekly)',
  steam_batch: 'Steam',
  steam: 'Steam',
  csfloat: 'CSFloat',
};

const MONO = 'var(--font-jetbrains-mono), monospace';

function summarizeHistory(history: PricePoint[]) {
  if (!history.length) return { currentPrice: null, priceChange24h: null, volume24h: null };

  const points = [...history].sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
  const latest = points[points.length - 1];
  const latestTs = new Date(latest.timestamp).getTime();
  const cutoff = latestTs - 24 * 60 * 60 * 1000;
  const windowPoints = points.filter(point => new Date(point.timestamp).getTime() >= cutoff);
  const comparisonPoints = windowPoints.length > 1 ? windowPoints : points.slice(-2);
  const first = comparisonPoints[0];
  const last = comparisonPoints[comparisonPoints.length - 1];

  const priceChange24h = first && last && first.price > 0 ? ((last.price - first.price) / first.price) * 100 : null;
  const volumeWindow = windowPoints.length ? windowPoints : points.slice(-2);
  const volume24h = volumeWindow.reduce((sum, point) => sum + (point.volume ?? 0), 0) || null;

  return { currentPrice: latest.price, priceChange24h, volume24h };
}

function buildSourceChartData(sourceData: MultiSourcePrices | null, selectedSources: string[], range: TimeRange): ChartRow[] {
  if (!sourceData) return [];

  const activeSources = selectedSources.length ? selectedSources : sourceData.sources;
  const now = Date.now();
  const cutoff = range === '24h' ? now - 24 * 60 * 60 * 1000
    : range === '7d' ? now - 7 * 24 * 60 * 60 * 1000
    : range === '30d' ? now - 30 * 24 * 60 * 60 * 1000
    : Number.NEGATIVE_INFINITY;

  const buckets = new Map<string, ChartRow>();

  for (const source of activeSources) {
    const points = sourceData.data[source] ?? [];
    for (const point of points) {
      const timestamp = new Date(point.timestamp).getTime();
      if (timestamp < cutoff) continue;

      const bucketKey = range === '24h'
        ? new Date(timestamp).toISOString().slice(0, 13)
        : new Date(timestamp).toISOString().slice(0, 10);
      const label = range === '24h'
        ? new Date(timestamp).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
        : new Date(timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' });

      const existing = buckets.get(bucketKey);
      if (!existing) {
        buckets.set(bucketKey, { label, timestamp, [source]: point.price });
      } else {
        existing[source] = point.price;
        existing.timestamp = Math.max(existing.timestamp, timestamp);
      }
    }
  }

  return [...buckets.values()].sort((a, b) => a.timestamp - b.timestamp);
}

function appendForecastBand(
  data: ChartRow[],
  prediction: PredictionResponse | null,
  primarySource: string,
  period: ForecastPeriod
): ChartRow[] {
  if (!prediction?.forecast || data.length === 0) return data;
  const f = prediction.forecast;
  const last = data[data.length - 1];
  const lastPrice = typeof last[primarySource] === 'number' ? (last[primarySource] as number) : null;
  if (lastPrice == null) return data;

  const endTs = last.timestamp + FORECAST_DAYS[period] * 24 * 60 * 60 * 1000;
  const endLabel = new Date(endTs).toLocaleDateString([], { month: 'short', day: 'numeric' });

  const rows = data.map((r) =>
    r === last ? { ...r, q10: lastPrice, q90span: 0, q50: lastPrice } : r
  );
  rows.push({ label: endLabel, timestamp: endTs, q10: f.low, q90span: f.high - f.low, q50: f.mid });
  return rows;
}

function formatCurrency(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return '\u2014';
  return `$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function formatPercent(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return '\u2014';
  return `${value > 0 ? '+' : ''}${value.toFixed(2)}%`;
}

function formatVolume(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return '\u2014';
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return `${value.toFixed(0)}`;
}

interface StrataTooltipItem {
  dataKey: string;
  value: number | string | null;
  color: string;
  payload?: ChartRow;
}

function StrataTooltip({ active, payload }: { active?: boolean; payload?: StrataTooltipItem[] }) {
  if (!active || !payload?.length) return null;
  const first = payload[0];
  if (first?.value == null) return null;

  const label = first.payload?.label;
  const rows = payload.filter(
    (p) => p.value != null && p.dataKey !== 'q10' && p.dataKey !== 'q90span'
  );

  return (
    <div className="bg-stock border border-border rounded-sm px-3 py-2.5 min-w-[180px]">
      <div className="dive-tag text-paper-tertiary mb-2">{label}</div>
      <div className="space-y-1">
        {rows.map((p) => (
          <div key={String(p.dataKey)} className="flex items-center justify-between gap-6">
            <span
              className="font-data text-[11px] flex items-center gap-2"
              style={{ color: p.dataKey === 'q50' ? 'var(--thermocline)' : 'var(--paper-tertiary)' }}
            >
              <span
                aria-hidden
                className="w-2 h-[2px] rounded-full inline-block"
                style={{ backgroundColor: p.dataKey === 'q50' ? 'var(--thermocline)' : p.color }}
              />
              {p.dataKey === 'q50' ? 'q50 forecast' : SOURCE_META[String(p.dataKey)] ?? String(p.dataKey)}
            </span>
            <span className="font-data text-xs text-paper">
              {formatCurrency(Number(p.value))}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ItemDetailPage() {
  const params = useParams();
  const itemId = decodeURIComponent(params.id as string);
  const [item, setItem] = useState<CatalogItem | null>(null);
  const [variants, setVariants] = useState<QualityVariant[]>([]);
  const [history, setHistory] = useState<PricePoint[]>([]);
  const [trends, setTrends] = useState<TrendResponse | null>(null);
  const [predictions, setPredictions] = useState<Partial<Record<ForecastPeriod, PredictionResponse | null>>>({});
  const [multiSourceData, setMultiSourceData] = useState<MultiSourcePrices | null>(null);
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  const [timeRange, setTimeRange] = useState<TimeRange>('30d');
  const [forecastPeriod, setForecastPeriod] = useState<ForecastPeriod>('7_days');
  const [eventImpacts, setEventImpacts] = useState<EventImpact[]>([]);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance | null>(null);
  const [accuracy, setAccuracy] = useState<LatestAccuracyRecord | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadItemData() {
      setIsLoading(true);
      setError(null);

      const days =
        timeRange === '24h' ? 1
          : timeRange === '7d' ? 7
          : timeRange === '30d' ? 30
          : 5000;

      try {
        const [itemResponse, variantsResponse, historyResponse, trendsResponse, sourceResponse, eventImpactsResponse, fiResponse, accuracyResponse] = await Promise.all([
          getItem(itemId),
          getItemVariants(itemId).catch(() => []),
          getPriceHistory(itemId, 5000, 0, 500),
          getItemTrends(itemId),
          getMultiSourcePrices(itemId, ['all'], days),
          getItemEventImpacts(itemId).catch(() => [] as EventImpact[]),
          getItemFeatureImportance(itemId).catch(() => null),
          getLatestAccuracy('forecast', -1).catch(() => null),
        ]);

        if (cancelled) return;

        setItem(itemResponse as CatalogItem);
        const variantList = Array.isArray(variantsResponse) ? variantsResponse as QualityVariant[] : [];
        setVariants(variantList);
        setHistory(Array.isArray(historyResponse) ? historyResponse as PricePoint[] : []);
        setTrends({
          item_id: trendsResponse?.item_id ?? '',
          item_name: trendsResponse?.item_name ?? '',
          current_price: trendsResponse?.current_price ?? null,
          trend_direction: trendsResponse?.trend_direction ?? 'insufficient_data',
          confidence: trendsResponse?.confidence ?? 'low',
          indicators: {
            sma_7: trendsResponse?.sma_7 ?? null,
            sma_30: trendsResponse?.sma_30 ?? null,
            volatility: trendsResponse?.volatility ?? null,
          },
          factors: trendsResponse?.factors ?? [],
          explanation: trendsResponse?.explanation ?? '',
        } as TrendResponse);

        const periodResults = await Promise.allSettled(
          FORECAST_PERIODS.map((p) => getItemPrediction(itemId, p))
        );
        const predictionMap: Partial<Record<ForecastPeriod, PredictionResponse | null>> = {};
        FORECAST_PERIODS.forEach((p, i) => {
          const res = periodResults[i];
          if (res.status === 'fulfilled' && res.value && !Array.isArray(res.value)) {
            const pr = res.value as Record<string, unknown>;
            const low = pr.forecast_low;
            const mid = pr.forecast_mid;
            const high = pr.forecast_high;
            if (typeof low === 'number' && typeof mid === 'number' && typeof high === 'number') {
              predictionMap[p] = {
                item_id: String(pr.item_id ?? itemId),
                current_price: (pr.current_price as number) ?? null,
                forecast: { low, mid, high },
                period_label: (pr.forecast_period as string) ?? p,
              };
            }
          }
        });
        setPredictions(predictionMap);

        setMultiSourceData(sourceResponse);
        setEventImpacts(Array.isArray(eventImpactsResponse) ? eventImpactsResponse as EventImpact[] : []);
        setFeatureImportance(fiResponse as FeatureImportance | null);

        const acc = accuracyResponse as LatestAccuracyRecord | null;
        if (acc && !Array.isArray(acc) && acc.metrics && typeof acc.metrics.interval_coverage === 'number') {
          setAccuracy(acc);
        }
        setSelectedSources(
          Array.isArray(sourceResponse?.sources) && sourceResponse.sources.length
            ? sourceResponse.sources
            : ['steam']
        );
      } catch (fetchError) {
        if (!cancelled) setError(fetchError instanceof Error ? fetchError.message : 'Failed to load item data');
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    loadItemData();
    return () => { cancelled = true; };
  }, [itemId, timeRange]);

  const availableSources = multiSourceData?.sources ?? ['steam'];
  const visibleSources = selectedSources.length ? selectedSources : availableSources;
  const primarySource = useMemo(() => {
    const preferred = visibleSources.find((s) => s === 'historical' || s === 'steam_historical');
    return preferred ?? visibleSources[0];
  }, [visibleSources]);
  const otherSources = visibleSources.filter((s) => s !== primarySource);

  const rawChartData = useMemo(
    () => buildSourceChartData(multiSourceData, visibleSources, timeRange),
    [multiSourceData, visibleSources, timeRange]
  );

  const showCorridor = timeRange === '30d' || timeRange === 'all';
  const chartData = useMemo(
    () =>
      showCorridor
        ? appendForecastBand(rawChartData, predictions[forecastPeriod] ?? null, primarySource, forecastPeriod)
        : rawChartData,
    [rawChartData, showCorridor, predictions, forecastPeriod, primarySource]
  );

  // The forecast row sits at the far right of the profile; the water darkens
  // from the last real close onward — the mesophotic zone.
  const forecastZonePct = useMemo(() => {
    const last = chartData[chartData.length - 1];
    if (chartData.length < 2 || typeof last?.q50 !== 'number') return null;
    return ((chartData.length - 1) / chartData.length) * 100;
  }, [chartData]);

  const summary = summarizeHistory(history);
  const latestPrice = summary.currentPrice;
  const trendDirection = trends?.trend_direction ?? 'insufficient_data';
  const trendFactors = trends?.factors ?? [];
  const hasPriceData = history.length > 0;
  const hasForecast = Object.values(predictions).some((p) => p?.forecast != null);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-ground">
        <Header />
        <div className="max-w-6xl mx-auto px-6 py-8">
          <div className="flex items-center gap-4 mb-8">
            <div className="w-16 h-16 rounded-xs bg-stock animate-pulse" />
            <div className="flex-1">
              <div className="h-8 w-64 bg-stock rounded-xs animate-pulse mb-2" />
              <div className="h-4 w-40 bg-recess rounded-xs animate-pulse" />
            </div>
          </div>
          <div className="bg-stock border border-border rounded-sm p-6">
            <div className="h-[380px] bg-recess/60 rounded-xs animate-pulse" />
          </div>
        </div>
      </div>
    );
  }

  if (error || !item) {
    return (
      <div className="min-h-screen bg-ground">
        <Header />
        <div className="max-w-6xl mx-auto px-6 py-8">
          <Link
            href="/market"
            className="dive-tag text-ink hover:text-ink-hover transition-colors duration-200 mb-6 inline-block"
          >
            &larr; Catalog
          </Link>
          <div className="bg-stock border border-border rounded-sm p-10 text-center">
            <h1 className="text-title text-paper mb-2">Item unavailable</h1>
            <p className="text-sm text-paper-secondary mb-6">
              {error || 'No backend item data was returned for this id.'}
            </p>
            <Link href="/market" className="btn btn-secondary">
              Back to Market
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-ground">
      <Header />

      <div className="max-w-6xl mx-auto px-6 py-8">
        <Link
          href="/market"
          className="dive-tag text-ink hover:text-ink-hover transition-colors duration-200 mb-6 inline-block"
        >
          &larr; Catalog
        </Link>

        {/* Specimen header */}
        <div className="mb-6">
          <span className="dive-tag text-paper-tertiary mb-2 block">{item.type}</span>
          <h1 className="text-headline text-paper mb-2">{item.name}</h1>
          <div className="flex flex-wrap items-center gap-x-5 gap-y-1 font-data text-xs text-paper-tertiary">
            <span>{itemId}</span>
            {item.release_date && (
              <span>{new Date(item.release_date).toLocaleDateString()}</span>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Primary plate — strata chart + wear tray */}
          <div className="lg:col-span-2">
            <div className="dive-card relative overflow-hidden">
              <span aria-hidden className="sounding-rule" />
              <span aria-hidden className="card-sweep" />
              <div className="flex items-center justify-between px-4 py-2.5 border-b border-divider">
                <span className="dive-tag text-paper-tertiary">Dive Profile &mdash; Price History</span>
                <span className="dive-tag text-paper-muted">dive {itemId}</span>
              </div>

              <div className="p-4">
                {/* Time ranges + forecast horizon */}
                <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                  <div className="flex gap-1">
                    {TIME_RANGES.map((range) => (
                      <button
                        key={range}
                        onClick={() => setTimeRange(range)}
                        className={`dive-tag px-2.5 py-1.5 rounded-xs transition-colors duration-200 ${
                          timeRange === range
                            ? 'text-ink bg-ink-subtle'
                            : 'text-paper-tertiary hover:text-paper'
                        }`}
                      >
                        {range}
                      </button>
                    ))}
                  </div>
                  {showCorridor && hasForecast && (
                    <div className="flex gap-1">
                      {FORECAST_PERIODS.map((p) => (
                        <button
                          key={p}
                          onClick={() => setForecastPeriod(p)}
                          className={`dive-tag px-2.5 py-1.5 rounded-xs transition-colors duration-200 ${
                            forecastPeriod === p
                              ? 'text-ink bg-ink-subtle'
                              : 'text-paper-tertiary hover:text-paper'
                          }`}
                        >
                          {FORECAST_DAYS[p]}d
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                {/* Source filter */}
                <div className="mb-4">
                  <PriceSourceFilter
                    selectedSources={visibleSources}
                    onSourceChange={setSelectedSources}
                    availableSources={availableSources}
                  />
                </div>

                {/* Dive profile chart */}
                {chartData.length > 0 ? (
                  <div
                    className="relative rounded-sm p-2 transition-colors duration-500"
                    style={{ backgroundColor: RANGE_DEPTH[timeRange] }}
                  >
                    {forecastZonePct !== null && (
                      <div
                        aria-hidden
                        className="absolute inset-0 rounded-sm pointer-events-none"
                        style={{
                          backgroundImage: `linear-gradient(90deg, transparent ${forecastZonePct}%, var(--recess) 100%)`,
                        }}
                      />
                    )}
                    <ResponsiveContainer width="100%" height={364}>
                      <ComposedChart data={chartData}>
                      <CartesianGrid horizontal stroke="var(--grid)" strokeWidth={1} vertical={false} />
                      <XAxis
                        dataKey="label"
                        tick={{ fontSize: 11, fill: 'var(--paper-tertiary)', fontFamily: MONO }}
                        axisLine={{ stroke: 'var(--border)' }}
                        tickLine={false}
                        minTickGap={40}
                      />
                      <YAxis
                        tick={{ fontSize: 11, fill: 'var(--paper-tertiary)', fontFamily: MONO }}
                        axisLine={false}
                        tickLine={false}
                        width={64}
                        domain={['auto', 'auto']}
                      />
                      <Tooltip content={<StrataTooltip />} cursor={{ stroke: 'var(--border)', strokeDasharray: '3 3' }} />
                      <Area dataKey="q10" stackId="corridor" stroke="none" fill="transparent" connectNulls isAnimationActive={false} />
                      <Area
                        dataKey="q90span"
                        stackId="corridor"
                        stroke="none"
                        fill="var(--thermocline)"
                        fillOpacity={0.24}
                        connectNulls
                        isAnimationActive={false}
                      />
                      <Line dataKey="q50" stroke="var(--thermocline)" strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />
                      <Line
                        dataKey={primarySource}
                        stroke="var(--paper)"
                        strokeWidth={1.5}
                        dot={false}
                        connectNulls
                        isAnimationActive={false}
                        name={SOURCE_META[primarySource] ?? primarySource}
                      />
                      {otherSources.map((source) => (
                        <Line
                          key={source}
                          dataKey={source}
                          stroke="var(--paper-tertiary)"
                          strokeWidth={1.25}
                          strokeDasharray="4 3"
                          dot={false}
                          connectNulls
                          isAnimationActive={false}
                          name={SOURCE_META[source] ?? source}
                        />
                      ))}
                    </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-[380px] text-sm text-paper-tertiary">
                    {hasPriceData
                      ? 'No price data available for the selected range'
                      : 'No price history recorded for this item'}
                  </div>
                )}
              </div>
            </div>

            {/* Wear tray */}
            {variants.length > 1 && (
              <div className="mt-6">
                <div className="dive-tag text-paper-tertiary mb-3">Wear Tray</div>
                <div className="flex flex-wrap gap-1.5">
                  {variants.map((v) => {
                    const isActive = v.item_id === itemId;
                    return (
                      <Link
                        key={v.item_id}
                        href={`/items/${encodeURIComponent(v.item_id)}`}
                        className={`wear-pill ${isActive ? 'wear-pill-active' : ''}`}
                      >
                        <span className="text-[11px] font-semibold uppercase tracking-[0.12em]">
                          {v.quality}
                        </span>
                        <span className="font-data text-xs">
                          {v.current_price != null ? formatCurrency(v.current_price) : '\u2014'}
                        </span>
                      </Link>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* Sidebar — curator tag, ledger stats, signals */}
          <aside className="space-y-6">
            {/* Curator tag */}
            <div className="bg-stock border border-border rounded-sm relative">
              <div className="flex items-center justify-between px-4 py-2.5 border-b border-divider">
                <span className="dive-tag text-thermocline">Dive Log Entry</span>
              </div>
              {hasForecast ? (
                <>
                  <div className="divide-y divide-divider">
                    {FORECAST_PERIODS.map((p) => {
                      const f = predictions[p]?.forecast;
                      if (!f) return null;
                      return (
                        <div key={p} className="px-4 py-3 flex items-baseline justify-between gap-4">
                          <div className="flex items-baseline gap-2">
                            <span className="dive-tag text-paper-tertiary">{FORECAST_DAYS[p]}d</span>
                            <span className="font-data text-[10px] text-paper-muted">q10&ndash;q90</span>
                          </div>
                          <div className="text-right">
                            <div className="text-data-lg text-thermocline leading-none">
                              {formatCurrency(f.mid)}
                            </div>
                            <div className="font-data text-[11px] text-paper-tertiary mt-1">
                              {formatCurrency(f.low)} &ndash; {formatCurrency(f.high)}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <div className="px-4 py-3 border-t border-divider">
                    {accuracy ? (
                      <p className="text-[10px] font-data text-paper-muted leading-relaxed">
                        Measured accuracy: {num(accuracy.metrics.interval_coverage)}% q10&ndash;q90
                        coverage on cohort &ge;${accuracy.price_tier ?? 1} at the{' '}
                        {accuracy.horizon_days != null ? `${accuracy.horizon_days}-day` : ''} horizon
                        &middot; {accuracy.evaluation_date ?? ''}
                      </p>
                    ) : (
                      <p className="text-[10px] font-data text-paper-muted">
                        No measured accuracy on record yet.
                      </p>
                    )}
                  </div>
                </>
              ) : (
                <div className="px-4 py-6">
                  <div className="flex items-center gap-3">
                    <span aria-hidden className="text-thermocline/60 font-data text-sm leading-none">&#8212;</span>
                    <p className="text-sm text-paper-secondary">Below working depth &mdash; no signal at this depth.</p>
                  </div>
                  <p className="text-[10px] font-data text-paper-muted mt-2">
                    Forecasts appear once the daily pipeline processes this item.
                  </p>
                </div>
              )}
            </div>

            {/* Ledger stats */}
            <div className="bg-stock border border-border rounded-sm">
              <div className="px-4 py-2.5 border-b border-divider">
                <span className="dive-tag text-paper-tertiary">Dive Log</span>
              </div>
              <div className="divide-y divide-divider">
                <LedgerRow
                  label="Current Price"
                  value={
                    <span className="text-data-lg text-paper">
                      {hasPriceData && latestPrice != null ? formatCurrency(latestPrice) : '\u2014'}
                    </span>
                  }
                />
                <LedgerRow
                  label="Change 24h"
                  value={
                    hasPriceData && summary.priceChange24h != null ? (
                      <span
                        className="inline-block px-1.5 py-0.5 rounded-xs font-data text-[11px] font-semibold"
                        style={{
                          backgroundColor: summary.priceChange24h >= 0 ? 'var(--up-subtle)' : 'var(--down-subtle)',
                          color: summary.priceChange24h >= 0 ? 'var(--up)' : 'var(--down)',
                        }}
                      >
                        {formatPercent(summary.priceChange24h)}
                      </span>
                    ) : (
                      <span className="font-data text-sm text-paper-muted">{'\u2014'}</span>
                    )
                  }
                />
                <LedgerRow
                  label="Volume 24h"
                  value={
                    <span className="font-data text-sm text-paper">
                      {hasPriceData ? formatVolume(summary.volume24h) : '\u2014'}
                    </span>
                  }
                />
                <LedgerRow
                  label="Trend"
                  value={
                    <span className="font-data text-sm capitalize" style={{ color: trendDirection === 'bullish' ? 'var(--up)' : trendDirection === 'bearish' ? 'var(--down)' : 'var(--paper-tertiary)' }}>
                      {trendDirection.replace('_', ' ')}
                    </span>
                  }
                />
                <LedgerRow
                  label="7d SMA"
                  value={
                    <span className="font-data text-sm text-paper">
                      {hasPriceData ? formatCurrency(trends?.indicators?.sma_7 ?? null) : '\u2014'}
                    </span>
                  }
                />
                <LedgerRow
                  label="30d SMA"
                  value={
                    <span className="font-data text-sm text-paper">
                      {hasPriceData ? formatCurrency(trends?.indicators?.sma_30 ?? null) : '\u2014'}
                    </span>
                  }
                />
                {trends?.indicators?.volatility != null && (
                  <LedgerRow
                    label="Volatility"
                    value={
                      <span className="font-data text-sm text-paper">
                        {trends.indicators.volatility.toFixed(1)}%
                      </span>
                    }
                  />
                )}
              </div>
            </div>

            {/* Signals */}
            <div className="bg-stock border border-border rounded-sm">
              <div className="px-4 py-2.5 border-b border-divider">
                <span className="dive-tag text-paper-tertiary">Signals</span>
              </div>
              <div className="px-4 py-3">
                {hasPriceData && trendFactors.length ? (
                  <ul className="space-y-2">
                    {trendFactors.map((factor) => (
                      <li key={factor} className="text-sm text-paper leading-snug flex gap-2">
                        <span aria-hidden className="text-ink font-data text-xs leading-snug mt-0.5">&#9642;</span>
                        {factor}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-paper-tertiary">
                    {hasPriceData ? 'No technical factors returned yet.' : 'No data to compute signals from'}
                  </p>
                )}
              </div>
            </div>

            {/* Event impacts */}
            {eventImpacts.length > 0 && (
              <div className="bg-stock border border-border rounded-sm">
                <div className="px-4 py-2.5 border-b border-divider">
                  <span className="dive-tag text-paper-tertiary">Event Impacts</span>
                </div>
                <div className="divide-y divide-divider">
                  {eventImpacts.slice(0, 5).map((imp) => (
                    <div key={imp.event_id} className="px-4 py-3">
                      <div className="flex items-center justify-between gap-3 mb-0.5">
                        <span className="text-xs font-medium text-paper capitalize">
                          {imp.event_type.replace('_', ' ')}
                        </span>
                        <span
                          className="font-data text-xs"
                          style={{ color: (imp.impact_pct_7day ?? 0) >= 0 ? 'var(--up)' : 'var(--down)' }}
                        >
                          {imp.impact_pct_7day != null
                            ? `${imp.impact_pct_7day > 0 ? '+' : ''}${imp.impact_pct_7day.toFixed(1)}%`
                            : '\u2014'}
                        </span>
                      </div>
                      <div className="text-xs text-paper-tertiary truncate">{imp.event_description}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Forecast drivers */}
            {featureImportance && Object.keys(featureImportance.horizons).length > 0 && (
              <div className="bg-stock border border-border rounded-sm">
                <div className="px-4 py-2.5 border-b border-divider">
                  <span className="dive-tag text-paper-tertiary">Forecast Drivers</span>
                </div>
                <div className="px-4 py-3 space-y-4">
                  {Object.entries(featureImportance.horizons).map(([horizon, features]) => (
                    <div key={horizon}>
                      <div className="dive-tag text-paper-muted mb-1.5">{horizon}d horizon</div>
                      <div className="space-y-1">
                        {features.slice(0, 5).map((fi) => (
                          <div key={fi.feature} className="flex items-center gap-2 text-xs">
                            <div className="flex-1 truncate text-paper-tertiary">
                              {fi.feature.replace(/_/g, ' ')}
                            </div>
                            <div className="font-data text-paper w-8 text-right">
                              {fi.importance.toFixed(0)}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </aside>
        </div>
      </div>
    </div>
  );
}

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function LedgerRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="px-4 py-3 flex items-center justify-between gap-4">
      <span className="dive-tag text-paper-tertiary">{label}</span>
      {value}
    </div>
  );
}
