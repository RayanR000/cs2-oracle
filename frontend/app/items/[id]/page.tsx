'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import { motion } from 'framer-motion';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Header, PriceSourceFilter } from '@/components';
import CountUpNumber from '@/components/CountUpNumber';
import {
  getItem,
  getItemPrediction,
  getItemTrends,
  getMultiSourcePrices,
  getPriceHistory,
  getItemVariants,
  getItemEventImpacts,
  getItemFeatureImportance,
  MultiSourcePrices,
  PricePoint,
  QualityVariant,
  EventImpact,
  FeatureImportance,
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
    rsi?: number | null;
    bollinger_upper?: number | null;
    bollinger_middle?: number | null;
    bollinger_lower?: number | null;
    macd?: number | null;
    macd_signal?: number | null;
    support?: number | null;
    resistance?: number | null;
  };
  factors?: string[];
  methodology?: string;
  timestamp?: string;
  message?: string;
}

interface PredictionResponse {
  item_id: string;
  item_name: string;
  current_price: number | null;
  forecast?: { low: number; mid: number; high: number };
  period_days?: number;
  period_label?: string;
  trend_direction?: string;
  confidence?: string;
  volatility?: number | null;
  methodology?: string;
  timestamp?: string;
  message?: string;
}

interface PriceSeriesRow {
  timestamp: number;
  label: string;
  [source: string]: number | string;
}

const SOURCE_CHART_META: Record<string, { label: string; color: string }> = {
  historical: { label: 'Historical', color: 'oklch(70% 0 0)' },
  aggregator_sync: { label: 'Live', color: 'var(--brand)' },
  market_csgo: { label: 'Market.CSGO', color: 'oklch(65% 0.14 250)' },
  steam_historical: { label: 'Steam (weekly)', color: 'oklch(70% 0 0)' },
  steam_batch: { label: 'Steam', color: 'oklch(70% 0 0)' },
  steam: { label: 'Steam', color: 'oklch(70% 0 0)' },
  csfloat: { label: 'CSFloat', color: 'var(--brand)' },
};

const TIME_RANGES = ['24h', '7d', '30d', 'all'] as const;
type TimeRange = (typeof TIME_RANGES)[number];

const EASE: [number, number, number, number] = [0.16, 1, 0.3, 1];

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

function buildSourceChartData(sourceData: MultiSourcePrices | null, selectedSources: string[], range: TimeRange): PriceSeriesRow[] {
  if (!sourceData) return [];

  const activeSources = selectedSources.length ? selectedSources : sourceData.sources;
  const now = Date.now();
  const cutoff = range === '24h' ? now - 24 * 60 * 60 * 1000
    : range === '7d' ? now - 7 * 24 * 60 * 60 * 1000
    : range === '30d' ? now - 30 * 24 * 60 * 60 * 1000
    : Number.NEGATIVE_INFINITY;

  const buckets = new Map<string, PriceSeriesRow>();

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
        buckets.set(bucketKey, { timestamp, label, [source]: point.price });
      } else {
        existing[source] = point.price;
        existing.timestamp = Math.max(existing.timestamp as number, timestamp);
      }
    }
  }

  return [...buckets.values()].sort((a, b) => (a.timestamp as number) - (b.timestamp as number));
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

export default function ItemDetailPage() {
  const params = useParams();
  // useParams returns the still-encoded URL segment; without decoding here,
  // api.ts encodes it a second time and every request 404s for ids
  // containing spaces or pipes.
  const itemId = decodeURIComponent(params.id as string);
  const [item, setItem] = useState<CatalogItem | null>(null);
  const [variants, setVariants] = useState<QualityVariant[]>([]);
  const [activeQuality, setActiveQuality] = useState<string>('');
  const [history, setHistory] = useState<PricePoint[]>([]);
  const [trends, setTrends] = useState<TrendResponse | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [multiSourceData, setMultiSourceData] = useState<MultiSourcePrices | null>(null);
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  const [timeRange, setTimeRange] = useState<TimeRange>('30d');
  const [forecastPeriod, setForecastPeriod] = useState<string>('7_days');
  const [eventImpacts, setEventImpacts] = useState<EventImpact[]>([]);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance | null>(null);
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
        const [itemResponse, variantsResponse, historyResponse, trendsResponse, predictionResponse, sourceResponse, eventImpactsResponse, fiResponse] = await Promise.all([
          getItem(itemId),
          getItemVariants(itemId).catch(() => []),
          getPriceHistory(itemId, 5000, 0, 500),
          getItemTrends(itemId),
          getItemPrediction(itemId, forecastPeriod),
          getMultiSourcePrices(itemId, ['all'], days),
          getItemEventImpacts(itemId).catch(() => [] as EventImpact[]),
          getItemFeatureImportance(itemId).catch(() => null),
        ]);

        if (cancelled) return;

        setItem(itemResponse as CatalogItem);
        const variantList = Array.isArray(variantsResponse) ? variantsResponse as QualityVariant[] : [];
        setVariants(variantList);

        const currentVariant = variantList.find(v => v.item_id === itemId);
        setActiveQuality(currentVariant?.quality ?? variantList[0]?.quality ?? 'Standard');

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
          explanation: trendsResponse?.explanation ?? '',
        } as TrendResponse);
        setPrediction({
          current_price: predictionResponse?.current_price ?? null,
          forecast: {
            low: predictionResponse?.forecast_low ?? 0,
            mid: predictionResponse?.forecast_mid ?? 0,
            high: predictionResponse?.forecast_high ?? 0,
          },
          period_label: predictionResponse?.forecast_period ?? forecastPeriod,
          trend_direction: predictionResponse?.trend_direction,
          confidence: predictionResponse?.confidence,
        } as PredictionResponse);
        setMultiSourceData(sourceResponse);
        setEventImpacts(Array.isArray(eventImpactsResponse) ? eventImpactsResponse as EventImpact[] : []);
        setFeatureImportance(fiResponse as FeatureImportance | null);
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
  }, [itemId, timeRange, forecastPeriod]);

  const sourceChartData = useMemo(
    () => buildSourceChartData(multiSourceData, selectedSources, timeRange),
    [multiSourceData, selectedSources, timeRange]
  );

  const availableSources = multiSourceData?.sources ?? ['steam'];
  const visibleSources = selectedSources.length ? selectedSources : availableSources;
  const summary = summarizeHistory(history);
  const latestPrice = summary.currentPrice;
  const trendDirection = trends?.trend_direction ?? 'insufficient_data';
  const confidence = trends?.confidence ?? 'low';
  const trendFactors = trends?.factors ?? [];
  const forecast = prediction?.forecast;
  const hasPriceData = history.length > 0;

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background-primary">
        <Header />
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-center gap-4 mb-8">
            <div className="w-20 h-20 rounded-sm bg-background-secondary animate-pulse" />
            <div className="flex-1">
              <div className="h-8 w-64 bg-background-secondary rounded-sm animate-pulse mb-2" />
              <div className="h-4 w-40 bg-background-tertiary rounded-sm animate-pulse" />
            </div>
          </div>
          <div className="widget-block p-6">
            <div className="h-[360px] bg-background-tertiary/30 rounded-sm animate-pulse" />
          </div>
        </div>
      </div>
    );
  }

  if (error || !item) {
    return (
      <div className="min-h-screen bg-background-primary">
        <Header />
        <div className="max-w-7xl mx-auto px-6 py-8">
          <Link href="/market" className="text-accent-primary hover:text-brand-hover text-xs font-bold uppercase tracking-[0.2em] mb-6 inline-block transition-colors">
            &larr; MARKET
          </Link>
          <div className="widget-block p-8 text-center">
            <h1 className="text-2xl font-semibold text-primary mb-2">Item unavailable</h1>
            <p className="text-sm text-secondary mb-4">{error || 'No backend item data was returned for this id.'}</p>
            <Link href="/market" className="text-xs font-bold uppercase tracking-widest text-accent-primary hover:text-brand-hover transition-colors">
              Back to Market
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const trendColor = trendDirection === 'bullish' ? 'var(--data-up)' : trendDirection === 'bearish' ? 'var(--data-down)' : 'var(--text-secondary)';

  return (
    <div className="min-h-screen bg-background-primary">
      <Header />

      <div className="max-w-7xl mx-auto px-6 py-8">
        <motion.div
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3, ease: EASE }}
        >
          <Link href="/market" className="text-accent-primary hover:text-brand-hover text-xs font-bold uppercase tracking-[0.2em] mb-6 inline-block transition-colors">
            &larr; MARKET
          </Link>
        </motion.div>

        {/* Quality Selector */}
        {variants.length > 1 && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05, duration: 0.35, ease: EASE }}
            className="mb-6"
          >
            <div className="text-[10px] uppercase tracking-[0.15em] text-muted mb-2 font-semibold">Quality</div>
            <div className="flex flex-wrap gap-1.5">
              {variants.map((v) => (
                <Link
                  key={v.item_id}
                  href={`/items/${encodeURIComponent(v.item_id)}`}
                  className={`px-3 py-2 text-xs font-bold uppercase tracking-widest rounded-sm border transition-all duration-200 ${
                    v.quality === activeQuality
                      ? 'bg-accent text-background-primary border-accent'
                      : 'bg-surface text-secondary border-border hover:bg-surface-hover hover:border-accent-primary'
                  }`}
                >
                  {v.quality}
                  {v.current_price != null && (
                    <span className="ml-1.5 font-data normal-case opacity-70">
                      {formatCurrency(v.current_price)}
                    </span>
                  )}
                </Link>
              ))}
            </div>
          </motion.div>
        )}

        {/* Item Header */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.45, ease: EASE }}
          className="widget-block p-6 mb-8"
        >
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <h1 className="text-3xl font-bold text-primary mb-2 tracking-tight">{item.name}</h1>
              <div className="flex flex-wrap items-center gap-3 text-xs font-data text-tertiary">
                <span className="uppercase tracking-wide">{item.type}</span>
                {item.release_date && <span>{new Date(item.release_date).toLocaleDateString()}</span>}
                <span className="tag-tech">{itemId}</span>
              </div>
            </div>

            <div className="text-right">
              <div className="text-4xl font-bold text-primary font-data mb-1">
                {hasPriceData ? (
                  <CountUpNumber from={latestPrice!} to={latestPrice!} decimals={2} formatFn={formatCurrency} />
                ) : (
                  <span className="text-tertiary">---</span>
                )}
              </div>
              <div
                className="font-data text-sm"
                style={{ color: (summary.priceChange24h ?? 0) >= 0 ? 'var(--data-up)' : 'var(--data-down)' }}
              >
                {hasPriceData ? `${formatPercent(summary.priceChange24h)} (24h)` : <span className="text-tertiary">No data</span>}
              </div>
            </div>
          </div>

          {!hasPriceData && (
            <div className="mt-4 rounded-sm border border-border bg-background-tertiary px-4 py-3 text-sm text-secondary">
              This item exists in the index but has no price history yet. Data will appear once the collection pipeline processes it.
            </div>
          )}

          {/* Metric Cards */}
          <div className="mt-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              label="Trend"
              value={trendDirection.replace('_', ' ')}
              sub={`Confidence ${confidence}`}
              accentColor={trendColor}
            />
            <MetricCard label="7d SMA" value={hasPriceData ? formatCurrency(trends?.indicators?.sma_7 ?? null) : '\u2014'} mono />
            <MetricCard label="30d SMA" value={hasPriceData ? formatCurrency(trends?.indicators?.sma_30 ?? null) : '\u2014'} mono />
            <MetricCard label="Volume 24h" value={hasPriceData ? formatVolume(summary.volume24h) : '\u2014'} mono />
          </div>
        </motion.div>

        {/* Chart + Sidebar */}
        <div className="grid grid-cols-1 gap-8 xl:grid-cols-4">
          {/* Chart */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.45, ease: EASE }}
            className="xl:col-span-3"
          >
            {/* Time Range Tabs */}
            <div className="flex items-center justify-between gap-4 mb-4">
              <div className="flex gap-0.5">
                {TIME_RANGES.map((range) => (
                  <button
                    key={range}
                    onClick={() => setTimeRange(range)}
                    className={`px-3 py-2 text-xs font-bold uppercase tracking-widest transition-all duration-200 rounded-sm ${
                      timeRange === range
                        ? 'text-accent-primary bg-accent-primary/10'
                        : 'text-tertiary hover:text-primary hover:bg-surface'
                    }`}
                  >
                    {range}
                  </button>
                ))}
              </div>
              <span className="tag-tech">price history</span>
            </div>

            {/* Source Filter */}
            <div className="mb-4">
              <PriceSourceFilter
                selectedSources={visibleSources}
                onSourceChange={setSelectedSources}
                availableSources={availableSources}
              />
            </div>

            {/* Chart */}
            <div className="widget-block p-4">
              {sourceChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={360}>
                  <LineChart data={sourceChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                    <XAxis dataKey="label" stroke="var(--text-tertiary)" style={{ fontSize: '12px' }} />
                    <YAxis
                      stroke="var(--text-tertiary)"
                      style={{ fontSize: '12px' }}
                      domain={['dataMin - 10', 'dataMax + 10']}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'var(--background-secondary)',
                        border: '1px solid var(--border)',
                        color: 'var(--text-primary)',
                        borderRadius: '4px',
                      }}
                      formatter={(value) => formatCurrency(Number(value))}
                    />
                    {visibleSources.map((source) => (
                      <Line
                        key={source}
                        type="monotone"
                        dataKey={source}
                        stroke={SOURCE_CHART_META[source]?.color ?? 'var(--text-secondary)'}
                        strokeWidth={2}
                        dot={false}
                        isAnimationActive={false}
                        connectNulls
                        name={SOURCE_CHART_META[source]?.label ?? source}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex items-center justify-center h-[360px] text-sm text-tertiary">
                  {hasPriceData ? 'No price data available for the selected range' : 'No price history recorded for this item'}
                </div>
              )}
            </div>
          </motion.div>

          {/* Sidebar */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.45, ease: EASE }}
            className="space-y-4"
          >
            {/* Prediction */}
            <div className="widget-block p-4">
              <div className="text-[10px] uppercase tracking-[0.15em] text-muted mb-3 font-semibold">Prediction</div>
              <div className="text-2xl font-bold text-primary font-data">
                {hasPriceData && forecast ? (
                  <CountUpNumber from={forecast.mid} to={forecast.mid} decimals={2} formatFn={formatCurrency} />
                ) : '\u2014'}
              </div>
              <div className="flex gap-1 mt-2">
                {(['3_days', '7_days', '14_days', '30_days'] as const).map((p) => (
                  <button
                    key={p}
                    onClick={() => setForecastPeriod(p)}
                    className={`text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded-xs transition-colors ${
                      forecastPeriod === p
                        ? 'bg-accent/20 text-accent'
                        : 'text-muted hover:text-secondary'
                    }`}
                  >
                    {p.replace('_', ' ')}
                  </button>
                ))}
              </div>
              <div className="text-xs text-tertiary mt-1">
                {hasPriceData ? (prediction?.period_label || forecastPeriod) + ' forecast' : 'Insufficient data'}
              </div>
            </div>

            {/* Forecast Band */}
            <div className="widget-block p-4">
              <div className="text-[10px] uppercase tracking-[0.15em] text-muted mb-3 font-semibold">Forecast band</div>
              {hasPriceData ? (
                <div className="space-y-1.5 font-data text-sm">
                  <div className="flex justify-between">
                    <span className="text-tertiary">Low</span>
                    <span className="text-primary">{formatCurrency(forecast?.low ?? null)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-tertiary">Mid</span>
                    <span className="text-primary font-medium">{formatCurrency(forecast?.mid ?? null)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-tertiary">High</span>
                    <span className="text-primary">{formatCurrency(forecast?.high ?? null)}</span>
                  </div>
                </div>
              ) : (
                <div className="text-sm text-tertiary">No price data to forecast from</div>
              )}
            </div>

            {/* Data Sources */}
            <div className="widget-block p-4">
              <div className="text-[10px] uppercase tracking-[0.15em] text-muted mb-3 font-semibold">Data sources</div>
              <div className="space-y-2">
                {availableSources.map((source) => (
                  <div key={source} className="flex items-center justify-between text-sm">
                    <span className="capitalize text-primary">{source}</span>
                    <span className="text-tertiary font-data text-xs">
                      {(multiSourceData?.data[source]?.length ?? 0).toString()} pts
                    </span>
                  </div>
                ))}
              </div>
              {!hasPriceData && (
                <div className="mt-2 text-xs text-tertiary">Awaiting collection...</div>
              )}
            </div>

            {/* Signals */}
            <div className="widget-block p-4">
              <div className="text-[10px] uppercase tracking-[0.15em] text-muted mb-3 font-semibold">Signals</div>
              <div className="space-y-2 text-sm text-primary">
                {hasPriceData && trendFactors.length ? (
                  trendFactors.map((factor) => (
                    <div key={factor} className="leading-snug">{factor}</div>
                  ))
                ) : (
                  <div className="text-tertiary">{hasPriceData ? 'No technical factors returned yet.' : 'No data to compute signals from'}</div>
                )}
              </div>
            </div>

            {/* Event Impacts */}
            {eventImpacts.length > 0 && (
              <div className="widget-block p-4">
                <div className="text-[10px] uppercase tracking-[0.15em] text-muted mb-3 font-semibold">Event impacts</div>
                <div className="space-y-3">
                  {eventImpacts.slice(0, 5).map((imp) => (
                    <div key={imp.event_id} className="text-xs">
                      <div className="flex items-center justify-between mb-0.5">
                        <span className="text-primary font-medium capitalize">{imp.event_type.replace('_', ' ')}</span>
                        <span className="font-data" style={{ color: (imp.impact_pct_7day ?? 0) >= 0 ? 'var(--data-up)' : 'var(--data-down)' }}>
                          {imp.impact_pct_7day != null ? `${imp.impact_pct_7day > 0 ? '+' : ''}${imp.impact_pct_7day.toFixed(1)}%` : '\u2014'}
                        </span>
                      </div>
                      <div className="text-tertiary truncate">{imp.event_description}</div>
                      {imp.confidence_score != null && (
                        <div className="text-tertiary mt-0.5">
                          Confidence: {imp.confidence_score.toFixed(2)} | Z-score: {imp.z_score?.toFixed(2) ?? '\u2014'}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Feature Importance */}
            {featureImportance && Object.keys(featureImportance.horizons).length > 0 && (
              <div className="widget-block p-4">
                <div className="text-[10px] uppercase tracking-[0.15em] text-muted mb-3 font-semibold">Forecast drivers</div>
                {Object.entries(featureImportance.horizons).map(([horizon, features]) => (
                  <div key={horizon} className="mb-3 last:mb-0">
                    <div className="text-[11px] font-semibold text-secondary mb-1.5 uppercase tracking-wide">{horizon}d horizon</div>
                    <div className="space-y-1">
                      {features.slice(0, 5).map((fi) => (
                        <div key={fi.feature} className="flex items-center gap-2 text-xs">
                          <div className="flex-1 truncate text-tertiary">{fi.feature.replace(/_/g, ' ')}</div>
                          <div className="font-data text-primary w-8 text-right">{fi.importance.toFixed(0)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value, sub, mono, accentColor }: { label: string; value: string; sub?: string; mono?: boolean; accentColor?: string }) {
  return (
    <div className="rounded-sm border border-border bg-background-tertiary p-4 group hover:border-border-accent transition-colors duration-200">
      <div className="text-[10px] uppercase tracking-[0.15em] text-muted mb-2 font-semibold">{label}</div>
      <div
        className={`text-lg font-semibold ${mono ? 'font-data' : ''} capitalize`}
        style={{ color: accentColor || 'var(--text-primary)' }}
      >
        {value}
      </div>
      {sub && <div className="text-xs text-tertiary mt-1">{sub}</div>}
    </div>
  );
}
