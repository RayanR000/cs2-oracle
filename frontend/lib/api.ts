/**
 * API client for CS2 Market Intelligence
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface TrendingItem {
  id: number;
  item_id: string;
  name: string;
  type: string;
  icon_url: string | null;
  latest_price: number;
}

export interface Item {
  id: number;
  item_id: string;
  name: string;
  type: 'skin' | 'case' | 'sticker';
  release_date?: string;
  created_at: string;
  updated_at: string;
}

export interface QualityVariant {
  item_id: string;
  name: string;
  quality: string;
  current_price: number | null;
  price_change_24h: number | null;
  volume_24h: number | null;
}

export interface GroupedMarketItem {
  base_name: string;
  type: string;
  icon_url: string | null;
  price_avg: number | null;
  price_min: number | null;
  price_max: number | null;
  price_change_24h: number | null;
  volatility: number | null;
  volume_24h: number | null;
  quality_count: number;
  qualities: QualityVariant[];
}

export interface PricePoint {
  timestamp: string;
  price: number;
  volume?: number;
  sma_7?: number;
  sma_30?: number;
}

export interface TrendAnalysis {
  item_id: number;
  item_name: string;
  current_price: number;
  trend_direction: 'bullish' | 'neutral' | 'bearish';
  confidence: 'low' | 'medium' | 'high';
  explanation: string;
}

export interface Prediction {
  item_id: number;
  item_name: string;
  current_price: number;
  forecast_low: number;
  forecast_high: number;
  forecast_period: string;
  trend_direction: string;
  confidence: string;
}

export interface Opportunity {
  item_id: number;
  item_name: string;
  current_price: number;
  opportunity_type: 'undervalued' | 'overheated' | 'momentum';
  opportunity_score: number;
  reason: string;
  current_trend: string;
  volatility?: number;
}

export interface SourcePrice {
  timestamp: string;
  price: number;
  volume?: number;
  median_price?: number;
}

export interface MultiSourcePrices {
  item_id: string;
  name: string;
  sources: string[];
  data: {
    [source: string]: SourcePrice[];
  };
}

// Items API
export async function getItemsCount(): Promise<number> {
  const response = await fetch(`${API_URL}/items/count`);
  if (!response.ok) throw new Error('Failed to fetch items count');
  return response.json();
}

export async function getItems(type?: string, skip = 0, limit = 50) {
  const url = new URL(`${API_URL}/items/`);
  if (type) url.searchParams.append('type', type);
  url.searchParams.append('skip', skip.toString());
  url.searchParams.append('limit', limit.toString());

  const response = await fetch(url.toString());
  if (!response.ok) throw new Error('Failed to fetch items');
  return response.json();
}

export async function searchItems(query: string) {
  const url = new URL(`${API_URL}/items/search`);
  url.searchParams.append('q', query);

  const response = await fetch(url.toString());
  return response.json();
}

export async function getTrendingItems(limit = 10) {
  const url = new URL(`${API_URL}/items/trending`);
  url.searchParams.append('limit', limit.toString());

  const response = await fetch(url.toString());
  if (!response.ok) throw new Error('Failed to fetch trending items');
  return response.json();
}

export async function getItem(itemId: string) {
  const response = await fetch(`${API_URL}/items/${encodeURIComponent(itemId)}`);
  if (!response.ok) throw new Error('Failed to fetch item');
  return response.json();
}

export async function getItemVariants(itemId: string): Promise<QualityVariant[]> {
  const response = await fetch(`${API_URL}/items/${encodeURIComponent(itemId)}/variants`);
  if (!response.ok) throw new Error('Failed to fetch item variants');
  return response.json();
}

export async function getPriceHistory(itemId: string, days = 30, skip = 0, limit = 100) {
  const url = new URL(`${API_URL}/items/${encodeURIComponent(itemId)}/price-history`);
  url.searchParams.append('days', days.toString());
  url.searchParams.append('skip', skip.toString());
  url.searchParams.append('limit', limit.toString());

  const response = await fetch(url.toString());
  if (!response.ok) throw new Error('Failed to fetch price history');
  return response.json();
}

export async function getItemTrends(itemId: string) {
  const response = await fetch(`${API_URL}/items/${encodeURIComponent(itemId)}/trends`);
  if (!response.ok) throw new Error('Failed to fetch item trends');
  return response.json();
}

export async function getItemPrediction(itemId: string, period = '7_days') {
  const url = new URL(`${API_URL}/items/${encodeURIComponent(itemId)}/prediction`);
  url.searchParams.append('period', period);

  const response = await fetch(url.toString());
  if (!response.ok) throw new Error('Failed to fetch item prediction');
  return response.json();
}

export async function getItemEvents(itemId: string, limit = 20) {
  const url = new URL(`${API_URL}/items/${encodeURIComponent(itemId)}/events`);
  url.searchParams.append('limit', limit.toString());

  const response = await fetch(url.toString());
  if (!response.ok) throw new Error('Failed to fetch item events');
  return response.json();
}

export async function getMultiSourcePrices(
  itemId: string,
  sources: string[] = ['all'],
  days: number = 5000
): Promise<MultiSourcePrices> {
  const sourceParam = sources.join(',');
  const url = new URL(`${API_URL}/items/${encodeURIComponent(itemId)}/prices`);
  url.searchParams.append('source', sourceParam);
  url.searchParams.append('days', days.toString());

  const response = await fetch(url.toString());
  if (!response.ok) throw new Error('Failed to fetch multi-source prices');
  return response.json();
}

// Market summary (optimized bulk endpoint - replaces N+1 pattern)
export async function getMarketSummary(
  type?: string,
  q?: string,
  skip = 0,
  limit = 50
) {
  const url = new URL(`${API_URL}/market/summary`);
  if (type) url.searchParams.append('type', type);
  if (q) url.searchParams.append('q', q);
  url.searchParams.append('skip', skip.toString());
  url.searchParams.append('limit', limit.toString());

  const response = await fetch(url.toString());
  if (!response.ok) throw new Error('Failed to fetch market summary');
  return response.json();
}

// Opportunities API
export async function getOpportunities(type?: string, limit = 20) {
  const url = new URL(`${API_URL}/opportunities/`);
  if (type) url.searchParams.append('type', type);
  url.searchParams.append('limit', limit.toString());

  const response = await fetch(url.toString());
  return response.json();
}

export async function getUndervaluedItems(limit = 10) {
  const url = new URL(`${API_URL}/opportunities/undervalued`);
  url.searchParams.append('limit', limit.toString());

  const response = await fetch(url.toString());
  return response.json();
}

export async function getOverheatedItems(limit = 10) {
  const url = new URL(`${API_URL}/opportunities/overheated`);
  url.searchParams.append('limit', limit.toString());

  const response = await fetch(url.toString());
  return response.json();
}

export async function getMomentumItems(limit = 10) {
  const url = new URL(`${API_URL}/opportunities/momentum`);
  url.searchParams.append('limit', limit.toString());

  const response = await fetch(url.toString());
  return response.json();
}

// Events API
export async function getEvents(type?: string, skip = 0, limit = 50) {
  const url = new URL(`${API_URL}/events/`);
  if (type) url.searchParams.append('type', type);
  url.searchParams.append('skip', skip.toString());
  url.searchParams.append('limit', limit.toString());

  const response = await fetch(url.toString());
  return response.json();
}

export async function getRecentEvents(limit = 20) {
  const url = new URL(`${API_URL}/events/recent`);
  url.searchParams.append('limit', limit.toString());

  const response = await fetch(url.toString());
  return response.json();
}

// Event Impacts API
export interface EventImpact {
  event_id: number;
  event_type: string;
  event_description: string;
  event_timestamp: string;
  price_day_before: number | null;
  price_day_1: number | null;
  price_day_3: number | null;
  price_day_7: number | null;
  impact_pct_1day: number | null;
  impact_pct_3day: number | null;
  impact_pct_7day: number | null;
  peak_impact_pct: number | null;
  peak_impact_day: number | null;
  duration_days: number | null;
  z_score: number | null;
  confidence_score: number | null;
}

export async function getItemEventImpacts(itemId: string, limit = 20): Promise<EventImpact[]> {
  const url = new URL(`${API_URL}/items/${encodeURIComponent(itemId)}/event-impacts`);
  url.searchParams.append('limit', limit.toString());
  const response = await fetch(url.toString());
  if (!response.ok) throw new Error('Failed to fetch event impacts');
  return response.json();
}

// Feature Importance API
export interface FeatureImportanceItem {
  feature: string;
  importance: number;
}

export interface FeatureImportance {
  item_id: string;
  item_name: string;
  horizons: {
    [horizon: string]: FeatureImportanceItem[];
  };
}

export async function getItemFeatureImportance(itemId: string): Promise<FeatureImportance> {
  const response = await fetch(`${API_URL}/items/${encodeURIComponent(itemId)}/feature-importance`);
  if (!response.ok) throw new Error('Failed to fetch feature importance');
  return response.json();
}

// Health check
export async function healthCheck() {
  const response = await fetch(`${API_URL}/health`);
  return response.json();
}

// Auth API
export async function getMe() {
  const response = await fetch(`${API_URL}/auth/me`, {
    credentials: 'include',
  });
  if (!response.ok) return null;
  return response.json();
}

export function getLoginUrl() {
  return `${API_URL}/auth/steam/login`;
}

export async function logout() {
  const response = await fetch(`${API_URL}/auth/logout`, {
    method: 'POST',
    credentials: 'include',
  });
  return response.json();
}

// Accuracy / Metrics API
export interface AccuracyRecord {
  id: number;
  prediction_type: string;
  evaluation_date: string;
  horizon_days: number | null;
  model_version: string | null;
  evaluation_window_days: number | null;
  sample_count: number;
  metrics: Record<string, unknown>;
  created_at: string;
}

export interface AccuracyMetric {
  mae?: number;
  rmse?: number;
  mape?: number;
  wmape?: number;
  directional_accuracy?: number;
  baseline_directional_accuracy?: number;
  interval_coverage?: number;
  [key: string]: unknown;
}

export interface LatestAccuracyRecord {
  id: number;
  prediction_type: string;
  evaluation_date: string | null;
  horizon_days: number | null;
  model_version: string | null;
  price_tier: number | null;
  evaluation_window_days: number | null;
  sample_count: number;
  metrics: AccuracyMetric;
  created_at: string | null;
}

export async function getAccuracy(
  predictionType?: string,
  priceTier?: number,
  limit = 50
): Promise<AccuracyRecord[]> {
  const url = new URL(`${API_URL}/accuracy/`);
  if (predictionType) url.searchParams.append('prediction_type', predictionType);
  if (priceTier !== undefined) url.searchParams.append('price_tier', priceTier.toString());
  url.searchParams.append('limit', limit.toString());
  const response = await fetch(url.toString());
  if (!response.ok) throw new Error('Failed to fetch accuracy');
  return response.json();
}

export async function getLatestAccuracy(
  predictionType?: string,
  priceTier?: number
): Promise<LatestAccuracyRecord | Record<string, LatestAccuracyRecord>> {
  const url = new URL(`${API_URL}/accuracy/latest`);
  if (predictionType) url.searchParams.append('prediction_type', predictionType);
  if (priceTier !== undefined) url.searchParams.append('price_tier', priceTier.toString());
  const response = await fetch(url.toString());
  if (!response.ok) throw new Error('Failed to fetch latest accuracy');
  return response.json();
}

export async function getAccuracySummary(predictionType?: string) {
  const url = new URL(`${API_URL}/accuracy/summary`);
  if (predictionType) url.searchParams.append('prediction_type', predictionType);
  const response = await fetch(url.toString());
  if (!response.ok) throw new Error('Failed to fetch accuracy summary');
  return response.json();
}

/**
 * Pesaran-Timmermann verdict for one horizon.
 *
 * `skill` / `perverse` — the statistic cleared +/- the hurdle t. `no_skill` —
 * it did not. `insufficient_dates` / `degenerate` — it could not be formed.
 * `untested` — the row predates the test and is not the same thing as a null.
 */
export type HeadlineVerdict =
  | 'skill'
  | 'no_skill'
  | 'perverse'
  | 'insufficient_dates'
  | 'degenerate'
  | 'untested';

export interface AccuracyHeadlineHorizon {
  horizon_days: number | null;
  model_version: string | null;
  evaluation_date: string | null;
  sample_count: number | null;
  verdict: HeadlineVerdict;
  pt_excess_pp: number | null;
  pt_t_stat: number | null;
  pt_p_value: number | null;
  pt_nw_lag: number | null;
  pt_n_dates: number | null;
  pt_n_dates_dropped: number | null;
  // Context for the verdict, never quoted alone — see getAccuracyHeadline.
  directional_accuracy: number | null;
  constant_call_accuracy: number | null;
  constant_call_direction: string | null;
  realised_down_rate: number | null;
  directional_accuracy_ci_clustered_lower: number | null;
  directional_accuracy_ci_clustered_upper: number | null;
  distinct_forecast_dates: number | null;
  date_coverage_sufficient: boolean | null;
  unchanged_pct: number | null;
  interval_coverage: number | null;
}

export interface AccuracyHeadline {
  cohort: string;
  price_tier: number;
  hurdle_t: number;
  min_forecast_dates: number;
  test: string;
  horizons: AccuracyHeadlineHorizon[];
}

/**
 * The published accuracy claim, as a significance test rather than a number.
 *
 * Prefer this over reading `directional_accuracy` off `getLatestAccuracy`. On
 * this data the realised down-rate swings between forecast dates while the
 * model's call distribution barely moves, so a bare hit rate reports which way
 * the market went, not whether the model knew. The backend computes
 * Pesaran-Timmermann per forecast date and takes a Newey-West t over dates;
 * `verdict` is that test's answer, and the accuracy fields are context for it.
 */
export async function getAccuracyHeadline(): Promise<AccuracyHeadline> {
  const response = await fetch(`${API_URL}/accuracy/headline`);
  if (!response.ok) throw new Error('Failed to fetch accuracy headline');
  return response.json();
}

// A/B Test API
export interface ABTestHorizonEntry {
  horizon_days: number;
  regime?: {
    sample_count: number;
    metrics: {
      mae: number;
      rmse: number;
      mape: number;
      directional_accuracy: number;
      interval_coverage: number;
      fold_count: number;
    };
  };
  global_only?: {
    sample_count: number;
    metrics: {
      mae: number;
      rmse: number;
      mape: number;
      directional_accuracy: number;
      interval_coverage: number;
      fold_count: number;
    };
  };
  delta?: {
    directional_accuracy_delta_pp: number;
    mae_delta: number;
    mape_delta: number;
    interval_coverage_delta_pp: number;
    regime_wins: boolean;
    regime_dir_acc: number;
    global_dir_acc: number;
    regime_mae: number;
    global_mae: number;
    regime_mape: number;
    global_mape: number;
    regime_int_cov: number;
    global_int_cov: number;
    regime_sample_count: number;
    global_sample_count: number;
    regime_fold_count: number;
    global_fold_count: number;
  };
}

export interface RegimeABTestResult {
  test_date: string | null;
  horizons: ABTestHorizonEntry[];
}

export async function getRegimeABTest(): Promise<RegimeABTestResult> {
  const response = await fetch(`${API_URL}/ab-test/regime`);
  if (!response.ok) throw new Error('Failed to fetch A/B test results');
  return response.json();
}

// Ensemble A/B Test API
export interface EnsembleABTestHorizonEntry {
  horizon_days: number;
  ens3?: {
    sample_count: number;
    metrics: {
      mae: number;
      rmse: number;
      mape: number;
      directional_accuracy: number;
      interval_coverage: number;
      fold_count: number;
    };
  };
  ens6?: {
    sample_count: number;
    metrics: {
      mae: number;
      rmse: number;
      mape: number;
      directional_accuracy: number;
      interval_coverage: number;
      fold_count: number;
    };
  };
  delta?: {
    directional_accuracy_delta_pp: number;
    mae_delta: number;
    mape_delta: number;
    interval_coverage_delta_pp: number;
    ens3_wins: boolean;
    ens3_dir_acc: number;
    ens6_dir_acc: number;
    ens3_mae: number;
    ens6_mae: number;
    ens3_mape: number;
    ens6_mape: number;
    ens3_int_cov: number;
    ens6_int_cov: number;
    ens3_sample_count: number;
    ens6_sample_count: number;
    ens3_fold_count: number;
    ens6_fold_count: number;
  };
}

export interface EnsembleABTestResult {
  test_date: string | null;
  horizons: EnsembleABTestHorizonEntry[];
}

export async function getEnsembleABTest(): Promise<EnsembleABTestResult> {
  const response = await fetch(`${API_URL}/ab-test/ensemble`);
  if (!response.ok) throw new Error('Failed to fetch ensemble A/B test results');
  return response.json();
}

// Social Sentiment API
export interface SocialMention {
  post_id: string;
  subreddit: string | null;
  post_title: string | null;
  post_score: number | null;
  sentiment_score: number | null;
  mentioned_at: string;
}

export interface SocialSentimentSummary {
  item_id: string;
  item_name: string;
  mentions_24h: number;
  mentions_7d: number;
  mention_velocity: number;
  avg_sentiment_7d: number;
  avg_score_7d: number;
  recent_mentions: SocialMention[];
}

export async function getItemSocialSentiment(itemId: string): Promise<SocialSentimentSummary> {
  const response = await fetch(`${API_URL}/items/${encodeURIComponent(itemId)}/social-sentiment`);
  if (!response.ok) throw new Error('Failed to fetch social sentiment');
  return response.json();
}

// Portfolio API
export async function getInventory() {
  const response = await fetch(`${API_URL}/portfolio/inventory`, {
    credentials: 'include',
  });
  if (!response.ok) {
    if (response.status === 401) return { error: 'unauthorized' };
    return { error: 'failed' };
  }
  return response.json();
}
