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
  sma_7?: number;
  sma_30?: number;
  volatility?: number;
  trend_score?: number;
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

export async function getAccuracy(
  predictionType?: string,
  limit = 50
): Promise<AccuracyRecord[]> {
  const url = new URL(`${API_URL}/accuracy/`);
  if (predictionType) url.searchParams.append('prediction_type', predictionType);
  url.searchParams.append('limit', limit.toString());
  const response = await fetch(url.toString());
  if (!response.ok) throw new Error('Failed to fetch accuracy');
  return response.json();
}

export async function getLatestAccuracy(predictionType?: string) {
  const url = new URL(`${API_URL}/accuracy/latest`);
  if (predictionType) url.searchParams.append('prediction_type', predictionType);
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
