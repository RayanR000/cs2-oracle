'use client';

import { useEffect, useMemo, useState, Suspense } from 'react';
import Link from 'next/link';
import { Header, ItemCard } from '@/components';
import { getMarketSummary, getTrendingItems, GroupedMarketItem, QualityVariant } from '@/lib/api';

type SortKey = 'name' | 'priceAvg' | 'priceChange24h' | 'volatility' | 'volume24h';

interface TrendingRow {
  item_id: string;
  name: string;
  type: string;
  icon_url: string | null;
  latest_price: number;
}

const PAGE_SIZE = 100;
const TYPES = ['all', 'skin', 'case', 'sticker'] as const;
const TYPE_LABELS: Record<string, string> = { all: 'All', skin: 'Skins', case: 'Cases', sticker: 'Stickers' };

function formatCurrency(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return '\u2014';
  return `$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function formatPercent(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return '\u2014';
  return `${value > 0 ? '+' : ''}${value.toFixed(1)}%`;
}

function formatVolume(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return '\u2014';
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return `${value.toFixed(0)}`;
}

function SortArrow({ direction }: { direction: 'asc' | 'desc' }) {
  return (
    <span className="text-ink text-[10px] leading-none">
      {direction === 'desc' ? '\u25BC' : '\u25B2'}
    </span>
  );
}

function SkeletonRow() {
  return (
    <tr className="ledger-row">
      <td className="px-5 py-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xs bg-recess animate-pulse" />
          <div className="h-4 w-48 bg-recess rounded-xs animate-pulse" />
        </div>
      </td>
      <td className="px-5 py-4">
        <div className="h-4 w-12 bg-recess rounded-xs animate-pulse" />
      </td>
      <td className="px-5 py-4">
        <div className="h-4 w-20 bg-recess rounded-xs animate-pulse ml-auto" />
      </td>
      <td className="px-5 py-4">
        <div className="h-6 w-16 bg-recess rounded-xs animate-pulse ml-auto" />
      </td>
      <td className="px-5 py-4">
        <div className="h-4 w-16 bg-recess rounded-xs animate-pulse ml-auto" />
      </td>
      <td className="px-5 py-4">
        <div className="h-4 w-12 bg-recess rounded-xs animate-pulse ml-auto" />
      </td>
    </tr>
  );
}

function readUrlParams() {
  if (typeof window === 'undefined') return {};
  const params = new URLSearchParams(window.location.search);
  const result: Record<string, string> = {};
  for (const [key, value] of params) result[key] = value;
  return result;
}

function MarketPageInner() {
  const [urlParams] = useState(readUrlParams);

  const initialType = TYPES.includes(urlParams.type as typeof TYPES[number])
    ? (urlParams.type as typeof TYPES[number]) : 'all';
  const initialQ = urlParams.q ?? '';
  const initialPage = parseInt(urlParams.page ?? '0', 10) || 0;
  const initialSortBy = (urlParams.sortBy as SortKey) ?? 'priceAvg';
  const initialSortOrder = urlParams.sortOrder === 'asc' ? ('asc' as const) : ('desc' as const);

  const [items, setItems] = useState<GroupedMarketItem[]>([]);
  const [trending, setTrending] = useState<TrendingRow[]>([]);
  const [sortBy, setSortBy] = useState<SortKey>(initialSortBy);
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>(initialSortOrder);
  const [searchQuery, setSearchQuery] = useState(initialQ);
  const [debouncedQuery, setDebouncedQuery] = useState(initialQ);
  const [activeType, setActiveType] = useState<string>(initialType);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pageOffset, setPageOffset] = useState(initialPage);
  const [hasMore, setHasMore] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(searchQuery), 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  const filterKey = `${debouncedQuery}|${activeType}`;
  const [prevFilterKey, setPrevFilterKey] = useState(filterKey);
  const page = prevFilterKey === filterKey ? pageOffset : 0;

  if (prevFilterKey !== filterKey) {
    setPrevFilterKey(filterKey);
    setPageOffset(0);
  }

  const fetchKey = `${debouncedQuery}|${activeType}|${page}`;

  useEffect(() => {
    let cancelled = false;

    async function loadMarketData() {
      setError(null);
      if (!items.length) setIsLoading(true);

      try {
        const raw = await Promise.all([
          getMarketSummary(
            activeType === 'all' ? undefined : activeType,
            debouncedQuery || undefined,
            page * PAGE_SIZE,
            PAGE_SIZE + 1,
          ),
          getTrendingItems(6),
        ]);
        const summaryResponse = raw[0];
        const trendingResponse = raw[1];

        const hasNext = Array.isArray(summaryResponse) && summaryResponse.length > PAGE_SIZE;
        const pageItems = Array.isArray(summaryResponse) ? summaryResponse.slice(0, PAGE_SIZE) : [];

        const summaryItems: GroupedMarketItem[] = pageItems.map((r: Record<string, unknown>) => ({
          base_name: String(r.base_name ?? ''),
          type: String(r.type ?? ''),
          icon_url: (r.icon_url as string) ?? null,
          price_avg: (r.price_avg as number) ?? null,
          price_min: (r.price_min as number) ?? null,
          price_max: (r.price_max as number) ?? null,
          price_change_24h: (r.price_change_24h as number) ?? null,
          volatility: (r.volatility as number) ?? null,
          volume_24h: (r.volume_24h as number) ?? null,
          quality_count: (r.quality_count as number) ?? 1,
          qualities: (r.qualities as QualityVariant[]) ?? [],
        }));

        if (!cancelled) {
          setItems(summaryItems);
          setHasMore(hasNext);
          const trendingArr = Array.isArray(trendingResponse) ? trendingResponse : [];
          setTrending(trendingArr.map((item: Record<string, unknown>) => ({
            item_id: String(item.item_id ?? ''),
            name: String(item.name ?? ''),
            type: String(item.type ?? ''),
            icon_url: (item.icon_url as string) ?? null,
            latest_price: (item.latest_price as number) ?? 0,
          })));
        }
      } catch (fetchError) {
        if (!cancelled) {
          setError(fetchError instanceof Error ? fetchError.message : 'Failed to load market data');
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    loadMarketData();
    return () => { cancelled = true; };
  }, [fetchKey]);

  useEffect(() => {
    const next = new URLSearchParams();
    if (activeType !== 'all') next.set('type', activeType);
    if (searchQuery) next.set('q', searchQuery);
    if (page > 0) next.set('page', String(page));
    if (sortBy !== 'priceAvg') next.set('sortBy', sortBy);
    if (sortOrder !== 'desc') next.set('sortOrder', sortOrder);
    const qs = next.toString();
    window.history.replaceState(null, '', `${window.location.pathname}${qs ? `?${qs}` : ''}`);
  }, [activeType, searchQuery, page, sortBy, sortOrder]);

  const setAndSearch = (type: string) => {
    setActiveType(type);
  };

  const sortedItems = useMemo(() => {
    return [...items].sort((a, b) => {
      if (sortBy === 'name') {
        return sortOrder === 'asc' ? a.base_name.localeCompare(b.base_name) : b.base_name.localeCompare(a.base_name);
      }
      const fieldMap: Record<string, keyof GroupedMarketItem> = {
        priceAvg: 'price_avg',
        priceChange24h: 'price_change_24h',
        volatility: 'volatility',
        volume24h: 'volume_24h',
      };
      const field = fieldMap[sortBy] ?? sortBy;
      const aVal = a[field] as number | null;
      const bVal = b[field] as number | null;
      return sortOrder === 'asc'
        ? (aVal ?? Number.NEGATIVE_INFINITY) - (bVal ?? Number.NEGATIVE_INFINITY)
        : (bVal ?? Number.NEGATIVE_INFINITY) - (aVal ?? Number.NEGATIVE_INFINITY);
    });
  }, [items, sortBy, sortOrder]);

  const handleSort = (column: SortKey) => {
    if (sortBy === column) {
      const next = sortOrder === 'asc' ? 'desc' : 'asc';
      setSortOrder(next);
      return;
    }
    setSortBy(column);
    setSortOrder('desc');
  };

  return (
    <div className="min-h-screen bg-ground">
      <Header />

      <div className="max-w-6xl mx-auto px-6 py-10">
        {/* Catalog header */}
        <div className="mb-8">
          <span className="specimen-tag text-paper-tertiary mb-3 block">The Catalog</span>
          <h1 className="text-headline text-paper mb-2">Market</h1>
          <p className="text-base text-paper-secondary max-w-xl">
            Drawers of every specimen in the collection — search, filter by type, and open
            the card to read its strata.
          </p>
        </div>

        {/* Search + type filter */}
        <div className="flex flex-col sm:flex-row gap-4 mb-10">
          <div className="relative flex-1 max-w-[480px]">
            <div className="relative flex items-center bg-stock border border-border rounded-sm transition-colors duration-200 focus-within:border-border-accent focus-within:bg-surface">
              <svg
                className="w-4 h-4 ml-4 text-paper-muted transition-colors duration-200 shrink-0"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                type="text"
                placeholder="Search items..."
                value={searchQuery}
                onChange={(e) => { setSearchQuery(e.target.value); }}
                className="flex-1 h-11 px-4 bg-transparent text-sm text-paper placeholder:text-paper-muted outline-none"
              />
            </div>
          </div>
          <div className="flex gap-1.5 items-center">
            {TYPES.map((t) => (
              <button
                key={t}
                onClick={() => setAndSearch(t)}
                className={`wear-pill text-[11px] font-semibold uppercase tracking-[0.12em] ${
                  activeType === t ? 'wear-pill-active' : ''
                }`}
              >
                {TYPE_LABELS[t]}
              </button>
            ))}
          </div>
        </div>

        {/* Specimen grid */}
        <div className="mb-10">
          <h2 className="text-title text-paper mb-5">Featured Specimens</h2>
          {trending.length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {trending.slice(0, 6).map((item) => (
                <ItemCard
                  key={item.item_id}
                  itemId={item.item_id}
                  name={item.name}
                  type={item.type}
                  imageUrl={item.icon_url ?? undefined}
                  currentPrice={item.latest_price}
                />
              ))}
            </div>
          ) : !isLoading ? (
            <div className="bg-stock border border-border rounded-sm px-5 py-4 text-sm text-paper-secondary">
              No featured items available.
            </div>
          ) : null}
        </div>

        {/* Error banner */}
        {error && (
          <div
            className="mb-6 rounded-sm border px-4 py-3 text-sm flex items-center justify-between"
            style={{
              backgroundColor: 'var(--down-subtle)',
              borderColor: 'color-mix(in oklch, var(--down) 30%, transparent)',
              color: 'var(--down)',
            }}
          >
            <span>{error}</span>
            <button
              onClick={() => {
                setError(null);
                setIsLoading(true);
              }}
              className="specimen-tag hover:opacity-80 transition-opacity"
            >
              Retry
            </button>
          </div>
        )}

        {/* Ledger table */}
        <div className="rounded-sm border border-border bg-stock">
          {isLoading && items.length > 0 && (
            <div className="h-0.5 w-full bg-ink/40 overflow-hidden rounded-full">
              <div className="h-full w-1/3 bg-ink animate-pulse" />
            </div>
          )}

          <div className="overflow-auto max-h-[70vh] rounded-sm">
            <table className="w-full text-sm min-w-[720px]">
              <thead className="sticky top-0 z-10">
                <tr className="bg-recess border-b border-border">
                  <th className="px-5 py-4 text-left w-[36%]">
                    <button onClick={() => handleSort('name')} className="specimen-tag text-paper-secondary hover:text-paper transition-colors duration-200 flex items-center gap-2">
                      Item {sortBy === 'name' && <SortArrow direction={sortOrder} />}
                    </button>
                  </th>
                  <th className="px-5 py-4 text-left w-[9%]">
                    <span className="specimen-tag text-paper-secondary">Type</span>
                  </th>
                  <th className="px-5 py-4 text-right w-[15%]">
                    <button onClick={() => handleSort('priceAvg')} className="specimen-tag text-paper-secondary hover:text-paper transition-colors duration-200 w-full flex justify-end items-center gap-2">
                      Avg Price {sortBy === 'priceAvg' && <SortArrow direction={sortOrder} />}
                    </button>
                  </th>
                  <th className="px-5 py-4 text-right w-[14%]">
                    <span className="specimen-tag text-paper-muted">Range</span>
                  </th>
                  <th className="px-5 py-4 text-right w-[11%]">
                    <button onClick={() => handleSort('priceChange24h')} className="specimen-tag text-paper-secondary hover:text-paper transition-colors duration-200 w-full flex justify-end items-center gap-2">
                      24h {sortBy === 'priceChange24h' && <SortArrow direction={sortOrder} />}
                    </button>
                  </th>
                  <th className="px-5 py-4 text-right w-[9%]">
                    <button onClick={() => handleSort('volume24h')} className="specimen-tag text-paper-secondary hover:text-paper transition-colors duration-200 w-full flex justify-end items-center gap-2">
                      Vol {sortBy === 'volume24h' && <SortArrow direction={sortOrder} />}
                    </button>
                  </th>
                  <th className="px-5 py-4 text-center w-[6%]">
                    <span className="specimen-tag text-paper-secondary">Quals</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {isLoading && !items.length ? (
                  Array.from({ length: 10 }).map((_, i) => <SkeletonRow key={i} />)
                ) : sortedItems.length ? (
                  sortedItems.map((item) => {
                    const firstVariant = item.qualities[0];
                    const linkId = firstVariant?.item_id ?? item.base_name;
                    return (
                      <tr key={item.base_name} className="ledger-row">
                        <td className="px-5 py-4">
                          <Link
                            href={`/items/${encodeURIComponent(linkId)}`}
                            className="flex items-center gap-3 font-medium text-paper transition-colors duration-200 hover:text-ink"
                          >
                            {item.icon_url ? (
                              <img src={item.icon_url} alt={item.base_name} className="w-8 h-8 rounded-xs object-cover shrink-0" loading="lazy" />
                            ) : (
                              <div className="w-8 h-8 rounded-xs bg-recess shrink-0" />
                            )}
                            <span className="truncate">{item.base_name}</span>
                          </Link>
                        </td>
                        <td className="px-5 py-4 text-left specimen-tag text-paper-tertiary">
                          {item.type}
                        </td>
                        <td className="px-5 py-4 text-right font-data font-medium text-paper">
                          {formatCurrency(item.price_avg)}
                        </td>
                        <td className="px-5 py-4 text-right font-data text-xs text-paper-tertiary">
                          {item.price_min != null && item.price_max != null
                            ? `${formatCurrency(item.price_min)} \u2013 ${formatCurrency(item.price_max)}`
                            : '\u2014'}
                        </td>
                        <td className="px-5 py-4 text-right">
                          <span
                            className="inline-block px-1.5 py-0.5 rounded-xs font-data text-[11px] font-semibold"
                            style={{
                              backgroundColor: (item.price_change_24h ?? 0) >= 0 ? 'var(--up-subtle)' : 'var(--down-subtle)',
                              color: (item.price_change_24h ?? 0) >= 0 ? 'var(--up)' : 'var(--down)',
                            }}
                          >
                            {formatPercent(item.price_change_24h)}
                          </span>
                        </td>
                        <td className="px-5 py-4 text-right font-data text-paper-secondary">
                          {formatVolume(item.volume_24h)}
                        </td>
                        <td className="px-5 py-4 text-center">
                          <span className="inline-block px-1.5 py-0.5 rounded-xs bg-recess font-data text-xs text-paper-secondary">
                            {item.quality_count}
                          </span>
                        </td>
                      </tr>
                    );
                  })
                ) : (
                  <tr>
                    <td colSpan={7} className="px-6 py-16 text-center">
                      <div className="flex flex-col items-center gap-3">
                        <span className="text-3xl text-paper-tertiary">{'\u2205'}</span>
                        {isLoading ? (
                          <p className="text-paper-secondary font-medium">Searching...</p>
                        ) : (
                          <>
                            <p className="text-paper-secondary font-medium">No items match your search</p>
                            <p className="text-xs text-paper-tertiary max-w-md">
                              Try broadening your search terms, or clear the filter to browse all items.
                            </p>
                            <button
                              onClick={() => { setSearchQuery(''); setDebouncedQuery(''); setActiveType('all'); setPageOffset(0); }}
                              className="btn btn-ghost mt-2"
                            >
                              Clear All Filters
                            </button>
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="flex items-center justify-between px-5 py-4 border-t border-divider">
            <button
              onClick={() => setPageOffset(p => Math.max(0, p - 1))}
              disabled={page === 0}
              className="btn btn-ghost disabled:opacity-30 disabled:cursor-not-allowed"
            >
              {'\u2190'} Previous
            </button>
            <span className="font-data text-xs text-paper-tertiary">Page {page + 1}</span>
            <button
              onClick={() => setPageOffset(p => p + 1)}
              disabled={!hasMore}
              className="btn btn-ghost disabled:opacity-30 disabled:cursor-not-allowed"
            >
              Next {'\u2192'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function MarketPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-ground flex items-center justify-center">
        <div className="h-0.5 w-48 bg-ink/40 rounded-full overflow-hidden">
          <div className="h-full w-1/3 bg-ink animate-pulse" />
        </div>
      </div>
    }>
      <MarketPageInner />
    </Suspense>
  );
}
