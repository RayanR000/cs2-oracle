'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useEffect, useState, useMemo, useRef } from 'react';
import { Header, ItemCard, Search, ExhibitStrata } from '@/components';
import {
  getTrendingItems,
  getItemsCount,
  searchItems,
  getPriceHistory,
  getItemPrediction,
  getAccuracyHeadline,
  type TrendingItem,
  type AccuracyHeadline,
  type AccuracyHeadlineHorizon,
  type HeadlineVerdict,
  type PricePoint,
} from '@/lib/api';

type LoadStatus = 'loading' | 'ready' | 'unavailable';

interface ExhibitData {
  history: PricePoint[];
  prediction: { low: number; mid: number; high: number } | null;
}

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function formatEvalDate(iso: string | null): string {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function formatLastClose(iso: string | null): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

/**
 * How each Pesaran-Timmermann verdict is stated on the placard.
 *
 * `no_skill` is deliberately not coloured as a failure and not softened into a
 * pending state: it is a completed measurement with a null result, and the
 * whole point of publishing a test rather than a hit rate is that the null is
 * sayable. `perverse` is the only red — it is a real, signed finding.
 */
const VERDICT_COPY: Record<
  HeadlineVerdict,
  { label: string; note: string; tone: 'up' | 'down' | 'neutral' }
> = {
  skill: {
    label: 'Skill',
    note: 'hit rate beats the per-date chance null',
    tone: 'up',
  },
  no_skill: {
    label: 'No skill',
    note: 'indistinguishable from the per-date chance null',
    tone: 'neutral',
  },
  perverse: {
    label: 'Anti-correlated',
    note: 'calls are significantly wrong, net of the market',
    tone: 'down',
  },
  insufficient_dates: {
    label: 'Untestable',
    note: 'too few forecast dates to separate model from market',
    tone: 'neutral',
  },
  degenerate: {
    label: 'Untestable',
    note: 'calls never vary, so the test has nothing to measure',
    tone: 'neutral',
  },
  untested: {
    label: 'Not tested',
    note: 'this run predates the significance test',
    tone: 'neutral',
  },
};

/**
 * The horizon shown on the placard. Fixed at 7d — the product's primary — and
 * NOT the best-scoring one, because picking the strongest verdict out of four
 * horizons is the multiple-testing problem the t > 3 hurdle exists to price in.
 */
const PLACARD_HORIZON_DAYS = 7;

function pickPlacardHorizon(
  headline: AccuracyHeadline | null
): AccuracyHeadlineHorizon | null {
  if (!headline?.horizons?.length) return null;
  return (
    headline.horizons.find((h) => h.horizon_days === PLACARD_HORIZON_DAYS) ??
    headline.horizons[0]
  );
}

export default function Home() {
  const router = useRouter();
  const [trending, setTrending] = useState<TrendingItem[]>([]);
  const [itemsCount, setItemsCount] = useState<number | null>(null);
  const [marketStatus, setMarketStatus] = useState<LoadStatus>('loading');
  const [headline, setHeadline] = useState<AccuracyHeadline | null>(null);
  const [accuracyStatus, setAccuracyStatus] = useState<LoadStatus>('loading');
  const [exhibitData, setExhibitData] = useState<ExhibitData | null>(null);
  const [exhibitStatus, setExhibitStatus] = useState<LoadStatus>('loading');
  const [exhibitVisible, setExhibitVisible] = useState(true);
  const exhibitRef = useRef<HTMLAnchorElement>(null);

  useEffect(() => {
    let cancelled = false;

    async function fetchMarketData() {
      const [trendingRes, countRes] = await Promise.allSettled([
        getTrendingItems(4),
        getItemsCount(),
      ]);
      if (cancelled) return;

      if (trendingRes.status === 'fulfilled' && Array.isArray(trendingRes.value)) {
        setTrending(trendingRes.value.slice(0, 4));
      }
      if (countRes.status === 'fulfilled') {
        setItemsCount(countRes.value);
      }
      setMarketStatus(
        trendingRes.status === 'fulfilled' || countRes.status === 'fulfilled' ? 'ready' : 'unavailable'
      );
    }

    async function fetchAccuracy() {
      try {
        const res = await getAccuracyHeadline();
        if (cancelled) return;
        // A verdict is the minimum for the placard. Interval coverage may be
        // absent on an older row without making the verdict unpublishable.
        if (pickPlacardHorizon(res)) {
          setHeadline(res);
          setAccuracyStatus('ready');
        } else {
          setAccuracyStatus('unavailable');
        }
      } catch {
        if (!cancelled) setAccuracyStatus('unavailable');
      }
    }

    fetchMarketData();
    fetchAccuracy();

    return () => {
      cancelled = true;
    };
  }, []);

  const heroItem = useMemo(() => {
    if (trending.length === 0) return null;
    return trending.find((i) => i.type === 'skin' || i.type === 'knife') ?? trending[0];
  }, [trending]);

  useEffect(() => {
    if (!heroItem) return;
    const itemId = heroItem.item_id;
    let cancelled = false;

    async function fetchExhibitData() {
      setExhibitStatus('loading');
      const [histRes, predRes] = await Promise.allSettled([
        getPriceHistory(itemId, 120, 0, 120),
        getItemPrediction(itemId, '7_days'),
      ]);
      if (cancelled) return;

      const history = histRes.status === 'fulfilled' && Array.isArray(histRes.value)
        ? (histRes.value as PricePoint[])
        : [];

      let prediction: ExhibitData['prediction'] = null;
      if (predRes.status === 'fulfilled' && predRes.value && !Array.isArray(predRes.value)) {
        const p = predRes.value as Record<string, unknown>;
        const low = p.forecast_low;
        const mid = p.forecast_mid;
        const high = p.forecast_high;
        if (typeof low === 'number' && typeof mid === 'number' && typeof high === 'number') {
          prediction = { low, mid, high };
        }
      }

      if (cancelled) return;
      setExhibitData({ history, prediction });
      setExhibitStatus('ready');
    }

    fetchExhibitData();
    return () => {
      cancelled = true;
    };
  }, [heroItem]);

  useEffect(() => {
    const el = exhibitRef.current;
    if (!el || typeof IntersectionObserver === 'undefined') return;
    const io = new IntersectionObserver(([entry]) => {
      setExhibitVisible(entry.isIntersecting);
    });
    io.observe(el);
    return () => io.disconnect();
  }, [heroItem]);

  const sideItems = useMemo(
    () => trending.filter((i) => i.item_id !== heroItem?.item_id).slice(0, 3),
    [trending, heroItem]
  );

  const lastCloseLabel = useMemo(() => {
    const h = exhibitData?.history ?? [];
    if (!h.length) return null;
    const sorted = [...h].sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );
    return formatLastClose(sorted[sorted.length - 1].timestamp);
  }, [exhibitData]);

  const placard = pickPlacardHorizon(headline);
  const verdict = placard ? VERDICT_COPY[placard.verdict] ?? VERDICT_COPY.untested : null;
  const coverage = placard ? num(placard.interval_coverage) : null;
  const directionalAccuracy = placard ? num(placard.directional_accuracy) : null;
  const constantCall = placard ? num(placard.constant_call_accuracy) : null;
  const excessPp = placard ? num(placard.pt_excess_pp) : null;
  const tStat = placard ? num(placard.pt_t_stat) : null;

  return (
    <div className="min-h-screen bg-ground">
      <Header />

      <main className="max-w-6xl mx-auto px-6">
        {/* --- SURFACE ZONE: the depth axis rules every line below --- */}
        <section className="pt-14 pb-14 lg:pt-20 lg:pb-16">
          <div className="lg:grid lg:grid-cols-[56px_minmax(0,1fr)] lg:gap-8">
            {/* Depth ruler — the instrument's one true axis */}
            <div aria-hidden className="hidden lg:block relative">
              <div className="absolute top-0 bottom-0 left-6 w-px bg-grid">
                {[
                  { top: '0%', label: 'SURFACE' },
                  { top: '33.33%', label: 'THERMOCLINE' },
                  { top: '66.66%', label: 'MESOPHOTIC' },
                  { top: '100%', label: 'THE FLOOR' },
                ].map((tick) => (
                  <div key={tick.label} className="absolute -translate-y-1/2" style={{ top: tick.top }}>
                    <span className="absolute top-0 left-0 w-2 h-px bg-border" />
                    <span
                      className="absolute top-0 right-2 whitespace-nowrap font-data text-[9px] tracking-widest text-paper-muted leading-none"
                      style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', transformOrigin: 'center' }}
                    >
                      {tick.label}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="min-w-0">
              {/* Placard */}
              <span className="dive-tag text-paper-tertiary mb-6 block">The Deep Dive</span>
              <h1 className="text-display text-paper mb-6">
                Thirteen years of CS2 prices, measured at depth.
              </h1>
              <p className="text-base text-paper-secondary max-w-lg leading-relaxed mb-8">
                Every item is a dive &mdash; its price history read as a profile through
                the water, its forecast drawn as the thermocline band where certainty
                changes, and the instrument&rsquo;s own accuracy kept in the same dive log.
              </p>

              {/* Finding aid */}
              <Search
                fetchOptions={(query) => searchItems(query)}
                onSelect={(option) => router.push(`/items/${encodeURIComponent(option.item_id)}`)}
              />

              {/* The surface reading — the featured dive under the drifting snow */}
              <div className="mt-10">
                {marketStatus === 'loading' ? (
                  <div className="dive-card overflow-hidden">
                    <div className="flex items-center justify-between px-4 py-2.5 border-b border-divider">
                      <div className="h-2.5 w-28 bg-surface rounded-xs animate-pulse" />
                      <div className="h-2.5 w-24 bg-surface rounded-xs animate-pulse" />
                    </div>
                    <div className="grid md:grid-cols-[260px_minmax(0,1fr)]">
                      <div className="aspect-square md:aspect-auto bg-recess flex items-center justify-center">
                        <div className="w-24 h-24 bg-surface rounded-xs animate-pulse" />
                      </div>
                      <div className="p-4 flex flex-col gap-3">
                        <div className="h-3 w-36 bg-surface rounded-xs animate-pulse" />
                        <div className="h-40 bg-recess rounded-xs animate-pulse" />
                      </div>
                    </div>
                  </div>
                ) : heroItem ? (
                  <Link
                    href={`/items/${encodeURIComponent(heroItem.item_id)}`}
                    ref={exhibitRef}
                    className={`group block relative ${exhibitVisible ? '' : 'exhibit-hidden'}`}
                  >
                    <div className="dive-card overflow-hidden">
                      <span aria-hidden className="sounding-rule" />
                      <span aria-hidden className="card-sweep" />
                      <div className="flex items-center justify-between px-4 py-2.5 border-b border-divider">
                        <span className="dive-tag text-paper-tertiary">Featured Dive</span>
                        <span className="flex items-center gap-2 dive-tag text-paper-muted">
                          <span aria-hidden className="w-1.5 h-1.5 rounded-full bg-paper-muted sonar-dot inline-block" />
                          7 markets &middot; {lastCloseLabel ? `last close ${lastCloseLabel}` : 'daily'}
                        </span>
                      </div>
                      <div className="grid md:grid-cols-[260px_minmax(0,1fr)]">
                        <div className="relative bg-recess overflow-hidden aspect-square md:aspect-auto">
                          {heroItem.icon_url ? (
                            <img
                              src={heroItem.icon_url}
                              alt={heroItem.name}
                              fetchPriority="high"
                              className="w-full h-full object-contain p-8 transition-transform duration-500 ease-out group-hover:scale-[1.03]"
                            />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center p-8">
                              <div className="dive-tag text-paper-muted">{heroItem.name}</div>
                            </div>
                          )}
                          <div aria-hidden className="snow-drift" />
                        </div>
                        <div className="flex flex-col min-w-0">
                          <div className="flex items-end justify-between gap-4 px-4 pt-3.5 pb-1">
                            <div className="min-w-0">
                              <div className="text-[13px] font-semibold text-paper mb-1 truncate">
                                {heroItem.name}
                              </div>
                              <div className="dive-tag text-paper-tertiary">{heroItem.type}</div>
                            </div>
                            <div className="font-data text-2xl text-paper shrink-0">
                              ${heroItem.latest_price?.toFixed(2)}
                            </div>
                          </div>
                          <ExhibitStrata
                            history={exhibitData?.history ?? []}
                            forecast={exhibitData?.prediction ?? null}
                            loading={exhibitStatus === 'loading'}
                            height={170}
                          />
                        </div>
                      </div>
                    </div>
                  </Link>
                ) : (
                  <div className="dive-card overflow-hidden">
                    <div className="px-4 py-2.5 border-b border-divider">
                      <span className="dive-tag text-paper-tertiary">Featured Dive</span>
                    </div>
                    <div className="aspect-square bg-recess flex items-center justify-center p-10">
                      <span className="dive-tag text-paper-muted">No dive on record</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* The collection's headline record — published accuracy, cohort and horizon named */}
          <div className="mt-12 lg:mt-16 lg:ml-24">
            {accuracyStatus === 'loading' ? (
              <div className="bg-stock border border-border rounded-sm">
                <div className="flex flex-col md:flex-row gap-6 p-6">
                  <div className="h-5 w-32 bg-surface animate-pulse rounded-xs" />
                  <div className="h-5 w-44 bg-surface animate-pulse rounded-xs" />
                  <div className="h-5 w-28 bg-surface animate-pulse rounded-xs" />
                </div>
              </div>
            ) : placard && verdict && headline ? (
              <Link
                href="/accuracy"
                className="group block bg-stock border border-border rounded-sm transition-colors duration-250 hover:border-border-accent hover:bg-surface"
              >
                <div className="flex flex-col md:flex-row md:items-stretch divide-y md:divide-y-0 md:divide-x divide-divider">
                  {/* The published claim is the TEST, not the hit rate. */}
                  <div className="px-6 py-5 md:w-72">
                    <div className="dive-tag text-paper-tertiary mb-2">
                      Directional Verdict
                    </div>
                    <div
                      className="text-data-lg"
                      style={{
                        color:
                          verdict.tone === 'up' ? 'var(--up)' :
                          verdict.tone === 'down' ? 'var(--down)' :
                          'var(--paper)',
                      }}
                    >
                      {verdict.label}
                      {tStat !== null && (
                        <span className="text-data-sm text-paper-tertiary ml-2">
                          t&thinsp;=&thinsp;{tStat.toFixed(2)}
                        </span>
                      )}
                    </div>
                    <div className="text-[10px] font-data text-paper-muted mt-1">
                      {verdict.note}
                    </div>
                  </div>
                  {/* Accuracy travels only as the triple that makes it readable. */}
                  <div className="px-6 py-5 flex-1">
                    <div className="dive-tag text-paper-tertiary mb-2">
                      Hit Rate vs Chance
                    </div>
                    <div className="text-data-lg text-paper">
                      {directionalAccuracy !== null ? `${directionalAccuracy.toFixed(1)}%` : '\u2014'}
                      {constantCall !== null && (
                        <span className="text-data-sm text-paper-tertiary ml-2">
                          vs {constantCall.toFixed(1)}% always-{placard.constant_call_direction ?? 'flat'}
                        </span>
                      )}
                    </div>
                    <div className="text-[10px] font-data text-paper-muted mt-1">
                      {excessPp !== null
                        ? `${excessPp >= 0 ? '+' : ''}${excessPp.toFixed(2)}pp over the per-date chance null`
                        : 'no per-date null could be formed'}
                    </div>
                  </div>
                  <div className="px-6 py-5 md:w-44">
                    <div className="dive-tag text-paper-tertiary mb-2">Coverage</div>
                    <div className="text-data-lg text-paper">
                      {coverage !== null ? `${coverage.toFixed(1)}%` : '\u2014'}
                    </div>
                    <div className="text-[10px] font-data text-paper-muted mt-1">
                      q10&ndash;q90 interval
                    </div>
                  </div>
                  <div className="px-6 py-5 md:w-44">
                    <div className="dive-tag text-paper-tertiary mb-2">Cohort</div>
                    <div className="text-data-lg text-paper">{headline.cohort}</div>
                    <div className="text-[10px] font-data text-paper-muted mt-1">
                      {placard.horizon_days != null ? `${placard.horizon_days}d` : '\u2014'}
                      {' \u00b7 '}
                      {placard.pt_n_dates ?? 0} dates
                      {' \u00b7 '}
                      {formatEvalDate(placard.evaluation_date)}
                    </div>
                  </div>
                  <div className="px-6 py-5 md:w-44 flex items-center">
                    <span className="dive-tag text-ink group-hover:text-ink-hover transition-colors duration-200">
                      Full Backtest &rarr;
                    </span>
                  </div>
                </div>
              </Link>
            ) : (
              <div className="bg-stock border border-border rounded-sm px-6 py-5">
                <span className="dive-tag text-paper-tertiary">Published Accuracy</span>
                <p className="text-sm text-paper-secondary mt-2">
                  No backtest on record yet — the dive log records its errors at the same
                  depth as its findings.
                </p>
              </div>
            )}
          </div>
        </section>

        {/* --- FEATURED SPECIMENS --- */}
        <section className="pb-14 lg:pb-16">
          <div className="flex items-end justify-between mb-6">
            <h2 className="text-title text-paper">Featured Dives</h2>
            <Link
              href="/market"
              className="dive-tag text-ink hover:text-ink-hover transition-colors duration-200"
            >
              View the Catalog &rarr;
            </Link>
          </div>

          {marketStatus === 'loading' ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {[0, 1, 2].map((i) => (
                <div key={i} className="bg-stock border border-border rounded-sm overflow-hidden">
                  <div className="aspect-square bg-surface animate-pulse" />
                  <div className="p-4 flex flex-col gap-2">
                    <div className="h-3 w-16 bg-surface rounded-xs animate-pulse" />
                    <div className="h-4 w-32 bg-surface rounded-xs animate-pulse" />
                  </div>
                </div>
              ))}
            </div>
          ) : marketStatus === 'unavailable' || sideItems.length === 0 ? (
            <div className="bg-stock border border-border rounded-sm p-10 text-center">
              <p className="text-sm text-paper-secondary">
                {marketStatus === 'unavailable'
                  ? 'Market data is unavailable right now — check back after the next daily archive update.'
                  : 'No market data yet — check back after the next daily archive update.'}
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {sideItems.map((item) => (
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
          )}
        </section>

        {/* --- HOLDINGS SUMMARY --- */}
        <section className="pb-16 lg:pb-24 border-t border-divider pt-12">
          <h2 className="text-title text-paper mb-6">The Ascent Stops</h2>
          <div className="bg-stock border border-border rounded-sm">
            <div className="flex flex-col md:flex-row divide-y md:divide-y-0 md:divide-x divide-divider">
              <div className="px-6 py-5 flex-1">
                <div className="dive-tag text-paper-tertiary mb-3">Items Tracked</div>
                <div className="text-data-lg text-paper">
                  {itemsCount !== null ? itemsCount.toLocaleString() : '\u2014'}
                </div>
              </div>
              <div className="px-6 py-5 flex-1">
                <div className="dive-tag text-paper-tertiary mb-3">Markets</div>
                <div className="text-data-lg text-paper">7</div>
                <div className="text-[10px] font-data text-paper-muted mt-1">daily closes</div>
              </div>
              <div className="px-6 py-5 flex-1">
                <div className="dive-tag text-paper-tertiary mb-3">Archive Depth</div>
                <div className="text-data-lg text-paper">13 yrs</div>
                <div className="text-[10px] font-data text-paper-muted mt-1">of price depth</div>
              </div>
              <div className="px-6 py-5 flex-1">
                <div className="dive-tag text-paper-tertiary mb-3">Cadence</div>
                <div className="text-data-lg text-paper">Daily</div>
                <div className="text-[10px] font-data text-paper-muted mt-1">archived at 23:00 UTC</div>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* --- FOOTER --- */}
      <footer className="border-t border-divider">
        <div className="max-w-6xl mx-auto px-6 py-10">
          <div className="flex flex-col sm:flex-row justify-between items-start gap-6">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-xs border border-border bg-stock flex items-center justify-center">
                <span className="font-data font-bold text-[7px] text-paper tracking-tighter">CS</span>
              </div>
              <span className="dive-tag text-paper-secondary">CS2 Oracle</span>
            </div>
            <div className="flex gap-8">
              <Link href="/market" className="dive-tag text-paper-tertiary hover:text-paper transition-colors duration-200">Market</Link>
              <Link href="/portfolio" className="dive-tag text-paper-tertiary hover:text-paper transition-colors duration-200">Portfolio</Link>
              <Link href="/accuracy" className="dive-tag text-paper-tertiary hover:text-paper transition-colors duration-200">Accuracy</Link>
            </div>
          </div>
          <div className="mt-8 pt-6 border-t border-divider">
            <p className="dive-tag text-paper-muted">
              &copy; {new Date().getFullYear()} CS2 Oracle
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
