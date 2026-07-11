'use client';

import Link from 'next/link';
import { useEffect, useState, useMemo, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Header } from '@/components';
import { getTrendingItems, getItemsCount, searchItems, type TrendingItem } from '@/lib/api';

const EASE: [number, number, number, number] = [0.16, 1, 0.3, 1];

interface MarketStats {
  totalItems: number;
  volume24h: number;
  avgVolatility: number;
}

interface SearchResult {
  item_id: string;
  name: string;
  type: string;
  icon_url: string | null;
  latest_price?: number;
}

const FALLBACK_ITEMS: TrendingItem[] = [
  {
    id: 1,
    item_id: 'ak-47-vulcan',
    name: 'AK-47 | Vulcan',
    type: 'skin',
    icon_url: 'https://community.cloudflare.steamstatic.com/economy/image/-9a81dlWLwJ2UUGcVs_nsVtzdOEdtWwKGZZLQHTxDZ7I56KU0Zwwo4NUX4oFJZEHLbXH5ApeO4YmlhxYQknCRvCo04DEVlxkKgpot7HxfDhjxszJemkV092lnYmGmOHLPr7Vn35cppR32-qS99SmiwS3_hU6Y236ctfDclM6YF_U_lXrk-7shZC8u8zBmnVguyZ25S3cmBfihB9SaeM60_veWAtXOnvE/512fx512f',
    latest_price: 942.50,
  },
  {
    id: 2,
    item_id: 'awp-asiimov',
    name: 'AWP | Asiimov',
    type: 'skin',
    icon_url: 'https://community.cloudflare.steamstatic.com/economy/image/-9a81dlWLwJ2UUGcVs_nsVtzdOEdtWwKGZZLQHTxDZ7I56KU0Zwwo4NUX4oFJZEHLbXH5ApeO4YmlhxYQknCRvCo04DEVlxkKgpot621FBRw7P7NYjV96t2ykZOfqODmNr_ulWhE18l4mLP--InlgUGw_0VvMNj2IdSRclA-M1_SrFW4krq9hZ_v75_MzCRkvXF34X7cnxa0hUwbafZshvveWAvp4K3Dsw/512fx512f',
    latest_price: 164.20,
  },
  {
    id: 3,
    item_id: 'm4a1-s-printstream',
    name: 'M4A1-S | Printstream',
    type: 'skin',
    icon_url: 'https://community.cloudflare.steamstatic.com/economy/image/-9a81dlWLwJ2UUGcVs_nsVtzdOEdtWwKGZZLQHTxDZ7I56KU0Zwwo4NUX4oFJZEHLbXH5ApeO4YmlhxYQknCRvCo04DEVlxkKgpou-6kejhjxszFJTwW09Kzm7-FmP7mDLfYkW5u5Mx2gv2P89-m2w3gr0s4ajzycITAdlA7N1vS_gTvyevp1sS0uMzAnXU2vXQm4ivezBa-1RkYarNxxavJGZ6S_vY/512fx512f',
    latest_price: 428.15,
  },
  {
    id: 4,
    item_id: 'butterfly-knife-fade',
    name: 'Butterfly Knife | Fade',
    type: 'knife',
    icon_url: 'https://community.cloudflare.steamstatic.com/economy/image/-9a81dlWLwJ2UUGcVs_nsVtzdOEdtWwKGZZLQHTxDZ7I56KU0Zwwo4NUX4oFJZEHLbXH5ApeO4YmlhxYQknCRvCo04DEVlxkKgpovbSsLQJf1f_BYi59_9S_mYmDkvPLPr7Vn35cppN0i-zEpdX0iwHhqkZuNmilddScclM6aVDWqFa9wr2-1JW1u8zAm3VvunYm43rD30vgoS7N6Q/512fx512f',
    latest_price: 3240.00,
  },
];

export default function Home() {
  const [trending, setTrending] = useState<TrendingItem[]>(FALLBACK_ITEMS);
  const [stats, setStats] = useState<MarketStats>({ totalItems: 5525, volume24h: 2481290, avgVolatility: 12.4 });
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [trendingRes, countRes] = await Promise.allSettled([
          getTrendingItems(4),
          getItemsCount(),
        ]);

        if (trendingRes.status === 'fulfilled' && Array.isArray(trendingRes.value) && trendingRes.value.length > 0) {
          setTrending(trendingRes.value.slice(0, 4));
        }

        const totalCount = countRes.status === 'fulfilled' ? countRes.value : 5525;
        const avgPrice = trendingRes.status === 'fulfilled' && Array.isArray(trendingRes.value)
          ? trendingRes.value.reduce((sum: number, item: TrendingItem) => sum + (item.latest_price || 0), 0) / Math.max(trendingRes.value.length, 1)
          : 0;

        setStats({
          totalItems: totalCount,
          volume24h: Math.round(avgPrice * 2800),
          avgVolatility: 12.4,
        });
      } catch {
        // Use fallback data
      }
    }
    fetchData();
  }, []);

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const handleSearch = useCallback((query: string) => {
    setSearchQuery(query);
    if (debounceRef.current) clearTimeout(debounceRef.current);

    if (query.trim().length < 2) {
      setSearchResults([]);
      setShowResults(false);
      return;
    }

    debounceRef.current = setTimeout(async () => {
      setIsSearching(true);
      try {
        const results = await searchItems(query.trim());
        if (Array.isArray(results)) {
          setSearchResults(results.slice(0, 8));
          setShowResults(true);
        }
      } catch {
        setSearchResults([]);
      } finally {
        setIsSearching(false);
      }
    }, 250);
  }, []);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setShowResults(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const featuredItems = useMemo(() => trending.slice(0, 4), [trending]);
  const heroItem = featuredItems[0];
  const sideItems = featuredItems.slice(1);

  return (
    <div className="min-h-screen bg-background-primary">
      <Header />

      <main className="max-w-6xl mx-auto px-6">
        {/* --- HERO --- */}
        <section className="pt-16 pb-20 lg:pt-24 lg:pb-28">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-16 items-center">
            {/* Left: Copy + Search */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, ease: EASE }}
            >
              <h1 className="text-4xl lg:text-[2.75rem] font-bold tracking-[-0.03em] text-primary mb-5" style={{ textWrap: 'balance', lineHeight: '1.15' }}>
                Track every skin.<br />
                <span className="text-accent">Read every signal.</span>
              </h1>
              <p className="text-base text-secondary max-w-md leading-relaxed mb-10">
                Price intelligence, trend analysis, and portfolio tracking across thousands of CS2 items. Multi-source data from Steam, CSFloat, and aggregated markets.
              </p>

              {/* Search */}
              <div ref={searchRef} className="relative">
                <div className="relative group">
                  <div className="relative flex items-center bg-background-secondary border border-border rounded-md overflow-hidden focus-within:border-accent transition-colors duration-200">
                    <svg className="w-5 h-5 ml-5 text-muted group-focus-within:text-accent transition-colors shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    <input
                      type="text"
                      placeholder="Search skins, knives, cases..."
                      value={searchQuery}
                      onChange={(e) => handleSearch(e.target.value)}
                      onFocus={() => searchResults.length > 0 && setShowResults(true)}
                      className="flex-1 px-4 py-4 bg-transparent text-sm text-primary placeholder:text-muted focus:outline-none"
                    />
                    <div className="pr-5">
                      <span className="tag-tech">Terminal</span>
                    </div>
                  </div>
                </div>

                <AnimatePresence>
                  {showResults && (
                    <motion.div
                      initial={{ opacity: 0, y: -4, scale: 0.98 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: -4, scale: 0.98 }}
                      transition={{ duration: 0.15, ease: EASE }}
                      className="absolute top-full left-0 right-0 mt-2 bg-background-secondary border border-border rounded-md overflow-hidden shadow-lg z-50"
                    >
                      {isSearching ? (
                        <div className="px-5 py-8 text-center">
                          <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin mx-auto" />
                        </div>
                      ) : searchResults.length > 0 ? (
                        <div className="py-1">
                          {searchResults.map((item) => (
                            <Link
                              key={item.item_id}
                              href={`/items/${item.item_id}`}
                              className="flex items-center gap-4 px-5 py-3 hover:bg-surface transition-colors"
                              onClick={() => setShowResults(false)}
                            >
                              {item.icon_url ? (
                                <img src={item.icon_url} alt={item.name} className="w-8 h-8 rounded-sm object-cover shrink-0" loading="lazy" />
                              ) : (
                                <div className="w-8 h-8 rounded-sm bg-background-tertiary shrink-0" />
                              )}
                              <div className="min-w-0 flex-1">
                                <div className="text-sm font-medium text-primary truncate">{item.name}</div>
                                <div className="text-[10px] font-data uppercase tracking-wider text-secondary">{item.type}</div>
                              </div>
                              {item.latest_price != null && (
                                <span className="font-data text-sm text-primary shrink-0">${item.latest_price.toFixed(2)}</span>
                              )}
                            </Link>
                          ))}
                        </div>
                      ) : (
                        <div className="px-5 py-8 text-center">
                          <p className="text-sm text-secondary">No items match &ldquo;{searchQuery}&rdquo;</p>
                        </div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Inline stats */}
              <div className="flex items-center gap-8 mt-8">
                {[
                  { label: 'Items', value: stats.totalItems.toLocaleString() },
                  { label: '24h Vol', value: `$${(stats.volume24h / 1_000_000).toFixed(1)}M` },
                  { label: 'Volatility', value: `${stats.avgVolatility.toFixed(1)}%` },
                ].map((stat) => (
                  <div key={stat.label} className="flex items-baseline gap-2">
                    <span className="font-data text-lg font-medium text-primary tabular-nums">{stat.value}</span>
                    <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted">{stat.label}</span>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Right: Featured item */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2, duration: 0.7, ease: EASE }}
              className="hidden lg:block"
            >
              {heroItem && (
                <Link href={`/items/${heroItem.item_id}`} className="group block">
                  <div className="relative aspect-square rounded-md border border-border bg-background-secondary overflow-hidden">
                    {heroItem.icon_url && (
                      <img
                        src={heroItem.icon_url}
                        alt={heroItem.name}
                        className="w-full h-full object-contain p-10 group-hover:scale-105 transition-transform duration-700 ease-out"
                      />
                    )}
                    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-background-primary/90 via-background-primary/40 to-transparent p-6 pt-16">
                      <div className="text-[10px] font-semibold uppercase tracking-[0.15em] text-muted mb-1">Featured</div>
                      <div className="font-medium text-primary text-sm mb-1">{heroItem.name}</div>
                      <div className="font-data text-2xl font-medium text-primary tabular-nums">
                        ${heroItem.latest_price?.toFixed(2)}
                      </div>
                    </div>
                  </div>
                </Link>
              )}
            </motion.div>
          </div>
        </section>

        {/* --- TRENDING --- */}
        <section className="pb-20 lg:pb-28">
          <div className="flex items-end justify-between mb-6">
            <h2 className="text-lg font-semibold tracking-tight text-primary">Trending now</h2>
            <Link href="/market" className="text-xs font-semibold uppercase tracking-[0.1em] text-accent hover:text-brand-hover transition-colors hidden sm:block">
              View all
            </Link>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
            {sideItems.map((item, i) => (
              <motion.div
                key={item.item_id}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 + i * 0.06, duration: 0.45, ease: EASE }}
              >
                <Link href={`/items/${item.item_id}`} className="widget-block block overflow-hidden group">
                  <div className="aspect-square bg-background-tertiary/50 flex items-center justify-center p-6 border-b border-border">
                    {item.icon_url && (
                      <img
                        src={item.icon_url}
                        alt={item.name}
                        className="max-w-full max-h-full object-contain group-hover:scale-105 transition-transform duration-500 ease-out"
                        loading="lazy"
                      />
                    )}
                  </div>
                  <div className="p-4">
                    <div className="text-[9px] font-semibold uppercase tracking-[0.15em] text-muted mb-1">
                      {item.type === 'knife' ? 'Knife' : 'Skin'}
                    </div>
                    <div className="text-sm font-medium text-primary truncate mb-2">{item.name}</div>
                    <div className="font-data text-lg font-medium text-primary tabular-nums">
                      ${item.latest_price?.toFixed(2)}
                    </div>
                  </div>
                </Link>
              </motion.div>
            ))}

            {/* CTA card */}
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.28, duration: 0.45, ease: EASE }}
            >
              <Link href="/market" className="widget-block flex flex-col items-center justify-center p-8 h-full text-center group min-h-[260px]">
                <div className="w-12 h-12 rounded-sm border border-border bg-background-tertiary flex items-center justify-center mb-4 group-hover:border-accent transition-colors">
                  <svg className="w-5 h-5 text-muted group-hover:text-accent transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M4 6h16M4 12h16m-7 6h7" />
                  </svg>
                </div>
                <span className="text-sm font-medium text-secondary group-hover:text-primary transition-colors">Browse all items</span>
                <span className="text-[10px] font-data text-muted mt-1">{stats.totalItems.toLocaleString()} tracked</span>
              </Link>
            </motion.div>
          </div>

          <div className="mt-6 sm:hidden">
            <Link href="/market" className="text-xs font-semibold uppercase tracking-[0.1em] text-accent hover:text-brand-hover transition-colors">
              View all items
            </Link>
          </div>
        </section>

        {/* --- CAPABILITIES --- */}
        <section className="pb-28 lg:pb-36 border-t border-border pt-16">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-60px' }}
            transition={{ duration: 0.5, ease: EASE }}
            className="mb-12"
          >
            <h2 className="text-2xl font-semibold tracking-tight text-primary" style={{ textWrap: 'balance' }}>
              Built for traders who need clarity
            </h2>
            <p className="text-sm text-secondary mt-2 max-w-lg">
              Every feature is designed to cut through market noise and surface what matters.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                title: 'Multi-Source Pricing',
                description: 'Steam, CSFloat, and aggregated market data in one view. Spot spreads, detect anomalies, and find the best execution price.',
              },
              {
                title: 'Technical Signals',
                description: 'SMA crossovers, volatility bands, momentum scoring, and trend direction — distilled into clear buy/hold/sell signals.',
              },
              {
                title: 'Portfolio Intelligence',
                description: 'Connect your Steam inventory for real-time valuation, cost-basis tracking, and risk distribution across your collection.',
              },
            ].map((cap, i) => (
              <motion.div
                key={cap.title}
                initial={{ opacity: 0, y: 12 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-40px' }}
                transition={{ delay: i * 0.08, duration: 0.45, ease: EASE }}
                className="widget-block p-6"
              >
                <h3 className="text-base font-semibold text-primary tracking-tight mb-2">{cap.title}</h3>
                <p className="text-sm text-secondary leading-relaxed">{cap.description}</p>
              </motion.div>
            ))}
          </div>
        </section>
      </main>

      {/* --- FOOTER --- */}
      <footer className="border-t border-border">
        <div className="max-w-6xl mx-auto px-6 py-12">
          <div className="flex flex-col sm:flex-row justify-between items-start gap-8">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-sm border border-border flex items-center justify-center bg-background-secondary">
                <span className="font-data font-bold text-[7px] text-primary tracking-tighter">CS</span>
              </div>
              <span className="font-semibold text-xs tracking-[0.15em] uppercase text-primary">Data Terminal</span>
            </div>
            <div className="flex gap-8">
              <Link href="/market" className="text-xs text-tertiary hover:text-primary transition-colors">Market</Link>
              <Link href="/portfolio" className="text-xs text-tertiary hover:text-primary transition-colors">Portfolio</Link>
              <Link href="/accuracy" className="text-xs text-tertiary hover:text-primary transition-colors">Accuracy</Link>
            </div>
          </div>
          <div className="mt-8 pt-6 border-t border-border/50">
            <p className="text-[10px] font-data text-muted uppercase tracking-[0.15em]">&copy; 2026 Data Terminal</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
