'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { Header } from '@/components';

interface Item {
  id: string;
  name: string;
  currentPrice: number;
  priceChange24h: number;
  volatility: number;
  volume24h: number;
}

const SAMPLE_ITEMS: Item[] = [
  { id: '1', name: 'AWP Dragon Lore', currentPrice: 2145, priceChange24h: 3.2, volatility: 12, volume24h: 1200 },
  { id: '2', name: 'M4A1 Hot Rod', currentPrice: 387, priceChange24h: -1.5, volatility: 8, volume24h: 3400 },
  { id: '3', name: 'Karambit Gamma Doppler', currentPrice: 1850, priceChange24h: 5.1, volatility: 15, volume24h: 850 },
  { id: '4', name: 'AK-47 Neon Rider', currentPrice: 125, priceChange24h: 2.8, volatility: 10, volume24h: 5600 },
  { id: '5', name: 'Desert Eagle Blaze', currentPrice: 650, priceChange24h: -0.5, volatility: 6, volume24h: 2100 },
  { id: '6', name: 'StatTrak M9 Bayonet', currentPrice: 3200, priceChange24h: 7.2, volatility: 18, volume24h: 420 },
  { id: '7', name: 'USP-S Kill Confirmed', currentPrice: 285, priceChange24h: 1.2, volatility: 5, volume24h: 1800 },
  { id: '8', name: 'Butterfly Knife Tiger Tooth', currentPrice: 2850, priceChange24h: 4.3, volatility: 14, volume24h: 650 },
];

export default function MarketPage() {
  const [items, setItems] = useState<Item[]>(SAMPLE_ITEMS);
  const [sortBy, setSortBy] = useState<keyof Item>('currentPrice');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [searchQuery, setSearchQuery] = useState('');
  const [hoveredRow, setHoveredRow] = useState<string | null>(null);

  useEffect(() => {
    let filtered = SAMPLE_ITEMS;
    if (searchQuery) {
      filtered = filtered.filter(item =>
        item.name.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    const sorted = [...filtered].sort((a, b) => {
      const aVal = a[sortBy];
      const bVal = b[sortBy];
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortOrder === 'asc' ? aVal - bVal : bVal - aVal;
      }
      return 0;
    });

    setItems(sorted);
  }, [sortBy, sortOrder, searchQuery]);

  const handleSort = (column: keyof Item) => {
    if (sortBy === column) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(column);
      setSortOrder('desc');
    }
  };

  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--background-primary)' }}>
      <Header />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        {/* Header */}
        <div className="mb-10">
          <div className="mb-6">
            <h1 className="text-4xl font-bold mb-2" style={{ color: 'var(--text-primary)' }}>Market Overview</h1>
            <p className="text-base" style={{ color: 'var(--text-secondary)' }}>Real-time prices and market data for Counter-Strike 2 items</p>
          </div>

          {/* Search Bar */}
          <input
            type="text"
            placeholder="Search items..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{
              maxWidth: '500px',
              padding: '11px 14px',
              backgroundColor: 'var(--surface)',
              borderColor: 'var(--border)',
              borderWidth: '1px',
              color: 'var(--text-primary)',
              borderRadius: '8px',
              fontSize: '14px',
              width: '100%',
              transition: 'all 0.2s ease'
            }}
            className="focus:outline-none"
            onFocus={(e) => {
              e.currentTarget.style.borderColor = 'var(--accent-primary)';
              e.currentTarget.style.backgroundColor = 'var(--surface-hover)';
              e.currentTarget.style.boxShadow = 'inset 0 0 0 1px var(--border-accent), 0 0 12px rgba(59, 130, 246, 0.1)';
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = 'var(--border)';
              e.currentTarget.style.backgroundColor = 'var(--surface)';
              e.currentTarget.style.boxShadow = 'none';
            }}
          />
        </div>

        {/* Market Table */}
        <div
          style={{
            borderColor: 'var(--border)',
            borderWidth: '1px',
            borderRadius: '10px',
            overflow: 'hidden',
            backgroundColor: 'var(--surface)'
          }}
          className="overflow-x-auto shadow-md"
        >
          <table className="w-full text-sm">
            <thead>
              <tr
                style={{
                  borderBottomColor: 'var(--border)',
                  borderBottomWidth: '1px',
                  backgroundColor: 'var(--background-tertiary)'
                }}
              >
                <th className="px-6 py-5 text-left text-xs font-semibold uppercase tracking-wide">
                  <button
                    onClick={() => handleSort('name')}
                    className="hover:opacity-80 transition flex items-center gap-2"
                    style={{ color: 'var(--text-secondary)' }}
                  >
                    Item {sortBy === 'name' && <span style={{ color: 'var(--accent-primary)' }}>▼</span>}
                  </button>
                </th>
                <th className="px-6 py-5 text-right text-xs font-semibold uppercase tracking-wide">
                  <button
                    onClick={() => handleSort('currentPrice')}
                    className="w-full flex justify-end hover:opacity-80 transition items-center gap-2"
                    style={{ color: 'var(--text-secondary)' }}
                  >
                    Price {sortBy === 'currentPrice' && <span style={{ color: 'var(--accent-primary)' }}>▼</span>}
                  </button>
                </th>
                <th className="px-6 py-5 text-right text-xs font-semibold uppercase tracking-wide">
                  <button
                    onClick={() => handleSort('priceChange24h')}
                    className="w-full flex justify-end hover:opacity-80 transition items-center gap-2"
                    style={{ color: 'var(--text-secondary)' }}
                  >
                    24h Change {sortBy === 'priceChange24h' && <span style={{ color: 'var(--accent-primary)' }}>▼</span>}
                  </button>
                </th>
                <th className="px-6 py-5 text-right text-xs font-semibold uppercase tracking-wide">
                  <button
                    onClick={() => handleSort('volatility')}
                    className="w-full flex justify-end hover:opacity-80 transition items-center gap-2"
                    style={{ color: 'var(--text-secondary)' }}
                  >
                    Vol {sortBy === 'volatility' && <span style={{ color: 'var(--accent-primary)' }}>▼</span>}
                  </button>
                </th>
                <th className="px-6 py-5 text-right text-xs font-semibold uppercase tracking-wide">
                  <button
                    onClick={() => handleSort('volume24h')}
                    className="w-full flex justify-end hover:opacity-80 transition items-center gap-2"
                    style={{ color: 'var(--text-secondary)' }}
                  >
                    Volume {sortBy === 'volume24h' && <span style={{ color: 'var(--accent-primary)' }}>▼</span>}
                  </button>
                </th>
              </tr>
            </thead>
            <tbody>
              {items.slice(0, 20).map((item, idx) => (
                <motion.tr
                  key={item.id}
                  initial={{ opacity: 0, y: -5 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.01 }}
                  style={{
                    borderBottomColor: 'var(--divider)',
                    borderBottomWidth: '1px',
                    backgroundColor: hoveredRow === item.id ? 'var(--background-tertiary)' : 'transparent',
                    transition: 'background-color 0.2s ease'
                  }}
                  className="group cursor-pointer"
                  onMouseEnter={() => setHoveredRow(item.id)}
                  onMouseLeave={() => setHoveredRow(null)}
                >
                  <td className="px-6 py-4">
                    <Link
                      href={`/items/${item.id}`}
                      className="font-medium transition-opacity hover:opacity-75"
                      style={{ color: hoveredRow === item.id ? 'var(--accent-primary)' : 'var(--text-primary)' }}
                    >
                      {item.name}
                    </Link>
                  </td>
                  <td className="px-6 py-4 text-right font-mono font-medium" style={{ color: 'var(--text-primary)' }}>
                    ${item.currentPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </td>
                  <td className="px-6 py-4 text-right">
                    <span
                      className="inline-block px-3 py-1.5 rounded font-mono font-semibold text-xs"
                      style={{
                        backgroundColor:
                          item.priceChange24h >= 0 ? 'var(--data-up-subtle)' : 'var(--data-down-subtle)',
                        color: item.priceChange24h >= 0 ? 'var(--data-up)' : 'var(--data-down)'
                      }}
                    >
                      {item.priceChange24h > 0 ? '+' : ''}{item.priceChange24h.toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'flex-end',
                        gap: '8px'
                      }}
                    >
                      <div
                        style={{
                          width: '24px',
                          height: '20px',
                          backgroundColor: 'var(--grid)',
                          borderRadius: '3px',
                          position: 'relative',
                          overflow: 'hidden'
                        }}
                      >
                        <div
                          style={{
                            position: 'absolute',
                            bottom: 0,
                            left: 0,
                            right: 0,
                            height: `${item.volatility}%`,
                            backgroundColor: 'var(--accent-primary)',
                            opacity: 0.6
                          }}
                        />
                      </div>
                      <span className="font-mono text-xs" style={{ color: 'var(--text-secondary)' }}>
                        {item.volatility}%
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-right font-mono" style={{ color: 'var(--text-secondary)' }}>
                    {(item.volume24h / 1000).toFixed(1)}K
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>

        {items.length === 0 && (
          <div className="text-center py-16" style={{ color: 'var(--text-secondary)' }}>
            <p className="text-base">No items match your search</p>
          </div>
        )}
      </div>
    </div>
  );
}
