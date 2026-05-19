'use client';

import { useState } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { Header } from '@/components';
import StatCard from '@/components/StatCard';

interface PortfolioItem {
  id: string;
  name: string;
  quantity: number;
  costBasis: number;
  currentPrice: number;
  quality: string;
}

const SAMPLE_PORTFOLIO: PortfolioItem[] = [
  {
    id: '1',
    name: 'AWP Dragon Lore',
    quantity: 1,
    costBasis: 1980,
    currentPrice: 2145,
    quality: 'Factory New',
  },
  {
    id: '2',
    name: 'M4A1 Hot Rod',
    quantity: 2,
    costBasis: 180,
    currentPrice: 387,
    quality: 'Minimal Wear',
  },
  {
    id: '3',
    name: 'Karambit Gamma Doppler',
    quantity: 1,
    costBasis: 1700,
    currentPrice: 1850,
    quality: 'Factory New',
  },
];

export default function PortfolioPage() {
  const [items] = useState<PortfolioItem[]>(SAMPLE_PORTFOLIO);
  const [hoveredRow, setHoveredRow] = useState<string | null>(null);

  const totalCost = items.reduce((sum, item) => sum + item.costBasis * item.quantity, 0);
  const totalValue = items.reduce((sum, item) => sum + item.currentPrice * item.quantity, 0);
  const totalGain = totalValue - totalCost;
  const gainPercent = totalCost > 0 ? ((totalGain / totalCost) * 100) : 0;

  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--background-primary)' }}>
      <Header />

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        {/* Header */}
        <div className="mb-10">
          <h1 className="text-4xl font-bold mb-2" style={{ color: 'var(--text-primary)' }}>Portfolio</h1>
          <p className="text-base" style={{ color: 'var(--text-secondary)' }}>Holdings, performance, and gains/losses</p>
        </div>

        {/* Portfolio Summary Stats */}
        <motion.div
          initial={{ opacity: 0, y: -15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, staggerChildren: 0.1 }}
          className="grid grid-cols-1 md:grid-cols-4 gap-5 mb-10"
        >
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}>
            <StatCard
              label="Total Cost"
              value={`$${totalCost.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`}
              highlight="primary"
            />
          </motion.div>
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.15 }}>
            <StatCard
              label="Current Value"
              value={`$${totalValue.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`}
              highlight="secondary"
            />
          </motion.div>
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
            <StatCard
              label="Gain/Loss"
              value={`$${Math.abs(totalGain).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`}
              isPositive={totalGain >= 0}
              change={totalGain >= 0 ? (totalGain / totalCost * 100) : -(Math.abs(totalGain) / totalCost * 100)}
            />
          </motion.div>
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.25 }}>
            <StatCard
              label="Return %"
              value={`${gainPercent >= 0 ? '+' : ''}${gainPercent.toFixed(1)}%`}
              isPositive={gainPercent >= 0}
            />
          </motion.div>
        </motion.div>

        {/* Portfolio Holdings Table */}
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
                <th className="px-6 py-5 text-left text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--text-secondary)' }}>Item</th>
                <th className="px-6 py-5 text-center text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--text-secondary)' }}>Quality</th>
                <th className="px-6 py-5 text-right text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--text-secondary)' }}>Qty</th>
                <th className="px-6 py-5 text-right text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--text-secondary)' }}>Cost</th>
                <th className="px-6 py-5 text-right text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--text-secondary)' }}>Price</th>
                <th className="px-6 py-5 text-right text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--text-secondary)' }}>Value</th>
                <th className="px-6 py-5 text-right text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--text-secondary)' }}>Gain/Loss</th>
                <th className="px-6 py-5 text-right text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--text-secondary)' }}>Return</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item, idx) => {
                const totalItemCost = item.costBasis * item.quantity;
                const totalItemValue = item.currentPrice * item.quantity;
                const itemGain = totalItemValue - totalItemCost;
                const itemGainPercent = totalItemCost > 0 ? ((itemGain / totalItemCost) * 100) : 0;

                return (
                  <motion.tr
                    key={item.id}
                    initial={{ opacity: 0, y: -5 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.05 }}
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
                    <td className="px-6 py-4 text-center text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>
                      {item.quality}
                    </td>
                    <td className="px-6 py-4 text-right font-mono" style={{ color: 'var(--text-secondary)' }}>
                      {item.quantity}
                    </td>
                    <td className="px-6 py-4 text-right font-mono" style={{ color: 'var(--text-primary)' }}>
                      ${item.costBasis.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                    </td>
                    <td className="px-6 py-4 text-right font-mono" style={{ color: 'var(--text-primary)' }}>
                      ${item.currentPrice.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                    </td>
                    <td className="px-6 py-4 text-right font-mono font-medium" style={{ color: 'var(--accent-secondary)' }}>
                      ${totalItemValue.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                    </td>
                    <td className="px-6 py-4 text-right">
                      <span
                        className="inline-block px-3 py-1.5 rounded font-mono font-semibold text-xs"
                        style={{
                          backgroundColor: itemGain >= 0 ? 'var(--data-up-subtle)' : 'var(--data-down-subtle)',
                          color: itemGain >= 0 ? 'var(--data-up)' : 'var(--data-down)'
                        }}
                      >
                        ${Math.abs(itemGain).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <span
                        className="inline-block px-3 py-1.5 rounded font-mono font-semibold text-xs"
                        style={{
                          backgroundColor: itemGainPercent >= 0 ? 'var(--data-up-subtle)' : 'var(--data-down-subtle)',
                          color: itemGainPercent >= 0 ? 'var(--data-up)' : 'var(--data-down)'
                        }}
                      >
                        {itemGainPercent >= 0 ? '+' : ''}{itemGainPercent.toFixed(1)}%
                      </span>
                    </td>
                  </motion.tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {items.length === 0 && (
          <div className="text-center py-16" style={{ color: 'var(--text-secondary)' }}>
            <p className="mb-4 text-base">No items in portfolio</p>
            <Link href="/market" className="font-medium transition-opacity hover:opacity-75" style={{ color: 'var(--accent-primary)' }}>
              Browse market →
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}
