'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Header } from '@/components';
import CountUpNumber from '@/components/CountUpNumber';

const QUALITY_TIERS = [
  { label: 'FN', fullName: 'Factory New', floatRange: [0, 0.07] },
  { label: 'MW', fullName: 'Minimal Wear', floatRange: [0.07, 0.15] },
  { label: 'FT', fullName: 'Field-Tested', floatRange: [0.15, 0.38] },
  { label: 'WW', fullName: 'Well-Worn', floatRange: [0.38, 0.45] },
  { label: 'BS', fullName: 'Battle-Scarred', floatRange: [0.45, 1.0] },
];

interface PriceData {
  date: string;
  price: number;
}

const mockPriceData: Record<string, PriceData[]> = {
  '24h': [
    { date: '12am', price: 2100 },
    { date: '4am', price: 2110 },
    { date: '8am', price: 2105 },
    { date: '12pm', price: 2130 },
    { date: '4pm', price: 2145 },
  ],
  '7d': [
    { date: 'Sun', price: 1980 },
    { date: 'Mon', price: 2020 },
    { date: 'Tue', price: 2045 },
    { date: 'Wed', price: 2078 },
    { date: 'Thu', price: 2100 },
    { date: 'Fri', price: 2125 },
    { date: 'Sat', price: 2145 },
  ],
  '30d': [
    { date: 'May 1', price: 1850 },
    { date: 'May 5', price: 1900 },
    { date: 'May 10', price: 1950 },
    { date: 'May 15', price: 2050 },
    { date: 'May 20', price: 2145 },
  ],
  'all': [
    { date: 'Jan', price: 1700 },
    { date: 'Feb', price: 1800 },
    { date: 'Mar', price: 1900 },
    { date: 'Apr', price: 2000 },
    { date: 'May', price: 2145 },
  ],
};

interface ItemDetail {
  id: string;
  name: string;
  currentPrice: number;
  priceChange24h: number;
  volatility: number;
  movingAvg7d: number;
  movingAvg30d: number;
  momentum: number;
  category: string;
  rarity: string;
  listed: number;
  volume24h: number;
}

const itemData: Record<string, ItemDetail> = {
  '1': {
    id: '1',
    name: 'AWP Dragon Lore',
    currentPrice: 2145,
    priceChange24h: 3.2,
    volatility: 12,
    movingAvg7d: 2078,
    movingAvg30d: 1950,
    momentum: 98,
    category: 'Rifle',
    rarity: '★★★★★',
    listed: 324,
    volume24h: 1200,
  },
};

const priceRangesByQuality: Record<string, { min: number; max: number }> = {
  'FN': { min: 2100, max: 2200 },
  'MW': { min: 1980, max: 2050 },
  'FT': { min: 1850, max: 1920 },
  'WW': { min: 1700, max: 1800 },
  'BS': { min: 1550, max: 1650 },
};

export default function ItemDetailPage({ params }: { params: { id: string } }) {
  const item = itemData[params.id] || itemData['1'];
  const [timeRange, setTimeRange] = useState<'24h' | '7d' | '30d' | 'all'>('30d');
  const [selectedQuality, setSelectedQuality] = useState<string>('FN');
  const [floatValue, setFloatValue] = useState<string>('0.0342');
  const [chartKey, setChartKey] = useState(0);

  const handleFloatChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setFloatValue(val);
    const float = parseFloat(val) || 0;

    for (const tier of QUALITY_TIERS) {
      if (float >= tier.floatRange[0] && float < tier.floatRange[1]) {
        setSelectedQuality(tier.label);
        break;
      }
    }
  };

  const handleQualityChange = (label: string) => {
    const tier = QUALITY_TIERS.find(t => t.label === label);
    if (tier) {
      setSelectedQuality(label);
      const midFloat = (tier.floatRange[0] + tier.floatRange[1]) / 2;
      setFloatValue(midFloat.toFixed(4));
    }
    setChartKey(prev => prev + 1);
  };

  const selectedPriceRange = priceRangesByQuality[selectedQuality];
  const priceData = mockPriceData[timeRange];

  return (
    <div className="min-h-screen bg-[#0f1419]">
      <Header />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Breadcrumb */}
        <Link href="/market" className="text-[#3b82f6] hover:underline text-xs font-medium mb-6 inline-block">
          ← MARKET
        </Link>

        {/* Header with Price Ticker */}
        <div className="bg-[#1a1f2e] border border-[#2d3748] rounded p-6 mb-8">
          <div className="flex justify-between items-start mb-6">
            <div>
              <h1 className="text-3xl font-bold text-[#d1d5db] mb-1">{item.name}</h1>
              <p className="text-[#6b7280] text-sm font-mono">{selectedQuality} • Float: {floatValue}</p>
            </div>
            <div className="text-right">
              <div className="text-4xl font-bold text-[#d1d5db] font-mono mb-1">
                ${selectedPriceRange.min.toFixed(0)}
              </div>
              <div className={`font-mono text-sm ${item.priceChange24h >= 0 ? 'text-[#10b981]' : 'text-[#ef4444]'}`}>
                {item.priceChange24h > 0 ? '+' : ''}{item.priceChange24h.toFixed(2)}% (24h)
              </div>
            </div>
          </div>

          {/* Quality Selector */}
          <div className="flex gap-2 flex-wrap">
            {QUALITY_TIERS.map((tier) => (
              <button
                key={tier.label}
                onClick={() => handleQualityChange(tier.label)}
                className={`px-3 py-1.5 text-xs font-medium rounded transition-colors ${
                  selectedQuality === tier.label
                    ? 'bg-[#3b82f6] text-white'
                    : 'bg-[#131820] text-[#6b7280] border border-[#2d3748] hover:border-[#3b82f6]'
                }`}
              >
                {tier.label}
              </button>
            ))}
            <input
              type="number"
              value={floatValue}
              onChange={handleFloatChange}
              min="0"
              max="1"
              step="0.0001"
              placeholder="Float"
              className="px-2 py-1.5 bg-[#131820] border border-[#2d3748] text-[#d1d5db] placeholder-[#6b7280] rounded focus:outline-none focus:border-[#3b82f6] font-mono text-xs transition-colors ml-auto"
            />
          </div>
        </div>

        {/* Image Preview */}
        <div className="bg-[#1a1f2e] border border-[#2d3748] rounded p-4 mb-8">
          <div className="aspect-video bg-[#131820] rounded flex items-center justify-center border border-[#2d3748]">
            <div className="text-center text-[#6b7280]">
              <div className="text-6xl mb-2">🎨</div>
              <p className="text-sm">Item image preview</p>
            </div>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-4 gap-6 mb-8">
          {/* Chart Section - 3 columns (takes up more space) */}
          <div className="col-span-3">
            {/* Time Range Tabs */}
            <div className="flex gap-1 mb-4 border-b border-[#2d3748] pb-3">
              {['24h', '7d', '30d', 'all'].map((range) => (
                <button
                  key={range}
                  onClick={() => {
                    setTimeRange(range as any);
                    setChartKey(prev => prev + 1);
                  }}
                  className={`px-3 py-2 text-xs font-medium transition-colors ${
                    timeRange === range
                      ? 'text-[#d1d5db] border-b-2 border-[#3b82f6]'
                      : 'text-[#6b7280] hover:text-[#d1d5db]'
                  }`}
                >
                  {range.toUpperCase()}
                </button>
              ))}
            </div>

            {/* Chart with Animation */}
            <motion.div
              key={chartKey}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
              className="bg-[#1a1f2e] border border-[#2d3748] p-4 rounded"
            >
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={priceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                  <XAxis dataKey="date" stroke="#6b7280" style={{ fontSize: '12px' }} />
                  <YAxis stroke="#6b7280" style={{ fontSize: '12px' }} domain={['dataMin - 50', 'dataMax + 50']} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #2d3748', color: '#d1d5db', borderRadius: '4px' }}
                    formatter={(value) => `$${value}`}
                  />
                  <motion.g
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                  >
                    <Line
                      type="monotone"
                      dataKey="price"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={true}
                      animationDuration={800}
                    />
                  </motion.g>
                </LineChart>
              </ResponsiveContainer>
            </motion.div>
          </div>

          {/* Stats Section - Compact right column */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="bg-[#1a1f2e] border border-[#2d3748] rounded p-4 space-y-3 h-fit"
          >
            <div>
              <p className="text-xs text-[#6b7280] mb-1">RANGE</p>
              <p className="text-lg font-semibold text-[#d1d5db] font-mono">
                ${selectedPriceRange.min}–${selectedPriceRange.max}
              </p>
            </div>

            <div className="border-t border-[#2d3748] pt-3 space-y-2">
              <div className="flex justify-between items-center text-xs">
                <span className="text-[#6b7280]">24h</span>
                <span className={`font-mono font-medium ${item.priceChange24h >= 0 ? 'text-[#10b981]' : 'text-[#ef4444]'}`}>
                  <CountUpNumber
                    from={0}
                    to={item.priceChange24h}
                    decimals={1}
                    duration={1}
                    formatFn={(val) => `${val > 0 ? '+' : ''}${val}%`}
                  />
                </span>
              </div>

              <div className="flex justify-between items-center text-xs">
                <span className="text-[#6b7280]">Vol</span>
                <span className="font-mono text-[#d1d5db] font-medium">
                  <CountUpNumber
                    from={0}
                    to={item.volatility}
                    decimals={0}
                    duration={1}
                    formatFn={(val) => `${val}%`}
                  />
                </span>
              </div>

              <div className="flex justify-between items-center text-xs">
                <span className="text-[#6b7280]">MA7d</span>
                <span className="font-mono text-[#d1d5db] font-medium">
                  $<CountUpNumber
                    from={0}
                    to={item.movingAvg7d}
                    decimals={0}
                    duration={1}
                  />
                </span>
              </div>

              <div className="flex justify-between items-center text-xs">
                <span className="text-[#6b7280]">MA30d</span>
                <span className="font-mono text-[#d1d5db] font-medium">
                  $<CountUpNumber
                    from={0}
                    to={item.movingAvg30d}
                    decimals={0}
                    duration={1}
                  />
                </span>
              </div>

              <div className="flex justify-between items-center text-xs border-t border-[#2d3748] pt-2 mt-2">
                <span className="text-[#6b7280]">Mom</span>
                <span className={`font-mono font-medium ${item.momentum >= 0 ? 'text-[#10b981]' : 'text-[#ef4444]'}`}>
                  <CountUpNumber
                    from={0}
                    to={item.momentum}
                    decimals={0}
                    duration={1}
                    formatFn={(val) => `${val > 0 ? '+' : ''}${val}`}
                  />
                </span>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Item Metrics */}
        <div className="bg-[#1a1f2e] border border-[#2d3748] rounded p-4 mb-8 grid grid-cols-4 gap-4">
          <div>
            <p className="text-xs text-[#6b7280] mb-1">Volume (24h)</p>
            <p className="text-lg font-semibold text-[#d1d5db] font-mono">{(item.volume24h / 1000).toFixed(1)}K</p>
          </div>
          <div>
            <p className="text-xs text-[#6b7280] mb-1">Listings</p>
            <p className="text-lg font-semibold text-[#d1d5db] font-mono">{item.listed}</p>
          </div>
          <div>
            <p className="text-xs text-[#6b7280] mb-1">Category</p>
            <p className="text-lg font-semibold text-[#d1d5db]">Rifle</p>
          </div>
          <div>
            <p className="text-xs text-[#6b7280] mb-1">Status</p>
            <p className="text-lg font-semibold text-[#10b981]">Active</p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-2 pb-16">
          {['Watchlist', 'Alert', 'Compare'].map((label) => (
            <motion.button
              key={label}
              whileHover={{ opacity: 0.9 }}
              whileTap={{ opacity: 0.8 }}
              className="px-4 py-2 bg-[#3b82f6] text-white font-medium text-sm rounded hover:bg-[#2563eb] transition-colors"
            >
              {label}
            </motion.button>
          ))}
        </div>
      </div>
    </div>
  );
}
