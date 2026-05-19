'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { Header } from '@/components';
import StatCard from '@/components/StatCard';

export default function Home() {
  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--background-primary)' }}>
      <Header />

      {/* Hero Section */}
      <div
        style={{
          backgroundColor: 'var(--background-tertiary)',
          borderBottomColor: 'var(--border)',
          borderBottomWidth: '1px'
        }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-14"
          >
            <div className="mb-6 inline-block">
              <span
                className="px-3 py-1.5 rounded-full text-xs font-semibold uppercase tracking-wide"
                style={{
                  backgroundColor: 'var(--data-up-subtle)',
                  color: 'var(--data-up)',
                  border: '1px solid var(--data-up)'
                }}
              >
                ✦ Live Market Data
              </span>
            </div>

            <h1 className="text-5xl md:text-6xl font-bold mb-6 tracking-tight" style={{ color: 'var(--text-primary)' }}>
              Professional Market Intelligence
            </h1>

            <p
              className="text-lg max-w-3xl mx-auto mb-10 leading-relaxed"
              style={{ color: 'var(--text-secondary)' }}
            >
              Real-time analytics for Counter-Strike 2 items. Track prices, analyze trends, and optimize your portfolio with institutional-grade market data.
            </p>

            <div className="flex justify-center gap-4">
              <Link
                href="/market"
                className="px-7 py-3.5 font-semibold rounded-lg transition-all duration-200 text-white"
                style={{
                  backgroundColor: 'var(--accent-primary)',
                  boxShadow: '0 8px 24px rgba(59, 130, 246, 0.25)'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.boxShadow = '0 12px 32px rgba(59, 130, 246, 0.35)';
                  e.currentTarget.style.transform = 'translateY(-2px)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.boxShadow = '0 8px 24px rgba(59, 130, 246, 0.25)';
                  e.currentTarget.style.transform = 'translateY(0)';
                }}
              >
                Launch Market
              </Link>
              <Link
                href="/portfolio"
                className="px-7 py-3.5 font-semibold rounded-lg transition-all duration-200"
                style={{
                  backgroundColor: 'var(--surface)',
                  color: 'var(--accent-primary)',
                  border: '1.5px solid var(--accent-primary)'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = 'var(--surface-hover)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'var(--surface)';
                }}
              >
                View Portfolio
              </Link>
            </div>
          </motion.div>

          {/* Hero Stats */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto"
          >
            <StatCard label="Live Items" value="1,247" />
            <StatCard label="Avg Volume" value="856K/day" highlight="secondary" />
            <StatCard label="Analysis Points" value="12K+" />
          </motion.div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        {/* Navigation Section */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-20"
        >
          <NavCard
            href="/market"
            title="Market Overview"
            description="Browse all items with real-time prices, volatility metrics, and 24h performance. Sort, search, and analyze market dynamics."
            accent="primary"
            icon="📊"
          />
          <NavCard
            href="/portfolio"
            title="Portfolio Tracker"
            description="Monitor your holdings, cost basis, and portfolio performance. Track gains/losses and returns in real-time."
            accent="secondary"
            icon="💼"
          />
        </motion.div>

        {/* Features Section */}
        <div className="mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="mb-12"
          >
            <h2 className="text-4xl font-bold mb-4" style={{ color: 'var(--text-primary)' }}>
              Advanced Analytics
            </h2>
            <p style={{ color: 'var(--text-secondary)' }}>
              Everything you need to make informed trading decisions
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <FeatureCard
              title="Price Charts & Analysis"
              description="Interactive charts with 24h, 7d, 30d, and all-time views. Track price movements, moving averages, and trend patterns."
              icon="📈"
              accent="primary"
            />
            <FeatureCard
              title="Quality & Float Analysis"
              description="Examine item conditions from Factory New to Battle-Scarred. Price variations by float value and condition tiers."
              icon="💎"
              accent="secondary"
            />
            <FeatureCard
              title="Technical Indicators"
              description="Moving averages, volatility metrics, momentum indicators, and market sentiment analysis for deeper insights."
              icon="🎯"
              accent="tertiary"
            />
            <FeatureCard
              title="Real-time Data Feed"
              description="Live market data, volume metrics, and instant price updates. Professional-grade market intelligence."
              icon="⚡"
              accent="primary"
            />
          </div>
        </div>

        {/* Stats Showcase */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          style={{
            backgroundColor: 'var(--background-tertiary)',
            borderColor: 'var(--border)',
            borderWidth: '1px',
            borderRadius: '12px',
            padding: '48px 32px',
            textAlign: 'center'
          }}
        >
          <h3 className="text-2xl font-bold mb-2" style={{ color: 'var(--text-primary)' }}>
            Ready to master the CS2 market?
          </h3>
          <p className="mb-8 text-base" style={{ color: 'var(--text-secondary)' }}>
            Start tracking prices, analyzing trends, and optimizing your portfolio today.
          </p>
          <Link
            href="/market"
            className="inline-block px-8 py-3.5 font-semibold rounded-lg text-white transition-all duration-200"
            style={{
              backgroundColor: 'var(--accent-primary)',
              boxShadow: '0 8px 24px rgba(59, 130, 246, 0.25)'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.boxShadow = '0 12px 32px rgba(59, 130, 246, 0.35)';
              e.currentTarget.style.transform = 'translateY(-2px)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.boxShadow = '0 8px 24px rgba(59, 130, 246, 0.25)';
              e.currentTarget.style.transform = 'translateY(0)';
            }}
          >
            Explore Market →
          </Link>
        </motion.div>
      </div>

      {/* Footer */}
      <footer
        style={{
          borderTopColor: 'var(--border)',
          borderTopWidth: '1px',
          backgroundColor: 'var(--background-secondary)',
          marginTop: '80px'
        }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 text-center text-sm" style={{ color: 'var(--text-tertiary)' }}>
          <p>CS2 Market Analyzer — Professional trading intelligence for Counter-Strike 2</p>
        </div>
      </footer>
    </div>
  );
}

function NavCard({
  href,
  title,
  description,
  accent,
  icon
}: {
  href: string;
  title: string;
  description: string;
  accent: 'primary' | 'secondary';
  icon?: string;
}) {
  const accentColor = accent === 'secondary' ? 'var(--accent-secondary)' : 'var(--accent-primary)';
  const accentBg = accent === 'secondary' ? 'rgba(245, 158, 11, 0.08)' : 'rgba(59, 130, 246, 0.08)';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      viewport={{ once: true }}
    >
      <Link
        href={href}
        className="group relative p-8 rounded-lg block transition-all duration-300"
        style={{
          backgroundColor: 'var(--surface)',
          borderWidth: '1.5px',
          borderColor: 'var(--border)'
        }}
        onMouseEnter={(e) => {
          const el = e.currentTarget as HTMLElement;
          el.style.borderColor = accentColor;
          el.style.backgroundColor = accentBg;
        }}
        onMouseLeave={(e) => {
          const el = e.currentTarget as HTMLElement;
          el.style.borderColor = 'var(--border)';
          el.style.backgroundColor = 'var(--surface)';
        }}
      >
        {icon && <span className="text-3xl mb-4 block">{icon}</span>}
        <h3 className="text-xl font-bold mb-3" style={{ color: 'var(--text-primary)' }}>
          {title}
        </h3>
        <p className="text-sm leading-relaxed mb-5" style={{ color: 'var(--text-secondary)' }}>
          {description}
        </p>
        <div className="flex items-center gap-2 text-sm font-semibold" style={{ color: accentColor }}>
          Explore →
        </div>
      </Link>
    </motion.div>
  );
}

function FeatureCard({
  title,
  description,
  icon,
  accent
}: {
  title: string;
  description: string;
  icon?: string;
  accent: 'primary' | 'secondary' | 'tertiary';
}) {
  let accentColor = 'var(--accent-primary)';
  let accentBg = 'rgba(59, 130, 246, 0.08)';

  if (accent === 'secondary') {
    accentColor = 'var(--accent-secondary)';
    accentBg = 'rgba(245, 158, 11, 0.08)';
  } else if (accent === 'tertiary') {
    accentColor = 'var(--accent-tertiary)';
    accentBg = 'rgba(99, 102, 241, 0.08)';
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      viewport={{ once: true }}
      style={{
        backgroundColor: accentBg,
        borderColor: accentColor,
        borderWidth: '1px',
        borderRadius: '10px',
        padding: '24px'
      }}
      className="group hover:shadow-md transition-all duration-300"
    >
      {icon && <span className="text-3xl block mb-4">{icon}</span>}
      <h3 className="text-lg font-bold mb-3" style={{ color: 'var(--text-primary)' }}>
        {title}
      </h3>
      <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
        {description}
      </p>
    </motion.div>
  );
}

