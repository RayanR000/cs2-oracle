'use client';

import Link from 'next/link';

export default function Header() {
  return (
    <header className="sticky top-0 z-50 border-b" style={{
      borderColor: 'var(--border)',
      backgroundColor: 'var(--background-secondary)',
      boxShadow: '0 1px 3px rgba(0,0,0,0.3)'
    }}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex justify-between items-center">
          <Link href="/" className="flex items-center gap-3 group">
            <div className="w-8 h-8 rounded border-2" style={{
              borderColor: 'var(--accent-primary)',
              backgroundColor: 'var(--bg-accent-subtle)'
            }} className="flex items-center justify-center">
              <span className="font-bold text-xs" style={{ color: 'var(--accent-primary)' }}>CS2</span>
            </div>
            <span className="font-semibold" style={{ color: 'var(--text-primary)', fontSize: '16px' }}>
              CS2
            </span>
            <span className="hidden sm:inline" style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>
              Market Analyzer
            </span>
          </Link>

          <nav className="hidden md:flex gap-12">
            <NavLink href="/" label="Home" />
            <NavLink href="/market" label="Market" />
            <NavLink href="/portfolio" label="Portfolio" />
          </nav>
        </div>
      </div>
    </header>
  );
}

function NavLink({ href, label }: { href: string; label: string }) {
  return (
    <Link
      href={href}
      className="relative text-sm font-medium transition-all duration-200 group"
      style={{ color: 'var(--text-secondary)' }}
    >
      {label}
      <span
        className="absolute bottom-0 left-0 w-0 h-0.5 transition-all duration-200 group-hover:w-full"
        style={{ backgroundColor: 'var(--accent-primary)' }}
      />
    </Link>
  );
}
