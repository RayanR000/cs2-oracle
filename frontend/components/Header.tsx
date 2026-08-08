'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useUser } from '@/lib/UserContext';
import { useTheme } from '@/lib/ThemeContext';
import { getLoginUrl } from '@/lib/api';

const NAV_LINKS = [
  { href: '/market', label: 'Market' },
  { href: '/portfolio', label: 'Portfolio' },
];

export default function Header() {
  const { user, loading, logout } = useUser();
  const { theme, toggleTheme } = useTheme();
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-[var(--z-sticky)] bg-ground/95 backdrop-blur-md border-b border-divider">
      <div className="max-w-6xl mx-auto px-6">
        <div className="flex justify-between items-center h-16">
          <Link href="/" className="flex items-center gap-3 group">
            <div className="w-8 h-8 rounded-sm border border-border bg-stock flex items-center justify-center transition-colors duration-200 group-hover:border-border-accent">
              <span className="font-data font-bold text-[10px] text-paper tracking-tighter">CS</span>
            </div>
            <div className="flex flex-col">
              <span className="specimen-tag text-paper leading-none">
                CS2 Oracle
              </span>
              <span className="specimen-tag text-paper-muted mt-1 leading-none">
                Specimen Archive
              </span>
            </div>
          </Link>

          <nav className="hidden md:flex items-center gap-8">
            {NAV_LINKS.map((link) => {
              const active = pathname === link.href || pathname.startsWith(`${link.href}/`);
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`specimen-tag relative py-2 transition-colors duration-200 ${
                    active ? 'text-ink' : 'text-paper-secondary hover:text-paper'
                  }`}
                >
                  {link.label}
                  {active && (
                    <span
                      aria-hidden
                      className="absolute -bottom-0.5 left-0 right-0 h-[2px] bg-ink"
                    />
                  )}
                </Link>
              );
            })}
          </nav>

          <div className="flex items-center gap-5">
            <button
              onClick={toggleTheme}
              className="w-8 h-8 rounded-sm border border-border bg-stock flex items-center justify-center transition-colors duration-200 text-paper-tertiary hover:text-paper hover:border-border-accent"
              aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
            >
              {theme === 'light' ? (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
              ) : (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
              )}
            </button>

            {loading ? (
              <div className="w-8 h-8 rounded-sm bg-surface animate-pulse" />
            ) : user ? (
              <div className="flex items-center gap-4">
                <div className="flex flex-col items-end hidden sm:flex">
                  <span className="text-[11px] font-semibold text-paper tracking-tight">
                    {user.username ?? user.steam_id}
                  </span>
                  <button
                    onClick={() => logout()}
                    className="specimen-tag text-paper-muted hover:text-paper transition-colors duration-200"
                  >
                    Logout
                  </button>
                </div>
                {user.avatar_url && (
                  <img
                    src={user.avatar_url}
                    alt={user.username ?? user.steam_id}
                    className="w-8 h-8 rounded-sm border border-border bg-stock"
                  />
                )}
              </div>
            ) : (
              <a href={getLoginUrl()} className="btn btn-primary">
                Authenticate
              </a>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
