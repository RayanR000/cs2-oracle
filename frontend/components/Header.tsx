'use client';

import Link from 'next/link';
import { useUser } from '@/lib/UserContext';
import { useTheme } from '@/lib/ThemeContext';
import { getLoginUrl } from '@/lib/api';

export default function Header() {
  const { user, loading, logout } = useUser();
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="sticky top-0 z-50 bg-background-primary/90 backdrop-blur-md border-b border-border">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex justify-between items-center">
          <Link href="/" className="flex items-center gap-4 group">
            <div className="w-8 h-8 rounded-sm border border-border flex items-center justify-center bg-background-secondary transition-all duration-300 group-hover:border-accent-primary">
              <span className="font-data font-bold text-[10px] text-primary tracking-tighter">CS</span>
            </div>
            <div className="flex flex-col">
              <span className="font-semibold text-sm leading-none tracking-tight text-primary">
                DATA TERMINAL
              </span>
              <span className="font-data text-[9px] text-muted tracking-[0.2em] uppercase mt-1">
                Analytical Report
              </span>
            </div>
          </Link>

          <nav className="hidden md:flex gap-10">
            <NavLink href="/market" label="MARKET" />
            <NavLink href="/portfolio" label="PORTFOLIO" />
            <NavLink href="/accuracy" label="ACCURACY" />
          </nav>

          <div className="flex items-center gap-5">
            <button
              onClick={toggleTheme}
              className="w-8 h-8 rounded-sm border border-border flex items-center justify-center bg-background-secondary hover:border-accent-primary hover:bg-surface transition-all duration-200 text-text-secondary hover:text-text-primary"
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
              <div className="w-8 h-8 rounded-full animate-pulse bg-surface" />
            ) : user ? (
              <div className="flex items-center gap-4">
                <div className="flex flex-col items-end hidden sm:flex">
                  <span className="text-[11px] font-bold text-primary tracking-tight">{user.username}</span>
                  <button
                    onClick={() => logout()}
                    className="text-[9px] text-muted hover:text-primary transition-colors uppercase tracking-[0.2em]"
                  >
                    Logout
                  </button>
                </div>
                {user.avatar_url && (
                  <img
                    src={user.avatar_url}
                    alt={user.username}
                    className="w-8 h-8 rounded-sm border border-border grayscale hover:grayscale-0 transition-all duration-300"
                  />
                )}
              </div>
            ) : (
              <a
                href={getLoginUrl()}
                className="flex items-center gap-3 px-5 py-2 rounded-sm text-[11px] font-bold uppercase tracking-widest transition-all duration-200 bg-accent text-background-primary hover:bg-brand-hover"
              >
                AUTHENTICATE
              </a>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}

function NavLink({ href, label }: { href: string; label: string }) {
  return (
    <Link
      href={href}
      className="relative text-[11px] font-bold uppercase tracking-[0.2em] transition-all duration-300 group text-secondary hover:text-primary py-1"
    >
      {label}
      <span className="absolute -bottom-0.5 left-0 h-[1.5px] w-0 transition-all duration-300 ease-out group-hover:w-full bg-accent-primary" />
    </Link>
  );
}
