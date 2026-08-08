'use client';

import { useEffect, useRef, useState } from 'react';

export interface SearchOption {
  item_id: string;
  name: string;
  type: string;
  icon_url: string | null;
  latest_price?: number;
}

interface SearchProps {
  placeholder?: string;
  fetchOptions: (query: string) => Promise<SearchOption[]>;
  onSelect: (option: SearchOption) => void;
  errorMessage?: string;
}

export default function Search({
  placeholder = 'Search the collection...',
  fetchOptions,
  onSelect,
  errorMessage = 'Search is unavailable right now \u2014 the market API is not responding.',
}: SearchProps) {
  const [query, setQuery] = useState('');
  const [options, setOptions] = useState<SearchOption[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const containerRef = useRef<HTMLDivElement>(null);
  const listboxRef = useRef<HTMLDivElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const seqRef = useRef(0);

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  useEffect(() => {
    const close = () => {
      setOpen(false);
      setActiveIndex(-1);
    };
    function handlePointerDown(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) close();
    }
    document.addEventListener('mousedown', handlePointerDown);
    window.addEventListener('scroll', close, { passive: true });
    return () => {
      document.removeEventListener('mousedown', handlePointerDown);
      window.removeEventListener('scroll', close);
    };
  }, []);

  useEffect(() => {
    const el = listboxRef.current?.querySelector(`[data-index="${activeIndex}"]`);
    el?.scrollIntoView({ block: 'nearest' });
  }, [activeIndex]);

  const handleChange = (value: string) => {
    setQuery(value);
    setActiveIndex(-1);
    if (debounceRef.current) clearTimeout(debounceRef.current);

    const trimmed = value.trim();
    if (trimmed.length < 2) {
      setOptions([]);
      setHasError(false);
      setOpen(false);
      return;
    }

    debounceRef.current = setTimeout(async () => {
      const seq = ++seqRef.current;
      setIsSearching(true);
      setHasError(false);
      try {
        const results = await fetchOptions(trimmed);
        if (seq !== seqRef.current) return;
        setOptions(Array.isArray(results) ? results.slice(0, 8) : []);
        setOpen(true);
      } catch {
        if (seq !== seqRef.current) return;
        setOptions([]);
        setHasError(true);
        setOpen(true);
      } finally {
        if (seq === seqRef.current) setIsSearching(false);
      }
    }, 250);
  };

  const selectOption = (option: SearchOption) => {
    onSelect(option);
    setOpen(false);
    setActiveIndex(-1);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Escape') {
      setOpen(false);
      setActiveIndex(-1);
      return;
    }
    if (!open || options.length === 0) return;

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIndex((i) => (i + 1) % options.length);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIndex((i) => (i <= 0 ? options.length - 1 : i - 1));
    } else if (e.key === 'Enter') {
      const target = activeIndex >= 0 ? options[activeIndex] : options[0];
      if (target) {
        e.preventDefault();
        selectOption(target);
      }
    }
  };

  return (
    <div ref={containerRef} className="relative">
      <div className="relative group flex items-center bg-stock border border-border rounded-sm transition-colors duration-200 focus-within:border-border-accent focus-within:bg-surface">
        <svg
          className="w-4 h-4 ml-4 text-paper-muted transition-colors duration-200 shrink-0 group-focus-within:text-paper"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          aria-hidden
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <input
          type="text"
          role="combobox"
          aria-expanded={open}
          aria-controls="home-search-listbox"
          aria-activedescendant={activeIndex >= 0 ? `search-option-${activeIndex}` : undefined}
          aria-autocomplete="list"
          value={query}
          placeholder={placeholder}
          onChange={(e) => handleChange(e.target.value)}
          onFocus={() => options.length > 0 && setOpen(true)}
          onKeyDown={handleKeyDown}
          className="input-search pl-11 bg-transparent"
        />
      </div>

      {open && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-stock border border-border rounded-sm z-[var(--z-dropdown)]">
          {isSearching ? (
            <div className="px-5 py-8 text-center">
              <span className="specimen-tag text-paper-muted">Searching...</span>
            </div>
          ) : hasError ? (
            <div className="px-5 py-8 text-center">
              <p className="text-sm text-paper-secondary">{errorMessage}</p>
            </div>
          ) : options.length > 0 ? (
            <div ref={listboxRef} role="listbox" id="home-search-listbox" className="py-1 max-h-80 overflow-y-auto">
              {options.map((item, i) => (
                <button
                  key={item.item_id}
                  type="button"
                  role="option"
                  id={`search-option-${i}`}
                  data-index={i}
                  aria-selected={i === activeIndex}
                  onMouseEnter={() => setActiveIndex(i)}
                  onClick={() => selectOption(item)}
                  className={`w-full flex items-center gap-4 px-4 py-3 text-left transition-colors ${
                    i === activeIndex ? 'bg-surface' : ''
                  }`}
                >
                  {item.icon_url ? (
                    <img
                      src={item.icon_url}
                      alt={item.name}
                      className="w-8 h-8 rounded-xs object-cover shrink-0"
                      loading="lazy"
                    />
                  ) : (
                    <div className="w-8 h-8 rounded-xs bg-recess shrink-0" />
                  )}
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-medium text-paper truncate">{item.name}</div>
                    <div className="specimen-tag text-paper-tertiary">{item.type}</div>
                  </div>
                  {item.latest_price != null && (
                    <span className="font-data text-sm text-paper shrink-0">${item.latest_price.toFixed(2)}</span>
                  )}
                </button>
              ))}
            </div>
          ) : (
            <div className="px-5 py-8 text-center">
              <p className="text-sm text-paper-secondary">No items match &ldquo;{query}&rdquo;</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
