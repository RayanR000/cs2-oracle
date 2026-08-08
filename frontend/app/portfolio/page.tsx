'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Header } from '@/components';
import StatCard from '@/components/StatCard';
import { useUser } from '@/lib/UserContext';
import { getInventory, getLoginUrl } from '@/lib/api';

interface InventoryItem {
  id: string;
  name: string;
  market_hash_name: string;
  quantity: number;
  current_price: number | null;
  image_url: string | null;
  type: string;
}

function formatCurrency(value: number) {
  return `$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export default function PortfolioPage() {
  const { user, loading: userLoading } = useUser();
  const [items, setItems] = useState<InventoryItem[]>([]);
  const [totalValue, setTotalValue] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchPortfolio() {
      if (userLoading) return;
      if (!user) { setLoading(false); return; }

      setLoading(true);
      try {
        const data = await getInventory();
        if (data.error) {
          if (data.error === 'unauthorized') {
            setError('Please sign in to view your portfolio');
          } else {
            setError('Make sure your Steam profile and inventory are set to public.');
          }
        } else {
          setItems(data.items || []);
          setTotalValue(data.total_value || 0);
        }
      } catch {
        setError('An unexpected error occurred');
      } finally {
        setLoading(false);
      }
    }

    fetchPortfolio();
  }, [user, userLoading]);

  if (userLoading || loading) {
    return (
      <div className="min-h-screen bg-ground">
        <Header />
        <div className="flex flex-col items-center justify-center h-[60vh] gap-4">
          <div className="h-0.5 w-48 bg-ink/40 rounded-full overflow-hidden">
            <div className="h-full w-1/3 bg-ink animate-pulse" />
          </div>
          <p className="text-sm text-paper-secondary">Opening your collection case...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="min-h-screen bg-ground">
        <Header />
        <div className="max-w-4xl mx-auto px-6 py-20 text-center">
          <span className="specimen-tag text-paper-tertiary mb-4 block">Your Collection Case</span>
          <h1 className="text-headline text-paper mb-4">Portfolio</h1>
          <p className="text-base text-paper-secondary max-w-2xl mx-auto leading-relaxed mb-10">
            Sign in with your Steam account to open your collection case — every inventory
            item becomes a specimen, priced against the archive.
          </p>
          <a href={getLoginUrl()} className="btn btn-primary">
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
              <path d="M11.979 0C5.678 0 .511 4.86.022 11.037l6.432 2.654c.545-.371 1.203-.59 1.912-.59.063 0 .125.004.188.006l2.83-4.146V8.92c0-2.607 2.113-4.72 4.72-4.72 2.607 0 4.72 2.113 4.72 4.72 0 2.607-2.113 4.72-4.72 4.72-.173 0-.341-.013-.506-.035l-4.14 2.831c.002.063.006.125.006.188 0 2.114-1.714 3.828-3.828 3.828-1.55 0-2.891-.918-3.504-2.236L0 15.352c.866 4.887 5.152 8.648 10.285 8.648 5.756 0 10.422-4.666 10.422-10.422C20.707 7.822 16.784 3.322 11.979 0zm2.741 12.01c-1.706 0-3.091-1.385-3.091-3.09 0-1.706 1.385-3.091 3.091-3.091 1.706 0 3.091 1.385 3.091 3.091 0 1.705-1.385 3.09-3.091 3.09zm-3.091-3.09c0 .416.084.81.233 1.168l-2.73 3.999c-.198-.016-.399-.026-.603-.026-1.127 0-2.146.486-2.854 1.261l-5.32-2.193c.312-4.143 3.49-7.447 7.554-8.156.002.016.006.033.006.05v.001zM10.285 17.548c0 1.312-1.063 2.375-2.375 2.375-1.312 0-2.375-1.063-2.375-2.375s1.063-2.375 2.375-2.375c.063 0 .125.004.188.006l2.193-3.193v.001c.416.486 1.035.789 1.724.789h.001c-.149-.358-.233-.752-.233-1.168s.084-.81.233-1.168h-.001c-.689 0-1.308.303-1.724.789l-2.193-3.193c-.063.002-.125.006-.188.006z" />
            </svg>
            Sign in with Steam
          </a>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-ground">
        <Header />
        <div className="max-w-4xl mx-auto px-6 py-20 text-center">
          <h1 className="text-title text-paper mb-4">Something went wrong</h1>
          <p className="mb-8 text-paper-secondary">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="btn btn-primary"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  const totalQuantity = items.reduce((sum, item) => sum + item.quantity, 0);

  return (
    <div className="min-h-screen bg-ground">
      <Header />

      <div className="max-w-6xl mx-auto px-6 py-10">
        <div className="mb-8 flex flex-col md:flex-row md:items-end md:justify-between gap-4">
          <div>
            <span className="specimen-tag text-paper-tertiary mb-3 block">Your Collection Case</span>
            <h1 className="text-headline text-paper mb-2">Portfolio</h1>
            <div className="flex items-center gap-3 text-sm text-paper-secondary">
              <span className="px-2 py-0.5 rounded-xs bg-recess font-data text-xs">
                Steam: {user.steam_id}
              </span>
              <span className="text-xs">{items.length} unique specimens</span>
            </div>
          </div>
          <button
            onClick={() => window.location.reload()}
            className="btn btn-ghost"
          >
            Refresh Inventory
          </button>
        </div>

        {/* Ledger stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-10">
          <StatCard
            label="Estimated Value"
            value={formatCurrency(totalValue)}
            annotation="TOTAL"
          />
          <StatCard
            label="Total Items"
            value={totalQuantity.toLocaleString()}
            annotation="QTY"
          />
          <StatCard
            label="Unique Items"
            value={items.length.toLocaleString()}
            annotation="SKUS"
          />
        </div>

        {/* Inventory ledger */}
        <div className="rounded-sm border border-border bg-stock">
          <div className="overflow-x-auto rounded-sm">
            <table className="w-full text-sm min-w-[680px]">
              <thead>
                <tr className="bg-recess border-b border-border">
                  <th className="px-5 py-4 text-left specimen-tag text-paper-secondary">Item</th>
                  <th className="px-5 py-4 text-left specimen-tag text-paper-secondary">Type</th>
                  <th className="px-5 py-4 text-right specimen-tag text-paper-secondary">Qty</th>
                  <th className="px-5 py-4 text-right specimen-tag text-paper-secondary">Current Price</th>
                  <th className="px-5 py-4 text-right specimen-tag text-paper-secondary">Total Value</th>
                  <th className="px-5 py-4 text-center specimen-tag text-paper-secondary">Market</th>
                </tr>
              </thead>
              <tbody>
                {items.map((item) => {
                  const totalItemValue = item.current_price ? (item.current_price * item.quantity) : 0;
                  return (
                    <tr key={item.id} className="ledger-row">
                      <td className="px-5 py-4">
                        <div className="flex items-center gap-3">
                          {item.image_url ? (
                            <img src={item.image_url} alt={item.name} className="w-10 h-8 object-contain rounded-xs" loading="lazy" />
                          ) : (
                            <div className="w-10 h-8 rounded-xs bg-recess" />
                          )}
                          <span className="font-medium text-paper">{item.name}</span>
                        </div>
                      </td>
                      <td className="px-5 py-4 text-left specimen-tag text-paper-tertiary">
                        {item.type}
                      </td>
                      <td className="px-5 py-4 text-right font-data text-paper-secondary">
                        {item.quantity}
                      </td>
                      <td className="px-5 py-4 text-right font-data text-paper">
                        {item.current_price ? formatCurrency(item.current_price) : 'N/A'}
                      </td>
                      <td className="px-5 py-4 text-right font-data font-medium text-paper">
                        {totalItemValue > 0 ? formatCurrency(totalItemValue) : 'N/A'}
                      </td>
                      <td className="px-5 py-4 text-center">
                        <Link
                          href={`/market?q=${encodeURIComponent(item.market_hash_name)}`}
                          className="specimen-tag text-ink hover:text-ink-hover transition-colors duration-200"
                        >
                          Analyze &rarr;
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Empty state */}
        {items.length === 0 && (
          <div className="text-center py-16">
            <div className="w-14 h-14 rounded-xs border border-border bg-stock mx-auto mb-6 flex items-center justify-center">
              <svg className="w-6 h-6 text-paper-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
              </svg>
            </div>
            <p className="mb-4 text-base text-paper-secondary">No specimens found in your public CS2 inventory</p>
            <p className="text-sm mb-6 max-w-md mx-auto text-paper-tertiary">
              Make sure your Steam profile and inventory are set to &ldquo;Public&rdquo; in your privacy settings.
            </p>
            <Link href="/market" className="btn btn-secondary">
              Browse Market &rarr;
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}
